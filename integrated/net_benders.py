"""Benders decomposition with column generation for the integrated model of the paper, one cluster at a time.

Master:     wages + overtime + Theta, service limits on the soft constraints, stops y and the cuts.
Subproblem: the routing LP over the open stops (no seat limit, lowest wage per stop, continuous variables), solved by
            column generation. Its duals, lifted to the closed stops, give a cut valid for every schedule.
Bounds:     UB = lowest cost of a real plan, starting from the sequential schedule; LB = best bound of the master.
Re-timing:  searches schedules near the best one with the trips found so far, and keeps a move only if the real cost falls.
A schedule without buses in planning is solved and reported as a baseline. It is not a start of the search, and its time is
not counted in benders_seconds.

Usage: python net_benders.py <data.xlsx> <out_prefix> --cluster <id prefix> [--master-times 1200,1200 --w-bus 1 --m-scale 1 --seq-only]
"""
import argparse
import io
import json
import math
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bus_scheduling_routing"))
import instance as I
import net_routing as N
from bus_riders import branch_cluster
from model import _solve, shift_events
from solver_cplex_new import StaffSchedulingSolver

WEEK_HOURS = 168


def set_start(m, start):
    """The only start of the next solve: the engine otherwise keeps the old ones, which a new cut can make infeasible, and ignores this one."""
    m.clear_mip_starts()
    m.get_cplex().MIP_starts.delete()
    m.add_mip_start(start, write_level=1)


def staff_args():
    saved, sys.argv = sys.argv, [sys.argv[0]]
    from args import get_args
    a = get_args()
    sys.argv = saved
    return a


class StaffModel:
    """The staff model of the paper with its cost terms and the slack variables of each soft-constraint level."""

    def __init__(self, args, data):
        self.solver = StaffSchedulingSolver(args, data)
        with redirect_stdout(io.StringIO()):
            self.solver.build_model()
        s, d = self.solver, data
        m = self.mdl = s.mdl
        self.j_reg = m.sum(d.wage_reg[i] * sh["duration"] * s.x[i, sh["id"]] for i in d.staff_ids for sh in d.shifts)
        self.j_over = m.sum(d.wage_ot[i] * s.o_daily[i, t] for i in d.staff_ids for t in d.days)
        expat, mgr = set(d.non_saudi_staff_ids), set(d.manager_ids)
        saudi = set(d.staff_ids) - expat
        crew = set(d.staff_ids) - mgr
        sids = [sh["id"] for sh in d.shifts]
        bds = [(b, t) for b in d.branch_operation_limits for t in d.days]
        self.levels = {
            1: m.sum(s.sl[k] for k in s.active_bt) + m.sum(s.v_idle[i] for i in d.staff_ids),
            2: m.sum(s.v_max_daily[i, t] for i in saudi & crew for t in d.days)
               + m.sum(s.v_min_onsite_not_new[k] for k in s.active_bt),
            3: m.sum(s.v_min_work_days[i] for i in d.staff_ids)
               + m.sum(s.v_max_daily[i, t] for i in mgr | (expat & crew) for t in d.days)
               + m.sum(s.v_max_shift[i, j] for i in (saudi & crew) | (expat & mgr) | (expat & crew) for j in sids)
               + m.sum(s.v_min_onsite_all[k] + s.v_min_onsite_mgr[k] for k in s.active_bt)
               + m.sum(s.sl_mgr_first[k] + s.sl_mgr_last[k] for k in bds),
            4: m.sum(s.v_min_onsite_not_mgr[k] for k in s.active_bt),
        }

    def value_of(self, x_on, threads=0):
        """Wages, overtime and the violation of each level of a fixed schedule, from the objective J of the staff model."""
        for key, var in self.solver.x.items():
            var.lb = var.ub = 1 if key in x_on else 0
        sol, _info = _solve(self.mdl, 600, 1e-9, None, threads)
        if not sol:
            raise RuntimeError("the schedule breaks a critical constraint")
        out = dict(j_reg=sol.get_value(self.j_reg), j_over=sol.get_value(self.j_over),
                   levels={p: sol.get_value(e) for p, e in self.levels.items()})
        for var in self.solver.x.values():
            var.lb, var.ub = 0, 1
        return out


class ClusterProblem:
    def __init__(self, cluster, data_file, args, threads=0, w_bus=1.0):
        self.cluster, self.threads, self.w_bus = cluster, threads, w_bus
        self.closed_price = "lift"  # "zero" only for the lifting ablation
        args.data_file = str(data_file)
        args.w_bus_trip = w_bus
        branches = sorted(b for b, c in branch_cluster(data_file).items() if c == cluster)
        self.data = I.BranchSubsetLoader(args, branches).load_and_preprocess()
        self.args = args
        self.net = N.Network(cluster, data_file)
        self.shift = {s["id"]: s for s in self.data.shifts}
        self.starts, self.ends = shift_events(self.data)

    def events(self, x_on):
        return [(i, self.shift[s]["start"], self.shift[s]["end"]) for i, s in x_on]

    def possible_stops(self):
        return [(b, a, t) for (b, a) in self.net.riders for t in range(WEEK_HOURS) if self.starts[t] or self.ends[t]]

    def add_stops(self, sm):
        """y and its link to the riders, as in the master."""
        m, x, y = sm.mdl, sm.solver.x, {}
        for (b, a, t) in self.possible_stops():
            riders = self.net.riders[b, a]
            load = m.sum(x[i, s] for i in riders for s in self.starts[t] + self.ends[t])
            y[b, a, t] = m.binary_var(name=f"y_{b}_{a[:6]}_{t}")
            m.add_constraint(y[b, a, t] <= load)
            m.add_constraint(load <= len(riders) * y[b, a, t])
        return y

    def low_cost(self, a, t, tb):
        return np.array([self.net.low_wage[b, a] for b in tb.branches])

    def cut(self, x_on, log=None):
        """Routing LP at the stops of x_on, with prices lifted to the closed stops. Returns the prices and the LP value."""
        net = self.net
        _q, _p, e = net.demand(self.events(x_on))
        open_stops = sorted(e)
        if not open_stops:
            return {}, 0.0, 0
        _master, value, pi, rho, mu, iters = N.column_generation(net, open_stops, {s: 1 for s in open_stops},
                                                                 self.low_cost, threads=self.threads, log=log)
        prices = dict(pi)
        arrs = N.extend_duals(rho, mu, net)
        by_hour = {}
        for b, a, t in self.possible_stops():
            by_hour.setdefault((a, t), []).append(b)
        for (a, t), bs in by_hour.items():
            closed = [b for b in sorted(bs) if (b, a, t) not in prices]
            if not closed:
                continue
            g = {}
            for k in range(len(net.buses)):
                tb = net.table(a, k)
                base = tb.drive_entry + tb.D @ self.low_cost(a, t, tb)
                dep = (60 * t + tb.delta) % N.WEEK
                g[k] = np.minimum.reduceat(base - N.rho_of(arrs[k, a], mu[k], dep, tb.dur), tb.start)
            priced = {b for b in bs if (b, a, t) in prices}
            for b in closed:
                best = np.inf
                for k in range(len(net.buses)):
                    tb = net.table(a, k)
                    known = np.array([x in priced for x in tb.branches])
                    others = np.array([prices.get((x, a, t), 0.0) for x in tb.branches])
                    keep = tb.mem[:, tb.col[b]] & ~(tb.mem & ~known & (np.array(tb.branches) != b)).any(axis=1)
                    if keep.any():
                        best = min(best, float((g[k] - tb.mem @ others)[keep].min()))
                prices[b, a, t] = best
                priced.add(b)
        if self.closed_price == "zero":
            # ablation: no lifting. A lower price keeps every dual constraint, so min(0, lifted) is still a valid cut,
            # but the master learns nothing about the cost of opening a new stop
            for s in prices:
                if s not in pi:
                    prices[s] = min(0.0, prices[s])
        # the cut must equal the LP value at its own stops
        check = sum(pi[s] for s in open_stops)
        assert abs(check - value) <= 1e-5 * max(1.0, abs(value)), (check, value)
        return prices, value, iters

    def sequential(self, time_limit=1200, gap=1e-3):
        """Step 1: the staff model min J of current practice, with its trip consolidation term."""
        self.args.w_bus_trip = self.w_bus
        sm = StaffModel(self.args, self.data)
        sol, info = _solve(sm.mdl, time_limit, gap, None, self.threads)
        return {key for key, var in sm.solver.x.items() if sol.get_value(var) > 0.5}, info

    def uncoordinated(self, limits, x0, time_limit=900, gap=1e-3):
        """Baseline without buses: lowest wages and overtime under the same service limits, then routing."""
        self.args.w_bus_trip = 0.0
        sm = StaffModel(self.args, self.data)
        for p, expr in sm.levels.items():
            sm.mdl.add_constraint(expr <= limits[p] + 1e-6)
        sm.mdl.minimize(sm.j_reg + sm.j_over)
        start = sm.mdl.new_solution()
        for key in x0:
            start.add_var_value(sm.solver.x[key], 1)
        sm.mdl.add_mip_start(start, write_level=1)
        sol, info = _solve(sm.mdl, time_limit, gap, None, self.threads)
        return {key for key, var in sm.solver.x.items() if sol.get_value(var) > 0.5}, info

    def real_wages(self, x_on):
        """Wages and overtime recomputed from the schedule alone, as an independent check of Z."""
        d, reg, hours = self.data, 0.0, {}
        for i, s in x_on:
            sh = self.shift[s]
            reg += d.wage_reg[i] * sh["duration"]
            hours[i, sh["day"]] = hours.get((i, sh["day"]), 0) + sh["duration"]
        expat, mgr = set(d.non_saudi_staff_ids), set(d.manager_ids)
        over = sum(d.wage_ot[i] * max(0, h - (9 if i in expat else 6)) for (i, _d), h in hours.items() if i not in mgr)
        return reg, over

    def plan_cost(self, sm, x_on):
        vals = sm.value_of(x_on, self.threads)
        plan = N.route_schedule(self.net, self.events(x_on), self.threads)
        return vals, plan, vals["j_reg"] + vals["j_over"] + plan.cost

    def benders(self, x0, master_times=(1200, 1200, 1200, 1200, 1200), master_gap=1e-3, target_gap=0.002, retime_rounds=1,
                retime_time=1200, bound_time=0, log=print):
        t_all = time.time()
        args = self.args
        args.w_bus_trip = self.w_bus
        evaluator = StaffModel(args, self.data)
        vals0, plan0, z0 = self.plan_cost(evaluator, x0)
        limits = vals0["levels"]
        log(f"sequential: Z {z0:,.2f}  wages {vals0['j_reg']:,.2f}  overtime {vals0['j_over']:,.2f}  transport {plan0.cost:,.2f}  "
            f"route gap {(plan0.cost - plan0.lp) / max(plan0.cost, 1e-9):.4%}  limits {limits}")
        x_u, info_u = self.uncoordinated(limits, x0)
        vals_u, plan_u, z_u = self.plan_cost(evaluator, x_u)
        log(f"no buses in planning: Z {z_u:,.2f}  wages {vals_u['j_reg']:,.2f}  overtime {vals_u['j_over']:,.2f}  transport {plan_u.cost:,.2f}  {info_u['status']}")
        args.w_bus_trip = 0.0
        sm = StaffModel(args, self.data)
        m, x = sm.mdl, sm.solver.x
        for p, expr in sm.levels.items():
            m.add_constraint(expr <= limits[p] + 1e-6, ctname=f"service_{p}")
        y = self.add_stops(sm)
        theta = m.continuous_var(lb=0, name="theta")
        m.minimize(sm.j_reg + sm.j_over + theta)
        best = dict(z=z0, x=set(x0), vals=vals0, plan=plan0, round=-1)
        lower, history, x_on, cuts, visited = -math.inf, [], set(x0), [], [(set(x0), plan0.cost), (set(x_u), plan_u.cost)]
        seen = set()
        t0 = time.time()
        def start_at(xs):
            start = m.new_solution()
            for key in xs:
                start.add_var_value(x[key], 1)
            open_now = self.net.demand(self.events(xs))[2]
            for s, var in y.items():
                start.add_var_value(var, 1 if s in open_now else 0)
            start.add_var_value(theta, max([sum(pr for s, pr in c.items() if s in open_now) for c in cuts], default=0.0) + 1e-6)
            set_start(m, start)

        for i, limit in enumerate(master_times):
            start_at(x_on)
            sol, info = _solve(m, limit, master_gap, None, self.threads)
            if not sol:
                log(f"round {i}: master found no solution ({info['status']})")
                break
            lower = max(lower, info["best_bound"])
            x_new = {key for key, var in x.items() if sol.get_value(var) > 0.5}
            stops = frozenset(self.net.demand(self.events(x_new))[2])
            repeat = stops in seen  # the same stops give the same cut
            seen.add(stops)
            value, iters = None, 0
            if not repeat:
                prices, value, iters = self.cut(x_new)
                cuts.append(prices)
                m.add_constraint(theta >= m.sum(pr * y[s] for s, pr in prices.items() if s in y), ctname=f"cut_{len(cuts)}")
            vals, plan, z = self.plan_cost(evaluator, x_new)
            visited.append((set(x_new), plan.cost))
            if z < best["z"]:
                best = dict(z=z, x=set(x_new), vals=vals, plan=plan, round=len(history))
            gap = (best["z"] - lower) / best["z"]
            history.append(dict(round=len(history), lb=lower, ub=z, best=best["z"], gap=gap, master=sol.objective_value,
                                master_gap=info["mip_gap"], theta=sol.get_value(theta), lp_at_cut=value, cg_iterations=iters,
                                transport=plan.cost, route_gap=(plan.cost - plan.lp) / max(plan.cost, 1e-9), repeat=repeat,
                                master_status=info["status"], seconds=time.time() - t0))
            log(f"round {len(history) - 1}: LB {lower:,.2f}  master {sol.objective_value:,.2f}  UB {z:,.2f}  best {best['z']:,.2f}  gap {gap:.3%}  "
                f"theta {history[-1]['theta']:,.1f} vs transport {plan.cost:,.1f}  {info['status']}  {time.time() - t0:.0f}s")
            x_on = x_new
            if gap <= target_gap or repeat:
                break
        if bound_time > 0 and history and history[-1]["gap"] > target_gap:
            # bound phase: once the cuts stop changing, one longer solve of the same master that only tries to raise the lower bound
            start_at(best["x"])
            m.parameters.emphasis.mip = 3
            sol, info = _solve(m, bound_time, master_gap, None, self.threads)
            m.parameters.emphasis.mip = 0
            lower = max(lower, info["best_bound"])
            history.append(dict(round="bound", lb=lower, ub=None, best=best["z"], gap=(best["z"] - lower) / best["z"], master_gap=info["mip_gap"],
                                master_status=info["status"], seconds=time.time() - t0))
            log(f"bound phase: LB {lower:,.2f}  best {best['z']:,.2f}  gap {history[-1]['gap']:.3%}  {info['status']}  {time.time() - t0:.0f}s")
        benders_z, benders_time = best["z"], time.time() - t0
        best = self.retime(best, limits, evaluator, retime_rounds, retime_time, log)
        visited.append((set(best["x"]), best["plan"].cost))
        gap = (best["z"] - lower) / best["z"]
        # every cut must lie below the real transport cost of every plan seen so far
        worst = 0.0
        for xs, cost in visited:
            ys = self.net.demand(self.events(xs))[2]
            for pr in cuts:
                worst = max(worst, sum(pr.get(s, 0.0) for s in ys) - cost)
        reg, over = self.real_wages(best["x"])
        check = reg + over + best["plan"].cost - best["z"]
        log(f"checks: largest cut violation {worst:.6f}; Z recomputed from the schedule differs by {check:.6f}")
        return dict(cluster=self.cluster[:8], z_seq=z0, seq_wages=vals0["j_reg"], seq_overtime=vals0["j_over"], seq=_parts(plan0), seq_levels=limits,
                    z_unc=z_u, unc_wages=vals_u["j_reg"], unc_overtime=vals_u["j_over"], unc=_parts(plan_u), unc_levels=vals_u["levels"],
                    z=best["z"], z_benders=benders_z, wages=best["vals"]["j_reg"], overtime=best["vals"]["j_over"],
                    plan=_parts(best["plan"]), lb=lower, gap=gap, saving=(z0 - best["z"]) / z0,
                    saving_vs_best_baseline=(min(z0, z_u) - best["z"]) / min(z0, z_u), limits=limits,
                    levels=best["vals"]["levels"], rounds=len(history), benders_seconds=benders_time, best_round=str(best["round"]),
                    cut_violation=worst, z_check=check, seconds=time.time() - t_all, history=history,
                    cuts=[[[*s, pr] for s, pr in c.items()] for c in cuts]), best["x"]

    def retime(self, best, limits, evaluator, rounds, time_limit, log):
        """Search near the best schedule with the current trips as columns: each shift stays, moves its start by one hour
        or changes its length by one hour. Trip costs are those at the current riders. A move is kept only if the real cost falls."""
        for k in range(rounds):
            self.args.w_bus_trip = 0.0
            sm = StaffModel(self.args, self.data)
            m, x = sm.mdl, sm.solver.x
            for p, expr in sm.levels.items():
                m.add_constraint(expr <= limits[p] + 1e-6)
            y = self.add_stops(sm)
            for var in x.values():
                var.ub = 0
            for i, s in best["x"]:
                c = self.shift[s]
                near = [x[i, o["id"]] for o in self.data.shifts
                        if (o["start"] == c["start"] and abs(o["duration"] - c["duration"]) <= 1)
                        or (o["duration"] == c["duration"] and abs(o["start"] - c["start"]) <= 1)]
                for var in near:
                    var.ub = 1
                m.add_constraint(m.sum(near) == 1)
            cost = self._add_transport(m, x, y, best)
            m.minimize(sm.j_reg + sm.j_over + cost)
            start = m.new_solution()
            for key in best["x"]:
                start.add_var_value(x[key], 1)
            m.add_mip_start(start, write_level=1)
            sol, info = _solve(m, time_limit, 1e-4, None, self.threads)
            if not sol:
                break
            new = {key for key, var in x.items() if sol.get_value(var) > 0.5}
            vals, plan, z = self.plan_cost(evaluator, new)
            log(f"re-timing {k}: estimate {sol.objective_value:,.2f}  real {z:,.2f}  best {best['z']:,.2f}  moved {len(new ^ best['x']) // 2}  {info['status']}")
            if z >= best["z"] - 1e-6:
                break
            best = dict(z=z, x=new, vals=vals, plan=plan, round=f"retime{k}")
        return best

    def _add_transport(self, m, x, y, best):
        """The routing model over the trips of the best plan and a direct trip for every stop."""
        net = self.net
        trips = set(best["plan"].trips)
        for s in y:
            trips.update(N.direct_trips(net, [s], None))
        lam = {h: m.binary_var() for h in trips}
        for s, var in y.items():
            m.add_constraint(m.sum(lam[h] for h in trips if s in h.stops) == var)
        for h in trips:
            if len(h.seq) > 1:
                big = sum(len(net.riders[b, h.acc]) for b in h.seq)
                a, t = h.acc, h.t
                drop = {b: m.sum(x[i, s] for i in net.riders[b, a] for s in self.starts[t]) for b in h.seq}
                pick = {b: m.sum(x[i, s] for i in net.riders[b, a] for s in self.ends[t]) for b in h.seq}
                for j in range(len(h.seq) + 1):
                    m.add_constraint(m.sum(drop[b] for b in h.seq[j:]) + m.sum(pick[b] for b in h.seq[:j])
                                     <= net.buses[h.k].capacity + big * (1 - lam[h]))
        # flow of buses over the event minutes of each bus type and accommodation
        n = {k: m.integer_var(lb=0) for k in range(len(net.buses))}
        cost = m.sum((h.drive + h.theta) * var for h, var in lam.items()) + m.sum(net.buses[k].fixed * n[k] for k in n)
        ev = {}
        arcs_in, arcs_out, at_start = {}, {}, {k: [] for k in n}
        for h, var in lam.items():
            ret = h.dep + h.dur
            ev.setdefault((h.k, h.acc), set()).update({h.dep, ret % N.WEEK})
            arcs_in.setdefault((h.k, h.acc, ret % N.WEEK), []).append(var)
            arcs_out.setdefault((h.k, h.acc, h.dep), []).append(var)
            if ret >= N.WEEK:
                at_start[h.k].append(var)
        empties = []
        if len(net.accs) > 1:
            for k, bus in enumerate(net.buses):
                for a in net.accs:
                    for a2 in net.accs:
                        if a == a2:
                            continue
                        dur = net.tau[a, a2]
                        for (kk, aa), mins in list(ev.items()):
                            if kk == k and aa == a:
                                for w in sorted(mins):
                                    var = m.continuous_var(lb=0)
                                    empties.append((bus.per_km * net.km[a, a2] + bus.per_min * dur, var))
                                    ret = w + dur
                                    ev.setdefault((k, a2), set()).add(ret % N.WEEK)
                                    arcs_out.setdefault((k, a, w), []).append(var)
                                    arcs_in.setdefault((k, a2, ret % N.WEEK), []).append(var)
                                    if ret >= N.WEEK:
                                        at_start[k].append(var)
        for key, mins in ev.items():
            mins = sorted(mins)
            for j, w in enumerate(mins):
                var = m.continuous_var(lb=0)
                nxt = mins[(j + 1) % len(mins)]
                arcs_out.setdefault((*key, w), []).append(var)
                arcs_in.setdefault((*key, nxt), []).append(var)
                if j == len(mins) - 1:
                    at_start[key[0]].append(var)
        for node in set(arcs_in) | set(arcs_out):
            m.add_constraint(m.sum(arcs_in.get(node, [])) == m.sum(arcs_out.get(node, [])))
        for k in n:
            m.add_constraint(m.sum(at_start[k]) <= n[k])
        return cost + m.sum(c * v for c, v in empties)


def _parts(plan):
    stops = [len(h.seq) for h in plan.trips]
    return dict(cost=plan.cost, lp=plan.lp, route_gap=(plan.cost - plan.lp) / max(plan.cost, 1e-9), driving=plan.driving,
                punctuality=plan.punctuality, empty=plan.empty, fleet=plan.fleet, buses=sum(plan.buses.values()),
                trips=len(stops), stops_per_trip=sum(stops) / max(1, len(stops)), max_stops=max(stops, default=0),
                empty_runs=float(sum(v for _g, v in plan.runs)))


def load_schedule(path, staff_ids):
    df = pd.read_csv(path)
    keep = set(staff_ids)
    return {(r.Staff, r.Shift) for r in df.itertuples() if r.Staff in keep}


def save_schedule(x_on, shift, path):
    rows = [dict(Staff=i, Shift=s, Day=shift[s].get("day"), Start=shift[s]["start"], Duration=shift[s]["duration"], End=shift[s]["end"])
            for i, s in sorted(x_on, key=str)]
    pd.DataFrame(rows).to_csv(path, index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("data")
    p.add_argument("out")
    p.add_argument("--cluster", required=True)
    p.add_argument("--threads", type=int, default=3)
    p.add_argument("--master-times", default="1200,1200,1200,1200,1200")
    p.add_argument("--retime-rounds", type=int, default=1)
    p.add_argument("--retime-time", type=float, default=1200)
    p.add_argument("--w-bus", type=float, default=1.0)
    p.add_argument("--seq-time", type=float, default=1200)
    p.add_argument("--seq-only", action="store_true")
    p.add_argument("--m-scale", type=float, default=1.0)
    p.add_argument("--bound-time", type=float, default=0, help="seconds of a final master solve with bound emphasis; 0 = none")
    p.add_argument("--closed-price", choices=["lift", "zero"], default="lift",
                   help="prices of closed stops in a cut: lifted (the method) or min(0, lifted) (ablation)")
    a = p.parse_args()
    cluster = next(c for c in set(branch_cluster(a.data).values()) if c.startswith(a.cluster))
    staff = staff_args()
    for name in ("M1", "M2", "M3", "M4"):
        setattr(staff, name, getattr(staff, name) * a.m_scale)
    prob = ClusterProblem(cluster, a.data, staff, a.threads, a.w_bus)
    prob.closed_price = a.closed_price
    seq_file = Path(f"{a.out}_x0.csv")
    if seq_file.exists():
        x0 = load_schedule(seq_file, prob.data.staff_ids)
    else:
        x0, seq_info = prob.sequential(a.seq_time)
        print("sequential staff model:", seq_info, flush=True)
        save_schedule(x0, prob.shift, seq_file)
    if a.seq_only:
        vals, plan, z = prob.plan_cost(StaffModel(prob.args, prob.data), x0)
        with open(f"{a.out}.json", "w") as f:
            json.dump(dict(cluster=cluster[:8], w_bus=a.w_bus, z=z, wages=vals["j_reg"], overtime=vals["j_over"], levels=vals["levels"], plan=_parts(plan)), f, indent=1, default=float)
        sys.exit(0)
    res, x = prob.benders(x0, [float(v) for v in a.master_times.split(",")], retime_rounds=a.retime_rounds, retime_time=a.retime_time,
                          bound_time=a.bound_time, log=lambda s: print(s, flush=True))
    with open(f"{a.out}.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    save_schedule(x, prob.shift, f"{a.out}_schedule.csv")
    print("DONE", {k: res[k] for k in ("z_seq", "z_unc", "z", "lb", "gap", "saving", "saving_vs_best_baseline", "seconds")}, flush=True)
