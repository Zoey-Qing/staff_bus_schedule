"""The surrogate-fit algorithm for staff scheduling and bus routing, one cluster at a time.

Upper level: min J_reg(x) + J_over(x) + V_hat(x) under the service limits and a trust region, with
             V_hat(x) = gamma + sum_w (pi_open_w y_w + pi_pass_w q_w).
Lower level: the routing model T_c (integrated/net_routing.py, imported unchanged) at the fixed schedule.
Feedback:    ridge regression of the observed cost on the features of the schedules seen so far.
The schedule with the lowest realized cost is kept. The clock starts once x0 is routed, as in Benders. The re-timing step of
Benders (imported unchanged) then runs on the best schedule.

Usage: python staff_surrogate.py --cluster <id prefix> --data <data.xlsx> --schedule <x0.csv> --time-limit <seconds> --out <file.json>
--penalties runs the penalty objective instead of the service limits, as a sensitivity check against E4.
--ablation static keeps the initial prices (no regression); --ablation lns drops the upper-level solve and keeps only the
neighbor moves around the best schedule; --ablation lns_fit does the same but refits the prices each round.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "integrated"))
sys.path.insert(0, str(ROOT / "staff_scheduling"))
sys.path.insert(0, str(ROOT / "bus_scheduling_routing"))
import instance as I  # noqa: E402
import net_benders as B  # noqa: E402
import net_routing as N  # noqa: E402
from bus_riders import branch_cluster  # noqa: E402
from model import _solve, shift_events  # noqa: E402
from solver_cplex_new import StaffSchedulingSolver  # noqa: E402

HOURS = 168
L1 = 0.1  # weight of the shift count in the staff model


def staff_args():
    saved, sys.argv = sys.argv, [sys.argv[0]]
    from args import get_args
    a = get_args()
    sys.argv = saved
    return a


class StaffModel:
    """The paper's staff model with its wage terms and the violation of each priority level as expressions."""

    def __init__(self, args, data):
        args.w_bus_trip = 0.0
        self.solver = StaffSchedulingSolver(args, data)
        with redirect_stdout(io.StringIO()):
            self.solver.build_model()
        s, d = self.solver, data
        m = self.mdl = s.mdl
        self.x = s.x
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
        self.M = {1: args.M1, 2: args.M2, 3: args.M3, 4: args.M4}
        self.penalty = m.sum(self.M[p] * e for p, e in self.levels.items())
        self.base = self.j_reg + self.j_over + self.penalty

    def value_of(self, x_on, threads=0):
        """Labor, penalty and violations of a fixed schedule."""
        for key, var in self.x.items():
            var.lb = var.ub = 1 if key in x_on else 0
        self.mdl.minimize(self.base)
        sol, _info = _solve(self.mdl, 600, 1e-9, None, threads)
        if not sol:
            raise RuntimeError("the schedule breaks a hard constraint")
        out = dict(labor=sol.get_value(self.j_reg) + sol.get_value(self.j_over), penalty=sol.get_value(self.penalty),
                   levels={p: sol.get_value(e) for p, e in self.levels.items()})
        for var in self.x.values():
            var.lb, var.ub = 0, 1
        return out


class Problem:
    def __init__(self, cluster, data_file, args, threads):
        self.threads = threads
        args.data_file = str(data_file)
        branches = sorted(b for b, c in branch_cluster(data_file).items() if c == cluster)
        self.data = I.BranchSubsetLoader(args, branches).load_and_preprocess()
        self.args = args
        self.net = N.Network(cluster, data_file)
        self.shift = {s["id"]: s for s in self.data.shifts}
        self.starts, self.ends = shift_events(self.data)
        self.evaluator = StaffModel(args, self.data)

    def events(self, x_on):
        return [(i, self.shift[s]["start"], self.shift[s]["end"]) for i, s in x_on]

    def evaluate(self, x_on):
        """The realized score of a schedule, from the lower level at this schedule."""
        vals = self.evaluator.value_of(x_on, self.threads)
        plan = N.route_schedule(self.net, self.events(x_on), self.threads)
        vals.update(v=plan.cost, theta=plan.punctuality, c_bus=plan.cost - plan.punctuality, trips=len(plan.trips),
                    buses=sum(plan.buses.values()), route_gap=(plan.cost - plan.lp) / max(plan.cost, 1e-9),
                    j_op=vals["labor"] + plan.cost, phi=vals["labor"] + vals["penalty"] + plan.cost)
        vals["objective"] = vals["phi"] + L1 * len(x_on)
        return vals


class Surrogate:
    def __init__(self, prob, nonneg, no_fixed, limits=None):
        self.prob = prob
        self.nonneg = nonneg
        sm = self.sm = StaffModel(prob.args, prob.data)
        m = self.m = sm.mdl
        self.x = sm.x
        # the same staff terms as the Benders limit run and penalty run
        self.goal = sm.j_reg + sm.j_over if limits else sm.base + L1 * m.sum(sm.x.values())
        if limits:
            for p, expr in sm.levels.items():
                m.add_constraint(expr <= limits[p] + 1e-6, ctname=f"service_{p}")
        net = prob.net
        self.y, self.q, self.keys = {}, {}, []
        for (b, a), riders in net.riders.items():
            for t in range(HOURS):
                for d, shifts in (("d", prob.starts[t]), ("p", prob.ends[t])):
                    if not shifts:
                        continue
                    g = (b, a, t, d)
                    load = m.sum(self.x[i, s] for i in riders for s in shifts)
                    self.y[g] = m.binary_var()
                    self.q[g] = load
                    m.add_constraint(self.y[g] <= load)
                    m.add_constraint(load <= len(riders) * self.y[g])
                    self.keys.append(g)
        self.index = {g: j for j, g in enumerate(self.keys)}
        n = len(self.keys)
        self.pi_open = np.zeros(n)
        for j, (b, a, _t, _d) in enumerate(self.keys):
            self.pi_open[j] = min((0.0 if no_fixed else bus.fixed) + bus.per_km * 2 * net.km[a, b] + bus.per_min * 2 * net.tau[a, b]
                                  for bus in net.buses)
        self.pi_pass = np.zeros(n)
        self.beta = 0.0
        self.A, self.v = [], []

    def pi(self):
        return np.r_[self.pi_open, self.pi_pass]

    def features(self, x_on):
        q, p, _e = self.prob.net.demand(self.prob.events(x_on))
        n = len(self.keys)
        fy, fq = np.zeros(n), np.zeros(n)
        for d, dd in (("d", q), ("p", p)):
            for (b, a, t), c in dd.items():
                j = self.index[b, a, t, d]
                fy[j], fq[j] = 1.0, c
        return np.r_[fy, fq]

    def fit(self, lam):
        """argmin sum (v_j - beta - a_j pi)^2 + lam |pi - pi_k|^2, in its dual form (fewer observations than coefficients)."""
        A, v = np.array(self.A), np.array(self.v)
        pi_k = self.pi()
        r = v - A @ pi_k
        Ac, rc = A - A.mean(axis=0), r - r.mean()
        pi = pi_k + Ac.T @ np.linalg.solve(Ac @ Ac.T + lam * np.eye(len(v)), rc)
        if self.nonneg:
            pi = np.maximum(pi, 0.0)
        n = len(self.keys)
        self.pi_open, self.pi_pass = pi[:n], pi[n:]
        self.beta = float((v - A @ pi).mean())

    def solve(self, center, radius, tl, gap, threads, noise=None, at_least=0, start=None):
        m, x = self.m, self.x
        a, b = self.pi_open, self.pi_pass
        if noise is not None:
            a, b = a * noise[0], b * noise[1]
        m.minimize(self.goal + m.sum(a[j] * self.y[g] + b[j] * self.q[g] for j, g in enumerate(self.keys)))
        cts = []
        if radius or at_least:
            dist = m.sum(1 - var if key in center else var for key, var in x.items())
            if radius:
                cts.append(m.add_constraint(dist <= 2 * radius))
            if at_least:
                cts.append(m.add_constraint(dist >= 2 * at_least))
        m.clear_mip_starts()
        m.get_cplex().MIP_starts.delete()  # the engine otherwise keeps old starts and ignores a new one
        if start is not None:
            s = m.new_solution()
            for key in start:
                s.add_var_value(x[key], 1)
            on = self.features(start)[:len(self.keys)]
            for j, g in enumerate(self.keys):
                s.add_var_value(self.y[g], on[j])
            m.add_mip_start(s, write_level=1)
        sol, info = _solve(m, tl, gap, None, threads)
        out = {key for key, var in x.items() if sol.get_value(var) > 0.5} if sol else None
        for c in cts:
            m.remove_constraint(c)
        return out, info


def retime(a, cluster, best_x, limits, log):
    """The re-timing step of Benders on the best schedule."""
    prob = B.ClusterProblem(cluster, a.data, staff_args(), a.threads)
    prob.args.w_bus_trip = 1.0
    evaluator = B.StaffModel(prob.args, prob.data)
    vals, plan, z = prob.plan_cost(evaluator, best_x)
    best = prob.retime(dict(z=z, x=set(best_x), vals=vals, plan=plan, round="search"), limits, evaluator, 1, 1200, log)
    return best["z"], best["x"], best["vals"], B._parts(best["plan"])


def save(a, res, x):
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=float)
    with open(Path(a.out).with_suffix(".schedule.csv"), "w") as f:
        f.write("Staff,Shift\n" + "".join(f"{i},{s}\n" for i, s in sorted(x)))


def run(a, log):
    cluster = next(c for c in set(branch_cluster(a.data).values()) if c.startswith(a.cluster))
    prob = Problem(cluster, a.data, staff_args(), a.threads)
    keep = set(prob.data.staff_ids)
    import pandas as pd
    x0 = {(r.Staff, r.Shift) for r in pd.read_csv(a.schedule).itertuples() if r.Staff in keep}
    e0 = prob.evaluate(x0)
    t0 = time.time()
    sg = Surrogate(prob, not a.allow_negative, not a.with_fixed, None if a.penalties else e0["levels"])
    log(f"sequential: Phi {e0['phi']:,.2f} J_op {e0['j_op']:,.2f} labor {e0['labor']:,.2f} penalty {e0['penalty']:,.2f} "
        f"C_bus {e0['c_bus']:,.2f} theta {e0['theta']:,.2f} levels {e0['levels']} trips {e0['trips']} buses {e0['buses']} "
        f"groups {len(sg.keys)}")
    rng = np.random.default_rng(a.seed)
    riders = list(prob.net.home)
    f0 = sg.features(x0)
    sg.A.append(f0)
    sg.v.append(e0["v"])
    sg.beta = e0["v"] - f0 @ sg.pi()  # level at x0, so that predictions start from a known point
    best, best_x, best_k = dict(e0), set(x0), -1
    score = "objective" if a.penalties else "j_op"
    best_at = 0.0
    left = lambda: a.time_limit - (time.time() - t0) if a.time_limit else math.inf
    x_on, cur_v, hist, dpred, dreal = set(x0), e0["v"], [], [], []
    for k in range(a.rounds):
        if left() <= 0:
            break
        base = sg.features(x_on)
        n_found = 0
        for _ in range(a.neighbors):
            if left() <= 0:
                break
            noise = (rng.uniform(1 - a.noise, 1 + a.noise, len(sg.keys)), rng.uniform(1 - a.noise, 1 + a.noise, len(sg.keys)))
            # free a few random riders and fix everyone else, so that the small model solves quickly
            free = set(rng.choice(riders, size=min(a.free, len(riders)), replace=False).tolist())
            for key, var in sg.x.items():
                if key[0] not in free:
                    var.lb = var.ub = 1 if key in x_on else 0
            xn, _info = sg.solve(x_on, a.near, min(a.near_time, left()), 0.02, a.threads, noise, at_least=1)
            for var in sg.x.values():
                var.lb, var.ub = 0, 1
            if xn is None:
                continue
            e = prob.evaluate(xn)
            f = sg.features(xn)
            dpred.append(float((f - base) @ sg.pi()))
            dreal.append(e["v"] - cur_v)
            sg.A.append(f)
            sg.v.append(e["v"])
            n_found += 1
            if e[score] < best[score] - 1e-6:
                best, best_x, best_k, best_at = e, set(xn), k, time.time() - t0
        if left() <= 0:
            break
        if a.ablation in ("lns", "lns_fit"):
            # no upper-level solve: the search goes on from the best schedule found so far; lns_fit still refits the
            # prices that perturb the neighbor moves
            if not n_found:
                continue
            if a.ablation == "lns_fit":
                sg.fit(a.lam)
            xs, e, info = set(best_x), best, dict(status="none", mip_gap=0.0)
            pred = float("nan")
        else:
            if a.ablation == "none":
                sg.fit(a.lam)
            xs, info = sg.solve(x_on, a.radius, min(a.master_time, left()), a.gap, a.threads, start=x_on)
            if xs is None:
                log(f"round {k}: no solution ({info['status']})")
                break
            e = prob.evaluate(xs)
            f = sg.features(xs)
            pred = sg.beta + f @ sg.pi()
            dpred.append(float((f - base) @ sg.pi()))
            dreal.append(e["v"] - cur_v)
            sg.A.append(f)
            sg.v.append(e["v"])
        if e[score] < best[score] - 1e-6:
            best, best_x, best_k, best_at = e, set(xs), k, time.time() - t0
        moved = len(xs ^ x_on) // 2
        corr = float(np.corrcoef(dpred, dreal)[0, 1]) if len(dpred) > 2 and np.std(dpred) > 0 and np.std(dreal) > 0 else float("nan")
        mae = float(np.mean(np.abs(np.array(dpred) - np.array(dreal))))
        hist.append(dict(e, round=k, best_score=best[score], best_phi=best["phi"], best_j_op=best["j_op"], predicted=pred, neighbors=n_found, moved=moved,
                         neg_open=int((sg.pi_open < 0).sum()), neg_pass=int((sg.pi_pass < 0).sum()), delta_corr=corr, delta_mae=mae,
                         status=info["status"], mip_gap=info["mip_gap"], seconds=time.time() - t0))
        log(f"round {k}: Z {e['j_op']:,.2f} best {best[score]:,.2f} ({score}) Phi {e['phi']:,.2f} labor {e['labor']:,.2f} penalty {e['penalty']:,.2f} "
            f"C_bus {e['c_bus']:,.2f} theta {e['theta']:,.2f} predicted {pred:,.2f} nbrs {n_found} moved {moved} "
            f"neg {hist[-1]['neg_open']}/{hist[-1]['neg_pass']} corr {corr:.2f} mae {mae:,.1f} levels {e['levels']} "
            f"{info['status']} gap {info['mip_gap']:.3%} {hist[-1]['seconds']:.0f}s")
        x_on, cur_v = xs, e["v"]
        save(a, dict(sequential=e0, best=best, best_round=best_k, best_seconds=best_at, history=hist, seconds=time.time() - t0), best_x)
    res = dict(sequential=e0, best=best, best_round=best_k, best_seconds=best_at, history=hist, dpred=dpred, dreal=dreal,
               seconds=time.time() - t0, z_search=best["j_op"], z=best["j_op"])
    if not a.penalties:
        res["z"], best_x, res["retime_vals"], res["retime_plan"] = retime(a, cluster, best_x, e0["levels"], log)
    return res, best_x


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cluster", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--schedule", required=True)
    p.add_argument("--rounds", type=int, default=100000)
    p.add_argument("--neighbors", type=int, default=4)
    p.add_argument("--near", type=int, default=6)
    p.add_argument("--free", type=int, default=8)
    p.add_argument("--near-time", type=float, default=30)
    p.add_argument("--noise", type=float, default=0.3)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--radius", type=int, default=40)
    p.add_argument("--allow-negative", action="store_true")
    p.add_argument("--with-fixed", action="store_true")
    p.add_argument("--penalties", action="store_true", help="penalties in the objective instead of the service limits")
    p.add_argument("--time-limit", type=float, required=True, help="seconds of search, counted from the routing of x0")
    p.add_argument("--master-time", type=float, default=300)
    p.add_argument("--gap", type=float, default=1e-3)
    p.add_argument("--threads", type=int, default=3)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--ablation", choices=["none", "static", "lns", "lns_fit"], default="none",
                   help="static: prices stay at the one-trip estimate (no fit); lns: no upper-level solve, only the neighbor moves; lns_fit: lns with the regression")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    res, x = run(a, lambda s: print(s, flush=True))
    save(a, res, x)
    print("DONE", res["z_search"], res["z"], flush=True)
