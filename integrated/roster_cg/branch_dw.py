"""Branch-roster column generation with Benders cuts (branch-price-and-Benders), one cluster at a time.

A new method next to net_benders.py, which it imports unchanged. The staff model is reformulated by Dantzig-Wolfe
decomposition with one block per branch, in the spirit of roster column generation for personnel scheduling
(Legrain and Omer 2024, Feng et al. 2024) and of Benders cuts inside a column master (Wang et al. 2026):

Column:   a complete weekly roster of the staff of one branch that meets every hard rule of the staff model.
          It carries its wages and overtime, its violation of each priority level, and the stops of its riders.
Master:   min  sum_c cost_c lam_c + theta
          s.t. sum_{c of branch b} lam_c = 1                          for every branch b
               sum_c lev_{l,c} lam_c <= Vbar_l                         for every level l (the service limits)
               theta >= sum_c (sum_{stops s of c} pi^k_s) lam_c        for every Benders cut k
Pricing:  for each branch, the staff model of that branch with its stops y, objective
          wages + overtime - sum_l dual_l lev_l + sum_k dual_k sum_s pi^k_s y_s - alpha_b, solved by CPLEX.
Bound:    Lagrangian bound  master LP value + sum_b min(0, best bound of the pricing of b), valid at every iteration.
          Every coupling row except the cuts lives inside a block, so the bound convexifies only the service limits and
          the cuts. The cuts are those of net_benders.ClusterProblem.cut, valid for every schedule, so the bound is a
          lower bound on the cost of the integrated model.
Primal:   the master as an integer program over the generated columns (one roster per branch), routed exactly;
          the routed schedule gives a new cut, and column generation resumes (outer Benders loop).

Usage: python branch_dw.py <data.xlsx> <benders_dir> <out_dir> --cluster <id prefix> [--time 6000 --cuts own|stored]
"""
import argparse
import copy
import io
import json
import math
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import cplex

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[2] / "staff_scheduling"))
sys.path.insert(0, str(HERE.parents[2] / "bus_scheduling_routing"))
import instance as I  # noqa: E402
import net_benders as B  # noqa: E402
from bus_riders import branch_cluster  # noqa: E402
from model import _solve  # noqa: E402

LEVELS = (1, 2, 3, 4)


class Block:
    """The staff model of one branch with the stops of its riders; its objective is set anew for each pricing."""

    def __init__(self, prob, branch, args):
        self.b = branch
        args.w_bus_trip = 0.0
        with redirect_stdout(io.StringIO()):
            data = I.BranchSubsetLoader(args, [branch]).load_and_preprocess()
        self.sm = B.StaffModel(args, data)
        self.staff = list(data.staff_ids)
        m, x = self.sm.mdl, self.sm.solver.x
        self.y = {}
        for (b, a, t) in prob.possible_stops():
            if b != branch:
                continue
            riders = prob.net.riders[b, a]
            load = m.sum(x[i, s] for i in riders for s in prob.starts[t] + prob.ends[t])
            self.y[b, a, t] = m.binary_var(name=f"y_{a[:6]}_{t}")
            m.add_constraint(self.y[b, a, t] <= load)
            m.add_constraint(load <= len(riders) * self.y[b, a, t])

    def column(self, x_on, prob):
        """Cost, level violations and open stops of a roster of this branch, with the slacks at their least values."""
        x_b = {(i, s) for i, s in x_on if i in set(self.staff)}
        # value_of solves with the current objective; with every slack and the overtime priced, each takes its least value
        self.sm.mdl.minimize(self.sm.mdl.sum(self.sm.levels[l] for l in LEVELS) + self.sm.j_over)
        vals = self.sm.value_of(x_b, 1)
        stops = {s for s in prob.net.demand(prob.events(x_b))[2] if s[0] == self.b}
        return dict(x=frozenset(x_b), cost=vals["j_reg"] + vals["j_over"], lev=vals["levels"], stops=frozenset(stops))

    def price(self, dual_lev, stop_price, alpha, time_limit, gap, threads):
        sm, m = self.sm, self.sm.mdl
        m.minimize(sm.j_reg + sm.j_over - m.sum(dual_lev[l] * sm.levels[l] for l in LEVELS)
                   + m.sum(stop_price.get(s, 0.0) * v for s, v in self.y.items()))
        sol, info = _solve(m, time_limit, gap, None, threads)
        bb = info["best_bound"]
        bb = bb if bb is not None and math.isfinite(bb) and abs(bb) < 1e19 else -math.inf
        if not sol:
            return None, bb - alpha, info
        info["objective"] = sol.objective_value
        x_on = {k for k, v in sm.solver.x.items() if sol.get_value(v) > 0.5}
        return x_on, bb - alpha, info


class Master:
    """Restricted master LP over the columns, through the CPLEX callable library so that columns can be added."""

    def __init__(self, branches, limits):
        self.c = cplex.Cplex()
        for stream in (self.c.set_log_stream, self.c.set_results_stream, self.c.set_warning_stream):
            stream(None)
        self.c.objective.set_sense(self.c.objective.sense.minimize)
        self.branches = branches
        self.c.linear_constraints.add(names=[f"conv_{k}" for k in range(len(branches))], senses="E" * len(branches),
                                      rhs=[1.0] * len(branches))
        self.c.linear_constraints.add(names=[f"lev_{l}" for l in LEVELS], senses="L" * 4, rhs=[float(limits[l]) for l in LEVELS])
        self.c.variables.add(obj=[1.0], lb=[0.0], names=["theta"])
        self.cols, self.cuts = [], []

    def add_cut(self, prices):
        k = len(self.cuts)
        self.cuts.append(prices)
        ind = ["theta"] + [f"lam_{j}" for j in range(len(self.cols))]
        val = [1.0] + [-sum(prices.get(s, 0.0) for s in col["stops"]) for col in self.cols]
        self.c.linear_constraints.add(lin_expr=[cplex.SparsePair(ind, val)], senses="G", rhs=[0.0], names=[f"cut_{k}"])

    def add_column(self, branch, col):
        j = len(self.cols)
        col = dict(col, branch=branch)
        self.cols.append(col)
        rows = [f"conv_{self.branches.index(branch)}"] + [f"lev_{l}" for l in LEVELS] + [f"cut_{k}" for k in range(len(self.cuts))]
        vals = [1.0] + [float(col["lev"][l]) for l in LEVELS] + [-sum(pr.get(s, 0.0) for s in col["stops"]) for pr in self.cuts]
        self.c.variables.add(obj=[col["cost"]], lb=[0.0], names=[f"lam_{j}"], columns=[cplex.SparsePair(rows, vals)])

    def solve_lp(self):
        self.c.set_problem_type(self.c.problem_type.LP)
        self.c.solve()
        duals = dict(zip(self.c.linear_constraints.get_names(), self.c.solution.get_dual_values()))
        return self.c.solution.get_objective_value(), duals

    def solve_ip(self, time_limit, threads):
        """One roster per branch: the master with binary columns."""
        n = self.c.variables.get_num()
        self.c.variables.set_types([(j, "C" if j == 0 else "B") for j in range(n)])
        self.c.parameters.timelimit.set(time_limit)
        self.c.parameters.threads.set(threads)
        self.c.parameters.mip.tolerances.mipgap.set(1e-6)
        self.c.solve()
        ok = self.c.solution.is_primal_feasible()
        chosen = [self.cols[j - 1] for j, v in enumerate(self.c.solution.get_values()) if j > 0 and v > 0.5] if ok else []
        value = self.c.solution.get_objective_value() if ok else math.inf
        self.c.variables.set_types([(j, "C") for j in range(n)])
        self.c.set_problem_type(self.c.problem_type.LP)
        return chosen, value


def run(prob, x0, limits, cuts0, time_budget, price_time, price_gap, threads, jobs, log, warm=()):
    t0 = time.time()
    branches = sorted(prob.data.branch_ids)
    blocks = {b: Block(prob, b, copy.copy(prob.args)) for b in branches}
    master = Master(branches, limits)
    evaluator = B.StaffModel(prob.args, prob.data)
    vals0, plan0, z0 = prob.plan_cost(evaluator, x0)
    best = dict(z=z0, x=set(x0), vals=vals0, plan=plan0, source="sequential")
    seen_stops = set()
    for prices in cuts0:
        master.add_cut(prices)
    if not cuts0:
        prices, _v, _it = prob.cut(x0)
        master.add_cut(prices)
        seen_stops.add(frozenset(prob.net.demand(prob.events(x0))[2]))
    keys = set()
    for b in branches:
        col = blocks[b].column(x0, prob)
        master.add_column(b, col)
        keys.add((b, col["x"]))
    # warm start: the branch rosters of given schedules (e.g. SG-LNS results) as columns, with a cut at each
    for name, xw in warm:
        vals, plan, z = prob.plan_cost(evaluator, xw)
        ok = all(vals["levels"][l] <= limits[l] + 1e-6 for l in LEVELS)
        log(f"warm {name}: routed Z {z:,.2f}  within limits {ok}")
        if ok and z < best["z"]:
            best = dict(z=z, x=set(xw), vals=vals, plan=plan, source=f"warm {name}")
        stops = frozenset(prob.net.demand(prob.events(xw))[2])
        if stops not in seen_stops:
            seen_stops.add(stops)
            master.add_cut(prob.cut(xw)[0])
        for b in branches:
            col = blocks[b].column(xw, prob)
            if (b, col["x"]) not in keys:
                master.add_column(b, col)
                keys.add((b, col["x"]))
    lower, history, outer = -math.inf, [], 0
    log(f"setup {time.time() - t0:.0f}s  sequential Z {z0:,.2f}  limits {limits}  cuts {len(master.cuts)}")
    while time.time() - t0 < time_budget:
        # column generation on the current cuts
        while time.time() - t0 < time_budget:
            value, duals = master.solve_lp()
            dual_lev = {l: duals[f"lev_{l}"] for l in LEVELS}
            stop_price = {}
            for k, pr in enumerate(master.cuts):
                mu = duals[f"cut_{k}"]
                if mu > 1e-12:
                    for s, v in pr.items():
                        stop_price[s] = stop_price.get(s, 0.0) + mu * v

            def job(b):
                return b, blocks[b].price(dual_lev, stop_price, duals[f"conv_{branches.index(b)}"], price_time, price_gap, threads)

            # CPLEX models solved from several Python threads hang, so the blocks are priced one after the other
            res = [job(b) for b in branches]
            bound = value + sum(min(0.0, rb) for _b, (_x, rb, _i) in res)
            lower = max(lower, bound)
            added = 0
            for b, (x_on, _rb, _info) in res:
                if x_on is None:
                    continue
                col = blocks[b].column(x_on, prob)
                rc = col["cost"] - sum(dual_lev[l] * col["lev"][l] for l in LEVELS) \
                    + sum(stop_price.get(s, 0.0) for s in col["stops"]) - duals[f"conv_{branches.index(b)}"]
                if rc < -1e-6 * max(1.0, abs(value)) and (b, col["x"]) not in keys:
                    master.add_column(b, col)
                    keys.add((b, col["x"]))
                    added += 1
            history.append(dict(outer=outer, lp=value, lb=lower, bound=bound, added=added, columns=len(master.cols),
                                cuts=len(master.cuts), best=best["z"], seconds=time.time() - t0,
                                pricing_gaps=[r[1][2]["mip_gap"] for r in res]))
            log(f"outer {outer}  LP {value:,.2f}  bound {bound:,.2f}  LB {lower:,.2f}  best {best['z']:,.2f}  "
                f"gap {(best['z'] - lower) / best['z']:.3%}  +{added} cols ({len(master.cols)})  {time.time() - t0:.0f}s")
            if added == 0:
                break
        # primal step: one roster per branch, routed exactly, and a new cut at it
        chosen, ip_value = master.solve_ip(min(300, max(30, time_budget - (time.time() - t0))), threads * jobs)
        if not chosen:
            log("integer master found no solution")
            break
        x_new = set().union(*[set(c["x"]) for c in chosen])
        vals, plan, z = prob.plan_cost(evaluator, x_new)
        ok = all(vals["levels"][l] <= limits[l] + 1e-6 for l in LEVELS)
        log(f"outer {outer}: integer master {ip_value:,.2f}  routed Z {z:,.2f}  levels {vals['levels']}  within limits {ok}")
        if ok and z < best["z"]:
            best = dict(z=z, x=x_new, vals=vals, plan=plan, source=f"outer {outer}")
        stops = frozenset(prob.net.demand(prob.events(x_new))[2])
        history.append(dict(outer=outer, ip=ip_value, routed=z, within=ok, best=best["z"], lb=lower, seconds=time.time() - t0))
        outer += 1
        if stops in seen_stops:
            log("the integer master repeats a stop set: no new cut")
            break
        seen_stops.add(stops)
        prices, _v, _it = prob.cut(x_new)
        master.add_cut(prices)
        if (best["z"] - lower) / best["z"] <= 1e-4:
            break
    return best, lower, history, master


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data")
    p.add_argument("benders_dir", help="folder with <stem>_<cluster>_x0.csv and the Benders json (limits, stored cuts)")
    p.add_argument("out_dir")
    p.add_argument("--cluster", required=True)
    p.add_argument("--time", type=float, default=None, help="seconds; default: the time of the Benders run")
    p.add_argument("--cuts", choices=["own", "stored"], default="own",
                   help="own: start from the cut at x0 only; stored: start from the cuts of the Benders run")
    p.add_argument("--price-time", type=float, default=120)
    p.add_argument("--price-gap", type=float, default=1e-4)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--jobs", type=int, default=1, help="threads multiplier of the integer master")
    p.add_argument("--warm", nargs="*", default=[], help="schedule csv files whose branch rosters start the master as columns")
    p.add_argument("--tag", default=None, help="output name suffix; default: the --cuts value")
    p.add_argument("--no-lift", action="store_true",
                   help="Benders cuts without lifting (closed stops at min(0, lifted price)): the literature method without our cut")
    a = p.parse_args()
    assert not (a.no_lift and (a.cuts == "stored" or a.warm)), "--no-lift is the literature baseline: use --cuts own and no --warm"
    t_all = time.time()
    cluster = next(c for c in set(branch_cluster(a.data).values()) if c.startswith(a.cluster))
    stem = Path(a.data).stem
    bj = Path(a.benders_dir) / f"{stem}_{cluster[:8]}.json"
    bres = json.load(open(bj))
    prob = B.ClusterProblem(cluster, a.data, B.staff_args(), a.threads)
    if a.no_lift:
        prob.closed_price = "zero"
    x0 = B.load_schedule(Path(a.benders_dir) / f"{stem}_{cluster[:8]}_x0.csv", prob.data.staff_ids)
    limits = {int(k): v for k, v in bres["limits"].items()}
    cuts0 = [{(b, ac, int(t)): pr for b, ac, t, pr in c} for c in bres["cuts"]] if a.cuts == "stored" else []
    budget = a.time if a.time else bres["benders_seconds"]
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    best, lower, history, master = run(prob, x0, limits, cuts0, budget, a.price_time, a.price_gap, a.threads, a.jobs,
                                       lambda s: print(s, flush=True),
                                       [(Path(w).name, B.load_schedule(w, prob.data.staff_ids)) for w in a.warm])
    z0 = bres["z_seq"]
    res = dict(cluster=cluster[:8], cuts_start=a.cuts, z_seq=z0, z=best["z"], lb=lower, gap=(best["z"] - lower) / best["z"],
               saving=(z0 - best["z"]) / z0, benders_lb=bres["lb"], benders_z=bres["z"], source=best["source"],
               wages=best["vals"]["j_reg"], overtime=best["vals"]["j_over"], levels=best["vals"]["levels"], limits=limits,
               plan=B._parts(best["plan"]), columns=len(master.cols), cuts=len(master.cuts), budget=budget,
               seconds=time.time() - t_all, history=history)
    name = f"{stem}_{cluster[:8]}_{a.tag or a.cuts}"
    with open(out / f"{name}.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    B.save_schedule(best["x"], prob.shift, out / f"{name}_schedule.csv")
    print(f"DONE z {best['z']:,.2f} lb {lower:,.2f} gap {res['gap']:.3%} saving {res['saving']:.3%}  (Benders: lb {bres['lb']:,.2f} z {bres['z']:,.2f})")


if __name__ == "__main__":
    main()
