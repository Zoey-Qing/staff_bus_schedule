"""Computational study of the Benders master of net_benders.py on one cluster.

Rebuilds the master exactly as in ClusterProblem.benders (staff model with w_bus_trip = 0, service limits of the
sequential schedule x0, stops y with the aggregated link of add_stops, theta and the stored cuts), then compares
formulations of the link between x and y:
  nocut : master without cuts (theta = 0), aggregated link
  A     : all stored cuts, aggregated link (current formulation)
  B     : A plus the disaggregated link y[b,a,t] >= x[i,s] for each rider i of (b,a) and shift s starting or ending at t
  C     : B without the constraints y <= load
  D     : A plus the disaggregated link as CPLEX user cuts
  E     : A plus one row per stop and rider: y[b,a,t] >= sum of x[i,s] over the shifts s starting or ending at t
  E2    : E with y continuous in [0, 1]
  theta : A with objective theta alone (diagnostic: how well the master bounds the transport term by itself)
For each variant: LP relaxation value, then the MIP from the x0 start for a fixed time, with bound, incumbent and gap.

Run from E:\\schedule\\integrated:  python master_study/master_study.py <variant> [--time 600 --threads 4]
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import cplex  # noqa: E402

import net_benders as NB  # noqa: E402
from bus_riders import branch_cluster  # noqa: E402
from model import _solve  # noqa: E402

DATA = Path(r"E:\schedule\staff_scheduling\Berkeley_Mock_Data_25_Stores_original_unbalanced.xlsx")
RUN = Path(r"E:\schedule\revision\results_new\original\benders_long\Berkeley_Mock_Data_25_Stores_original_unbalanced_a1e8b5a7")
PREFIX = "a1e8"


def build(variant, threads, log):
    cluster = next(c for c in set(branch_cluster(DATA).values()) if c.startswith(PREFIX))
    prob = NB.ClusterProblem(cluster, DATA, NB.staff_args(), threads, 1.0)
    x0 = NB.load_schedule(Path(f"{RUN}_x0.csv"), prob.data.staff_ids)
    stored = json.load(open(f"{RUN}.json"))
    limits = {int(p): v for p, v in stored["limits"].items()}
    cuts = [{(b, a, int(t)): pr for b, a, t, pr in c} for c in stored["cuts"]]

    # the levels of x0 must equal the stored limits (the master of the run used these)
    prob.args.w_bus_trip = prob.w_bus
    t = time.time()
    levels = NB.StaffModel(prob.args, prob.data).value_of(x0, threads)["levels"]
    log(f"levels of x0 {levels} vs stored limits {limits} ({time.time() - t:.0f}s)")
    assert all(abs(levels[p] - limits[p]) < 1e-6 for p in limits)

    # master, as in ClusterProblem.benders
    t = time.time()
    prob.args.w_bus_trip = 0.0
    sm = NB.StaffModel(prob.args, prob.data)
    m, x = sm.mdl, sm.solver.x
    for p, expr in sm.levels.items():
        m.add_constraint(expr <= limits[p] + 1e-6, ctname=f"service_{p}")
    y = prob.add_stops(sm)
    theta = m.continuous_var(lb=0, name="theta")
    m.minimize(theta if variant == "theta" else sm.j_reg + sm.j_over + theta)
    use_cuts = [] if variant == "nocut" else cuts
    for k, prices in enumerate(use_cuts):
        m.add_constraint(theta >= m.sum(pr * y[s] for s, pr in prices.items() if s in y), ctname=f"cut_{k + 1}")

    added = 0
    if variant in ("B", "C", "D"):
        disagg = [y[b, a, t] >= x[i, s] for (b, a, t) in y for i in prob.net.riders[b, a] for s in prob.starts[t] + prob.ends[t]]
        added = len(disagg)
        if variant == "D":
            for ct in disagg:
                m.add_user_cut_constraint(ct)
        else:
            m.add_constraints(disagg)
    if variant in ("E", "E2"):
        # one row per stop and rider: valid because the staff model's rest constraint (sum of x over the shifts whose
        # [start, end + min_break) covers t is <= 1) already bounds the right-hand side by 1, also in the LP
        rows = [y[b, a, t] >= m.sum(x[i, s] for s in prob.starts[t] + prob.ends[t]) for (b, a, t) in y for i in prob.net.riders[b, a]]
        added = len(rows)
        m.add_constraints(rows)
        if variant == "E2":
            for var in y.values():
                var.set_vartype(m.continuous_vartype)
                var.lb, var.ub = 0, 1
    removed = 0
    if variant == "C":
        # add_stops adds, per stop, y <= load and load <= n * y; drop the first kind
        rm = _upper_links(m, y)
        removed = len(rm)
        m.remove_constraints(rm)
    log(f"variant {variant}: {m.number_of_variables} variables, {m.number_of_constraints} constraints, "
        f"{len(use_cuts)} cuts, {added} disaggregated links added, {removed} y <= load removed, {len(y)} stops "
        f"({time.time() - t:.0f}s to build)")

    # start from x0, its open stops and the largest cut value there
    open0 = prob.net.demand(prob.events(x0))[2]
    theta0 = max([sum(pr for s, pr in c.items() if s in open0) for c in use_cuts], default=0.0) + 1e-6
    start = m.new_solution()
    for key in x0:
        start.add_var_value(x[key], 1)
    for s, var in y.items():
        start.add_var_value(var, 1 if s in open0 else 0)
    start.add_var_value(theta, theta0)
    return dict(prob=prob, sm=sm, m=m, x=x, y=y, theta=theta, start=start, theta0=theta0, added=added, removed=removed,
                n_cuts=len(use_cuts), x0=x0)


def _upper_links(m, y):
    """The constraints y <= load of add_stops: y - load <= 0 with y coefficient +1 and only x variables otherwise."""
    yset = {v.index for v in y.values()}
    out = []
    for ct in m.iter_linear_constraints():
        if ct.sense.name != "LE":
            continue
        lhs, rhs = ct.left_expr, ct.right_expr
        lv = list(lhs.iter_variables())
        coef = 1 if hasattr(lhs, "lb") else lhs.unchecked_get_coef(lv[0]) if len(lv) == 1 else 0  # a Var has lb
        if len(lv) == 1 and lv[0].index in yset and coef == 1:
            rv = list(rhs.iter_variables())
            if rv and all(v.index not in yset for v in rv):
                out.append(ct)
    return out


def max_events(prob, x_on):
    """Largest number of chosen shifts of one rider that start or end in the same hour."""
    cnt = {}
    for i, s in x_on:
        if i in prob.net.home:
            for t in {prob.shift[s]["start"] % 168, prob.shift[s]["end"] % 168}:
                cnt[i, t] = cnt.get((i, t), 0) + 1
    return max(cnt.values(), default=0)


def lp_relaxation(m, threads, parts):
    """LP value of a copy of the engine model (user cuts are not part of the copy), and the value of each named part."""
    cpx = cplex.Cplex(m.get_cplex())
    cpx.set_log_stream(None)
    cpx.set_results_stream(None)
    cpx.set_warning_stream(None)
    cpx.parameters.threads.set(threads)
    cpx.MIP_starts.delete()
    cpx.set_problem_type(cpx.problem_type.LP)
    t = time.time()
    cpx.solve()
    vals = cpx.solution.get_values()
    split = {k: sum(c * vals[v.index] for v, c in e.iter_terms()) + e.constant for k, e in parts.items()}
    return cpx.solution.get_objective_value(), cpx.solution.get_status_string(), time.time() - t, split


def root_bound(text):
    """The last node-log line of node 0 (root bound after the cut rounds), kept as text."""
    root = None
    for line in text.splitlines():
        if re.match(r"^\s*\*?\s+0\+?\s+[0-9]+\s", line):
            root = line.strip()
    return root


def main():
    p = argparse.ArgumentParser()
    p.add_argument("variant", choices=["nocut", "A", "B", "C", "D", "E", "E2", "theta"])
    p.add_argument("--time", type=float, default=600)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--gap", type=float, default=1e-3)
    p.add_argument("--emphasis", type=int, default=0, help="CPLEX MIP emphasis; 3 = move the best bound")
    p.add_argument("--tag", default="")
    a = p.parse_args()
    out = HERE / f"run_{a.variant}{a.tag}"
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    if a.variant == "D":
        # D waits while the flag file exists, so that E and E2 can run first
        while (HERE / "hold_D").exists():
            time.sleep(20)
    t_all = time.time()
    b = build(a.variant, a.threads, log)
    m = b["m"]
    lp, lp_status, lp_sec, split = lp_relaxation(m, a.threads, dict(wages=b["sm"].j_reg, overtime=b["sm"].j_over, theta=b["theta"] + 0))
    log(f"LP relaxation {lp:,.2f} ({lp_status}, {lp_sec:.0f}s)  {split}")

    NB.set_start(m, b["start"])
    m.parameters.emphasis.mip = a.emphasis
    sol, info = _solve(m, a.time, a.gap, f"{out}_cplex.log", a.threads)
    res = dict(variant=a.variant, cuts=b["n_cuts"], added=b["added"], removed=b["removed"], variables=m.number_of_variables,
               constraints=m.number_of_constraints, user_cuts=m.number_of_user_cut_constraints if a.variant == "D" else 0,
               lp=lp, lp_seconds=lp_sec, lp_wages=split["wages"], lp_overtime=split["overtime"], lp_theta=split["theta"], theta_start=b["theta0"], mip_status=info["status"], best_bound=info["best_bound"],
               mip_seconds=info["seconds"], nodes=info["nodes"])
    if sol:
        res.update(incumbent=sol.objective_value, gap=info["mip_gap"], inc_wages=sol.get_value(b["sm"].j_reg),
                   inc_overtime=sol.get_value(b["sm"].j_over), inc_theta=sol.get_value(b["theta"]),
                   inc_open=sum(1 for v in b["y"].values() if sol.get_value(v) > 0.5),
                   inc_same_as_x0=({k for k, v in b["x"].items() if sol.get_value(v) > 0.5} == b["x0"]))
        x_inc = {k for k, v in b["x"].items() if sol.get_value(v) > 0.5}
        NB.save_schedule(x_inc, b["prob"].shift, f"{out}_incumbent.csv")
        res["inc_max_events_per_rider_hour"] = max_events(b["prob"], x_inc)
    text = open(f"{out}_cplex.log", encoding="utf-8").read()
    res["root_line"] = root_bound(text)
    res["total_seconds"] = time.time() - t_all
    log(json.dumps(res, default=float))
    with open(f"{out}.json", "w") as f:
        json.dump(res, f, indent=1, default=float)


if __name__ == "__main__":
    main()
