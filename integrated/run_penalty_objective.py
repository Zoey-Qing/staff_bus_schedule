"""E4: the integrated model with the soft constraints priced in the objective, instead of held under service limits.

Master: min J_reg + J_over + Theta + J_pen + J_reg1. It starts from the cuts of the limit run, which hold for every schedule.

Usage: python run_penalty_objective.py <data.xlsx> <limit_prefix> <out_prefix> --cluster <id prefix> [--threads 3 --master-times 1200,1200,1200]
"""
import argparse
import json
import math
import sys
import time

import net_benders as B
from net_benders import ClusterProblem, StaffModel, _parts, _solve, branch_cluster, load_schedule, save_schedule, set_start, staff_args

L1 = 0.1


def terms(prob, staff, vals, plan, z, n_shifts):
    pen = {p: getattr(staff, f"M{p}") * v for p, v in vals["levels"].items()}
    total = z + sum(pen.values()) + L1 * n_shifts
    return dict(objective=total, z=z, wages=vals["j_reg"], overtime=vals["j_over"], transport=plan.cost, j_pen=pen, j_reg1=L1 * n_shifts,
                levels=vals["levels"], shifts=n_shifts, plan=_parts(plan))


def run(prob, staff, limit_prefix, master_times, log):
    t0 = time.time()
    prob.args.w_bus_trip = prob.w_bus
    evaluator = StaffModel(prob.args, prob.data)
    refs = {}
    for name, path in (("sequential", f"{limit_prefix}_x0.csv"), ("limits", f"{limit_prefix}_schedule.csv")):
        xs = load_schedule(path, prob.data.staff_ids)
        vals, plan, z = prob.plan_cost(evaluator, xs)
        refs[name] = terms(prob, staff, vals, plan, z, len(xs))
        refs[name]["x"] = xs
        log(f"{name}: objective {refs[name]['objective']:,.2f}  Z {z:,.2f}  levels {vals['levels']}")
    prob.args.w_bus_trip = 0.0
    sm = StaffModel(prob.args, prob.data)
    m, x = sm.mdl, sm.solver.x
    y = prob.add_stops(sm)
    theta = m.continuous_var(lb=0, name="theta")
    m.minimize(sm.j_reg + sm.j_over + theta + m.sum(getattr(staff, f"M{p}") * e for p, e in sm.levels.items()) + L1 * m.sum(x.values()))
    for c in json.load(open(f"{limit_prefix}.json"))["cuts"]:
        prices = {tuple(row[:3]): row[3] for row in c}
        assert set(y) <= set(prices)
        m.add_constraint(theta >= m.sum(pr * y[s] for s, pr in prices.items() if s in y))
    best_name = min(refs, key=lambda k: refs[k]["objective"])
    best, x_on, lower, history = dict(refs[best_name]), refs[best_name]["x"], -math.inf, []
    for i, limit in enumerate(master_times):
        start = m.new_solution()
        for key in x_on:
            start.add_var_value(x[key], 1)
        set_start(m, start)
        sol, info = _solve(m, limit, 1e-3, None, prob.threads)
        if not sol:
            log(f"round {i}: no solution ({info['status']})")
            break
        lower = max(lower, info["best_bound"])
        x_on = {key for key, var in x.items() if sol.get_value(var) > 0.5}
        vals, plan, z = prob.plan_cost(evaluator, x_on)
        now = terms(prob, staff, vals, plan, z, len(x_on))
        if now["objective"] < best["objective"]:
            best = dict(now, x=set(x_on))
        history.append(dict(round=i, lb=lower, ub=now["objective"], best=best["objective"], master_gap=info["mip_gap"], status=info["status"], seconds=time.time() - t0))
        log(f"round {i}: LB {lower:,.2f}  objective {now['objective']:,.2f}  best {best['objective']:,.2f}  Z {z:,.2f}  {info['status']}")
        prices, _value, _iters = prob.cut(x_on)
        m.add_constraint(theta >= m.sum(pr * y[s] for s, pr in prices.items() if s in y))
    x_best = best.pop("x")
    return dict(cluster=prob.cluster[:8], best=best, lb=lower, gap=(best["objective"] - lower) / best["objective"], history=history,
                sequential={k: v for k, v in refs["sequential"].items() if k != "x"}, limits={k: v for k, v in refs["limits"].items() if k != "x"},
                seconds=time.time() - t0), x_best


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("data")
    p.add_argument("limit_prefix")
    p.add_argument("out")
    p.add_argument("--cluster", required=True)
    p.add_argument("--threads", type=int, default=3)
    p.add_argument("--master-times", default="1200,1200,1200")
    a = p.parse_args()
    cluster = next(c for c in set(branch_cluster(a.data).values()) if c.startswith(a.cluster))
    staff = staff_args()
    prob = ClusterProblem(cluster, a.data, staff, a.threads)
    res, x = run(prob, staff, a.limit_prefix, [float(v) for v in a.master_times.split(",")], lambda s: print(s, flush=True))
    with open(f"{a.out}.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    save_schedule(x, prob.shift, f"{a.out}_schedule.csv")
    print("DONE", res["best"]["objective"], res["lb"], res["gap"], flush=True)
