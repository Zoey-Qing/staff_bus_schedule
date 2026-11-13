"""Transport-aware staff scheduling MIP, a baseline in the spirit of Ormeci, Salman and Yucel (2014, Omega 43:41-53).

Their call-centre model adds the shuttle trips needed at shift starts and ends to the staff scheduling MIP, without routing.
Here, one cluster at a time, with net_benders.py imported unchanged:

  min  J_reg(x) + J_over(x) + sum_{b,a,t} sum_k c_{k,a,b} n_{b,a,t,k}
  s.t. the staff model with the service limits V_l(x) <= Vbar_l (as in the integrated model),
       sum_k Q_k n_{b,a,t,k} >= load_{b,a,t,d}(x)        for d = drop-off and pick-up: riders of branch b living at a whose
                                                          shift starts (d) or ends (p) at hour t; one trip does both
       n integer >= 0.
  c_{k,a,b} is the direct round trip of bus type k between a and b (per km and per minute; the fixed bus cost is not
  counted, since the model has no fleet). Every trip is direct: no pooling of stops, no empty runs, no fleet limit.
The schedule is then routed exactly (net_benders.plan_cost), as every other method. The MIP starts from x0 and gets the
time T less the routing.

Usage: python ormeci_baseline.py <data.xlsx> <benders_dir> <out_dir> --cluster <prefix> [--time T] [--threads 3]
"""
import argparse
import json
import math
import time
from pathlib import Path

import branch_dw as D

B, LEVELS = D.B, D.LEVELS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data")
    p.add_argument("benders_dir")
    p.add_argument("out_dir")
    p.add_argument("--cluster", required=True)
    p.add_argument("--time", type=float, default=None, help="seconds in all; default: the time of the Benders run")
    p.add_argument("--route-reserve", type=float, default=300, help="seconds kept for routing the result")
    p.add_argument("--gap", type=float, default=1e-4)
    p.add_argument("--threads", type=int, default=3)
    p.add_argument("--tag", default="ormeci")
    a = p.parse_args()
    t_all = time.time()
    cluster = next(c for c in set(D.branch_cluster(a.data).values()) if c.startswith(a.cluster))
    stem = Path(a.data).stem
    bd = Path(a.benders_dir)
    bres = json.load(open(bd / f"{stem}_{cluster[:8]}.json"))
    total = a.time or bres["benders_seconds"]
    prob = B.ClusterProblem(cluster, a.data, B.staff_args(), a.threads)
    x0 = B.load_schedule(bd / f"{stem}_{cluster[:8]}_x0.csv", prob.data.staff_ids)
    limits = {int(k): v for k, v in bres["limits"].items()}
    evaluator = B.StaffModel(prob.args, prob.data)
    vals0, plan0, z0 = prob.plan_cost(evaluator, x0)

    prob.args.w_bus_trip = 0.0
    sm = B.StaffModel(prob.args, prob.data)
    m, x, net = sm.mdl, sm.solver.x, prob.net
    for l, expr in sm.levels.items():
        m.add_constraint(expr <= limits[l] + 1e-6, ctname=f"service_{l}")
    trip_cost = []
    for (b, ac), riders in net.riders.items():
        for t in range(B.WEEK_HOURS):
            if not (prob.starts[t] or prob.ends[t]):
                continue
            # one set of trips per hour serves both the drop-offs (shift starts) and the pick-ups (shift ends), as a
            # direct trip in the routing model does
            n = {k: m.integer_var(lb=0, name=f"n_{b}_{ac[:6]}_{t}_{k}") for k in range(len(net.buses))}
            seats = m.sum(net.buses[k].capacity * n[k] for k in n)
            for shifts in (prob.starts[t], prob.ends[t]):
                if shifts:
                    m.add_constraint(seats >= m.sum(x[i, s] for i in riders for s in shifts))
            for k, bus in enumerate(net.buses):
                trip_cost.append((bus.per_km * 2 * net.km[ac, b] + bus.per_min * 2 * net.tau[ac, b]) * n[k])
    m.minimize(sm.j_reg + sm.j_over + m.sum(trip_cost))
    start = m.new_solution()
    for key in x0:
        start.add_var_value(x[key], 1)
    m.add_mip_start(start, write_level=1)
    left = total - (time.time() - t_all) - a.route_reserve
    sol, info = D._solve(m, max(60, left), a.gap, None, a.threads)
    assert sol, info["status"]
    xs = {key for key, var in x.items() if sol.get_value(var) > 0.5}
    vals, plan, z = prob.plan_cost(evaluator, xs)
    within = all(vals["levels"][l] <= limits[l] + 1e-6 for l in LEVELS)
    reg, over = prob.real_wages(xs)
    res = dict(cluster=cluster[:8], z_seq=z0, z=z, saving=(z0 - z) / z0, mip_objective=sol.objective_value,
               mip_bound=info["best_bound"], mip_gap=info["mip_gap"], mip_status=info["status"], wages=vals["j_reg"],
               overtime=vals["j_over"], levels=vals["levels"], limits=limits, within_limits=within, plan=B._parts(plan),
               z_check=reg + over + plan.cost - z, budget=total, seconds=time.time() - t_all)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = f"{stem}_{cluster[:8]}_{a.tag}"
    with open(out / f"{name}.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    B.save_schedule(xs, prob.shift, out / f"{name}_schedule.csv")
    print(f"DONE z {z:,.2f} saving {res['saving']:.3%}  MIP {sol.objective_value:,.2f} ({info['status']}, gap {info['mip_gap']:.2%})  "
          f"within limits {within}  z check {res['z_check']:.2e}  time {res['seconds']:,.0f}s", flush=True)


if __name__ == "__main__":
    main()
