"""Exact routing check for one schedule of a cluster with one accommodation.

Lists every trip of the routing model of the paper (every route over at most five open stops of one hour that fits
the seats, on every bus type, at every departure minute of the window) and solves the integer program to optimality.
This gives T(x) exactly, which is compared with the routing solver (column generation + integer program over the
generated trips), with the linear relaxation, and with the routing LP of the cut (no seats, lowest wages).

With one accommodation there are no empty runs, so the waiting arcs between event minutes give the full network.

Usage: python verify_routing.py <data.xlsx> <schedule.csv> --cluster <id prefix> --out <file.json>
"""
import argparse
import json
import time

import numpy as np

import net_benders as B
import net_routing as N


def all_trips(net, q, p, e):
    trips = []
    by = {}
    for (b, a, t) in e:
        by.setdefault((a, t), set()).add(b)
    for (a, t), open_b in by.items():
        for k, bus in enumerate(net.buses):
            tb = net.table(a, k)
            isopen = np.array([b in open_b for b in tb.branches])
            valid = ~(tb.mem & ~isopen).any(axis=1)
            ends = np.append(tb.start, len(tb.delta))
            wage = np.array([e.get((b, a, t), 0.0) for b in tb.branches])
            for r in np.flatnonzero(valid):
                seq = tb.seqs[r]
                if not N.fits(seq, a, t, q, p, bus.capacity):
                    continue
                for j in range(tb.start[r], ends[r + 1]):
                    theta = float(tb.D[j] @ wage)
                    trips.append(N.Trip(k, a, t, seq, int(tb.delta[j]), int((60 * t + tb.delta[j]) % N.WEEK), int(tb.dur[j]),
                                        float(tb.drive[r]), theta))
    return trips


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("data")
    p.add_argument("schedule")
    p.add_argument("--cluster", required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--time-limit", type=float, default=3600)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    cluster = next(c for c in set(B.branch_cluster(a.data).values()) if c.startswith(a.cluster))
    prob = B.ClusterProblem(cluster, a.data, B.staff_args(), a.threads)
    net = prob.net
    assert len(net.accs) == 1, "only clusters with one accommodation"
    x = B.load_schedule(a.schedule, prob.data.staff_ids)
    events = prob.events(x)
    q, pp, e = net.demand(events)
    stops = sorted(e)
    t0 = time.time()
    plan = N.route_schedule(net, events, a.threads)
    t_solver = time.time() - t0
    # routing LP of the cut at the same stops
    _m, lp_cut, *_r = N.column_generation(net, stops, {s: 1 for s in stops}, prob.low_cost, threads=a.threads)
    trips = all_trips(net, q, pp, e)
    master = N.Master(net, stops, {s: 1 for s in stops})
    for h in trips:
        master.add_trip(h)
    t1 = time.time()
    lp_full, *_d = master.solve_lp(a.threads)
    c, row, names, ev = master.build(True, a.threads)
    c.parameters.timelimit.set(a.time_limit)
    c.parameters.mip.tolerances.mipgap.set(0.0)
    c.parameters.mip.tolerances.absmipgap.set(1e-6)
    c.solve()
    exact, bound = c.solution.get_objective_value(), c.solution.MIP.get_best_objective()
    res = dict(cluster=cluster[:8], stops=len(stops), trips=len(trips), solver=plan.cost, solver_lp=plan.lp, solver_seconds=t_solver,
               full_lp=lp_full, exact=exact, exact_bound=bound, exact_status=c.solution.get_status_string(),
               exact_seconds=time.time() - t1, lp_cut=lp_cut,
               solver_excess_pct=100 * (plan.cost - exact) / exact, cut_lp_below_exact=lp_cut <= bound + 1e-6)
    print(json.dumps(res, indent=1), flush=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
