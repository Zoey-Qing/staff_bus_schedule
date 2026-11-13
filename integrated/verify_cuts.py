"""Numerical check of the Benders cut of net_benders.py on one cluster (Proposition 2 of the paper).

At the stops of a schedule x it builds the cut exactly as net_benders does (column generation, then lifting), and then:
  1. measures the largest violation of every dual constraint of the full routing LP: every trip over every possible stop
     of the hour, with every bus type and departure minute, every empty run at every minute, and every waiting arc;
  2. repeats check 1 with the closed stops priced at zero instead of lifted, to show what lifting is needed for;
  3. solves the routing LP at random stop sets y near the current one and checks  sum pi_s y_s <= LP(y).

Usage: python verify_cuts.py <data.xlsx> <x0.csv> --cluster <id prefix> --random 10 --out <file.json>
"""
import argparse
import json
import sys
import time

import numpy as np

import net_benders as B
import net_routing as N


def capture_cg():
    """Wrap column_generation so that the duals of its last call can be read back."""
    box, orig = {}, N.column_generation

    def wrapped(*args, **kw):
        out = orig(*args, **kw)
        box["out"] = out
        return out
    N.column_generation = wrapped
    return box


def dual_violations(prob, prices, rho, mu):
    """Largest violation of each family of dual constraints of the routing LP at these prices."""
    net = prob.net
    arrs = N.extend_duals(rho, mu, net)
    by_hour = {}
    for b, a, t in prob.possible_stops():
        by_hour.setdefault((a, t), []).append(b)
    worst_trip, n_trips, n_bad = -np.inf, 0, 0
    for (a, t), _bs in by_hour.items():
        for k in range(len(net.buses)):
            tb = net.table(a, k)
            base = tb.drive_entry + tb.D @ prob.low_cost(a, t, tb)
            dep = (60 * t + tb.delta) % N.WEEK
            g = np.minimum.reduceat(base - N.rho_of(arrs[k, a], mu[k], dep, tb.dur), tb.start)
            pr = np.array([prices.get((b, a, t), 0.0) for b in tb.branches])
            viol = tb.mem @ pr - g          # > 0 means sum pi + rho > cost for the best departure of the route
            worst_trip = max(worst_trip, float(viol.max()))
            n_trips += len(viol)
            n_bad += int((viol > 1e-6).sum())
    worst_run = -np.inf
    for k, bus in enumerate(net.buses):
        for a in net.accs:
            for a2 in net.accs:
                if a == a2:
                    continue
                dur = net.tau[a, a2]
                cost = bus.per_km * net.km[a, a2] + bus.per_min * dur
                deps = np.arange(N.WEEK)
                ret = deps + dur
                lhs = arrs[k, a2][ret % N.WEEK] - arrs[k, a][deps] + np.where(ret >= N.WEEK, mu[k], 0.0)
                worst_run = max(worst_run, float((lhs - cost).max()))
    worst_wait = -np.inf
    for k in range(len(net.buses)):
        for a in net.accs:
            arr = arrs[k, a]
            nxt = np.roll(arr, -1)
            lhs = nxt - arr
            lhs[-1] += mu[k]                # the arc from the last minute of the week to minute 0
            worst_wait = max(worst_wait, float(lhs.max()))
    worst_fleet = max(float(m) for m in mu) if mu else 0.0  # mu <= 0
    worst_fleet = max(worst_fleet, max(-(m + bus.fixed) for m, bus in zip(mu, net.buses)))
    return dict(trip=worst_trip, trip_routes=n_trips, trip_routes_violated=n_bad, empty_run=worst_run, waiting=worst_wait,
                fleet=worst_fleet)


def lp_value(prob, y_open):
    """Routing LP of the cut at the open stops y_open."""
    stops = sorted(y_open)
    if not stops:
        return 0.0
    _m, value, *_rest = N.column_generation(prob.net, stops, {s: 1 for s in stops}, prob.low_cost, threads=prob.threads)
    return value


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("data")
    p.add_argument("schedule")
    p.add_argument("--cluster", required=True)
    p.add_argument("--random", type=int, default=10)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    cluster = next(c for c in set(B.branch_cluster(a.data).values()) if c.startswith(a.cluster))
    prob = B.ClusterProblem(cluster, a.data, B.staff_args(), a.threads)
    x0 = B.load_schedule(a.schedule, prob.data.staff_ids)
    box = capture_cg()
    t0 = time.time()
    prices, value, iters = prob.cut(x0)
    _master, _v, pi, rho, mu, _it = box["out"]
    open_now = set(pi)
    possible = prob.possible_stops()
    res = dict(cluster=cluster[:8], open_stops=len(open_now), possible_stops=len(possible), lp_value=value, cg_iterations=iters,
               cut_seconds=time.time() - t0)
    res["lifted"] = dual_violations(prob, prices, rho, mu)
    zero = {s: (prices[s] if s in open_now else 0.0) for s in prices}
    res["zero_closed"] = dual_violations(prob, zero, rho, mu)
    closed_prices = np.array([prices[s] for s in possible if s not in open_now])
    res["closed_price_stats"] = dict(n=len(closed_prices), min=float(closed_prices.min()), median=float(np.median(closed_prices)),
                                     max=float(closed_prices.max()), negative=int((closed_prices < 0).sum()))
    print(json.dumps(res, indent=1), flush=True)
    # random stop sets near the current one: drop each open stop with probability 0.2 and open as many closed stops
    rng = np.random.default_rng(a.seed)
    closed = [s for s in possible if s not in open_now]
    tests = []
    for r in range(a.random):
        keep = {s for s in open_now if rng.random() > 0.2}
        add = rng.choice(len(closed), size=min(len(closed), len(open_now) - len(keep)), replace=False)
        y = keep | {closed[j] for j in add}
        t1 = time.time()
        lp = lp_value(prob, y)
        cut = sum(prices[s] for s in y)
        cut0 = sum(zero[s] for s in y)
        tests.append(dict(stops=len(y), lp=lp, cut=cut, slack=lp - cut, zero_cut=cut0, zero_slack=lp - cut0, seconds=time.time() - t1))
        print(f"random {r}: stops {len(y)}  LP {lp:,.2f}  cut {cut:,.2f}  slack {lp - cut:,.4f}  zero-price cut {cut0:,.2f}  "
              f"slack {lp - cut0:,.4f}  {time.time() - t1:.0f}s", flush=True)
    res["random"] = tests
    res["min_slack"] = min((t["slack"] for t in tests), default=None)
    res["min_zero_slack"] = min((t["zero_slack"] for t in tests), default=None)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=float)
    print("DONE", flush=True)
    sys.exit(0)
