"""SG-LNS from the sequential schedule x0 with the surrogate's opening prices taken from one lifted Benders cut at x0.

Phase 1 of the method in integrated/roster_cg/sg_bpb.py. staff_surrogate.py and benders_sglns.py are imported unchanged.
No finished Benders run is needed: the cut is computed here, at x0, by net_benders.ClusterProblem.cut (the routing LP at
the stops of x0 by column generation, with prices lifted to the closed stops; valid for every schedule). Its time counts in
the time limit.

  prices   the opening coefficient of each (branch, accommodation, hour, direction) group is the cut price of its stop;
           a stop with both shift starts and shift ends at that hour splits its price evenly between the two directions
           (benders_sglns.seed_prices). The per-rider coefficients start at 0.
  start    x0, the only point of the regression history.
  search   the lns_fit loop of staff_surrogate.py: free eight riders, perturb the prices, repair, route, refit; then the
           re-timing of net_benders.py.
With --prices direct the opening coefficients stay at the one-trip estimate of staff_surrogate.py (the earlier SG-LNS).

Usage: python sglns_cutprices.py --cluster <id> --data <data.xlsx> --schedule <x0.csv> --time-limit <s> --out <file.json>
       [--prices cut|direct] [--seed 1]
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import benders_sglns as BS
import staff_surrogate as S
from staff_surrogate import branch_cluster


def run(a, log):
    cluster = next(c for c in set(branch_cluster(a.data).values()) if c.startswith(a.cluster))
    prob = S.Problem(cluster, a.data, S.staff_args(), a.threads)
    keep = set(prob.data.staff_ids)
    x0 = {(r.Staff, r.Shift) for r in pd.read_csv(a.schedule).itertuples() if r.Staff in keep}
    e0 = prob.evaluate(x0)
    limits = e0["levels"]
    log(f"sequential: Z {e0['j_op']:,.2f}  transport {e0['v']:,.2f}  levels {limits}")
    t0 = time.time()
    left = lambda: a.time_limit - (time.time() - t0)  # noqa: E731
    sg = S.Surrogate(prob, True, True, limits)
    lp_value = float("nan")
    if a.prices == "cut":
        cp = S.B.ClusterProblem(cluster, a.data, S.staff_args(), a.threads)
        if a.no_lift:
            cp.closed_price = "zero"  # the cut of the ablation without lifting
        prices, lp_value, iters = cp.cut(x0)
        log(f"cut at x0: LP value {lp_value:,.2f}  CG iterations {iters}  stops {len(prices)}  lifted {not a.no_lift}")
        BS.seed_prices(sg, prob, prices)
    price_seconds = time.time() - t0
    f0 = sg.features(x0)
    sg.A.append(f0)
    sg.v.append(e0["v"])
    sg.beta = e0["v"] - f0 @ sg.pi()  # the prediction at x0 is the real transport cost of x0
    rng = np.random.default_rng(a.seed)
    riders = list(prob.net.home)
    best, best_x, best_k, best_at = dict(e0), set(x0), -1, 0.0
    hist, dpred, dreal = [], [], []
    for k in range(a.rounds):
        if left() <= 0:
            break
        x_on, cur_v = set(best_x), best["v"]
        base = sg.features(x_on)
        n_found = 0
        for _ in range(a.neighbors):
            if left() <= 0:
                break
            noise = (rng.uniform(1 - a.noise, 1 + a.noise, len(sg.keys)), rng.uniform(1 - a.noise, 1 + a.noise, len(sg.keys)))
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
            if e["j_op"] < best["j_op"] - 1e-6:
                best, best_x, best_k, best_at = e, set(xn), k, time.time() - t0
        if not n_found:
            continue
        sg.fit(a.lam)
        corr = float(np.corrcoef(dpred, dreal)[0, 1]) if len(dpred) > 2 and np.std(dpred) > 0 and np.std(dreal) > 0 else float("nan")
        hist.append(dict(round=k, best_j_op=best["j_op"], neighbors=n_found, delta_corr=corr, seconds=time.time() - t0))
        log(f"round {k}: best {best['j_op']:,.2f}  nbrs {n_found}  corr {corr:.2f}  {hist[-1]['seconds']:.0f}s")
    res = dict(cluster=a.cluster, prices=a.prices, seed=a.seed, sequential=e0, z_seq=e0["j_op"], lp_at_x0=lp_value,
               price_seconds=price_seconds, best=best, best_round=best_k, best_seconds=best_at, history=hist,
               dpred=dpred, dreal=dreal, seconds=time.time() - t0, z_search=best["j_op"])
    res["z"], best_x, res["retime_vals"], res["retime_plan"] = S.retime(a, cluster, best_x, limits, log)
    return res, best_x


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cluster", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--schedule", required=True, help="the sequential schedule x0")
    p.add_argument("--prices", choices=["cut", "direct"], default="cut")
    p.add_argument("--no-lift", action="store_true", help="the cut without lifting (closed stops at min(0, lifted price))")
    p.add_argument("--rounds", type=int, default=100000)
    p.add_argument("--neighbors", type=int, default=4)
    p.add_argument("--near", type=int, default=6)
    p.add_argument("--free", type=int, default=8)
    p.add_argument("--near-time", type=float, default=30)
    p.add_argument("--noise", type=float, default=0.3)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--time-limit", type=float, required=True)
    p.add_argument("--threads", type=int, default=3)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    res, x = run(a, lambda s: print(s, flush=True))
    S.save(a, res, x)
    print("DONE", res["z_search"], res["z"], flush=True)
