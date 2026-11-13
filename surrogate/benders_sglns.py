"""Benders-seeded SG-LNS: the surrogate-guided neighborhood search started from what the Benders decomposition found.

Benders + CG (integrated/net_benders.py) and SG-LNS (staff_surrogate.py) are imported unchanged; this file only reads their
outputs. The Benders run of a cluster supplies three things:
  start    its best schedule (after re-timing) replaces x0 as the first incumbent;
  prices   the lifted stop prices pi of the routing LP at that schedule (ClusterProblem.cut, valid for every schedule)
           replace the one-trip estimate as the initial opening coefficients of the surrogate;
  history  x0 and the Benders schedule both enter the regression history.
A stop (b, a, t) of Benders holds both directions; when hour t has both shift starts and shift ends, its price is split
evenly between the drop-off and the pick-up group, so that a stop used in both directions still costs pi once.
The search itself is the lns_fit loop of staff_surrogate.py: free eight riders, perturb the prices, repair, route, refit.
The service limits are those of x0, as in Benders. The time budget starts once the start schedule is routed and includes
the computation of the prices.

Usage: python benders_sglns.py --cluster <id> --data <data.xlsx> --benders-dir <dir> --time-limit <s> --out <file.json>
       [--start x0|benders] [--prices direct|benders] [--seed 1]
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import staff_surrogate as S  # noqa: E402
from staff_surrogate import B, branch_cluster  # noqa: E402


def load(path, keep):
    return {(r.Staff, r.Shift) for r in pd.read_csv(path).itertuples() if r.Staff in keep}


def benders_prices(a, cluster, x_on, log):
    """Lifted stop prices of the routing LP at x_on, from the verified Benders code."""
    cp = B.ClusterProblem(cluster, a.data, S.staff_args(), a.threads)
    if getattr(a, "no_lift", False):
        cp.closed_price = "zero"  # ablation: the cut without lifting (closed stops at min(0, lifted price))
    prices, value, iters = cp.cut(x_on)
    log(f"benders prices at the start schedule: LP value {value:,.2f}  CG iterations {iters}  stops {len(prices)}  "
        f"negative {sum(v < 0 for v in prices.values())}")
    return prices, value


def seed_prices(sg, prob, prices):
    """Opening coefficients of the groups from the stop prices; a stop with both directions splits its price."""
    for j, (b, a_, t, _d) in enumerate(sg.keys):
        both = bool(prob.starts[t]) and bool(prob.ends[t])
        sg.pi_open[j] = max(prices.get((b, a_, t), 0.0), 0.0) / (2.0 if both else 1.0)
    sg.pi_pass[:] = 0.0


def run(a, log):
    cluster = next(c for c in set(branch_cluster(a.data).values()) if c.startswith(a.cluster))
    bd = Path(a.benders_dir)
    stem = Path(a.data).stem
    prob = S.Problem(cluster, a.data, S.staff_args(), a.threads)
    keep = set(prob.data.staff_ids)
    x0 = load(bd / f"{stem}_{a.cluster}_x0.csv", keep)
    xb = load(bd / f"{stem}_{a.cluster}_schedule.csv", keep)
    e0 = prob.evaluate(x0)
    eb = prob.evaluate(xb)
    limits = e0["levels"]
    ok = all(eb["levels"][p] <= limits[p] + 1e-6 for p in limits)
    log(f"sequential: Z {e0['j_op']:,.2f}  transport {e0['v']:,.2f}  levels {e0['levels']}")
    log(f"benders:    Z {eb['j_op']:,.2f}  transport {eb['v']:,.2f}  levels {eb['levels']}  within limits {ok}")
    if not ok:
        raise RuntimeError("the Benders schedule breaks the service limits of x0")
    start_x, start_e = (set(xb), eb) if a.start == "benders" else (set(x0), e0)
    t0 = time.time()
    left = lambda: a.time_limit - (time.time() - t0)

    sg = S.Surrogate(prob, True, True, limits)
    prices, lp_value = ({}, float("nan"))
    if a.prices == "benders":
        prices, lp_value = benders_prices(a, cluster, start_x, log)
        seed_prices(sg, prob, prices)
    price_seconds = time.time() - t0
    for x, e in ((x0, e0), (xb, eb)):
        sg.A.append(sg.features(x))
        sg.v.append(e["v"])
    sg.beta = float(np.mean([v - f @ sg.pi() for f, v in zip(sg.A, sg.v)]))

    rng = np.random.default_rng(a.seed)
    riders = list(prob.net.home)
    best, best_x, best_k, best_at = dict(start_e), set(start_x), -1, 0.0
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
            if min(a.near_time, left()) <= 0:  # the clock can run out between the check above and here
                for var in sg.x.values():
                    var.lb, var.ub = 0, 1
                break
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
        if k % 10 == 0:
            S.save(a, dict(partial=True, best=best, history=hist), best_x)
    res = dict(cluster=a.cluster, start=a.start, prices=a.prices, seed=a.seed, sequential=e0, benders=eb, z_seq=e0["j_op"],
               z_benders=eb["j_op"], benders_lp_at_start=lp_value, price_seconds=price_seconds, best=best, best_round=best_k,
               best_seconds=best_at, history=hist, dpred=dpred, dreal=dreal, seconds=time.time() - t0, z_search=best["j_op"])
    res["z"], best_x, res["retime_vals"], res["retime_plan"] = S.retime(a, cluster, best_x, limits, log)
    res["z"] = min(res["z"], best["j_op"])
    return res, best_x


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cluster", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--benders-dir", required=True)
    p.add_argument("--start", choices=["x0", "benders"], default="benders")
    p.add_argument("--prices", choices=["direct", "benders"], default="benders")
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
    p.add_argument("--no-lift", action="store_true", help="ablation: stop prices from the cut without lifting")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    res, x = run(a, lambda s: print(s, flush=True))
    S.save(a, res, x)
    print("DONE", res["z_search"], res["z"], flush=True)
