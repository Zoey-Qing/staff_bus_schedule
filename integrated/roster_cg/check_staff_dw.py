"""Checks of staff_dw.py on one cluster.

1. Accounting: for the sequential schedule x0 and the SG-LNS schedules, the wages, overtime and level violations from the
   staff columns plus the least master slacks equal those of net_benders.StaffModel.value_of.
2. Pricing: at the duals of the master, adaptive and full pricing give the same least reduced cost for every employee, and the
   roster returned has that reduced cost when evaluated column by column.

Usage: python check_staff_dw.py <data.xlsx> <benders_dir> --cluster <prefix> [--warm a.csv b.csv]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import staff_dw as S

B = S.B


def master_levels(P, cols):
    """Level totals of a schedule given by its columns: own violations plus the least slacks of the coupling rows."""
    g = lambda k, v: getattr(P.prob.args, k, v)  # noqa: E731
    lev = {l: sum(c["lev"][l] for c in cols) + P.lev_const[l] for l in S.LEVELS}
    cnt = {}
    for c in cols:
        i = c["i"]
        b, mgr, nn = P.branch[i], P.rules.cat[i]["mgr"], i in P.not_new
        for t in c["hours"]:
            for key in ("all", "mgr" if mgr else "nm") + (("nn",) if nn else ()):
                cnt[key, b, t] = cnt.get((key, b, t), 0) + 1
    dem = P.data.branch_demand
    for (b, t) in P.active:
        a = cnt.get(("all", b, t), 0)
        lev[1] += max(0, dem[b, t] - a)
        lev[2] += max(0, g("min_onsite_not_new_soft", 1) - cnt.get(("nn", b, t), 0))
        lev[3] += max(0, g("min_onsite_all_soft", 2) - a) + max(0, g("min_onsite_mgr_soft", 1) - cnt.get(("mgr", b, t), 0))
        lev[4] += max(0, g("min_onsite_not_mgr_soft", 2) - cnt.get(("nm", b, t), 0))
    for (b, t) in P.open_close:
        lev[3] += max(0, 1 - cnt.get(("mgr", b, t), 0))
    return lev


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data")
    p.add_argument("benders_dir")
    p.add_argument("--cluster", required=True)
    p.add_argument("--warm", nargs="*", default=[])
    a = p.parse_args()
    cluster = next(c for c in set(S.branch_cluster(a.data).values()) if c.startswith(a.cluster))
    stem = Path(a.data).stem
    bres = json.load(open(Path(a.benders_dir) / f"{stem}_{cluster[:8]}.json"))
    prob = B.ClusterProblem(cluster, a.data, B.staff_args(), 4)
    x0 = B.load_schedule(Path(a.benders_dir) / f"{stem}_{cluster[:8]}_x0.csv", prob.data.staff_ids)
    limits = {int(k): v for k, v in bres["limits"].items()}
    t0 = time.time()
    P = S.Problem(prob, "adaptive")
    print(f"problem {time.time() - t0:.1f}s  staff {len(P.staff)}  day rosters per day "
          f"{sorted({tuple(len(d['h']) for d in dr.days) for dr in P.dr.values()})}")
    sm = B.StaffModel(prob.args, prob.data)
    worst = 0.0
    for name, x in [("x0", x0)] + [(Path(w).name, B.load_schedule(w, prob.data.staff_ids)) for w in a.warm]:
        cols = []
        for i in P.staff:
            pick = P.shifts_by_day(i, [s for j, s in x if j == i])
            assert pick is not None, (name, i)
            cols.append(P.column(i, pick))
        v = sm.value_of(x, 4)
        lev = master_levels(P, cols)
        wages = sum(c["cost"] for c in cols)
        d_w = wages - v["j_reg"] - v["j_over"]
        d_l = {l: lev[l] - v["levels"][l] for l in S.LEVELS}
        worst = max(worst, abs(d_w), *map(abs, d_l.values()))
        print(f"{name}: wages+overtime columns {wages:,.4f} model {v['j_reg'] + v['j_over']:,.4f}  levels columns "
              f"{ {l: round(lev[l], 4) for l in S.LEVELS} } model { {l: round(v['levels'][l], 4) for l in S.LEVELS} }")
    print(f"ACCOUNTING largest difference {worst:.2e}")

    master = S.Master(P, limits)
    for i in P.staff:
        master.add_column(P.column(i, P.shifts_by_day(i, [s for j, s in x0 if j == i])))
    master.add_cut(prob.cut(x0)[0])
    full = S.Problem.__new__(S.Problem)
    full.__dict__.update(P.__dict__)
    full.pricer = {i: S.Pricer(P.dr[i], P.rules, "full") for i in P.staff}
    for it in range(3):
        value, pi = master.solve_lp()
        ta = tf = 0.0
        diff, added, rounds = 0.0, 0, []
        for i in P.staff:
            rc_day, g_w = master.reduced_costs(i, pi)
            t = time.time()
            la, pa, ra = P.pricer[i].price(rc_day, g_w)
            ta += time.time() - t
            t = time.time()
            lf, pf, _ = full.pricer[i].price(rc_day, g_w)
            tf += time.time() - t
            rounds.append(ra)
            diff = max(diff, abs(la - lf))
            if pa is not None:
                w = sum(1 for j in pa if j is not None)
                direct = sum(rc_day[d][j] for d, j in enumerate(pa) if j is not None) + g_w[w]
                diff = max(diff, abs(direct - la))
                if la < -1e-6:
                    added += master.add_column(P.column(i, pa))
        print(f"iteration {it}: LP {value:,.2f}  adaptive {ta:.1f}s (rounds mean {np.mean(rounds):.1f} max {max(rounds)})  "
              f"full {tf:.1f}s  largest difference {diff:.2e}  +{added} columns")
    print("CHECK DONE")


if __name__ == "__main__":
    main()
