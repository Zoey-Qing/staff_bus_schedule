"""Checks of sg_bpb.py / branch_dw.py on one cluster (a few minutes, no experiment).

1. Decomposition: for each given schedule, the branch columns added over the branches give the wages, overtime and levels of
   StaffModel.value_of on the whole cluster, and their stops are the stops of the schedule.
2. Reduced costs: for a column added to the master, sg_bpb.reduced_cost equals obj - (column of the CPLEX matrix) . duals.
3. Pricing: at the LP duals, the branch pricing objective at its incumbent minus the convexity dual equals the reduced cost of
   the column built from it, and the Lagrangian bound lies below the LP value.

Usage: python check_sg_cg.py <data.xlsx> <benders_dir> --cluster <prefix> [--schedules a.csv ...]
"""
import argparse
import copy
import json
from pathlib import Path

import sg_bpb as S

D, B, LEVELS = S.D, S.B, S.LEVELS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data")
    p.add_argument("benders_dir")
    p.add_argument("--cluster", required=True)
    p.add_argument("--schedules", nargs="*", default=[])
    p.add_argument("--price-time", type=float, default=60)
    a = p.parse_args()
    cluster = next(c for c in set(D.branch_cluster(a.data).values()) if c.startswith(a.cluster))
    stem = Path(a.data).stem
    bd = Path(a.benders_dir)
    bres = json.load(open(bd / f"{stem}_{cluster[:8]}.json"))
    prob = B.ClusterProblem(cluster, a.data, B.staff_args(), 4)
    limits = {int(k): v for k, v in bres["limits"].items()}
    x0 = B.load_schedule(bd / f"{stem}_{cluster[:8]}_x0.csv", prob.data.staff_ids)
    branches = sorted(prob.data.branch_ids)
    blocks = {b: D.Block(prob, b, copy.copy(prob.args)) for b in branches}
    sm = B.StaffModel(prob.args, prob.data)

    worst1 = 0.0
    for name, x in [("x0", x0)] + [(Path(f).name, B.load_schedule(f, prob.data.staff_ids)) for f in a.schedules]:
        cols = [blocks[b].column(x, prob) for b in branches]
        v = sm.value_of(x, 4)
        d_cost = sum(c["cost"] for c in cols) - v["j_reg"] - v["j_over"]
        d_lev = {l: sum(c["lev"][l] for c in cols) - v["levels"][l] for l in LEVELS}
        stops_ok = set().union(*[c["stops"] for c in cols]) == set(prob.net.demand(prob.events(x))[2])
        reg, over = prob.real_wages(x)
        worst1 = max(worst1, abs(d_cost), *map(abs, d_lev.values()), abs(reg + over - v["j_reg"] - v["j_over"]))
        print(f"1 {name}: cost diff {d_cost:.2e}  level diffs { {l: round(d, 9) for l, d in d_lev.items()} }  stops equal {stops_ok}  "
              f"real_wages diff {reg + over - v['j_reg'] - v['j_over']:.2e}")
        assert stops_ok

    master = S.Master(branches, limits)
    master.add_cut(prob.cut(x0)[0])
    for b in branches:
        master.add_column(b, blocks[b].column(x0, prob))
    value, duals, _xv = master.solve_lp()
    names = master.c.linear_constraints.get_names()
    pi = master.c.solution.get_dual_values()
    lev = {l: duals[f"lev_{l}"] for l in LEVELS}
    mu = {k: duals[f"cut_{k}"] for k in range(len(master.cuts))}
    sp = S.stop_prices(master.cuts, mu)
    worst2 = worst3 = 0.0
    bound = sum(lev[l] * limits[l] for l in LEVELS)
    for kb, b in enumerate(branches):
        x_on, raw, info = blocks[b].price(lev, sp, 0.0, a.price_time, 1e-6, 4)
        bound += raw
        col = blocks[b].column(x_on, prob)
        rc = S.reduced_cost(col, kb, master.cuts, duals)
        worst3 = max(worst3, abs(info["objective"] - duals[f"conv_{kb}"] - rc))
        master.add_column(b, col)
        j = len(master.cols)
        ind, val = master.c.variables.get_cols(f"lam_{j - 1}").unpack()
        rc_cplex = master.c.objective.get_linear(f"lam_{j - 1}") - sum(v * pi[i] for i, v in zip(ind, val))
        worst2 = max(worst2, abs(rc_cplex - rc))
        print(f"2/3 {b}: reduced cost {rc:,.4f}  from CPLEX matrix {rc_cplex:,.4f}  pricing objective - conv "
              f"{info['objective'] - duals[f'conv_{kb}']:,.4f}  pricing gap {info['mip_gap']:.1e}")
    print(f"LP {value:,.2f}  Lagrangian bound {bound:,.2f}  bound <= LP {bound <= value + 1e-6}")
    print(f"CHECK 1 decomposition largest difference {worst1:.2e}")
    print(f"CHECK 2 reduced cost vs CPLEX matrix largest difference {worst2:.2e}")
    print(f"CHECK 3 pricing objective vs column reduced cost largest difference {worst3:.2e}")
    assert len(names) == len(pi)


if __name__ == "__main__":
    main()
