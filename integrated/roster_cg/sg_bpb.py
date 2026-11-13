"""SG-LNS-guided branch-roster column generation with lifted Benders cuts (the method of the paper), one cluster at a time.

A new method next to net_benders.py, branch_dw.py and surrogate/staff_surrogate.py, which it uses unchanged. Everything
runs within one time budget T, counted from the start of the run:

Phase 1  SG-LNS from the sequential schedule x0 (surrogate/sglns_cutprices.py), for a share of T, with its re-timing.
         Its surrogate starts from the stop prices of one lifted Benders cut at x0, so the heuristic is guided by the
         Benders prices from the start (--sglns-prices direct: the one-trip estimate, as an ablation).
Phase 2  Column generation over branch rosters (Dantzig-Wolfe, one block per branch):
         Master   min sum_c cost_c lam_c + theta
                  s.t. sum_{c of b} lam_c = 1 (every branch b);  sum_c lev_{l,c} lam_c <= Vbar_l (every level l);
                       theta >= sum_c (sum_{s in stops(c)} pi^k_s) lam_c (every lifted Benders cut k).
         Columns  the branch rosters of x0 and of the SG-LNS schedule, with a lifted cut at each.
         Pricing  (a) heuristic, in the spirit of SG-LNS: per branch, the roster of largest weight in the master LP is kept
                  except for a few staff (riders first), and the branch pricing MIP is solved over those staff;
                  (b) exact: the full branch pricing MIP, when (a) finds no column and at least every --exact-every
                  iterations. Exact pricing runs at smoothed duals (Wentges 1997); if that finds no column with negative
                  reduced cost at the LP duals, it is repeated at the LP duals.
         Bound    L(pi) = sum_l dual_l Vbar_l + sum_b min over rosters of b of (cost - sum_l dual_l lev_l
                  + sum_k mu_k cut_k), from the best bounds of the exact pricing MIPs. Valid for every pi with dual_l <= 0,
                  mu_k >= 0 and sum_k mu_k <= 1 (see revision/method_notes/sg_cg_correctness.md).
         Primal   the master as an integer program over the columns, routed exactly, a new lifted cut, and column
                  generation resumes; at the end, the re-timing of net_benders.py on the best plan.
Checks   every cut below the real transport cost of every routed plan; Z recomputed from the schedule; the levels of the best
         plan within the limits; bound below the best plan; at the LP duals, the pricing objective at its incumbent equals
         the reduced cost of the column built from it; at every integer-master plan, the chosen columns add up to the staff
         model on the whole cluster.

Switches for the ablations: --sglns-frac 0 (no SG-LNS phase), --exact-every 0 (no heuristic pricing), --smooth 0
(no smoothing), --no-lift (cuts without lifting), --benders-start (cuts and schedule of a finished Benders run).

Usage: python sg_bpb.py <data.xlsx> <benders_dir> <out_dir> --cluster <prefix> [--time T] [--seed 1]
       (benders_dir gives x0, the service limits and, by default, T = the time of the Benders run)
"""
import argparse
import copy
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import cplex

import branch_dw as D

B, LEVELS = D.B, D.LEVELS
ROOT = Path(__file__).resolve().parents[2]


class Master(D.Master):
    def solve_lp(self):
        self.c.set_problem_type(self.c.problem_type.LP)
        self.c.solve()
        assert self.c.solution.is_primal_feasible(), self.c.solution.get_status_string()
        duals = dict(zip(self.c.linear_constraints.get_names(), self.c.solution.get_dual_values()))
        return self.c.solution.get_objective_value(), duals, self.c.solution.get_values()

    def solve_ip(self, time_limit, threads, start):
        """The integer master (theta continuous, columns binary), started from one column per branch (start: column
        indices) with theta at its least value; then back to the LP."""
        n = self.c.variables.get_num()
        self.c.variables.set_types([(j, "C" if j == 0 else "B") for j in range(n)])
        vals = [0.0] * n
        for j in start:
            vals[j + 1] = 1.0
        vals[0] = max([sum(sum(pr.get(s, 0.0) for s in self.cols[j]["stops"]) for j in start) for pr in self.cuts] + [0.0])
        self.c.MIP_starts.delete()
        self.c.MIP_starts.add(cplex.SparsePair(list(range(n)), vals), self.c.MIP_starts.effort_level.repair)
        self.c.parameters.timelimit.set(time_limit)
        self.c.parameters.threads.set(threads)
        self.c.parameters.mip.tolerances.mipgap.set(1e-6)
        self.c.solve()
        ok = self.c.solution.is_primal_feasible()
        chosen = [self.cols[j - 1] for j, v in enumerate(self.c.solution.get_values()) if j > 0 and v > 0.5] if ok else []
        value = self.c.solution.get_objective_value() if ok else math.inf
        self.c.MIP_starts.delete()
        self.c.variables.set_types([(j, "C") for j in range(n)])
        self.c.set_problem_type(self.c.problem_type.LP)
        return chosen, value


def reduced_cost(col, kb, cuts, duals):
    """Reduced cost of a column at the LP duals."""
    return col["cost"] - sum(duals[f"lev_{l}"] * col["lev"][l] for l in LEVELS) \
        + sum(duals[f"cut_{k}"] * sum(pr.get(s, 0.0) for s in col["stops"]) for k, pr in enumerate(cuts)) - duals[f"conv_{kb}"]


def stop_prices(cuts, mu):
    out = {}
    for k, pr in enumerate(cuts):
        if mu[k] > 1e-12:
            for s, v in pr.items():
                out[s] = out.get(s, 0.0) + mu[k] * v
    return out


def run(prob, x0, limits, cuts0, warm, a, log):
    t0 = time.time()
    stop_at = a.time - a.retime_time
    branches = sorted(prob.data.branch_ids)
    blocks = {b: D.Block(prob, b, copy.copy(prob.args)) for b in branches}
    master = Master(branches, limits)
    evaluator = B.StaffModel(prob.args, prob.data)
    vals0, plan0, z0 = prob.plan_cost(evaluator, x0)
    best = dict(z=z0, x=set(x0), vals=vals0, plan=plan0, round="sequential")
    seen, visited, keys = set(), [(set(x0), plan0.cost)], set()
    rng = random.Random(a.seed)
    riders = {i for rs in prob.net.riders.values() for i in rs}

    def stops_of(x):
        return frozenset(prob.net.demand(prob.events(x))[2])

    def add_cut_at(x):
        st = stops_of(x)
        if st not in seen:
            seen.add(st)
            master.add_cut(prob.cut(x)[0])

    def add_col(b, col):
        if (b, col["x"]) in keys:
            return 0
        master.add_column(b, col)
        keys.add((b, col["x"]))
        return 1

    for prices in cuts0:
        master.add_cut(prices)
    add_cut_at(x0)
    for b in branches:
        add_col(b, blocks[b].column(x0, prob))
    for name, xw in warm:
        vals, plan, z = prob.plan_cost(evaluator, xw)
        ok = all(vals["levels"][l] <= limits[l] + 1e-6 for l in LEVELS)
        visited.append((set(xw), plan.cost))
        log(f"start {name}: routed Z {z:,.2f}  within limits {ok}")
        if ok and z < best["z"]:
            best = dict(z=z, x=set(xw), vals=vals, plan=plan, round=f"start {name}")
        add_cut_at(xw)
        for b in branches:
            add_col(b, blocks[b].column(xw, prob))
    log(f"setup {time.time() - t0:.0f}s  sequential Z {z0:,.2f}  best start {best['z']:,.2f} ({best['round']})  "
        f"cuts {len(master.cuts)}  columns {len(master.cols)}  limits {limits}")

    lower, center, l_center, history, outer, lp_checks, acct = -math.inf, None, -math.inf, [], 0, [], []

    def smoothed(duals, alpha):
        lev = {l: alpha * center["lev"][l] + (1 - alpha) * duals[f"lev_{l}"] for l in LEVELS}
        mu = {k: alpha * center["mu"].get(k, 0.0) + (1 - alpha) * duals[f"cut_{k}"] for k in range(len(master.cuts))}
        return lev, mu

    while time.time() - t0 < stop_at:
        since_exact = 0
        while time.time() - t0 < stop_at:
            value, duals, xv = master.solve_lp()
            tp = time.time()
            if since_exact < a.exact_every:
                # heuristic pricing at the smoothed duals (the LP duals before the first exact pricing)
                since_exact += 1
                lev, mu = smoothed(duals, a.smooth) if center is not None else \
                    ({l: duals[f"lev_{l}"] for l in LEVELS}, {k: duals[f"cut_{k}"] for k in range(len(master.cuts))})
                sp, added = stop_prices(master.cuts, mu), 0
                for kb, b in enumerate(branches):
                    js = [j for j, col in enumerate(master.cols) if col["branch"] == b]
                    ref = master.cols[max(js, key=lambda j: xv[j + 1])]["x"]
                    staff = blocks[b].staff
                    ride = [i for i in staff if i in riders]
                    free = set(rng.sample(ride, min(len(ride), a.free // 2)))
                    rest = [i for i in staff if i not in free]
                    free |= set(rng.sample(rest, min(len(rest), a.free - len(free))))
                    xvars = blocks[b].sm.solver.x
                    for (i, s), var in xvars.items():
                        if i not in free:
                            var.lb = var.ub = 1 if (i, s) in ref else 0
                    x_on, _rb, _info = blocks[b].price(lev, sp, 0.0, a.heur_time, 1e-4, a.threads)
                    for var in xvars.values():
                        var.lb, var.ub = 0, 1
                    if x_on is None:
                        continue
                    col = blocks[b].column(x_on, prob)
                    if reduced_cost(col, kb, master.cuts, duals) < -1e-6 * max(1.0, abs(value)):
                        added += add_col(b, col)
                history.append(dict(outer=outer, kind="heuristic", lp=value, added=added, columns=len(master.cols), lb=lower,
                                    best=best["z"], seconds=time.time() - t0))
                log(f"outer {outer}  LP {value:,.2f}  heuristic +{added} cols ({len(master.cols)})  LB {lower:,.2f}  "
                    f"best {best['z']:,.2f}  {time.time() - tp:.0f}s pricing  {time.time() - t0:.0f}s")
                if added:
                    continue
            since_exact = 0
            alpha = a.smooth if center is not None else 0.0
            while True:
                if alpha:
                    lev, mu = smoothed(duals, alpha)
                else:
                    lev, mu = {l: duals[f"lev_{l}"] for l in LEVELS}, {k: duals[f"cut_{k}"] for k in range(len(master.cuts))}
                assert sum(mu.values()) <= 1 + 1e-6 and min(mu.values(), default=0) >= -1e-6 and max(lev.values()) <= 1e-6
                # remove round-off so that the conditions of the bound hold exactly
                lev = {l: min(0.0, v) for l, v in lev.items()}
                mu = {k: max(0.0, v) for k, v in mu.items()}
                tot = sum(mu.values())
                if tot > 1.0:
                    mu = {k: v / tot for k, v in mu.items()}
                sp = stop_prices(master.cuts, mu)
                # each branch gets at most its share of the time left, so that the budget holds
                share = max(10.0, (stop_at - (time.time() - t0)) / len(branches))
                res = [(kb, b, *blocks[b].price(lev, sp, 0.0, min(a.price_time, share), a.price_gap, a.threads))
                       for kb, b in enumerate(branches)]
                bound = sum(lev[l] * limits[l] for l in LEVELS) + sum(raw for _kb, _b, _x, raw, _i in res)

                if bound > l_center:
                    center, l_center = dict(lev=lev, mu=mu), bound
                lower = max(lower, bound)
                added = 0
                for kb, b, x_on, _raw, info in res:
                    if x_on is not None:
                        col = blocks[b].column(x_on, prob)
                        if alpha == 0.0:
                            # the pricing objective at its incumbent equals the reduced cost of the column built from it
                            lp_checks.append(info["objective"] - duals[f"conv_{kb}"] - reduced_cost(col, kb, master.cuts, duals))
                        if reduced_cost(col, kb, master.cuts, duals) < -1e-6 * max(1.0, abs(value)):
                            added += add_col(b, col)
                history.append(dict(outer=outer, kind="exact", alpha=alpha, lp=value, bound=bound, lb=lower, added=added,
                                    columns=len(master.cols), best=best["z"], seconds=time.time() - t0,
                                    pricing_gaps=[i["mip_gap"] for _kb, _b, _x, _r, i in res]))
                log(f"outer {outer}  LP {value:,.2f}  exact a={alpha:.2f}  bound {bound:,.2f}  LB {lower:,.2f}  best {best['z']:,.2f}  "
                    f"gap {(best['z'] - lower) / best['z']:.3%}  +{added} cols ({len(master.cols)})  {time.time() - tp:.0f}s pricing  "
                    f"{time.time() - t0:.0f}s")
                if added or alpha == 0.0 or time.time() - t0 >= stop_at:
                    break
                alpha = 0.0  # mispricing: repeat at the LP duals
            if added == 0:
                break
        if time.time() - t0 >= stop_at and outer > 0:
            break
        # start: the rosters of the best plan if all of them are columns, else those of x0 (the first column of each
        # branch), so that the start always meets the service limits
        start = []
        for b in branches:
            staff_b = set(blocks[b].staff)
            own = frozenset(k for k in best["x"] if k[0] in staff_b)
            start.append(next((j for j, c in enumerate(master.cols) if c["branch"] == b and c["x"] == own), None))
        if None in start:
            start = [next(j for j, c in enumerate(master.cols) if c["branch"] == b) for b in branches]
        chosen, ip_value = master.solve_ip(min(300, max(30, stop_at - (time.time() - t0))), a.threads, start)
        if not chosen:
            log("integer master found no solution")
            break
        x_new = set().union(*[set(c["x"]) for c in chosen])
        vals, plan, z = prob.plan_cost(evaluator, x_new)
        visited.append((set(x_new), plan.cost))
        # the chosen branch columns add up to the staff model on the whole cluster
        acct.append(max([abs(sum(c["cost"] for c in chosen) - vals["j_reg"] - vals["j_over"])]
                        + [abs(sum(c["lev"][l] for c in chosen) - vals["levels"][l]) for l in LEVELS]))
        ok = all(vals["levels"][l] <= limits[l] + 1e-6 for l in LEVELS)
        log(f"outer {outer}: integer master {ip_value:,.2f}  routed Z {z:,.2f}  within limits {ok}")
        if ok and z < best["z"]:
            best = dict(z=z, x=set(x_new), vals=vals, plan=plan, round=f"outer {outer}")
        history.append(dict(outer=outer, kind="primal", ip=ip_value, routed=z, within=ok, best=best["z"], lb=lower, seconds=time.time() - t0))
        outer += 1
        if stops_of(x_new) in seen:
            log("the integer master repeats a stop set: no new cut")
            break
        add_cut_at(x_new)
        if (best["z"] - lower) / best["z"] <= 1e-4:
            break
    z_before = best["z"]
    best = prob.retime(best, limits, evaluator, 1, max(60, a.time - (time.time() - t0)), log)
    visited.append((set(best["x"]), best["plan"].cost))
    worst = max(sum(pr.get(s, 0.0) for s in stops_of(xs)) - cost for xs, cost in visited for pr in master.cuts)
    reg, over = prob.real_wages(best["x"])
    z_check = reg + over + best["plan"].cost - best["z"]
    within = all(best["vals"]["levels"][l] <= limits[l] + 1e-6 for l in LEVELS)
    checks = dict(cut_violation=worst, z_check=z_check, within_limits=within, bound_below_best=lower <= best["z"] + 1e-6,
                  pricing_vs_column=max(map(abs, lp_checks), default=0.0), columns_vs_model=max(acct, default=0.0))
    log(f"checks: largest cut violation {worst:.6f}; Z recomputed differs by {z_check:.6f}; levels within limits {within}; "
        f"bound below best {checks['bound_below_best']}; pricing objective vs column reduced cost {checks['pricing_vs_column']:.2e}; "
        f"chosen columns vs staff model {checks['columns_vs_model']:.2e}")
    return best, z_before, lower, history, master, checks


def sglns(a, cluster, x0_csv, seconds, out, log):
    """Phase 1: SG-LNS from x0, its opening prices from one lifted cut at x0 (surrogate/sglns_cutprices.py), for the given
    time (cut included), then its re-timing. Returns its schedule file."""
    cmd = [sys.executable, "-u", str(ROOT / "surrogate" / "sglns_cutprices.py"), "--cluster", cluster, "--data", a.data,
           "--schedule", str(x0_csv), "--time-limit", f"{seconds:.0f}", "--seed", str(a.seed), "--threads", str(a.threads),
           "--prices", a.sglns_prices, "--out", str(out)] + (["--no-lift"] if a.no_lift else [])
    with open(Path(out).with_suffix(".log"), "w") as f:
        code = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    assert code == 0, f"SG-LNS failed with exit code {code}"
    res = json.load(open(out))
    log(f"SG-LNS: Z {res['z']:,.2f} (search {res['z_search']:,.2f})")
    return Path(out).with_suffix(".schedule.csv")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data")
    p.add_argument("benders_dir", help="gives x0 (<stem>_<cluster>_x0.csv), the service limits and the default time")
    p.add_argument("out_dir")
    p.add_argument("--cluster", required=True)
    p.add_argument("--time", type=float, default=None, help="seconds in all (SG-LNS, column generation, re-timing)")
    p.add_argument("--sglns-frac", type=float, default=0.3, help="share of the time for the SG-LNS search; 0: no SG-LNS phase")
    p.add_argument("--sglns-prices", choices=["cut", "direct"], default="cut",
                   help="opening prices of the SG-LNS surrogate: the lifted cut at x0, or (ablation) the one-trip estimate")
    p.add_argument("--retime-time", type=float, default=600)
    p.add_argument("--free", type=int, default=8, help="staff freed per branch in heuristic pricing")
    p.add_argument("--heur-time", type=float, default=20)
    p.add_argument("--exact-every", type=int, default=3, help="0: exact pricing only")
    p.add_argument("--smooth", type=float, default=0.7, help="0: no smoothing")
    p.add_argument("--price-time", type=float, default=120)
    p.add_argument("--price-gap", type=float, default=1e-4)
    p.add_argument("--no-lift", action="store_true", help="ablation: cuts without lifting")
    p.add_argument("--benders-start", action="store_true", help="ablation: start from the cuts and schedule of the Benders run")
    p.add_argument("--threads", type=int, default=3)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--tag", default="sgcg")
    a = p.parse_args()
    assert not (a.no_lift and a.benders_start), "the Benders cuts are lifted: --no-lift with --benders-start is not a clean ablation"
    t_all = time.time()
    # phase 1 runs its search for sglns_frac * T, then up to 1200 s of re-timing; phase 2 needs room after that
    if a.time and a.sglns_frac > 0:
        assert (1 - a.sglns_frac) * a.time - 1200 >= a.retime_time + 300, "too little time for phase 2: lower --sglns-frac or raise --time"
    log = lambda s: print(f"[{time.time() - t_all:.0f}s] {s}", flush=True)  # noqa: E731
    cluster = next(c for c in set(D.branch_cluster(a.data).values()) if c.startswith(a.cluster))
    stem = Path(a.data).stem
    bd = Path(a.benders_dir)
    bres = json.load(open(bd / f"{stem}_{cluster[:8]}.json"))
    total = a.time or bres["benders_seconds"]
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = f"{stem}_{cluster[:8]}_{a.tag}_s{a.seed}"
    x0_csv = bd / f"{stem}_{cluster[:8]}_x0.csv"
    starts = []
    if a.sglns_frac > 0:
        starts.append(("SG-LNS", sglns(a, cluster, x0_csv, a.sglns_frac * total, out / f"{name}_sglns.json", log)))
    prob = B.ClusterProblem(cluster, a.data, B.staff_args(), a.threads)
    if a.no_lift:
        prob.closed_price = "zero"
    x0 = B.load_schedule(x0_csv, prob.data.staff_ids)
    limits = {int(k): v for k, v in bres["limits"].items()}
    cuts0 = []
    if a.benders_start:
        cuts0 = [{(b, ac, int(t)): pr for b, ac, t, pr in c} for c in bres["cuts"]]
        starts.append(("benders", bd / f"{stem}_{cluster[:8]}_schedule.csv"))
    warm = [(n, B.load_schedule(f, prob.data.staff_ids)) for n, f in starts]
    a.time = total - (time.time() - t_all)
    log(f"phase 2 gets {a.time:,.0f}s of {total:,.0f}s")
    assert a.time >= a.retime_time + 300, "too little time left for column generation: lower --sglns-frac or raise --time"
    best, z_before, lower, history, master, checks = run(prob, x0, limits, cuts0, warm, a, log)
    z0 = bres["z_seq"]
    res = dict(cluster=cluster[:8], z_seq=z0, z=best["z"], z_before_retime=z_before, lb=lower if math.isfinite(lower) else None,
               gap=(best["z"] - lower) / best["z"] if math.isfinite(lower) else None,
               saving=(z0 - best["z"]) / z0, benders_lb=bres["lb"], benders_z=bres["z"], source=str(best["round"]),
               wages=best["vals"]["j_reg"], overtime=best["vals"]["j_over"], levels=best["vals"]["levels"], limits=limits,
               plan=B._parts(best["plan"]), columns=len(master.cols), cuts=len(master.cuts), budget=total,
               seconds=time.time() - t_all, checks=checks,
               settings={k: v for k, v in vars(a).items() if k not in ("data", "benders_dir", "out_dir")}, history=history)
    with open(out / f"{name}.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    B.save_schedule(best["x"], prob.shift, out / f"{name}_schedule.csv")
    log(f"DONE z {best['z']:,.2f} lb {lower:,.2f} gap {(best['z'] - lower) / best['z']:.3%} saving {res['saving']:.3%}  time {res['seconds']:,.0f}s of {total:,.0f}s  "
        f"(Benders: lb {bres['lb']:,.2f} z {bres['z']:,.2f})")


if __name__ == "__main__":
    main()
