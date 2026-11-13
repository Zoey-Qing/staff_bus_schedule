"""Summary of the ablations WITHOUT dual smoothing (seed 1). Main method = ../ablation_m5eq/<c>/A4_no_smooth.

saving_pct = (Z0 - Z) / Z0;  share_max_pct = (Z0 - Z) / (Z0 - LB*), LB* = best lower bound of the cluster found by any
run in ../compare_all, ../ablation_m5eq or here;  gap_pct = (Z - LB) / Z with the run's own stage-1 bound.
A1 and A2 reuse the stage 1 of the main method (same LB). "with_smoothing" is the smoothed main method of ../compare_all
(M5eq); M3 and L3 come from ../compare_all (no smoothing there). Writes summary.csv and pivot.csv next to this file.
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE.parent
CA, AB = R / "compare_all", R / "ablation_m5eq"
S = "Berkeley_Mock_Data_25_Stores_original_unbalanced"
CL = ["1aea822f", "662b3495", "a1e8b5a7", "ba5b9796", "df75e3dc"]
NAME = {c: f"C{k + 1}" for k, c in enumerate(CL)}
LABEL = {"main": "main method (no smoothing)", "A1_prices_direct": "stage-2 prices: one-trip estimate",
         "A2_start_x0": "stage 2 starts from x0", "A3_no_heur": "exact pricing only", "A5_split06": "stage 1 = 0.6 Delta",
         "A6_no_lift": "cuts and prices without lifting", "M2_stage1_only": "stage 1 only (whole budget)",
         "with_smoothing": "main method with dual smoothing", "M3": "stage 2 only (from x0, one-trip prices)",
         "L3": "literature: branch CG, no lifting"}
ORDER = list(LABEL)
rows_ca = list(csv.DictReader(open(CA / "summary.csv")))


def ca(c, m):
    return next(r for r in rows_ca if r["cluster"] == c and r["method"] == m)


def lim_ok(levels, limits):
    g = lambda d, l: float(d[str(l)] if str(l) in d else d[l])
    return all(g(levels, l) <= g(limits, l) + 1e-6 for l in (1, 2, 3, 4))


def s1_ok(s1):
    ch = s1["checks"]
    return bool(ch["within_limits"]) and bool(ch["bound_below_best"]) and ch["cut_violation"] <= 1e-4


def two_stage(d, lb=None):
    c = d.parent.name if d.parent.name in CL else d.parent.parent.name
    j = json.load(open(d / f"{S}_{c}_s1.json"))
    lim = j["sequential"]["levels"]
    ok = lim_ok(j["retime_vals"]["levels"], lim) if j.get("retime_vals") else lim_ok(j["best"]["levels"], lim)
    if lb is None:
        s1 = json.load(open(d / "stage1" / f"{S}_{c}_stage1_s1.json"))
        lb, ok = s1["lb"], ok and s1_ok(s1)
    return dict(z=j["z"], z0=j["z_seq"], lb=lb, ok=ok)


out = []
for c in CL:
    main = two_stage(AB / c / "A4_no_smooth")
    z0 = main["z0"]
    rows = {"main": main}
    for v in ("A1_prices_direct", "A2_start_x0"):
        rows[v] = two_stage(HERE / c / v, lb=main["lb"])
    for v in ("A3_no_heur", "A5_split06", "A6_no_lift"):
        rows[v] = two_stage(HERE / c / v)
    m2 = json.load(open(HERE / c / "M2_stage1_only" / f"{S}_{c}_m2_s1.json"))
    rows["M2_stage1_only"] = dict(z=m2["z"], z0=m2["z_seq"], lb=m2["lb"], ok=s1_ok(m2))
    for v, m in (("with_smoothing", "M5eq"), ("M3", "M3"), ("L3", "L3")):
        r = ca(c, m)
        rows[v] = dict(z=float(r["z"]), z0=z0, lb=float(r["lb"]) if r["lb"] else None, ok=True)
    for r in rows.values():
        assert abs(r["z0"] - z0) < 1e-3 * z0, (c, r)
    # best bound over every run of the cluster
    lbs = [float(r["lb"]) for r in rows_ca if r["cluster"] == c and r["lb"]]
    lbs += [r["lb"] for r in rows.values() if r["lb"] is not None]
    for f in (AB / c).glob(f"*/stage1/{S}_{c}_stage1_s1.json"):
        lbs.append(json.load(open(f))["lb"])
    lb_star = max(lbs)
    for v in ORDER:
        r = rows[v]
        z = r["z"]
        out.append(dict(cluster=NAME[c], id=c, variant=v, what=LABEL[v], z=round(z, 2),
                        saving_pct=round(100 * (z0 - z) / z0, 3), share_max_pct=round(100 * (z0 - z) / (z0 - lb_star), 1),
                        lb=None if r["lb"] is None else round(r["lb"], 2),
                        gap_pct=None if r["lb"] is None else round(100 * (z - r["lb"]) / z, 3),
                        lb_star=round(lb_star, 2), z0=round(z0, 2), checks_ok=r["ok"]))

with open(HERE / "summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)
with open(HERE / "pivot.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["variant", "what"] + [f"{NAME[c]}_{k}" for c in CL for k in ("saving", "share", "gap")]
               + ["mean_saving", "mean_share"])
    for v in ORDER:
        rs = [next(r for r in out if r["id"] == c and r["variant"] == v) for c in CL]
        w.writerow([v, LABEL[v]] + [x for r in rs for x in (r["saving_pct"], r["share_max_pct"], r["gap_pct"])]
                   + [round(sum(r["saving_pct"] for r in rs) / 5, 3), round(sum(r["share_max_pct"] for r in rs) / 5, 1)])
print(open(HERE / "pivot.csv").read())
for c in CL:
    r = next(r for r in out if r["id"] == c)
    print(NAME[c], "Z0", r["z0"], "LB*", r["lb_star"], "max saving %", round(100 * (r["z0"] - r["lb_star"]) / r["z0"], 2))
for r in out:
    if r["checks_ok"] is False:
        print("CHECK FAILED", r)
