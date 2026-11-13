"""Summary of the M5-eq ablations (seed 1) against the full method and the single-stage rows of compare_all/.

saving_pct = (Z0 - Z) / Z0;  share_max_pct = (Z0 - Z) / (Z0 - LB*), LB* = the best lower bound found for the cluster by
any method in compare_all/ or here;  gap_pct = (Z - LB) / Z with the variant's own stage-1 bound.
A1 and A2 reuse the stage 1 of compare_all/ (same LB). On df75e3dc, A1 is taken from A1_prices_direct_rerun/ (see its NOTE).
Writes summary.csv and pivot.csv next to this file.
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CA = HERE.parent / "compare_all"
S = "Berkeley_Mock_Data_25_Stores_original_unbalanced"
CL = ["1aea822f", "662b3495", "a1e8b5a7", "ba5b9796", "df75e3dc"]
NAME = {c: f"C{k + 1}" for k, c in enumerate(CL)}
VARIANTS = ["A1_prices_direct", "A2_start_x0", "A3_no_heur", "A4_no_smooth", "A5_split06", "A6_no_lift"]
LABEL = {"M5eq": "full method", "A1_prices_direct": "stage-2 prices: one-trip estimate",
         "A2_start_x0": "stage 2 starts from x0", "A3_no_heur": "exact pricing only", "A4_no_smooth": "no dual smoothing",
         "A5_split06": "stage 1 = 0.6 Delta", "A6_no_lift": "cuts and prices without lifting",
         "M2": "stage 1 only", "M3": "stage 2 only (from x0, one-trip prices)", "L3": "stage 1 only, no lifting"}

rows_ca = list(csv.DictReader(open(CA / "summary.csv")))


def ca(c, m):
    return next((r for r in rows_ca if r["cluster"] == c and r["method"] == m), None)


def within(levels, limits):
    return all(float(levels[str(l)] if str(l) in levels else levels[l]) <= float(limits[str(l)] if str(l) in limits else limits[l]) + 1e-6
               for l in (1, 2, 3, 4))


def stage2_row(path, lb):
    d = json.load(open(path))
    lim = d["sequential"]["levels"]
    ok = within(d["retime_vals"]["levels"], lim) if d.get("retime_vals") else within(d["best"]["levels"], lim)
    return dict(z=d["z"], z0=d["z_seq"], lb=lb, within=ok)


out = []
for c in CL:
    z0 = float(ca(c, "M5eq")["z"]) / (1 - float(ca(c, "M5eq")["saving_pct"]) / 100)
    lb_full = float(ca(c, "M5eq")["lb"])
    rows = [dict(variant="M5eq", z=float(ca(c, "M5eq")["z"]), lb=lb_full, within=True, wall=None)]
    for v in VARIANTS:
        if v == "A2_start_x0":
            d = HERE / c / "A1_prices_direct" / "A2_start_x0"
        elif v == "A1_prices_direct" and (HERE / c / "A1_prices_direct_rerun").exists():
            d = HERE / c / "A1_prices_direct_rerun"
        else:
            d = HERE / c / v
        f = d / f"{S}_{c}_s1.json"
        if not f.exists():
            rows.append(dict(variant=v, z=None, lb=None, within=None, wall=None))
            continue
        if v in ("A1_prices_direct", "A2_start_x0"):
            lb, ok1 = lb_full, True
        else:
            s1 = json.load(open(d / "stage1" / f"{S}_{c}_stage1_s1.json"))
            lb = s1["lb"]
            ok1 = all(bool(s1["checks"][k]) for k in ("within_limits", "bound_below_best")) and s1["checks"]["cut_violation"] <= 1e-4
        r = stage2_row(f, lb)
        wall = (d / "wall.txt").read_text().split() if (d / "wall.txt").exists() else None
        rows.append(dict(variant=v, z=r["z"], lb=lb, within=r["within"] and ok1,
                         wall=dict(zip(wall[0::2], map(float, wall[1::2]))) if wall else None))
        z0 = r["z0"]
    for m in ("M2", "M3", "L3"):
        r = ca(c, m)
        rows.append(dict(variant=m, z=float(r["z"]), lb=float(r["lb"]) if r["lb"] else None, within=True, wall=None))
    lbs = [float(r["lb"]) for r in rows_ca if r["cluster"] == c and r["lb"]] + [r["lb"] for r in rows if r["lb"] is not None]
    lb_star = max(lbs)
    for r in rows:
        z = r["z"]
        out.append(dict(cluster=NAME[c], id=c, variant=r["variant"], what=LABEL[r["variant"]],
                        z=None if z is None else round(z, 2),
                        saving_pct=None if z is None else round(100 * (z0 - z) / z0, 3),
                        share_max_pct=None if z is None else round(100 * (z0 - z) / (z0 - lb_star), 1),
                        lb=None if r["lb"] is None else round(r["lb"], 2),
                        gap_pct=None if (z is None or r["lb"] is None) else round(100 * (z - r["lb"]) / z, 3),
                        lb_star=round(lb_star, 2), z0=round(z0, 2), checks_ok=r["within"],
                        total_s=(r["wall"] or {}).get("total"), T=(r["wall"] or {}).get("T")))

with open(HERE / "summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)
order = ["M5eq"] + VARIANTS + ["M2", "M3", "L3"]
with open(HERE / "pivot.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["variant", "what"] + [f"{NAME[c]}_{k}" for c in CL for k in ("saving", "share", "gap")]
               + ["mean_saving", "mean_share"])
    for v in order:
        rs = [next(r for r in out if r["id"] == c and r["variant"] == v) for c in CL]
        sv = [r["saving_pct"] for r in rs]
        sh = [r["share_max_pct"] for r in rs]
        w.writerow([v, LABEL[v]] + [x for r in rs for x in (r["saving_pct"], r["share_max_pct"], r["gap_pct"])]
                   + [round(sum(sv) / 5, 3) if None not in sv else None, round(sum(sh) / 5, 1) if None not in sh else None])
print(open(HERE / "pivot.csv").read())
for r in out:
    if r["checks_ok"] is False:
        print("CHECK FAILED", r)
