"""Tables for E5 (penalty scale M) and E7-E11 (routing settings) on the original data.

Usage: python make_tables_extra.py   (run from this folder)
"""
import csv
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CLUSTERS = ["1aea822f", "662b3495", "a1e8b5a7", "ba5b9796", "df75e3dc"]
LABEL = {c: f"Cluster {i + 1}" for i, c in enumerate(CLUSTERS)}


def e5():
    """M scaled by 0.1, 1, 10; all runs with 20-minute masters, so the M = 1 row comes from benders/."""
    out = []
    for c in ("1aea822f", "a1e8b5a7"):
        for m, path in (("0.1", f"e5_e6/e5_m0.1_{c}.json"), ("1", glob.glob(os.path.join(ROOT, f"benders/*{c}.json"))[0]),
                        ("10", f"e5_e6/e5_m10_{c}.json")):
            r = json.load(open(os.path.join(ROOT, path)))
            out.append((LABEL[c], m, r["z_seq"], r["z"], 100 * r["saving"], 100 * r["gap"]))
    with open(os.path.join(HERE, "table_e5_penalty_scale.tex"), "w") as f:
        f.write("\\begin{tabular}{lrrrrr}\n\\toprule\nCluster & Scale of $M$ & Sequential $\\tilde Z$ & Integrated $\\tilde Z$ & Saving (\\%) & Gap (\\%) \\\\\n\\midrule\n")
        for i, (c, m, zs, z, sv, gp) in enumerate(out):
            if i == 3:
                f.write("\\midrule\n")
            f.write(f"{c if m == '0.1' else ''} & {m} & {zs:,.0f} & {z:,.0f} & {sv:.2f} & {gp:.2f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def e7_e11():
    """Routing cost of the sequential schedule under each setting, summed over the clusters the setting applies to."""
    rows = list(csv.DictReader(open(os.path.join(ROOT, "e7_e11_routing.csv"))))
    base = {r["cluster"]: r for r in rows if r["setting"] == "base"}
    names = {"base": "Base (30-min window, 40 km/h, 1-min grid)", "window15": "Window 15 min", "window60": "Window 60 min",
             "speed30": "Speed 30 km/h", "speed50": "Speed 50 km/h", "fixed_x4": "Fixed bus cost $\\times 4$",
             "theta_off": "No punctuality in routing", "step5": "5-min departure grid", "step10": "10-min departure grid",
             "stationed": "Buses stationed at each accommodation"}
    with open(os.path.join(HERE, "table_e7_e11_routing.tex"), "w") as f:
        f.write("\\begin{tabular}{lrrrrrr}\n\\toprule\nSetting & Clusters & Bus cost & Change (\\%) & Driving & Punctuality & Buses \\\\\n\\midrule\n")
        for key, name in names.items():
            rs = [r for r in rows if r["setting"] == key]
            tot = lambda k, rr: sum(float(r[k]) for r in rr)
            b = [base[r["cluster"]] for r in rs]
            ch = 100 * (tot("cost", rs) - tot("cost", b)) / tot("cost", b)
            f.write(f"{name} & {len(rs)} & {tot('cost', rs):,.0f} & {ch:+.1f} & {tot('driving', rs):,.0f} & {tot('punctuality', rs):,.0f} & "
                    f"{int(tot('buses', rs))} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


if __name__ == "__main__":
    e5()
    e7_e11()
    print("done")
