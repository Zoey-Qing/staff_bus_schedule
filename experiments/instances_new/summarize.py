"""Summary of the sensitivity runs (S_*) and new instances (G*, T*) of 2026-11-11 -> summary.csv.
z0 is taken from the search output (benders_sglns re-evaluates x0 under the scenario's parameters); the z_seq field of
the stage1 json is the copied base value and is NOT used. Run: python summarize.py
"""
import csv
import glob
import json
import os

R = "E:/schedule/revision"
rows = []


def row(name, base):
    s1 = glob.glob(base + "/stage1/*.json")
    s2 = glob.glob(base + "/*_s1.json")
    if not (s1 and s2):
        return
    s1, s2 = json.load(open(s1[0])), json.load(open(s2[0]))
    if "z_seq" not in s2:  # search still running (partial file)
        return
    z0, z, lb, x1 = s2["z_seq"], s2["z"], s1["lb"], s2["z_benders"]
    rows.append(dict(name=name, z0=round(z0, 2), z_x1=round(x1, 2), z=round(z, 2), lb=round(lb, 2),
                     saving_pct=round(100 * (z0 - z) / z0, 3), gap_pct=round(100 * (z - lb) / z, 3),
                     share_pct=round(100 * (z0 - z) / (z0 - lb), 2)))


row("base_C2", R + "/results_new/original/ablation_m5eq/a1e8b5a7/A4_no_smooth")
for d in sorted(glob.glob(R + "/instances_new/runs/*")):
    n = os.path.basename(d)
    if not n.endswith("_failed"):
        row(n, d)
with open(R + "/instances_new/summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, rows[0].keys())
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print(r)
