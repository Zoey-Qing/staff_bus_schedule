"""Build summary.csv of the equal-time comparison (saving vs x0, lower bound, gap, seconds)."""
import csv, glob, json, os
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H = os.path.join(R, "compare_all")
rows = []
for bf in sorted(glob.glob(os.path.join(R, "benders", "*.json"))):
    c = bf[-13:-5]
    b = json.load(open(bf)); z0 = b["z_seq"]; T = b["benders_seconds"]
    def add(m, z, lb, sec, note=""):
        rows.append(dict(cluster=c, method=m, z=round(z, 2), saving_pct=round(100 * (z0 - z) / z0, 3),
                         lb=round(lb, 2) if lb else "", gap_pct=round(100 * (z - lb) / z, 3) if lb else "",
                         seconds=round(sec) if sec else "", T=round(T), note=note))
    add("B1", z0, None, None)
    add("M1", b["z"], b["lb"], T, "reused benders/")
    n = json.load(open(glob.glob(os.path.join(R, "benders_nolift", f"*{c}.json"))[0]))
    add("L2", n["z_benders"], n["lb"], n["benders_seconds"], "reused benders_nolift/, z before re-timing")
    s = json.load(open(glob.glob(os.path.join(R, "ablation_lns_fit", f"*{c}_s1.json"))[0]))
    add("M3", s["z"], None, s["seconds"], "reused ablation_lns_fit/ s1")
    d = os.path.join(H, c)
    for m, pat in (("L1", "L1/*.json"), ("L3", "L3/*.json"), ("L4", "L4/*.json"), ("M2", "M2/*.json")):
        f = glob.glob(os.path.join(d, pat))
        if f:
            j = json.load(open(f[0])); add(m, j["z"], j.get("lb"), j.get("seconds"))
    f = glob.glob(os.path.join(d, "m5eq_*_s1.json")); s1 = glob.glob(os.path.join(d, "m5eq_stage1", "*.json"))
    if f and s1:
        j = json.load(open(f[0])); k = json.load(open(s1[0]))
        wall = open(os.path.join(d, "m5eq_wall.txt")).read().split()
        add("M5eq", min(j["z"], k["z"]), k["lb"], float(wall[wall.index("total") + 1]), "bound from stage 1")
with open(os.path.join(H, "summary.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows: print(r)
