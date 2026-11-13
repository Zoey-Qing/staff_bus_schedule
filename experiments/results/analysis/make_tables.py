"""LaTeX tables from analyze_existing.py and verify_cuts.py. Writes to ../../../manuscript/tables/.

  table_gap_parts.tex   Benders gap split into the master's own gap and the cut's underestimate of transport
  table_cut_check.tex   numerical check of the cuts (dual feasibility and random stop sets)
  table_surrogate.tex   prediction and ranking quality of the surrogate, fitted against fixed coefficients
"""
import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "..", "..", "manuscript", "tables")
BASE = ["1aea822f", "662b3495", "a1e8b5a7", "ba5b9796", "df75e3dc"]
NET100 = ["1273d27b", "27ea17d0", "41835ef9", "46591601", "69f22899"]
ORDER = {"base": BASE, "far2": BASE, "net100": NET100}
SETNAME = {"base": "Base network", "far2": "Far network", "net100": "Large network"}


def label(st, c):
    return f"Cluster {ORDER[st].index(c) + 1}"


def gap_parts():
    d = pd.read_csv(os.path.join(HERE, "gap_parts.csv"), dtype={"cluster": str})
    with open(os.path.join(OUT, "table_gap_parts.tex"), "w") as f:
        f.write("\\begin{tabular}{@{}lrrrrr@{}}\n\\toprule\n & Gap & Master & Transport & $\\Theta/\\widetilde T$ & Rounds \\\\\n"
                " & (\\%) & gap (\\%) & under (\\%) & (\\%) & \\\\\n\\midrule\n")
        for st in ("base", "far2", "net100"):
            f.write(f"\\multicolumn{{6}}{{@{{}}l}}{{\\textit{{{SETNAME[st]}}}}} \\\\\n")
            g = d[d.set == st].copy()
            g["k"] = g.cluster.map(ORDER[st].index)
            for r in g.sort_values("k").itertuples():
                f.write(f"{label(st, r.cluster)} & {r.gap_pct:.2f} & {r.master_gap_pct:.2f} & {r.cut_under_pct:.2f} & "
                        f"{100 * r.theta_share:.1f} & {r.rounds} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def cut_check():
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "verify_cuts", "*_*.json"))):
        name = os.path.basename(f)[:-5]
        if name.startswith("test"):
            continue
        st, c = name.split("_")
        r = json.load(open(f))
        slack = [t["slack"] / t["lp"] for t in r["random"]]
        rows.append(dict(set=st, cluster=c, open=r["open_stops"], possible=r["possible_stops"], routes=r["lifted"]["trip_routes"],
                         viol=max(r["lifted"]["trip"], r["lifted"]["empty_run"], r["lifted"]["waiting"], r["lifted"]["fleet"]),
                         tests=len(slack), min_slack=100 * min(slack), neg=r["closed_price_stats"]["negative"]))
    with open(os.path.join(OUT, "table_cut_check.tex"), "w") as f:
        f.write("\\begin{tabular}{@{}lrrrrrr@{}}\n\\toprule\n & Open & Possible & Routes & Largest dual & Random & Smallest \\\\\n"
                " & stops & stops & per hour, bus type & violation & stop sets & slack (\\%) \\\\\n\\midrule\n")
        for st in ("base", "net100"):
            g = [r for r in rows if r["set"] == st]
            if not g:
                continue
            f.write(f"\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{SETNAME[st]}}}}} \\\\\n")
            for r in sorted(g, key=lambda r: ORDER[st].index(r["cluster"])):
                v = "$<10^{-13}$" if r["viol"] < 1e-13 else f"{r['viol']:.1e}"
                f.write(f"{label(st, r['cluster'])} & {r['open']:,} & {r['possible']:,} & {r['routes']:,} & {v} & {r['tests']} & "
                        f"{r['min_slack']:.1f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    return rows


def surrogate():
    q = pd.read_csv(os.path.join(HERE, "surrogate_quality.csv"), dtype={"cluster": str})
    s = pd.read_csv(os.path.join(HERE, "seeds.csv"), dtype={"cluster": str})
    q, s = q[q.set == "base"], s[s.set == "base"]
    with open(os.path.join(OUT, "table_surrogate.tex"), "w") as f:
        f.write("\\begin{tabular}{@{}lrrrrrr@{}}\n\\toprule\n & \\multicolumn{2}{c}{Rank correlation} & \\multicolumn{2}{c}{Mean abs.\\ error}"
                " & \\multicolumn{2}{c}{Saving (\\%)} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
                "Cluster & Fixed & Fitted & Fixed & Fitted & Fixed & Fitted \\\\\n\\midrule\n")
        for c in BASE:
            a = q[(q.cluster == c) & (q.method == "LNS fixed prices")]
            b = q[(q.cluster == c) & (q.method == "SG-LNS")]
            sa = s[(s.cluster == c) & (s.method == "LNS fixed prices")].saving_pct
            sb = s[(s.cluster == c) & (s.method == "SG-LNS")].saving_pct
            f.write(f"{label('base', c)} & {a.spearman.mean():.2f} & {b.spearman.mean():.2f} & {a.mae.mean():.1f} & {b.mae.mean():.1f} & "
                    f"{sa.mean():.2f} & {sb.mean():.2f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def routing_check():
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "verify_routing", "*.json"))):
        name = os.path.basename(f)[:-5].split("_")
        r = json.load(open(f))
        r.pop("cluster", None); rows.append(dict(set=name[0], cluster=name[1], which="Benders" if len(name) > 2 else "Sequential", **r))
    with open(os.path.join(OUT, "table_routing_check.tex"), "w") as f:
        f.write("\\begin{tabular}{@{}llrrrrr@{}}\n\\toprule\n & & Trips & LP, column & LP, all & Routing & Exact \\\\\n"
                "Cluster & Schedule & listed & generation & trips & solver & optimum \\\\\n\\midrule\n")
        for r in sorted(rows, key=lambda r: (ORDER["base"].index(r["cluster"]) if r["set"] == "base" else 9, r["which"] != "Sequential")):
            lab = label("base", r["cluster"]) if r["set"] == "base" else r["cluster"]
            f.write(f"{lab} & {r['which']} & {r['trips']:,} & {r['solver_lp']:,.2f} & {r['full_lp']:,.2f} & {r['solver']:,.2f} & "
                    f"{r['exact']:,.2f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def nolift():
    with open(os.path.join(OUT, "table_nolift.tex"), "w") as f:
        f.write("\\begin{tabular}{@{}lrrrrrr@{}}\n\\toprule\n & \\multicolumn{2}{c}{Lower bound} & \\multicolumn{2}{c}{Gap (\\%)}"
                " & \\multicolumn{2}{c}{Saving (\\%)} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
                "Cluster & Lifted & Not lifted & Lifted & Not lifted & Lifted & Not lifted \\\\\n\\midrule\n")
        for c in BASE:
            a = json.load(open(glob.glob(os.path.join(ROOT, "benders", f"*{c}.json"))[0]))
            b = json.load(open(glob.glob(os.path.join(ROOT, "benders_nolift", f"*{c}.json"))[0]))
            f.write(f"{label('base', c)} & {a['lb']:,.0f} & {b['lb']:,.0f} & {100 * a['gap']:.2f} & {100 * b['gap']:.2f} & "
                    f"{100 * a['saving']:.2f} & {100 * b['saving']:.2f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def net_rows(bdir, ldir):
    """Per cluster: sequential cost, bus share, saving and gap of both methods, buses, from the Benders and SG-LNS (seed 1) files."""
    rows = []
    for fb in sorted(glob.glob(os.path.join(ROOT, bdir, "*.json"))):
        d = json.load(open(fb))
        stem = os.path.basename(fb)[:-5]
        fl = os.path.join(ROOT, ldir, f"{stem}_s1.json")
        s = json.load(open(fl)) if os.path.exists(fl) and "z" in json.load(open(fl)) else None
        rows.append(dict(inst=stem[:-9], cluster=stem[-8:], zseq=d["z_seq"], bus=100 * d["seq"]["cost"] / d["z_seq"], lb=d["lb"],
                         sb=100 * d["saving"], gb=100 * (d["z"] - d["lb"]) / d["z"],
                         ss=None if s is None else 100 * (d["z_seq"] - s["z"]) / d["z_seq"],
                         gs=None if s is None else 100 * (s["z"] - d["lb"]) / s["z"],
                         nseq=d["seq"]["buses"], nb=d["plan"]["buses"], ns=None if s is None else s["retime_plan"]["buses"]))
    return rows


def net_table(name, groups):
    fmt = lambda v: "--" if v is None else f"{v:.2f}"
    with open(os.path.join(OUT, name), "w") as f:
        f.write("\\begin{tabular}{@{}lrrrrrrr@{}}\n\\toprule\n & Sequential & Bus & \\multicolumn{2}{c}{Saving (\\%)} & \\multicolumn{2}{c}{Gap (\\%)}"
                " & Buses \\\\\n\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
                "Cluster & $\\widetilde Z$ & share (\\%) & Benders & SG-LNS & Benders & SG-LNS & S/B/L \\\\\n\\midrule\n")
        for title, rows in groups:
            f.write(f"\\multicolumn{{8}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\\n")
            for k, r in enumerate(rows):
                f.write(f"Cluster {k + 1} & {r['zseq']:,.0f} & {r['bus']:.1f} & {fmt(r['sb'])} & {fmt(r['ss'])} & {fmt(r['gb'])} & "
                        f"{fmt(r['gs'])} & {r['nseq']}/{r['nb']}/{'--' if r['ns'] is None else r['ns']} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def small_and_families():
    small = net_rows("small/benders", "small/lns_fit")
    net_table("table_small.tex", [(f"Small network, seed {s}", [r for r in small if r["inst"] == f"syn_small_s{s}"]) for s in (1, 2, 3)])
    fam = net_rows("families/benders", "families/lns_fit")
    names = {"syn_m_compact_s1": "Compact", "syn_m_far2_s1": "Far (distances doubled)", "syn_m_2acc_s1": "Two accommodations per cluster"}
    net_table("table_families.tex", [(names[i], [r for r in fam if r["inst"] == i]) for i in names])
    return small, fam


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    gap_parts()
    cut_check()
    surrogate()
    routing_check()
    nolift()
    small_and_families()
    print("done")
