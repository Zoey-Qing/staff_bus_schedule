"""Figures and tables for the original-data results (20-minute masters; 60-minute runs as the longer-budget check).

The main heuristic is the surrogate-guided LNS (staff_surrogate.py --ablation lns_fit); the full surrogate with its
upper-level solve and the other ablations appear in the ablation table.

Usage: python make_figures.py   (run from this folder; writes PDF and PNG figures and CSV/TeX tables here)

Figures are sized for a two-column Elsevier journal: the full text width is 7.0 in (about 180 mm), so they can be
included at width=\\textwidth without scaling and all text stays at 7 pt or larger.
"""
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CLUSTERS = ["1aea822f", "662b3495", "a1e8b5a7", "ba5b9796", "df75e3dc"]
LABEL = {c: f"Cluster {i + 1}" for i, c in enumerate(CLUSTERS)}

FULL_W = 7.0  # full text width of a two-column Elsevier page (in)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                     "axes.grid": True, "axes.axisbelow": True, "grid.color": "#e3e3e3", "grid.linewidth": 0.5,
                     "lines.linewidth": 1.4, "legend.frameon": False, "hatch.linewidth": 0.6,
                     "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "savefig.dpi": 300})
# Okabe-Ito colorblind-safe palette. Methods: vermillion (Benders + CG) and blue (Surrogate).
# Cost parts use bluish green (staff) and orange (bus), distinct from the method colors.
C_BD, C_SG, C_LB = "#D55E00", "#0072B2", "#6b6b6b"
MAIN, MAIN_NAME, MAIN_SHORT = "ablation_lns_fit", "Surrogate-guided LNS", "SG-LNS"
C_STAFF, C_BUS = "#009E73", "#E69F00"
C_W0, C_W10 = "#d0d0d0", "#707070"
MINUS = "−"


def money(v, _=None):
    """Dollar tick label with thousands separator and a typographic minus."""
    s = f"{abs(v):,.0f}"
    return s if round(v) == 0 else (MINUS + s if v < 0 else s)


def pct(v, _=None):
    """Percent tick label with a typographic minus."""
    v = round(v, 6)
    s = f"{abs(v):g}"
    return s if v == 0 else (MINUS + s if v < 0 else s)


def money_axis(ax, n=5):
    ax.yaxis.set_major_formatter(FuncFormatter(money))
    ax.yaxis.set_major_locator(MaxNLocator(n))


def top_legend(fig, handles, ncol):
    """One shared legend centred above the row of panels."""
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=ncol, frameon=False,
               columnspacing=1.8, handletextpad=0.6, handlelength=2.4, borderaxespad=0.0)


def load(folder, c, suffix=""):
    return json.load(open(glob.glob(os.path.join(ROOT, folder, f"*{c}{suffix}.json"))[0]))


def surrogate_final(s):
    """Staff and bus cost of the reported surrogate plan: the re-timed one when it was kept, else the search best."""
    rv, rp = s.get("retime_vals"), s.get("retime_plan")
    if rv and abs(rv["j_reg"] + rv["j_over"] + rp["cost"] - s["z"]) < 1e-3:
        return rv["j_reg"] + rv["j_over"], rp["cost"]
    return s["best"]["labor"], s["best"]["v"]


def best_path(s):
    """Time, best cost so far, and staff and bus cost of the current plan after each round, starting from the sequential plan.

    The best plan can come from a neighbour evaluated inside a round, whose parts are not logged, so only its total is used.
    """
    e0 = s["sequential"]
    t, best, staff, bus = [0.0], [e0["j_op"]], [e0["labor"]], [e0["v"]]
    for h in s["history"]:
        t.append(h["seconds"]); best.append(h["best_j_op"]); staff.append(h["labor"]); bus.append(h["v"])
    t.append(s["seconds"]); best.append(s["z_search"]); staff.append(staff[-1]); bus.append(bus[-1])
    return np.array(t) / 3600, np.minimum.accumulate(best), np.array(staff), np.array(bus)


def benders_path(b):
    t, ub, lb, best = [0.0], [b["z_seq"]], [np.nan], b["z_seq"]
    for h in b["history"]:
        best = min(best, h["ub"])
        t.append(h["seconds"]); ub.append(best); lb.append(h["lb"])
    return np.array(t) / 3600, np.array(ub), np.array(lb)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"{name}.{ext}"))
    plt.close(fig)


def row_of_panels(h=1.85):
    return plt.subplots(1, 5, figsize=(FULL_W, h))


def fig_convergence():
    """Saving over time for both methods with the same budget.

    Step lines: incumbent of the search. Dot joined by a dotted segment: final plan after the re-timing step.
    """
    fig, axes = row_of_panels()
    for ax, c in zip(axes, CLUSTERS):
        s, b = load(MAIN, c, "_s1"), load("benders", c)
        z0 = b["z_seq"]
        tb, ub, _ = benders_path(b)
        tb2, yb = np.append(tb, tb[-1]), 100 * (z0 - np.append(ub, ub[-1])) / z0
        fb = 100 * (z0 - b["z"]) / z0
        t, best, _, _ = best_path(s)
        ys = 100 * (z0 - best) / z0
        fs = 100 * (z0 - s["z"]) / z0
        for tt, yy, ff, col in ((tb2, yb, fb, C_BD), (t, ys, fs, C_SG)):
            ax.step(tt, yy, where="post", color=col, zorder=3)
            ax.plot([tt[-1], tt[-1]], [yy[-1], ff], ":", color=col, lw=1.1, zorder=3)
            ax.plot(tt[-1], ff, "o", color=col, ms=3.8, mec="white", mew=0.5, zorder=4, clip_on=False)
        ax.set_title(LABEL[c]); ax.set_xlabel("Time (h)")
        ax.set_ylim(bottom=-0.02)
        ax.yaxis.set_major_locator(MaxNLocator(5)); ax.yaxis.set_major_formatter(FuncFormatter(pct))
        ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
    axes[0].set_ylabel("Saving over sequential (%)")
    fig.tight_layout(w_pad=0.6)
    top_legend(fig, [Line2D([], [], color=C_BD, label="Benders + CG"), Line2D([], [], color=C_SG, label=MAIN_NAME),
                     Line2D([], [], color="#444", ls=":", lw=1.1, marker="o", ms=3.8, mec="white", mew=0.5,
                            label="Final plan after re-timing")], ncol=3)
    save(fig, "fig_convergence")


def fig_surrogate_parts():
    """Staff and bus parts of the best plan after each round of the LNS, and the best total so far, relative to the sequential plan."""
    fig, axes = row_of_panels()
    for ax, c in zip(axes, CLUSTERS):
        s = load(MAIN, c, "_s1")
        t, best, staff, bus = best_path(s)
        ax.axhline(0, color="#999", lw=0.6)
        ax.step(t, staff - staff[0], where="post", color=C_STAFF, lw=0.9)
        ax.step(t, bus - bus[0], where="post", color=C_BUS, lw=0.9)
        ax.step(t, best - best[0], where="post", color="k", lw=1.2, ls="--")
        ax.set_title(LABEL[c]); ax.set_xlabel("Time (h)")
        ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
        money_axis(ax)
    axes[0].set_ylabel("Change from sequential ($)")
    fig.tight_layout(w_pad=0.6)
    top_legend(fig, [Line2D([], [], color=C_STAFF, lw=0.9, label="Staff cost"), Line2D([], [], color=C_BUS, lw=0.9, label="Bus cost"),
                     Line2D([], [], color="k", lw=1.2, ls="--", label="Total cost, best plan so far")], ncol=3)
    save(fig, "fig_surrogate_cost_parts")


def stacked_bar(ax, x, parts, colors, w, **kw):
    """Stack bars so that positive parts grow upward from 0 and negative parts grow downward from 0."""
    pos = neg = 0.0
    for v, col in zip(parts, colors):
        ax.bar(x, v, w, bottom=pos if v >= 0 else neg, color=col, edgecolor="#333", lw=0.5, **kw)
        if v >= 0:
            pos += v
        else:
            neg += v


def fig_final_parts():
    """Final change of staff and bus cost for both methods: Benders + CG solid (left bar), SG-LNS hatched (right bar)."""
    fig, ax = plt.subplots(figsize=(FULL_W * 0.6, 2.4))
    x, w = np.arange(len(CLUSTERS)), 0.36
    methods = (("Benders + CG", "B+CG", {}), (MAIN_NAME, MAIN_SHORT, {"hatch": "/////"}))
    ticks, ticklabels = [], []
    with plt.rc_context({"hatch.linewidth": 0.5}):
        for k, (name, short, style) in enumerate(methods):
            dstaff, dbus = [], []
            for c in CLUSTERS:
                b = load("benders", c)
                if name == MAIN_NAME:
                    st, bu = surrogate_final(load(MAIN, c, "_s1"))
                else:
                    st, bu = b["wages"] + b["overtime"], b["plan"]["cost"]
                dstaff.append(st - b["seq_wages"] - b["seq_overtime"]); dbus.append(bu - b["seq"]["cost"])
            xs = x + (k - 0.5) * w
            for xi, ds, db in zip(xs, dstaff, dbus):
                stacked_bar(ax, xi, (ds, db), (C_STAFF, C_BUS), w * 0.88, **style)
            ax.plot(xs, np.add(dstaff, dbus), "D", color="k", mfc="white", mew=1.0, ms=4, zorder=5)
            ticks += list(xs); ticklabels += [short] * len(xs)
        ax.axhline(0, color="#333", lw=0.6)
        ax.set_xticks(ticks, ticklabels, fontsize=7)
        ax.tick_params(axis="x", length=0, pad=2)
        for xi, c in zip(x, CLUSTERS):
            ax.annotate(LABEL[c], (xi, 0), xycoords=("data", "axes fraction"), xytext=(0, -13),
                        textcoords="offset points", ha="center", va="top", fontsize=8)
        ax.grid(axis="x", visible=False)
        ax.set_xlim(-0.6, len(CLUSTERS) - 0.4)
        money_axis(ax, 6)
        ax.set_ylabel("Change from sequential ($)")
        fig.tight_layout()
        # Handle order fills columns, so the rows read: parts and total / method fills.
        ax.legend(handles=[Patch(fc=C_STAFF, ec="#333", lw=0.5, label="Staff cost"),
                           Patch(fc="#bdbdbd", ec="#333", lw=0.5, label="Benders + CG (B+CG)"),
                           Patch(fc=C_BUS, ec="#333", lw=0.5, label="Bus cost"),
                           Patch(fc="#bdbdbd", ec="#333", lw=0.5, hatch="/////", label=f"{MAIN_NAME} ({MAIN_SHORT})"),
                           Line2D([], [], ls="", marker="D", color="k", mfc="white", mew=1.0, ms=4, label="Total")],
                  loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=7.5, columnspacing=1.2,
                  handlelength=1.6, borderaxespad=0.2)
        save(fig, "fig_final_cost_parts")


def fig_benders_bounds():
    """Best plan and lower bound of Benders by round, 20- and 60-minute masters."""
    fig, axes = row_of_panels(1.95)
    for ax, c in zip(axes, CLUSTERS):
        for folder, ls, filled in (("benders", "--", False), ("benders_long", "-", True)):
            b = load(folder, c)
            r = [h["round"] for h in b["history"]]
            for key, col, mk in (("best", C_BD, "o"), ("lb", C_LB, "s")):
                ax.plot(r, [h[key] for h in b["history"]], ls, marker=mk, ms=3, color=col, lw=1.1,
                        mfc=col if filled else "white")
        ax.set_title(LABEL[c]); ax.set_xlabel("Round"); ax.set_xticks(range(5))
        money_axis(ax, 5)
    axes[0].set_ylabel("Cost ($)")
    fig.tight_layout(w_pad=0.4)
    top_legend(fig, [Line2D([], [], color=C_BD, ls="-", marker="o", ms=3, lw=1.1, label="Best plan, 60-min master"),
                     Line2D([], [], color=C_LB, ls="-", marker="s", ms=3, lw=1.1, label="Lower bound, 60-min master"),
                     Line2D([], [], color=C_BD, ls="--", marker="o", ms=3, mfc="white", lw=1.1, label="Best plan, 20-min master"),
                     Line2D([], [], color=C_LB, ls="--", marker="s", ms=3, mfc="white", lw=1.1,
                            label="Lower bound, 20-min master")], ncol=4)
    save(fig, "fig_benders_bounds")


def fig_prediction():
    """Predicted against realised change of bus cost for every evaluated schedule (prediction made before the point is added)."""
    fig, axes = row_of_panels(1.8)
    for ax, c in zip(axes, CLUSTERS):
        s = load(MAIN, c, "_s1")
        p, r = np.array(s["dpred"]), np.array(s["dreal"])
        lo, hi = min(p.min(), r.min()), max(p.max(), r.max())
        ax.plot([lo, hi], [lo, hi], color="#999", lw=0.6)
        ax.scatter(p, r, s=5, color=C_SG, alpha=0.5, lw=0)
        ax.set_title(LABEL[c]); ax.set_xlabel("Predicted ($)")
        ax.text(0.04, 0.97, f"$r$ = {np.corrcoef(p, r)[0, 1]:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=7.5)
        money_axis(ax, 4)
        ax.xaxis.set_major_formatter(FuncFormatter(money)); ax.xaxis.set_major_locator(MaxNLocator(3))
    axes[0].set_ylabel("Realised ($)")
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_surrogate_prediction")


def fig_e6_wbus():
    """E6: fixed bus weights against the integrated model, relative to the sequential plan (W_bus = 1).

    (a) Change of total cost Z (%). (b) Change of the number of service violations, summed over priorities 1-4
    (lower is better service). The LNS keeps the sequential service levels, so its change in (b) is 0.
    """
    series = (("$W_{\\mathrm{bus}} = 0$", C_W0), ("$W_{\\mathrm{bus}} = 10$", C_W10),
              ("Integrated, Benders + CG", C_BD), ("Integrated, " + MAIN_SHORT, C_SG))
    dz, dv = [[] for _ in series], [[] for _ in series]
    for c in CLUSTERS:
        b = load("benders", c)
        z0, v0 = b["z_seq"], sum(b["seq_levels"].values())
        w0 = json.load(open(os.path.join(ROOT, "e5_e6", f"e6_w0_{c}.json")))
        w10 = json.load(open(os.path.join(ROOT, "e5_e6", f"e6_w10_{c}.json")))
        sg = load(MAIN, c, "_s1")
        for k, (z, lv) in enumerate(((w0["z"], w0["levels"]), (w10["z"], w10["levels"]),
                                     (b["z"], b["levels"]), (sg["z"], b["seq_levels"]))):
            dz[k].append(100 * (z - z0) / z0); dv[k].append(sum(lv.values()) - v0)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(FULL_W, 3.5), sharex=True)
    x, w = np.arange(len(CLUSTERS)), 0.2
    for k, (name, col) in enumerate(series):
        xs = x + (k - 1.5) * w
        a1.bar(xs, dz[k], w * 0.9, color=col, edgecolor="#333", lw=0.4)
        a2.bar(xs, dv[k], w * 0.9, color=col, edgecolor="#333", lw=0.4)
        for xi, v in zip(xs, dv[k]):
            lab = "0" if round(v) == 0 else f"{v:+.0f}".replace("-", MINUS)
            a2.text(xi, v, lab, ha="center", va="top" if v < 0 else "bottom", fontsize=7, color="#333")
    for ax in (a1, a2):
        ax.axhline(0, color="#333", lw=0.6)
        ax.set_xticks(x, [LABEL[c] for c in CLUSTERS], fontsize=8)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="x", visible=False)
    a1.tick_params(axis="x", labelbottom=False)
    a1.set_ylabel("Change of $Z$ (%)")
    a1.yaxis.set_major_formatter(FuncFormatter(pct))
    a2.set_ylabel("Change of violations")
    a2.yaxis.set_major_locator(MaxNLocator(6, integer=True)); a2.yaxis.set_major_formatter(FuncFormatter(pct))
    lo, hi = min(map(min, dv)), max(map(max, dv))
    a2.set_ylim(min(lo, 0) - 0.12 * (hi - lo), hi + 0.12 * (hi - lo))
    a1.set_title("(a) Total cost relative to the sequential plan", loc="left")
    a2.set_title("(b) Service violations (priorities 1–4) relative to the sequential plan", loc="left")
    fig.tight_layout(h_pad=0.8)
    top_legend(fig, [Patch(fc=col, ec="#333", lw=0.4, label=name) for name, col in series], ncol=4)
    save(fig, "fig_e6_wbus")


def signed(v):
    """Signed integer with thousands separator; a value that rounds to 0 prints as 0 (not +0 or -0)."""
    return "0" if round(v) == 0 else f"{v:+,.0f}"


def tables():
    rows = []
    for c in CLUSTERS:
        # main setting: 20-minute masters (about 2 h per cluster); the 60-minute runs are the longer-budget check
        b, b60 = load("benders", c), load("benders_long", c)
        s = load(MAIN, c, "_s1")
        lb, z0 = max(b["lb"], b60["lb"]), b["z_seq"]
        st, bu = surrogate_final(s)
        rows.append(dict(cluster=LABEL[c], z_seq=z0, bd60=b60["z"], hours60=b60["benders_seconds"] / 3600, bd=b["z"], sg=s["z"], lb=lb,
                         bd_save=100 * (z0 - b["z"]) / z0, sg_save=100 * (z0 - s["z"]) / z0,
                         bd_gap=100 * (b["z"] - lb) / b["z"], sg_gap=100 * (s["z"] - lb) / s["z"], hours=b["benders_seconds"] / 3600,
                         sg_best_h=s["best_seconds"] / 3600, rounds_bd=len(b["history"]), rounds_sg=len(s["history"]),
                         d_staff_bd=b["wages"] + b["overtime"] - b["seq_wages"] - b["seq_overtime"], d_bus_bd=b["plan"]["cost"] - b["seq"]["cost"],
                         d_staff_sg=st - b["seq_wages"] - b["seq_overtime"], d_bus_sg=bu - b["seq"]["cost"], bus_seq=b["seq"]["cost"]))
    keys = list(rows[0])
    with open(os.path.join(HERE, "table_main.csv"), "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(f"{r[k]:.4f}" if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
    with open(os.path.join(HERE, "table_main.tex"), "w") as f:
        f.write("\\begin{tabular}{lrrrrrrrr}\n\\toprule\n")
        f.write(" & Sequential & \\multicolumn{3}{c}{Benders + CG} & \\multicolumn{3}{c}{" + MAIN_SHORT + "} & Lower \\\\\n")
        f.write("\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\n")
        f.write("Cluster & $\\tilde Z$ & $\\tilde Z$ & Saving (\\%) & Gap (\\%) & $\\tilde Z$ & Saving (\\%) & Gap (\\%) & bound \\\\\n\\midrule\n")
        for r in rows:
            bb, sb = ("\\textbf{%s}", "%s") if r["bd"] < r["sg"] else ("%s", "\\textbf{%s}")
            zb, zs = bb % format(r["bd"], ",.0f"), sb % format(r["sg"], ",.0f")
            f.write(f"{r['cluster']} & {r['z_seq']:,.0f} & {zb} & {r['bd_save']:.2f} & {r['bd_gap']:.2f} & "
                    f"{zs} & {r['sg_save']:.2f} & {r['sg_gap']:.2f} & {r['lb']:,.0f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    with open(os.path.join(HERE, "table_cost_parts.tex"), "w") as f:
        f.write("\\begin{tabular}{lrrrrr}\n\\toprule\n & Bus cost & \\multicolumn{2}{c}{Benders + CG} & \\multicolumn{2}{c}{" + MAIN_SHORT + "} \\\\\n")
        f.write("\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\nCluster & sequential & $\\Delta$ staff & $\\Delta$ bus & $\\Delta$ staff & $\\Delta$ bus \\\\\n\\midrule\n")
        for r in rows:
            f.write(f"{r['cluster']} & {r['bus_seq']:,.0f} & {signed(r['d_staff_bd'])} & {signed(r['d_bus_bd'])} & {signed(r['d_staff_sg'])} & {signed(r['d_bus_sg'])} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    with open(os.path.join(HERE, "table_time.tex"), "w") as f:
        f.write("\\begin{tabular}{lrrrrrr}\n\\toprule\n & \\multicolumn{3}{c}{20-min masters} & \\multicolumn{2}{c}{60-min masters} & " + MAIN_SHORT + " \\\\\n")
        f.write("\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\nCluster & Budget (h) & Benders (\\%) & " + MAIN_SHORT + " (\\%) & Budget (h) & Benders (\\%) & best found (h) \\\\\n\\midrule\n")
        for r in rows:
            f.write(f"{r['cluster']} & {r['hours']:.1f} & {r['bd_save']:.2f} & {r['sg_save']:.2f} & {r['hours60']:.1f} & "
                    f"{100 * (r['z_seq'] - r['bd60']) / r['z_seq']:.2f} & {r['sg_best_h']:.1f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


ABLATION = (("Benders + CG", "benders", (None,)), ("LNS, fixed prices", "ablation_lns", (1, 2, 3)),
            ("Upper-level solve, fixed prices", "ablation_static", (1,)),
            ("Full surrogate (upper-level solve, regression)", "surrogate", (1, 2, 3)),
            (MAIN_NAME + " (LNS, regression)", MAIN, (1, 2, 3)))


def table_ablation():
    """Saving (%) of each variant at the Benders budget: mean over the seeds, with the range when there are several."""
    z0 = {c: load("benders", c)["z_seq"] for c in CLUSTERS}
    cell = lambda v: f"{np.mean(v):.2f}" if len(v) == 1 else f"{np.mean(v):.2f} ({min(v):.2f}--{max(v):.2f})"
    with open(os.path.join(HERE, "table_ablation.tex"), "w") as f:
        f.write("\\begin{tabular}{l" + "r" * len(CLUSTERS) + "}\n\\toprule\n")
        f.write("Variant & " + " & ".join(LABEL[c] for c in CLUSTERS) + " \\\\\n\\midrule\n")
        for name, folder, seeds in ABLATION:
            per = {c: [100 * (z0[c] - load(folder, c, "" if k is None else f"_s{k}")["z"]) / z0[c] for k in seeds] for c in CLUSTERS}
            f.write(f"{name} & " + " & ".join(cell(per[c]) for c in CLUSTERS) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def table_far2():
    """Far x2 network: accommodations at twice their distance from the centre of their cluster's branches (seed 1)."""
    rows = []
    for c in CLUSTERS:
        b = load("far2/benders", c)
        rows.append(dict(cluster=LABEL[c], z_seq=b["z_seq"], bus=b["seq"]["cost"], lb=b["lb"], bd=b["z"],
                         sg=load("far2/surrogate", c, "_s1")["z"], main=load("far2/lns_fit", c, "_s1")["z"]))
    with open(os.path.join(HERE, "table_far2.tex"), "w") as f:
        f.write("\\begin{tabular}{lrrrrrrr}\n\\toprule\n & Sequential & Bus cost & \\multicolumn{3}{c}{Saving (\\%)} & Gap of & Lower \\\\\n")
        f.write("\\cmidrule(lr){4-6}\nCluster & $\\tilde Z$ & sequential & Benders + CG & Full surrogate & " + MAIN_SHORT + " & "
                + MAIN_SHORT + " (\\%) & bound \\\\\n\\midrule\n")
        for r in rows:
            sv = {k: 100 * (r["z_seq"] - r[k]) / r["z_seq"] for k in ("bd", "sg", "main")}
            f.write(f"{r['cluster']} & {r['z_seq']:,.0f} & {r['bus']:,.0f} & {sv['bd']:.2f} & {sv['sg']:.2f} & {sv['main']:.2f} & "
                    f"{100 * (r['main'] - r['lb']) / r['main']:.2f} & {r['lb']:,.0f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


NET100 = ["1273d27b", "27ea17d0", "41835ef9", "46591601", "69f22899"]


def table_scale():
    """Larger network net100_k10x2_s1: 20 branches and about 200 staff per cluster (seed 1, equal time)."""
    with open(os.path.join(HERE, "table_scale.tex"), "w") as f:
        f.write("\\begin{tabular}{lrrrrrrr}\n\\toprule\n & Sequential & \\multicolumn{2}{c}{Saving (\\%)} & "
                "\\multicolumn{2}{c}{Gap (\\%)} & Lower & Time \\\\\n\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n"
                "Cluster & $\\tilde Z$ & Benders + CG & " + MAIN_SHORT + " & Benders + CG & " + MAIN_SHORT
                + " & bound & (h) \\\\\n\\midrule\n")
        for i, c in enumerate(NET100):
            b, m = load("net100/benders", c), load("net100/lns_fit", c, "_s1")
            zs, lb = b["z_seq"], b["lb"]
            f.write(f"Cluster {i + 1} & {zs:,.0f} & {100 * (zs - b['z']) / zs:.2f} & {100 * (zs - m['z']) / zs:.2f} & "
                    f"{100 * (b['z'] - lb) / b['z']:.2f} & {100 * (m['z'] - lb) / m['z']:.2f} & {lb:,.0f} & "
                    f"{b['seconds'] / 3600:.1f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


if __name__ == "__main__":
    fig_convergence()
    fig_surrogate_parts()
    fig_final_parts()
    fig_benders_bounds()
    fig_prediction()
    fig_e6_wbus()
    tables()
    table_ablation()
    table_far2()
    table_scale()
    print("done")
