"""Result figures (2026-11-12, revised after a figure review).
fig_share:  share of the saving bound per method and cluster (numbers of tab:literature, seed 1). L2 and L3 obtain
            (almost) nothing and are named in the figure instead of drawn as empty bars.
fig_trips:  C2, bus plans of x0 and of our plan: change of trips per hour of the day (summed over the week), and
            trips by number of riders (data: results_new/original/reroute/trips_a1e8b5a7.csv from trips_dump.py).
fig_search: cost reduction of x1 over a search three times as long as in the method, updated vs fixed direct-trip
            estimate, C3 and C4, in % of Z(x0) (data: results_new/original/search_long/).
Run: python fig_results.py
"""
import csv
import glob
import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import make_figures as MF

R = "E:/schedule/revision/results_new/original"
OURS = MF.ACCENT          # #1f5a96, our method everywhere
BASE = "#a3a3a3"          # sequential plan / baselines
LIFT = "#8fb3da"          # reduced variant of our method (hatched, so it survives grayscale)
TICK, LAB, LEG, ANN = 7.5, 8, 7.5, 7


def style(ax, grid=True):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=TICK, width=0.5, length=2.5)
    if grid:
        ax.yaxis.grid(True, color="#e8e8e8", lw=0.5)
        ax.set_axisbelow(True)


def fig_share():
    series = [  # tab:literature, share of the saving bound (%), C1..C4, mean
        ("L1 transport-aware staff model", BASE, None, [9.2, 14.3, -0.5, 9.9, 8.2]),
        ("Decomposition without lifting", LIFT, "////", [16.8, 12.4, 2.0, 39.7, 17.7]),
        ("Our method", OURS, None, [33.5, 54.2, 19.2, 48.9, 39.0]),
    ]
    groups = ["C1", "C2", "C3", "C4", "Mean"]
    x = np.array([0, 1, 2, 3, 4.4])
    fig, ax = plt.subplots(figsize=(7.1, 2.4))
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.11, top=0.97)
    w = 0.25
    for j, (name, col, hatch, v) in enumerate(series):
        xs = x + (j - 1) * w
        ax.bar(xs, v, w * 0.92, color=col, hatch=hatch, edgecolor=OURS if hatch else col, linewidth=0.3,
               label=name)
        for xi, vi in zip(xs, v):
            ax.text(xi, max(vi, 0) + 1.0, f"{vi:.1f}".replace("-", "$-$"), ha="center", va="bottom", fontsize=ANN,
                    color=OURS if col == OURS else "#333333", fontweight="bold" if col == OURS else None)
    ax.axvspan(3.85, 4.95, color="#f0f0f0", lw=0, zorder=0)
    ax.axhline(0, color=MF.INK, lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=LAB)
    ax.set_ylabel(r"share of the saving bound (\%)", fontsize=LAB)
    ax.set_ylim(-3, 62)
    ax.set_xlim(-0.55, 4.95)
    style(ax)
    ax.legend(loc="upper left", fontsize=LEG, frameon=False, handlelength=1.6)
    ax.text(0.375, 0.97, r"L2 classical Benders and L3 staff-roster CG: 0 on every cluster,"
                       "\n" r"except L2 on C4 (3.1)", transform=ax.transAxes, fontsize=ANN, color="#555555",
            va="top")
    MF.save(fig, "fig_share")


def fig_trips():
    rows = list(csv.DictReader(open(f"{R}/reroute/trips_a1e8b5a7.csv")))
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.3), gridspec_kw=dict(width_ratios=[1.6, 1]))
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.2, top=0.9, wspace=0.27)
    # left: change of trips per hour of the day
    ax = axes[0]
    hours = np.arange(24)
    cnt = {p: Counter(int(r["t"]) % 24 for r in rows if r["plan"] == p) for p in ("x0", "ours")}
    diff = np.array([cnt["ours"].get(h, 0) - cnt["x0"].get(h, 0) for h in hours])
    ax.bar(hours, diff, 0.72, color=[OURS if d < 0 else BASE for d in diff])
    ax.axhline(0, color=MF.INK, lw=0.5)
    n0, n1 = sum(cnt["x0"].values()), sum(cnt["ours"].values())
    ax.text(0.98, 0.95, rf"{n0} $\rightarrow$ {n1} trips per week ($-${n0 - n1})", transform=ax.transAxes,
            ha="right", va="top", fontsize=ANN)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_xlim(-0.7, 23.7)
    ax.set_xlabel("hour of the day", fontsize=LAB)
    ax.set_ylabel(r"change in trips per week", fontsize=LAB)
    ax.set_title(r"\textbf{(a) Our plan minus $\boldsymbol{x^0}$, by hour}", fontsize=LAB, pad=3)
    style(ax)
    # right: trips by number of riders
    ax = axes[1]
    k = np.arange(1, 7)
    for off, (p, col, lab) in zip((-0.2, 0.2), (("x0", BASE, "$x^0$"), ("ours", OURS, "our plan"))):
        trips = [r for r in rows if r["plan"] == p]
        c = Counter(min(int(r["drop"]) + int(r["pick"]), 6) for r in trips)
        mean = sum(int(r["drop"]) + int(r["pick"]) for r in trips) / len(trips)
        ax.bar(k + off, [c.get(i, 0) for i in k], 0.38, color=col, label=f"{lab}: {mean:.2f} riders per trip")
    ax.set_xticks(k)
    ax.set_xticklabels(["1", "2", "3", "4", "5", r"$\ge$6"])
    ax.set_xlabel("riders on the trip", fontsize=LAB)
    ax.set_ylabel("trips per week", fontsize=LAB)
    ax.set_title(r"\textbf{(b) Trips by number of riders}", fontsize=LAB, pad=3)
    ax.legend(fontsize=LEG - 0.5, frameon=False, loc="upper right")
    style(ax)
    MF.save(fig, "fig_trips")


def fig_search():
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.2))
    fig.subplots_adjust(left=0.075, right=0.965, bottom=0.2, top=0.9, wspace=0.25)
    for ax, (name, c) in zip(axes, (("(a) C3", "ba5b9796"), ("(b) C4", "df75e3dc"))):
        base = json.load(open(glob.glob(f"{R}/ablation_m5eq/{c}/A4_no_smooth/*_s1.json")[0]))
        x1, z0 = base["z_benders"], base["z_seq"]
        for v, col, ls, lab in (("learned", OURS, "-", "estimate updated after every round (SG-LNS)"),
                                ("fixed_direct", "#D55E00", (0, (4, 2)), "fixed estimate (direct-trip costs)")):
            d = json.load(open(glob.glob(f"{R}/search_long/{c}/{v}/*_s1.json")[0]))
            T = d["seconds"]
            ts, red, best = [0.0], [0.0], x1
            for r in d["history"]:
                best = min(best, r["best_j_op"])
                ts.append(3 * r["seconds"] / T)
                red.append(100 * (x1 - best) / z0)
            ts.append(3.0)
            red.append(100 * (x1 - min(best, d["z_search"])) / z0)
            ax.step(ts, red, where="post", color=col, lw=1.2, ls=ls, label=lab)
            ax.text(3.03, red[-1], f"{red[-1]:.2f}", fontsize=ANN, color=col, va="center")
        ax.axvline(1, color="#9a9a9a", lw=0.6)
        if name.endswith("C4"):
            ax.text(1.04, 0.97, "time of\nour method", transform=ax.get_xaxis_transform(), fontsize=ANN,
                    color="#555555", va="top")
        ax.set_xlim(0, 3)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(["0", r"1$\times$", r"2$\times$", r"3$\times$"])
        ax.set_ylim(bottom=-0.005)
        ax.set_title(rf"\textbf{{{name}}}", fontsize=LAB, pad=3)
        ax.set_xlabel("search time, in multiples of the time in our method", fontsize=LAB, labelpad=1)
        style(ax)
    axes[0].set_ylabel(r"cost reduction (\% of $\widetilde Z(x^0)$)", fontsize=LAB)
    leg = axes[0].legend(fontsize=LEG - 0.5, frameon=True, loc="lower right", framealpha=1, edgecolor="none",
                         borderpad=0.3)
    leg.get_frame().set_facecolor("white")
    MF.save(fig, "fig_search")


if __name__ == "__main__":
    fig_share()
    fig_trips()
    fig_search()
    print("done")
