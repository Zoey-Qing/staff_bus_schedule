"""Figure fig_convergence: saving of the incumbent and saving bound of the lower bound over time, per cluster, for the
main run (seed 1). Data: results_new/original/ablation_m5eq/<c>/A4_no_smooth (stage1 history, search history, wall.txt).
The search history is shifted by the wall time of the column generation. Run: python fig_convergence.py
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import make_figures as MF

R = "E:/schedule/revision/results_new/original/ablation_m5eq"
INST = [("C1", "1aea822f"), ("C2", "a1e8b5a7"), ("C3", "ba5b9796"), ("C4", "df75e3dc")]


def series(c):
    base = f"{R}/{c}/A4_no_smooth"
    s1 = json.load(open(glob.glob(f"{base}/stage1/*.json")[0]))
    s2 = json.load(open(glob.glob(f"{base}/*_s1.json")[0]))
    wall = dict(zip(*[iter(open(f"{base}/wall.txt").read().split())] * 2))
    t1, T = float(wall["stage1"]), float(wall["T"])
    z0 = s1["z_seq"]
    tb, best, tl, lb = [0.0], [z0], [], []
    for r in s1["history"]:
        tb.append(r["seconds"]); best.append(min(best[-1], r["best"]))
        if r["lb"] > -1e300:
            tl.append(r["seconds"]); lb.append(r["lb"])
    tb.append(t1); best.append(s1["z"])  # after re-timing
    for r in s2["history"]:
        tb.append(t1 + r["seconds"]); best.append(min(best[-1], r["best_j_op"]))
    tb.append(t1 + s2["seconds"]); best.append(min(best[-1], s2["z"]))
    tl.append(t1); lb.append(max(lb))
    lbmax = []
    m = -1e300
    for v in lb:
        m = max(m, v); lbmax.append(m)
    return z0, T, t1, tb, best, tl, lbmax


def main():
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 2.1), sharey=False)
    fig.subplots_adjust(wspace=0.35, bottom=0.27, top=0.86)
    for ax, (name, c) in zip(axes, INST):
        z0, T, t1, tb, best, tl, lb = series(c)
        sav = [100 * (z0 - v) / z0 for v in best]
        bnd = [100 * (z0 - v) / z0 for v in lb]
        ax.step([t / T for t in tb], sav, where="post", color=MF.INK, lw=1.0)
        ax.step([t / T for t in tl], bnd, where="post", color=MF.ACCENT, lw=1.0, ls=(0, (3, 2)))
        ax.axvline(t1 / T, color="#b5b5b5", lw=0.6)
        ax.set_title(rf"\textbf{{{name}}}", fontsize=7.5, pad=3)
        ax.set_xlim(0, 1); ax.set_ylim(0, max(bnd) * 1.08 if bnd else 1)
        ax.set_xlabel(r"time / $\Delta$", fontsize=6.5, labelpad=1)
        ax.tick_params(labelsize=6, width=0.5, length=2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel(r"\% of $\widetilde Z(x^0)$", fontsize=6.5)
    h = [plt.Line2D([], [], color=MF.INK, lw=1.0), plt.Line2D([], [], color=MF.ACCENT, lw=1.0, ls=(0, (3, 2))),
         plt.Line2D([], [], color="#b5b5b5", lw=0.6)]
    fig.legend(h, ["saving of the best plan", "saving bound $(\\widetilde Z(x^0)-\\mathrm{LB})/\\widetilde Z(x^0)$",
                   "end of column generation"], loc="lower center", ncol=3, fontsize=6.3, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    MF.save(fig, "fig_convergence")


if __name__ == "__main__":
    main()
    print("done")
