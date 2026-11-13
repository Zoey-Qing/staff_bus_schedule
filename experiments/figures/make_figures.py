"""Figures for the manuscript, as PDF and PNG (no TikZ).

fig_method   : workflow of the two-stage method (figure*, full text width).
fig_clusters : synthetic base network (single column).

Run from any directory:  python make_figures.py
Data: staff_scheduling/Berkeley_Mock_Data_25_Stores_original_unbalanced.xlsx (synthetic).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker  # noqa: F401
import matplotlib.patches  # noqa: F401
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DATA = os.path.join(ROOT, "staff_scheduling", "Berkeley_Mock_Data_25_Stores_original_unbalanced.xlsx")

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{stix}",
    "font.family": "serif",
    "font.size": 7.5,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
})

INK = "#1a1a1a"      # schedules, boxes
MUTED = "#555555"
ACCENT = "#1f5a96"   # prices, duals, bounds
PANEL = "#f2f2f2"


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"{name}.{ext}"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------
# Method workflow
# ----------------------------------------------------------------------------------------------
def box(ax, x0, y0, x1, y1, text, ec=INK, fc="white", lw=0.7, size=7.5, weight=None):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.05",
                                ec=ec, fc=fc, lw=lw, zorder=2))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center", fontsize=size, zorder=3,
            linespacing=1.25)


def panel(ax, x0, y0, x1, y1, title):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.08",
                                ec="#9a9a9a", fc=PANEL, lw=0.6, zorder=0))
    ax.text(x0 + 0.12, y1 - 0.13, title, ha="left", va="top", fontsize=8, zorder=3)


def arrow(ax, pts, color=INK, dashed=False, lw=0.8):
    """Poly-line arrow through the points; head at the last point."""
    ls = (0, (3, 2)) if dashed else "-"
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, ls=ls, zorder=1, solid_capstyle="butt")
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>,head_length=4,head_width=2.2",
                                 color=color, lw=lw, linestyle=ls, shrinkA=0, shrinkB=0, zorder=1))


def label(ax, x, y, s, color=INK, ha="center", va="center", size=6.8, rot=0):
    ax.text(x, y, s, ha=ha, va=va, fontsize=size, color=color, rotation=rot, zorder=3,
            bbox=dict(fc=PANEL, ec="none", pad=0.6) if color is not None else None)


def fig_method():
    W, H = 7.1, 3.35
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    # ---------------- stage 1 ----------------
    panel(ax, 0.05, 0.12, 3.62, 3.02, r"\textbf{Column generation with lifted cuts} (time $\Delta_1$)")
    # top row: master LP <-> pricing
    box(ax, 0.30, 1.98, 1.62, 2.58, "Master LP over\nbranch rosters\n" r"(start: $x^0$ and its cut)", size=7.2)
    box(ax, 2.12, 1.98, 3.44, 2.58, "Pricing per branch\n(heuristic / exact)")
    arrow(ax, [(1.62, 2.40), (2.12, 2.40)], color=ACCENT, dashed=True)
    label(ax, 1.87, 2.68, r"duals $\alpha,\beta,\mu$", color=ACCENT, size=7.0)
    arrow(ax, [(2.12, 2.16), (1.62, 2.16)])
    label(ax, 1.87, 1.90, r"rosters, $\bar c_r<0$", color=MUTED, size=7.0)
    # input
    # bottom row: integer master -> route -> routing LP + lifting
    box(ax, 0.30, 0.38, 1.20, 0.98, "Integer\nmaster")
    box(ax, 1.42, 0.38, 2.32, 0.98, "Route $\\hat x$\n$\\widetilde Z(\\hat x)$")
    box(ax, 2.54, 0.38, 3.44, 0.98, "Routing LP\n+ lifting", ec=ACCENT)
    arrow(ax, [(0.62, 1.98), (0.62, 0.98)])
    label(ax, 0.66, 1.62, "no new\nroster", color=MUTED, ha="left", size=7.0)
    arrow(ax, [(1.20, 0.68), (1.42, 0.68)])
    label(ax, 1.31, 0.83, r"$\hat x$", color=None, size=6.8)
    arrow(ax, [(2.32, 0.68), (2.54, 0.68)])
    # lifted cut back to the master
    arrow(ax, [(2.99, 0.98), (2.99, 1.48), (1.30, 1.48), (1.30, 1.98)], color=ACCENT, dashed=True)
    label(ax, 2.15, 1.48, r"cut $m$: $\theta\ge\sum_\sigma\pi^m_\sigma y_\sigma$", color=ACCENT, size=7.0)
    ax.text(1.87, 0.22, r"$x^{\mathrm{best}}$: cheapest routed $\hat x$; re-timing gives $x^1$", ha="center",
            va="center", fontsize=7.0, color=MUTED)

    # ---------------- hand-off ----------------
    arrow(ax, [(3.62, 1.30), (4.18, 1.30)], lw=0.9)
    ax.text(3.90, 1.42, r"$x^1$", ha="center", va="bottom", fontsize=7.2)
    arrow(ax, [(3.62, 1.05), (4.18, 1.05)], color=ACCENT, dashed=True)
    ax.text(3.90, 0.93, r"$\pi$ at $x^1$", ha="center", va="top", fontsize=7.0, color=ACCENT)

    # ---------------- stage 2 ----------------
    panel(ax, 4.18, 0.12, 6.18, 3.02, r"\textbf{Neighborhood search} (time $\Delta_2$)")
    box(ax, 4.36, 2.04, 6.00, 2.62,
        r"Start: $\phi_\kappa$ from $\pi$, $\psi=0$" "\n" r"archive $\mathcal A=\{x^0,x^1\}$", ec=ACCENT)
    box(ax, 4.36, 1.16, 5.12, 1.66, "Destroy:\nfree $F$ riders")
    box(ax, 5.24, 1.16, 6.00, 1.66, "Repair MIP\nwith $L+\\widehat T$")
    box(ax, 5.24, 0.34, 6.00, 0.84, "Route; keep\nif cheaper")
    box(ax, 4.36, 0.34, 5.12, 0.84, "Update the\nestimate")
    arrow(ax, [(4.74, 2.04), (4.74, 1.66)], color=ACCENT, dashed=True)
    arrow(ax, [(5.12, 1.41), (5.24, 1.41)])
    arrow(ax, [(5.62, 1.16), (5.62, 0.84)])
    arrow(ax, [(5.24, 0.59), (5.12, 0.59)])
    arrow(ax, [(4.74, 0.84), (4.74, 1.16)], color=ACCENT, dashed=True)
    ax.text(5.18, 1.00, r"$J$ per round", ha="center", va="center", fontsize=7.0, color=MUTED)

    # ---------------- output ----------------
    box(ax, 6.34, 0.62, 7.06, 2.40,
        "Output\n\n" r"$x^2$ and" "\n" "its bus plan\n\n" r"$\mathrm{LB}\le z^*$" "\n\n" "saving bound\n"
        r"$\widetilde Z(x^0)-\mathrm{LB}$" "\n\n" r"gap" "\n"
        r"$\frac{\widetilde Z(x^2)-\mathrm{LB}}{\widetilde Z(x^2)}$", size=7.0)
    arrow(ax, [(6.18, 1.10), (6.34, 1.10)], lw=0.9)
    # bound from exact pricing straight to the output
    arrow(ax, [(3.10, 2.58), (3.10, 3.18), (6.70, 3.18), (6.70, 2.40)], color=ACCENT, dashed=True)
    ax.text(4.90, 3.21, r"exact pricing: $\mathrm{LB}=\max\big\{\mathrm{LB},\ \sum_b\underline z_b-\sum_l\beta_l\bar V_l\big\}$",
            ha="center", va="bottom", fontsize=7.0, color=ACCENT)

    # legend
    ax.plot([6.36, 6.56], [0.40, 0.40], color=INK, lw=0.8)
    ax.text(6.60, 0.40, "schedules", va="center", fontsize=7.0)
    ax.plot([6.36, 6.56], [0.22, 0.22], color=ACCENT, lw=0.8, ls=(0, (3, 2)))
    ax.text(6.60, 0.22, "prices, bounds", va="center", fontsize=7.0)

    ax.set_ylim(0, 3.45)
    save(fig, "fig_method")


# ----------------------------------------------------------------------------------------------
# Base network
# ----------------------------------------------------------------------------------------------
# Cluster labels used in the paper (ids sorted alphabetically).
PAPER = {"1aea822f": "C1", "662b3495": "C2", "a1e8b5a7": "C3", "ba5b9796": "C4", "df75e3dc": "C5"}
COLORS = {"C1": "#0072B2", "C2": "#D55E00", "C3": "#009E73", "C4": "#CC79A7", "C5": "#E69F00"}


def fig_clusters():
    x = pd.ExcelFile(DATA)
    br = x.parse("branches").rename(columns={"longitute": "longitude"})
    ac = x.parse("accommodation")
    st = x.parse("staff")
    lat0 = br.latitude.mean(); lon0 = br.longitude.mean()
    kx = 111.32 * np.cos(np.radians(lat0)); ky = 110.57

    def xy(df):
        return (df.longitude - lon0) * kx, (df.latitude - lat0) * ky

    br["x"], br["y"] = xy(br); ac["x"], ac["y"] = xy(ac)
    br["c"] = br.cluster_id.str[:8].map(PAPER); ac["c"] = ac.cluster_id.str[:8].map(PAPER)
    links = st.dropna(subset=["accommodation_id"]).groupby(["branch_id", "accommodation_id"]).size().reset_index()

    fig, ax = plt.subplots(figsize=(3.4, 3.15))
    bpos = br.set_index("branch_id")[["x", "y"]]; apos = ac.set_index("id")[["x", "y"]]
    for _, r in links.iterrows():
        if r.branch_id in bpos.index and r.accommodation_id in apos.index:
            b = bpos.loc[r.branch_id]; a = apos.loc[r.accommodation_id]
            ax.plot([a.x, b.x], [a.y, b.y], color="#b5b5b5", lw=0.5, zorder=1)
    for c, g in br.groupby("c"):
        ax.scatter(g.x, g.y, s=16, marker="o", fc=COLORS[c], ec="white", lw=0.5, zorder=3)
        ga = ac[ac.c == c]
        ax.scatter(ga.x, ga.y, s=26, marker="s", fc="white", ec=COLORS[c], lw=1.1, zorder=3)
        pts = np.vstack([g[["x", "y"]].values, ga[["x", "y"]].values])
        cx, cy = pts[:, 0].min(), pts[:, 1].max()
        ax.text(cx - 0.6, cy + 0.6, c, fontsize=8, color=INK, ha="right", va="bottom", zorder=4)
    # legend
    ax.scatter([], [], s=16, marker="o", fc=MUTED, ec="white", label="branch")
    ax.scatter([], [], s=26, marker="s", fc="white", ec=MUTED, lw=1.1, label="accommodation")
    ax.plot([], [], color="#b5b5b5", lw=0.5, label="staff housed at accommodation")
    ax.legend(loc="lower right", fontsize=6.5, frameon=False, handlelength=1.4, borderaxespad=0.2)
    ax.set_aspect("equal")
    ax.set_xlabel("east--west (km)"); ax.set_ylabel("north--south (km)")
    ax.tick_params(labelsize=6.5, width=0.5, length=2.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()
    ax.set_xlim(xmin - 2.5, xmax + 1); ax.set_ylim(ymin - 1, ymax + 2)
    save(fig, "fig_clusters")


# ----------------------------------------------------------------------------------------------
# Instances: one column per cluster (the four test instances; 662b3495 is not used)
# ----------------------------------------------------------------------------------------------
INST = [("C1", "1aea822f"), ("C2", "a1e8b5a7"), ("C3", "ba5b9796"), ("C4", "df75e3dc")]
X0 = os.path.join(ROOT, "revision", "results_new", "original", "ablation_m5eq", "{c}", "A4_no_smooth", "input",
                  "Berkeley_Mock_Data_25_Stores_original_unbalanced_{c}_x0.csv")
DAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]


def fig_instances():
    sys_path = os.path.join(ROOT, "bus_scheduling_routing")
    import sys
    sys.path.insert(0, sys_path)
    from bus_riders import is_bus_rider
    x = pd.ExcelFile(DATA)
    br = x.parse("branches").rename(columns={"longitute": "longitude"})
    ac = x.parse("accommodation")
    st = x.parse("staff")
    dem = x.parse("staff_demand")
    br["c8"] = br.cluster_id.str[:8]; ac["c8"] = ac.cluster_id.str[:8]
    st = st.merge(br[["branch_id", "c8"]], on="branch_id")
    st["rider"] = is_bus_rider(st["Nationality"])
    dem = dem.merge(br[["branch_id", "c8"]], on="branch_id")

    fig, axes = plt.subplots(3, 4, figsize=(7.1, 5.6), gridspec_kw=dict(height_ratios=[1.15, 1, 1]))
    fig.subplots_adjust(wspace=0.32, hspace=0.78, bottom=0.11)
    hours = np.arange(24)
    for j, (name, c) in enumerate(INST):
        b = br[br.c8 == c]; a = ac[ac.c8 == c]; s = st[st.c8 == c]
        lat0, lon0 = b.latitude.mean(), b.longitude.mean()
        kx, ky = 111.32 * np.cos(np.radians(lat0)), 110.57
        bx, by = (b.longitude - lon0) * kx, (b.latitude - lat0) * ky
        axx, axy = (a.longitude - lon0) * kx, (a.latitude - lat0) * ky
        bpos = dict(zip(b.branch_id, zip(bx, by))); apos = dict(zip(a.id, zip(axx, axy)))
        # (a) layout: rider links, branches, accommodations
        ax = axes[0, j]
        links = s[s.rider].groupby(["branch_id", "accommodation_id"]).size()
        for (bi, ai), n in links.items():
            if bi in bpos and ai in apos:
                ax.plot([apos[ai][0], bpos[bi][0]], [apos[ai][1], bpos[bi][1]], color="#b5b5b5",
                        lw=0.4 + 0.12 * n, zorder=1, solid_capstyle="round")
        ax.scatter(bx, by, s=14, marker="o", fc=INK, ec="white", lw=0.5, zorder=3)
        ax.scatter(axx, axy, s=24, marker="s", fc="white", ec=ACCENT, lw=1.0, zorder=3)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(rf"\textbf{{{name}}}: {len(b)} branches, {len(s)} staff, {int(s.rider.sum())} riders",
                     fontsize=7, pad=3)
        ax.tick_params(labelsize=7, width=0.5, length=2)
        ax.set_xlabel("km", fontsize=7.5, labelpad=1)
        if j == 0:
            ax.set_ylabel("km", fontsize=7.5, labelpad=1)
        # (b) staff demand by hour of the day: mean over the seven days, band from min to max
        ax = axes[1, j]
        d = dem[dem.c8 == c].groupby(["day", "hour"]).staff_demand.sum().unstack("hour").reindex(columns=hours).fillna(0)
        ax.fill_between(hours, d.min().values, d.max().values, color="#cfcfcf", lw=0, step="mid")
        ax.step(hours, d.mean().values, where="mid", color=INK, lw=0.9)
        ax.set_xlim(-0.5, 23.5); ax.set_xticks([0, 6, 12, 18, 23])
        ax.tick_params(labelsize=7, width=0.5, length=2)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("hour of the day", fontsize=7.5, labelpad=1)
        if j == 0:
            ax.set_ylabel("staff demanded", fontsize=7.5)
        # (c) rider shift starts and ends in the sequential schedule, by hour of the day, over the week
        ax = axes[2, j]
        x0 = pd.read_csv(X0.format(c=c))
        x0 = x0[x0.Staff.isin(set(s.staff_id[s.rider]))]
        starts = np.bincount(x0.Start.values % 24, minlength=24)
        ends = np.bincount(x0.End.values % 24, minlength=24)
        ax.bar(hours, starts, width=0.8, color=INK, lw=0, label="shift starts (drop-offs)")
        ax.bar(hours, -ends, width=0.8, color=ACCENT, lw=0, label="shift ends (pick-ups)")
        ax.axhline(0, color=INK, lw=0.5)
        ax.set_xlim(-0.5, 23.5); ax.set_xticks([0, 6, 12, 18, 23])
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _p: f"{abs(int(v))}"))
        ax.tick_params(labelsize=7, width=0.5, length=2)
        ax.set_xlabel("hour of the day", fontsize=7.5, labelpad=1)
        if j == 0:
            ax.set_ylabel("riders per week", fontsize=7.5)
        for r in range(3):
            for sp in ("top", "right"):
                axes[r, j].spines[sp].set_visible(False)
    # shared legends below the rows
    h1 = [plt.Line2D([], [], ls="none", marker="o", ms=4, mfc=INK, mec="white"),
          plt.Line2D([], [], ls="none", marker="s", ms=4.5, mfc="white", mec=ACCENT),
          plt.Line2D([], [], color="#b5b5b5", lw=1.0)]
    fig.legend(h1, ["branch", "accommodation", "riders housed there (width: number)"], loc="upper center",
               bbox_to_anchor=(0.5, axes[0, 0].get_position().y0 - 0.045), ncol=3, fontsize=7.5, frameon=False, handlelength=1.6)
    h2 = [plt.Line2D([], [], color=INK, lw=0.9), matplotlib.patches.Patch(color="#cfcfcf")]
    fig.legend(h2, ["mean over the days", "range over the days"], loc="upper center", bbox_to_anchor=(0.5, axes[1, 0].get_position().y0 - 0.045),
               ncol=2, fontsize=7.5, frameon=False, handlelength=1.6)
    h3 = [matplotlib.patches.Patch(color=INK), matplotlib.patches.Patch(color=ACCENT)]
    fig.legend(h3, ["shift starts (drop-offs)", "shift ends (pick-ups)"], loc="upper center",
               bbox_to_anchor=(0.5, axes[2, 0].get_position().y0 - 0.045), ncol=2, fontsize=7.5, frameon=False, handlelength=1.6)
    save(fig, "fig_instances")


if __name__ == "__main__":
    fig_method()
    fig_clusters()
    fig_instances()
    print("done")
