"""Figure fig_problem: how a staff schedule creates stops, trips and the fleet (a small illustrative example, not data).

Example: two branches b1, b2, two accommodations a1, a2, five riders.
  r1 b1 a1 7-15   r2 b1 a1 7-16   r3 b2 a1 7-17   r4 b1 a1 15-23   r5 b2 a2 7-17
Stops (q drop-offs, p pick-ups): (b1,a1,7) q2; (b2,a1,7) q1; (b2,a2,7) q1; (b1,a1,15) q1 p1; (b1,a1,16) p1;
  (b2,a1,17) p1; (b2,a2,17) p1; (b1,a1,23) p1.
Trips: 7:00 a1->b1->b2->a1 and a2->b2->a2; 15:00, 16:00, 23:00 a1->b1; 17:00 a1->b2 and a2->b2. Fleet: 2 buses.
Run: python fig_problem.py (uses the style of make_figures.py).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, ConnectionPatch
import make_figures as MF

INK, MUTED, ACCENT = MF.INK, MF.MUTED, MF.ACCENT
B1, B2 = INK, ACCENT  # colour of the two branches

RIDERS = [("r1", "b1", "a1", 7, 15), ("r2", "b1", "a1", 7, 16), ("r3", "b2", "a1", 7, 17),
          ("r4", "b1", "a1", 15, 23), ("r5", "b2", "a2", 7, 17)]


def stops():
    out = {}
    for _r, b, a, s, e in RIDERS:
        out.setdefault((b, a, s), [0, 0])[0] += 1
        out.setdefault((b, a, e), [0, 0])[1] += 1
    return out


def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=7, width=0.5, length=2)


def main():
    fig = plt.figure(figsize=(7.1, 2.25))
    xs = [0.055, 0.345, 0.585, 0.80]
    w = [0.19, 0.18, 0.14, 0.195]
    axs = [fig.add_axes([x, 0.2, ww, 0.62]) for x, ww in zip(xs, w)]
    titles = ["(a) Staff schedule", "(b) Stops", "(c) Trips at 7:00", "(d) Buses over the day"]
    for ax, t in zip(axs, titles):
        ax.set_title(t, fontsize=8, pad=4, loc="left")

    # (a) shifts of the riders
    ax = axs[0]
    for k, (r, b, a, s, e) in enumerate(RIDERS):
        y = len(RIDERS) - 1 - k
        ax.barh(y, e - s, left=s, height=0.5, color=B1 if b == "b1" else B2, lw=0)
        ax.plot(s, y + 0.42, marker="v", ms=3.5, color=INK, ls="none")
        ax.plot(e, y + 0.42, marker="^", ms=3.5, mfc="white", mec=INK, mew=0.7, ls="none")
    ax.set_yticks(range(len(RIDERS)))
    ax.set_yticklabels([f"$r_{k}$ ({a})" for k, (_r, _b, a, _s, _e) in reversed(list(enumerate(RIDERS, 1)))], fontsize=7)
    ax.set_xlim(5, 24.5); ax.set_ylim(-0.6, len(RIDERS) - 0.1); ax.set_xticks([6, 12, 18, 24])
    ax.set_xlabel("hour", fontsize=7, labelpad=1)
    style(ax)

    # (b) stops by branch-accommodation pair and hour
    ax = axs[1]
    rows = [("b1", "a1"), ("b2", "a1"), ("b2", "a2")]
    for (b, a, t), (q, p) in stops().items():
        y = len(rows) - 1 - rows.index((b, a))
        col = B1 if b == "b1" else B2
        if q:
            ax.plot(t - 0.25, y + 0.12, marker="v", ms=3 + 1.5 * q, color=col, ls="none")
        if p:
            ax.plot(t + 0.25, y - 0.12, marker="^", ms=3 + 1.5 * p, mfc="white", mec=col, mew=0.8, ls="none")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"({b},{a})" for b, a in reversed(rows)], fontsize=7)
    ax.set_xlim(5, 24.5); ax.set_ylim(-0.6, len(rows) - 0.4); ax.set_xticks([6, 12, 18, 24])
    ax.set_xlabel("hour", fontsize=7, labelpad=1)
    ax.text(7.3, 2.45, r"$q{=}2$", fontsize=7, color=MUTED, va="center")
    ax.text(15.3, 2.42, r"$q{=}1,\,p{=}1$", fontsize=7, color=MUTED, va="center")
    style(ax)

    # (c) the two trips at 7:00 on a small map
    ax = axs[2]
    P = {"a1": (0.0, 0.0), "a2": (2.6, 2.4), "b1": (1.4, 0.2), "b2": (2.2, 1.2)}

    def route(seq, col):
        for u, v in zip(seq[:-1], seq[1:]):
            ax.add_patch(FancyArrowPatch(P[u], P[v], arrowstyle="-|>,head_length=3.5,head_width=2", color=col,
                                         lw=0.8, shrinkA=5, shrinkB=5, connectionstyle="arc3,rad=0.12"))
    route(["a1", "b1", "b2", "a1"], INK)
    route(["a2", "b2", "a2"], ACCENT)
    for n in ("a1", "a2"):
        ax.plot(*P[n], marker="s", ms=6, mfc="white", mec=INK, mew=1.0)
    ax.plot(*P["b1"], marker="o", ms=5.5, color=B1, mec="white", mew=0.5)
    ax.plot(*P["b2"], marker="o", ms=5.5, color=B2, mec="white", mew=0.5)
    for n, (dx, dy, ha) in {"a1": (-0.15, -0.35, "left"), "a2": (0.0, 0.3, "center"), "b1": (0.0, -0.4, "center"),
                            "b2": (0.25, 0.0, "left")}.items():
        ax.text(P[n][0] + dx, P[n][1] + dy, n, fontsize=7, ha=ha, va="center")
    ax.text(-0.5, 2.55, "bus 1: 3 drop-offs\nbus 2: 1 drop-off", fontsize=7, color=MUTED, va="center")
    ax.set_xlim(-0.5, 3.2); ax.set_ylim(-0.7, 2.9); ax.set_aspect("equal"); ax.axis("off")

    # (d) time-space view: one lane per accommodation, trips leave and return, buses wait in between
    ax = axs[3]
    lane = {"a1": 1.0, "a2": 0.0}
    trips = [("a1", 7, 1), ("a2", 7, 2), ("a1", 15, 1), ("a1", 16, 1), ("a1", 17, 1), ("a2", 17, 2), ("a1", 23, 1)]
    dur = 0.7
    for a, t, bus in trips:
        y = lane[a]; col = INK if bus == 1 else ACCENT
        xx = np.linspace(t - 0.15, t - 0.15 + dur, 20)
        ax.plot(xx, y + 0.32 * np.sin(np.pi * (xx - xx[0]) / dur), color=col, lw=1.0)
    for bus, a, pts in ((1, "a1", [7, 15, 16, 17, 23]), (2, "a2", [7, 17])):
        y = lane[a]; col = INK if bus == 1 else ACCENT
        for s, e in zip(pts[:-1], pts[1:]):
            ax.plot([s - 0.15 + dur, e - 0.15], [y, y], color=col, lw=0.6, ls=(0, (2, 1.5)))
    ax.set_yticks([0, 1]); ax.set_yticklabels(["a2", "a1"], fontsize=7)
    ax.set_xlim(5, 24.5); ax.set_ylim(-0.5, 1.6); ax.set_xticks([6, 12, 18, 24])
    ax.set_xlabel("hour", fontsize=7, labelpad=1)
    ax.text(9.5, 1.45, "bus 1", fontsize=7, color=INK); ax.text(9.5, 0.42, "bus 2", fontsize=7, color=ACCENT)
    style(ax)

    # arrows between the panels
    for i in range(3):
        x0 = xs[i] + w[i] + 0.006; x1 = xs[i + 1] - {0: 0.068, 1: 0.012, 2: 0.04}[i]
        fig.add_artist(FancyArrowPatch((x0, 0.51), (x1, 0.51), transform=fig.transFigure,
                                       arrowstyle="-|>,head_length=4,head_width=2.2", color=MUTED, lw=0.9))
    # legend
    h = [plt.Line2D([], [], color=B1, lw=4), plt.Line2D([], [], color=B2, lw=4),
         plt.Line2D([], [], marker="v", color=INK, ls="none", ms=4),
         plt.Line2D([], [], marker="^", mfc="white", mec=INK, ls="none", ms=4),
         plt.Line2D([], [], color=INK, lw=1.0), plt.Line2D([], [], color=INK, lw=0.6, ls=(0, (2, 1.5)))]
    fig.legend(h, ["branch b1", "branch b2", "shift start (drop-off)", "shift end (pick-up)", "trip", "bus waits"],
               loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=6, fontsize=7, frameon=False, handlelength=1.6,
               columnspacing=1.4)
    MF.save(fig, "fig_problem")


if __name__ == "__main__":
    main()
    print("done")
