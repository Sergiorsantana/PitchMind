"""Radar tático 2D (vista de cima) com matplotlib."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Arc, Circle, Rectangle  # noqa: E402
from scipy.spatial import ConvexHull, QhullError  # noqa: E402

from pitchmind.models import FrameState, PITCH_LENGTH as L, PITCH_WIDTH as W  # noqa: E402

COLORS = {"A": "#2f7de1", "B": "#e5484d", "ball": "#f5d90a"}


def draw_pitch(ax):
    ax.add_patch(Rectangle((-3, -3), L + 6, W + 6, color="#1f6f3a", zorder=0))
    for k in range(0, 10, 2):  # faixas do gramado
        ax.add_patch(Rectangle((k * L / 10, 0), L / 10, W, color="#237a41", zorder=0))
    kw = dict(color="white", lw=1.2, fill=False)
    ax.add_patch(Rectangle((0, 0), L, W, **kw))
    ax.plot([L / 2, L / 2], [0, W], color="white", lw=1.2)
    ax.add_patch(Circle((L / 2, W / 2), 9.15, **kw))
    for x0, s in ((0, 1), (L, -1)):
        ax.add_patch(Rectangle((x0 if s > 0 else x0 - 16.5, W / 2 - 20.15), 16.5, 40.3, **kw))
        ax.add_patch(Rectangle((x0 if s > 0 else x0 - 5.5, W / 2 - 9.15), 5.5, 18.3, **kw))
        ax.add_patch(Arc((x0 + s * 11, W / 2), 18.3, 18.3, theta1=-53 if s > 0 else 127,
                         theta2=53 if s > 0 else 233, color="white", lw=1.2))
    ax.set_xlim(-3, L + 3)
    ax.set_ylim(-3, W + 3)
    ax.set_aspect("equal")
    ax.axis("off")


def radar(frame: FrameState, title: str = "", labels: dict[str, list[str]] | None = None, hulls: bool = True):
    fig, ax = plt.subplots(figsize=(8, 5.4), dpi=110)
    fig.patch.set_facecolor("#0e1117")
    draw_pitch(ax)
    for team in ("A", "B"):
        pts = frame.team(team)
        if hulls and len(pts) >= 4:
            out = pts[np.argsort(pts[:, 0] if team == "A" else -pts[:, 0])][1:]
            try:
                h = ConvexHull(out)
                ax.fill(*out[h.vertices].T, color=COLORS[team], alpha=0.25)
            except QhullError:
                pass
        ax.scatter(pts[:, 0], pts[:, 1], s=140, c=COLORS[team], edgecolors="white", lw=1.2, zorder=3)
        for i, (x, y) in enumerate(pts):
            lab = (labels or {}).get(team, [None] * len(pts))[i] if labels else None
            if lab:
                ax.text(x, y, lab, ha="center", va="center", fontsize=7, color="white", zorder=4, weight="bold")
    if frame.ball is not None:
        ax.scatter(*frame.ball, s=60, c=COLORS["ball"], edgecolors="black", zorder=5)
    if title:
        ax.set_title(title, color="white", fontsize=11)
    fig.tight_layout()
    return fig
