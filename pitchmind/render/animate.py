"""Animação "ao vivo": radar tático + métricas e eventos atualizados em tempo real.

    pitchmind animate --start 250 --seconds 40 --out docs/img/live.gif
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from scipy.spatial import ConvexHull, QhullError  # noqa: E402

from pitchmind.models import FrameState, Window  # noqa: E402
from pitchmind.render.radar import COLORS, draw_pitch  # noqa: E402

_EV_PT = {"high_press": "PRESSÃO ALTA", "low_block": "BLOCO BAIXO", "counter": "CONTRA-ATAQUE",
          "build_up": "CONSTRUÇÃO"}


def _hull(pts: np.ndarray, team: str) -> np.ndarray | None:
    if len(pts) < 4:
        return None
    out = pts[np.argsort(pts[:, 0] if team == "A" else -pts[:, 0])][1:]
    try:
        return out[ConvexHull(out).vertices]
    except QhullError:
        return None


def animate(frames: list[FrameState], windows: list[Window], out: str | Path,
            start_s: float = 250.0, seconds: float = 40.0, speed: float = 4.0, fps_out: int = 10,
            numbers: dict[str, list[str]] | None = None) -> Path:
    """Gera um GIF: cada quadro do GIF avança `speed / fps_out` segundos de jogo."""
    src_fps = len(frames) / (frames[-1].t + 1e-9)
    step = max(1, int(round(src_fps * speed / fps_out)))
    i0 = int(start_s * src_fps)
    idx = list(range(i0, min(len(frames), i0 + int(seconds * src_fps)), step))
    win_of = {w.window_id: w for w in windows}
    wsize = windows[1].t_start - windows[0].t_start if len(windows) > 1 else 10.0
    hist = {"t": [], "A": [], "B": []}  # série do controle de campo

    fig = plt.figure(figsize=(10, 4.6), dpi=80)
    fig.patch.set_facecolor("#0e1117")
    ax = fig.add_axes([0.01, 0.04, 0.64, 0.86])
    side = fig.add_axes([0.69, 0.10, 0.29, 0.36])
    draw_pitch(ax)
    title = fig.text(0.02, 0.94, "", color="white", fontsize=13, weight="bold", family="monospace")
    banner = fig.text(0.665, 0.94, "", color="#0e1117", fontsize=11, weight="bold",
                      bbox=dict(boxstyle="round,pad=0.35", fc="#f5d90a", ec="none"))
    panel = fig.text(0.69, 0.86, "", color="white", fontsize=9.5, family="monospace", va="top")

    hulls = {t: ax.fill([], [], color=COLORS[t], alpha=0.25, zorder=2)[0] for t in "AB"}
    dots = {t: ax.scatter([], [], s=150, c=COLORS[t], edgecolors="white", lw=1.2, zorder=3) for t in "AB"}
    labels = {t: [ax.text(0, 0, "", ha="center", va="center", fontsize=6.5, color="white", weight="bold",
                          zorder=4) for _ in range(11)] for t in "AB"}
    ball = ax.scatter([], [], s=70, c=COLORS["ball"], edgecolors="black", zorder=5)
    trail, = ax.plot([], [], color=COLORS["ball"], lw=1.5, alpha=0.6, zorder=4)
    trail_pts: list[np.ndarray] = []

    side.set_facecolor("#0e1117")
    for s in side.spines.values():
        s.set_color("#444")
    side.tick_params(colors="#aaa", labelsize=7)
    side.set_ylim(20, 80)
    side.set_title("controle de campo (%)", color="#aaa", fontsize=8)
    lA, = side.plot([], [], color=COLORS["A"], lw=2)
    lB, = side.plot([], [], color=COLORS["B"], lw=2)

    def update(k: int):
        f = frames[idx[k]]
        w = win_of.get(int(f.t // wsize))
        m = w.metrics if w else {}
        for t in "AB":
            pts = f.team(t)
            dots[t].set_offsets(pts)
            h = _hull(pts, t)
            hulls[t].set_xy(h if h is not None else np.zeros((0, 2)))
            nums = (numbers or {}).get(t, [])
            for j, txt in enumerate(labels[t]):
                if j < len(pts) and j < len(nums):
                    txt.set_position(pts[j])
                    txt.set_text(nums[j])
        if f.ball is not None:
            ball.set_offsets([f.ball])
            trail_pts.append(f.ball)
        del trail_pts[:-8]
        if trail_pts:
            trail.set_data(*np.array(trail_pts).T)

        title.set_text(f"{int(f.t // 60):02d}:{int(f.t % 60):02d}   PitchMind · ao vivo")
        evs = [f"{_EV_PT.get(e.split('_', 1)[1], e)} ({e[0]})" for e in (w.events if w else [])]
        banner.set_text(" · ".join(evs) if evs else "")
        banner.set_visible(bool(evs))
        g = lambda key, fmt="{:.0f}": fmt.format(m[key]) if key in m else "–"  # noqa: E731
        panel.set_text(
            f"{'':12s}{'A':>6s}{'B':>6s}\n"
            f"{'posse %':12s}{g('A_possession'):>6s}{g('B_possession'):>6s}\n"
            f"{'bloco (m)':12s}{g('A_block_height'):>6s}{g('B_block_height'):>6s}\n"
            f"{'linha (m)':12s}{g('A_def_line'):>6s}{g('B_def_line'):>6s}\n"
            f"{'área (m²)':12s}{g('A_compactness'):>6s}{g('B_compactness'):>6s}\n"
            f"{'pressão':12s}{g('A_pressers', '{:.1f}'):>6s}{g('B_pressers', '{:.1f}'):>6s}\n"
            f"bola {g('ball_speed', '{:.1f}')} m/s")
        n_a = len(f.players_a) + len(f.players_b)
        if n_a:
            from scipy.spatial import cKDTree
            from pitchmind.transform.features import _GRID
            _, own = cKDTree(np.vstack([f.players_a, f.players_b])).query(_GRID)
            pa = float((own < len(f.players_a)).mean() * 100)
            hist["t"].append(f.t)
            hist["A"].append(pa)
            hist["B"].append(100 - pa)
            lA.set_data(hist["t"], hist["A"])
            lB.set_data(hist["t"], hist["B"])
            side.set_xlim(frames[idx[0]].t, max(frames[idx[-1]].t, frames[idx[0]].t + 1))
        return []

    anim = FuncAnimation(fig, update, frames=len(idx), blit=False)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=fps_out))
    plt.close(fig)
    return out
