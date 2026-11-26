"""Transform: tracking -> métricas táticas quantitativas por janela.

Todas as métricas de time são calculadas na "perspectiva de ataque" do
próprio time (x'=0 no próprio gol, x'=105 no gol adversário), assim
"altura do bloco = 60" significa a mesma coisa para A e para B.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree

from pitchmind.models import FrameState, Window, PITCH_LENGTH, PITCH_WIDTH, TEAMS

# grade para pitch control (Voronoi discretizado) — 21 x 14 células
_GX, _GY = np.meshgrid(np.linspace(2.5, PITCH_LENGTH - 2.5, 21), np.linspace(2.4, PITCH_WIDTH - 2.4, 14))
_GRID = np.c_[_GX.ravel(), _GY.ravel()]

# dicionário de métricas: nome -> (descrição PT, unidade). Usado pelo planner e pelo verificador.
METRIC_DOCS: dict[str, tuple[str, str]] = {}
for _t in TEAMS:
    METRIC_DOCS.update({
        f"{_t}_block_height": (f"altura média do bloco do time {_t} (m a partir do próprio gol)", "m"),
        f"{_t}_def_line": (f"altura da linha defensiva do time {_t}", "m"),
        f"{_t}_width": (f"largura ocupada pelo time {_t}", "m"),
        f"{_t}_depth": (f"profundidade (distância último->primeiro homem de linha) do time {_t}", "m"),
        f"{_t}_compactness": (f"área do fecho convexo dos jogadores de linha do time {_t} (menor = mais compacto)", "m²"),
        f"{_t}_pressers": (f"nº médio de jogadores do time {_t} a menos de 10 m da bola", "jogadores"),
        f"{_t}_pitch_control": (f"% do campo controlado pelo time {_t} (Voronoi)", "%"),
        f"{_t}_possession": (f"% de posse de bola do time {_t} na janela", "%"),
        f"{_t}_field_tilt": (f"% do tempo com a bola no terço final ofensivo do time {_t}", "%"),
    })
METRIC_DOCS["ball_speed"] = ("velocidade média da bola", "m/s")
METRIC_DOCS["ball_progression_A"] = ("avanço líquido da bola rumo ao gol de B (positivo = A progrediu)", "m")


def _attack_frame(pts: np.ndarray, team: str) -> np.ndarray:
    if team == "A" or len(pts) == 0:
        return pts
    out = pts.copy()
    out[:, 0] = PITCH_LENGTH - out[:, 0]
    out[:, 1] = PITCH_WIDTH - out[:, 1]
    return out


def _hull_area(pts: np.ndarray) -> float:
    if len(pts) < 3:
        return float("nan")
    try:
        return float(ConvexHull(pts).volume)  # em 2D, volume = área
    except QhullError:
        return float("nan")


def frame_metrics(f: FrameState) -> dict[str, float]:
    m: dict[str, float] = {}
    for team in TEAMS:
        pts = _attack_frame(f.team(team), team)
        if len(pts) < 4:
            continue
        outfield = pts[np.argsort(pts[:, 0])][1:]  # remove o mais recuado (goleiro)
        xs = np.sort(outfield[:, 0])
        m[f"{team}_block_height"] = float(xs.mean())
        m[f"{team}_def_line"] = float(xs[:4].mean())
        m[f"{team}_depth"] = float(xs[-1] - xs[0])
        m[f"{team}_width"] = float(np.percentile(outfield[:, 1], 90) - np.percentile(outfield[:, 1], 10))
        m[f"{team}_compactness"] = _hull_area(outfield)
        if f.ball is not None:
            d = np.linalg.norm(f.team(team) - f.ball, axis=1)
            m[f"{team}_pressers"] = float((d < 10).sum())

    a, b = f.players_a, f.players_b
    if len(a) and len(b):
        _, owner = cKDTree(np.vstack([a, b])).query(_GRID)
        m["A_pitch_control"] = float((owner < len(a)).mean() * 100)
        m["B_pitch_control"] = 100 - m["A_pitch_control"]
        if f.ball is not None:
            da, db = np.linalg.norm(a - f.ball, axis=1).min(), np.linalg.norm(b - f.ball, axis=1).min()
            m["_owner_A"] = float(da < db)
            m["_ball_x"] = float(f.ball[0])
    return m


def aggregate_windows(frames: list[FrameState], window_s: float = 10.0) -> list[Window]:
    """Agrega frames em janelas fixas. Unidade de indexação do RAG."""
    if not frames:
        return []
    windows: list[Window] = []
    t_end = frames[-1].t
    n = int(np.ceil((t_end + 1e-9) / window_s)) or 1
    buckets: list[list[FrameState]] = [[] for _ in range(n)]
    for f in frames:
        buckets[min(int(f.t // window_s), n - 1)].append(f)

    for wid, fs in enumerate(buckets):
        if not fs:
            continue
        per = [frame_metrics(f) for f in fs]
        keys = {k for d in per for k in d if not k.startswith("_")}
        agg = {k: float(np.nanmean([d[k] for d in per if k in d])) for k in sorted(keys)}

        owners = [d["_owner_A"] for d in per if "_owner_A" in d]
        if owners:
            agg["A_possession"] = float(np.mean(owners) * 100)
            agg["B_possession"] = 100 - agg["A_possession"]
        bx = np.array([d["_ball_x"] for d in per if "_ball_x" in d])
        balls = [(f.t, f.ball) for f in fs if f.ball is not None]
        if len(bx):
            agg["A_field_tilt"] = float((bx > 2 * PITCH_LENGTH / 3).mean() * 100)
            agg["B_field_tilt"] = float((bx < PITCH_LENGTH / 3).mean() * 100)
            agg["ball_progression_A"] = float(bx[-1] - bx[0])
        if len(balls) > 1:
            ts = np.array([t for t, _ in balls])
            ps = np.array([p for _, p in balls])
            dist = np.linalg.norm(np.diff(ps, axis=0), axis=1)
            dt = np.diff(ts)
            agg["ball_speed"] = float(dist.sum() / max(dt.sum(), 1e-6))

        phases = [f.phase for f in fs if f.phase]
        true_phase = max(sorted(set(phases)), key=phases.count) if phases else None
        mid = fs[len(fs) // 2]
        windows.append(Window(window_id=wid, t_start=fs[0].t, t_end=fs[-1].t, metrics=agg,
                              true_phase=true_phase, keyframe_t=mid.t))
    return windows
