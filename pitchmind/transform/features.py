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


def _aggregate(wid: int, fs: list[FrameState]) -> Window:
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
    return Window(window_id=wid, t_start=fs[0].t, t_end=fs[-1].t, metrics=agg,
                  true_phase=true_phase, keyframe_t=mid.t)


def aggregate_windows(frames: list[FrameState], window_s: float = 10.0) -> list[Window]:
    """Agrega frames em janelas FIXAS. Unidade de indexação do RAG."""
    if not frames:
        return []
    t_end = frames[-1].t
    n = int(np.ceil((t_end + 1e-9) / window_s)) or 1
    buckets: list[list[FrameState]] = [[] for _ in range(n)]
    for f in frames:
        buckets[min(int(f.t // window_s), n - 1)].append(f)
    return [_aggregate(i, fs) for i, fs in enumerate(b for b in buckets if b)]


def possession_owner(frames: list[FrameState], smooth_s: float = 2.0) -> np.ndarray:
    """Dono da posse por frame (1 = A, 0 = B), suavizado por mediana móvel.

    Bola não detectada herda o último dono conhecido (oclusão não é troca de posse).
    """
    raw = np.full(len(frames), np.nan)
    for i, f in enumerate(frames):
        if f.ball is not None and len(f.players_a) and len(f.players_b):
            raw[i] = float(np.linalg.norm(f.players_a - f.ball, axis=1).min()
                           < np.linalg.norm(f.players_b - f.ball, axis=1).min())
    last = 1.0
    for i in range(len(raw)):  # forward-fill
        last = raw[i] = last if np.isnan(raw[i]) else raw[i]
    fps = len(frames) / max(frames[-1].t - frames[0].t, 1e-6) if len(frames) > 1 else 1.0
    k = max(1, int(smooth_s * fps)) | 1
    pad = np.pad(raw, k // 2, mode="edge")
    return np.array([np.median(pad[i:i + k]) for i in range(len(raw))])


def aggregate_possession_windows(frames: list[FrameState], min_s: float = 3.0, max_s: float = 12.0,
                                 smooth_s: float = 2.0) -> list[Window]:
    """Janelas ADAPTATIVAS: cortam na troca de posse (suavizada), não no relógio.

    - uma posse longa é dividida em pedaços de até `max_s`;
    - segmentos mais curtos que `min_s` são fundidos ao vizinho anterior
      (disputas rápidas de bola não viram janelas próprias).
    Resultado: cada janela descreve UMA fase (com bola / sem bola) de um time.
    """
    if not frames:
        return []
    owner = possession_owner(frames, smooth_s)
    segs: list[list[int]] = [[0]]
    for i in range(1, len(frames)):
        cur = segs[-1]
        if owner[i] != owner[cur[0]] or frames[i].t - frames[cur[0]].t >= max_s:
            segs.append([i])
        else:
            cur.append(i)
    merged: list[list[int]] = []
    for sg in segs:
        dur = frames[sg[-1]].t - frames[sg[0]].t
        if merged and dur < min_s:
            merged[-1].extend(sg)
        else:
            merged.append(sg)
    if len(merged) > 1 and frames[merged[0][-1]].t - frames[merged[0][0]].t < min_s:
        merged[1] = merged[0] + merged[1]
        merged.pop(0)
    return [_aggregate(i, [frames[j] for j in sg]) for i, sg in enumerate(merged)]


def phase_purity(windows: list[Window], frames: list[FrameState]) -> float:
    """Fração de frames cuja fase real coincide com a fase majoritária da sua janela."""
    hit = tot = 0
    j = 0
    for f in frames:
        while j < len(windows) - 1 and f.t > windows[j].t_end + 1e-9:
            j += 1
        if f.phase is not None:
            tot += 1
            hit += f.phase == windows[j].true_phase
    return hit / max(tot, 1)
