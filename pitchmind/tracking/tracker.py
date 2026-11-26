"""Rastreamento multi-objeto estilo ByteTrack + ReID por aparência (à la BoT-SORT).

Implementação enxuta em NumPy/SciPy (sem GPU, sem dependências extras):

1. Kalman de velocidade constante sobre a caixa (cx, cy, w, h);
2. associação em DOIS estágios (ByteTrack): primeiro detecções de alta confiança,
   depois as de baixa confiança com os tracks que sobraram, o que recupera jogadores
   parcialmente ocluídos em vez de descartá-los;
3. custo = mistura de (1 − IoU) e distância de cosseno entre embeddings (ReID);
4. tracks perdidos ficam "em espera" por `max_lost` quadros e podem ser
   re-identificados só pela aparência + proximidade, sem gerar um novo ID.

O embedding padrão é um histograma HSV do tronco (`color_embedding`), suficiente para
separar times e ajudar na re-associação. Qualquer modelo ReID (OSNet etc.) pode ser
plugado passando `embed_fn`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


# ------------------------------------------------------------------ util
def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU entre caixas xyxy: a (n,4), b (m,4) -> (n,m)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def color_embedding(frame: np.ndarray, box: np.ndarray, bins: tuple[int, int] = (12, 6)) -> np.ndarray:
    """Embedding de aparência: histograma H-S do tronco, normalizado (L2)."""
    x1, y1, x2, y2 = box.astype(int)
    h, w = y2 - y1, x2 - x1
    crop = frame[max(y1 + h // 6, 0): max(y1 + h // 2, 1), max(x1 + w // 5, 0): max(x2 - w // 5, 1)]
    if crop.size == 0:
        return np.zeros(bins[0] * bins[1], np.float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, list(bins), [0, 180, 0, 256]).ravel()
    return (hist / (np.linalg.norm(hist) + 1e-9)).astype(np.float32)


def _xyxy_to_z(b: np.ndarray) -> np.ndarray:
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1]], float)


def _z_to_xyxy(z: np.ndarray) -> np.ndarray:
    cx, cy, w, h = z[:4]
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


# ------------------------------------------------------------------ Kalman
class _Kalman:
    """Estado [cx, cy, w, h, vcx, vcy, vw, vh], velocidade constante."""

    def __init__(self, box: np.ndarray):
        self.x = np.r_[_xyxy_to_z(box), np.zeros(4)]
        h = max(self.x[3], 1.0)
        self.P = np.diag(np.r_[[2 * 0.05 * h] * 4, [10 * 0.00625 * h] * 4]) ** 2
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)
        self.H = np.eye(4, 8)

    def predict(self) -> np.ndarray:
        h = max(self.x[3], 1.0)
        Q = np.diag(np.r_[[0.05 * h] * 4, [0.00625 * h] * 4]) ** 2
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + Q
        return _z_to_xyxy(self.x)

    def update(self, box: np.ndarray) -> None:
        h = max(self.x[3], 1.0)
        R = np.diag([0.05 * h] * 4) ** 2
        y = _xyxy_to_z(box) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P


# ------------------------------------------------------------------ tracks
@dataclass
class Track:
    track_id: int
    box: np.ndarray
    score: float
    kf: _Kalman
    emb: Optional[np.ndarray] = None
    hits: int = 1
    lost: int = 0
    history: list[tuple[int, np.ndarray]] = field(default_factory=list)  # (frame_idx, box)

    def predict(self) -> None:
        self.box = self.kf.predict()

    def update(self, box: np.ndarray, score: float, emb: Optional[np.ndarray], frame_idx: int, alpha: float = 0.9):
        self.kf.update(box)
        self.box = _z_to_xyxy(self.kf.x)
        self.score, self.hits, self.lost = score, self.hits + 1, 0
        if emb is not None:  # média móvel exponencial do embedding (BoT-SORT)
            self.emb = emb if self.emb is None else alpha * self.emb + (1 - alpha) * emb
            self.emb = self.emb / (np.linalg.norm(self.emb) + 1e-9)
        self.history.append((frame_idx, box.copy()))


@dataclass
class TrackerConfig:
    high_thresh: float = 0.5      # detecções de alta confiança (1º estágio)
    low_thresh: float = 0.1       # abaixo disso, descarta
    new_track_thresh: float = 0.6
    match_thresh: float = 0.8     # custo máximo aceito no 1º estágio
    low_match_thresh: float = 0.5 # custo máximo (só IoU) no 2º estágio
    reid_weight: float = 0.5      # peso da aparência no custo combinado
    reid_thresh: float = 0.25     # distância de cosseno máx. para re-identificar um track perdido
    max_lost: int = 30            # quadros que um track perdido fica em espera
    min_hits: int = 2             # hits mínimos para o track ser reportado


class ByteTracker:
    def __init__(self, cfg: TrackerConfig | None = None,
                 embed_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = color_embedding):
        self.cfg = cfg or TrackerConfig()
        self.embed_fn = embed_fn
        self.tracks: list[Track] = []
        self.lost: list[Track] = []
        self.removed: list[Track] = []
        self._next_id = 1
        self.frame_idx = -1

    # ---------- custo
    def _cost(self, tracks: list[Track], boxes: np.ndarray, embs: Optional[np.ndarray], use_reid: bool) -> np.ndarray:
        c = 1 - iou_matrix(np.array([t.box for t in tracks]).reshape(-1, 4), boxes)
        if use_reid and embs is not None and all(t.emb is not None for t in tracks) and len(tracks) and len(boxes):
            app = 1 - np.clip(np.array([t.emb for t in tracks]) @ embs.T, 0, 1)
            # aparência só conta quando há alguma sobreposição espacial (evita trocas distantes)
            c = np.where(c < 1.0, (1 - self.cfg.reid_weight) * c + self.cfg.reid_weight * app, c)
        return c

    @staticmethod
    def _match(cost: np.ndarray, thresh: float):
        if cost.size == 0:
            return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
        r, c = linear_sum_assignment(cost)
        pairs = [(i, j) for i, j in zip(r, c) if cost[i, j] <= thresh]
        mi, mj = {i for i, _ in pairs}, {j for _, j in pairs}
        return pairs, [i for i in range(cost.shape[0]) if i not in mi], [j for j in range(cost.shape[1]) if j not in mj]

    # ---------- passo
    def update(self, boxes: np.ndarray, scores: np.ndarray, frame: Optional[np.ndarray] = None) -> list[Track]:
        """boxes (n,4) xyxy, scores (n,). Retorna os tracks ativos e confirmados."""
        cfg = self.cfg
        self.frame_idx += 1
        boxes = np.asarray(boxes, float).reshape(-1, 4)
        scores = np.asarray(scores, float).reshape(-1)
        keep = scores >= cfg.low_thresh
        boxes, scores = boxes[keep], scores[keep]
        embs = None
        if frame is not None and self.embed_fn is not None and len(boxes):
            embs = np.stack([self.embed_fn(frame, b) for b in boxes])

        for t in self.tracks + self.lost:
            t.predict()

        hi, lo = np.where(scores >= cfg.high_thresh)[0], np.where(scores < cfg.high_thresh)[0]
        pool = self.tracks + self.lost

        # 1º estágio: alta confiança x todos os tracks (IoU + ReID)
        e_hi = embs[hi] if embs is not None else None
        pairs, um_t, um_d = self._match(self._cost(pool, boxes[hi], e_hi, True), cfg.match_thresh)
        for i, j in pairs:
            pool[i].update(boxes[hi][j], scores[hi][j], e_hi[j] if e_hi is not None else None, self.frame_idx)
        rest = [pool[i] for i in um_t]

        # 2º estágio (ByteTrack): baixa confiança x tracks ativos restantes, só IoU
        active_rest = [t for t in rest if t.lost == 0]
        pairs2, um_t2, _ = self._match(self._cost(active_rest, boxes[lo], None, False), cfg.low_match_thresh)
        for i, j in pairs2:
            active_rest[i].update(boxes[lo][j], scores[lo][j], embs[lo][j] if embs is not None else None,
                                  self.frame_idx)
        still = [active_rest[i] for i in um_t2] + [t for t in rest if t.lost > 0]

        # ReID de longo prazo: detecções altas sem par x tracks perdidos, só aparência + distância
        new_det = list(um_d)
        lost_pool = [t for t in still if t.emb is not None]
        if embs is not None and lost_pool and new_det:
            app = 1 - np.clip(np.array([t.emb for t in lost_pool]) @ e_hi[new_det].T, 0, 1)
            ctr_t = np.array([[(t.box[0] + t.box[2]) / 2, (t.box[1] + t.box[3]) / 2] for t in lost_pool])
            bd = boxes[hi][new_det]
            ctr_d = np.c_[(bd[:, 0] + bd[:, 2]) / 2, (bd[:, 1] + bd[:, 3]) / 2]
            hgt = np.array([max(t.box[3] - t.box[1], 1) for t in lost_pool])
            dist = np.linalg.norm(ctr_t[:, None] - ctr_d[None], axis=2) / (hgt[:, None] * 3)
            cost = np.where(dist < 1.0, app, 9.0)
            p3, _, _ = self._match(cost, cfg.reid_thresh)
            for i, j in p3:
                d = new_det[j]
                lost_pool[i].update(boxes[hi][d], scores[hi][d], e_hi[d], self.frame_idx)
            used = {new_det[j] for _, j in p3}
            new_det = [d for d in new_det if d not in used]

        # novos tracks
        for d in new_det:
            if scores[hi][d] >= cfg.new_track_thresh:
                t = Track(self._next_id, boxes[hi][d], scores[hi][d], _Kalman(boxes[hi][d]),
                          e_hi[d] if e_hi is not None else None)
                t.history.append((self.frame_idx, boxes[hi][d].copy()))
                self._next_id += 1
                self.tracks.append(t)

        # atualiza estados
        matched_now = {id(t) for t in self.tracks + self.lost if t.history and t.history[-1][0] == self.frame_idx}
        everything = self.tracks + [t for t in self.lost if t not in self.tracks]
        self.tracks, self.lost = [], []
        for t in everything:
            if id(t) in matched_now:
                self.tracks.append(t)
            else:
                t.lost += 1
                (self.lost if t.lost <= cfg.max_lost else self.removed).append(t)
        return [t for t in self.tracks if t.hits >= cfg.min_hits]

    def all_tracks(self) -> list[Track]:
        return self.tracks + self.lost + self.removed
