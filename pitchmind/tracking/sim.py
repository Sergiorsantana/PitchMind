"""Cenário sintético de rastreamento com ground truth, para avaliar o tracker.

Jogadores (retângulos com cor do time + leve variação individual) se movem numa
imagem de transmissão. Inclui: cruzamentos, oclusões (detecção some ou vem com
confiança baixa), jitter de caixa e falsos positivos.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import cv2
import numpy as np

TEAM_BGR = {"A": (200, 90, 40), "B": (40, 50, 210)}


def simulate_tracking(n_players: int = 14, n_frames: int = 300, size=(640, 360), seed: int = 0,
                      occl_rate: float = 0.08, render: bool = True):
    rng = np.random.default_rng(seed)
    W, H = size
    pos = np.c_[rng.uniform(60, W - 60, n_players), rng.uniform(60, H - 40, n_players)]
    vel = rng.normal(0, 2.2, (n_players, 2))
    teams = ["A" if i % 2 == 0 else "B" for i in range(n_players)]
    shade = rng.integers(-35, 35, (n_players, 3))
    occl_left = np.zeros(n_players, int)
    out = []
    for f in range(n_frames):
        vel += rng.normal(0, 0.35, vel.shape)
        vel = np.clip(vel, -4, 4)
        pos += vel
        for d, lim in ((0, (20, W - 20)), (1, (40, H - 10))):
            hit = (pos[:, d] < lim[0]) | (pos[:, d] > lim[1])
            vel[hit, d] *= -1
            pos[:, d] = np.clip(pos[:, d], *lim)
        h = 34 + pos[:, 1] / H * 20  # perspectiva: mais perto da câmera, maior
        w = h * 0.45
        gt = np.c_[pos[:, 0] - w / 2, pos[:, 1] - h, pos[:, 0] + w / 2, pos[:, 1]]

        img = None
        if render:
            img = np.full((H, W, 3), (60, 140, 60), np.uint8)
            for i in np.argsort(pos[:, 1]):  # desenha de trás para frente
                x1, y1, x2, y2 = gt[i].astype(int)
                bh, bw = y2 - y1, x2 - x1
                col = np.clip(np.array(TEAM_BGR[teams[i]]) + shade[i], 0, 255).tolist()
                cx = (x1 + x2) // 2
                cv2.circle(img, (cx, y1 + bh // 9), max(bh // 9, 2), (150, 180, 220), -1, cv2.LINE_AA)  # cabeça
                cv2.rectangle(img, (x1, y1 + bh // 5), (x2, y1 + int(bh * 0.55)), col, -1)        # camisa
                cv2.rectangle(img, (x1 + 1, y1 + int(bh * 0.55)), (x2 - 1, y1 + int(bh * 0.72)), (30, 30, 30), -1)
                leg = max(bw // 5, 2)
                swing = int(3 * np.sin(f * 0.5 + i))
                cv2.line(img, (cx - leg, y1 + int(bh * 0.72)), (cx - leg - swing, y2), (150, 180, 220), leg)
                cv2.line(img, (cx + leg, y1 + int(bh * 0.72)), (cx + leg + swing, y2), (150, 180, 220), leg)

        boxes, scores, ids = [], [], []
        start = rng.random(n_players) < occl_rate / 6
        occl_left[start & (occl_left == 0)] = rng.integers(3, 15, int((start & (occl_left == 0)).sum()))
        for i in range(n_players):
            jitter = rng.normal(0, 1.5, 4)
            if occl_left[i] > 0:
                occl_left[i] -= 1
                if rng.random() < 0.5:   # metade das oclusões: some
                    continue
                boxes.append(gt[i] + jitter)          # outra metade: baixa confiança
                scores.append(float(rng.uniform(0.15, 0.45)))
            else:
                boxes.append(gt[i] + jitter)
                scores.append(float(rng.uniform(0.6, 0.95)))
            ids.append(i)
        if rng.random() < 0.15:  # falso positivo
            x, y = rng.uniform(0, W - 20), rng.uniform(0, H - 40)
            boxes.append(np.array([x, y, x + 18, y + 40]))
            scores.append(float(rng.uniform(0.2, 0.55)))
            ids.append(-1)
        out.append(dict(frame=img, boxes=np.array(boxes), scores=np.array(scores), gt_ids=ids, gt_boxes=gt))
    return out


def evaluate_tracker(tracker, seq) -> dict[str, float]:
    """IDF1 simplificado (pureza) e nº de trocas de ID, casando por IoU com o GT."""
    from pitchmind.tracking.tracker import iou_matrix
    from scipy.optimize import linear_sum_assignment

    assign = defaultdict(list)  # gt_id -> [track_id por quadro em que foi casado]
    total_gt = 0
    for step in seq:
        active = tracker.update(step["boxes"], step["scores"], step["frame"])
        gt = step["gt_boxes"]
        total_gt += len(gt)
        if not active:
            continue
        iou = iou_matrix(gt, np.array([t.box for t in active]))
        r, c = linear_sum_assignment(-iou)
        for i, j in zip(r, c):
            if iou[i, j] >= 0.3:
                assign[i].append(active[j].track_id)
    switches = sum(sum(1 for a, b in zip(v, v[1:]) if a != b) for v in assign.values())
    matched = sum(len(v) for v in assign.values())
    pure = sum(Counter(v).most_common(1)[0][1] for v in assign.values() if v)
    return {"id_switches": float(switches), "coverage": matched / total_gt, "id_purity": pure / max(matched, 1),
            "n_ids": float(len({t for v in assign.values() for t in v}))}
