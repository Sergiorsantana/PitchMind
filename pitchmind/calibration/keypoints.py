"""Calibração automática de câmera por keypoints do campo.

Em vez de 4 pontos clicados à mão, um detector de keypoints (ex.: YOLOv8-pose treinado
nos 32 pontos do campo, como no Roboflow Sports) dá pontos (pixel, id, confiança). Daqui:

1. `PITCH_KEYPOINTS`: modelo métrico do campo (FIFA 105×68) com 32 pontos nomeados;
2. `estimate_homography`: RANSAC + checagens de sanidade (nº de inliers, erro de
   reprojeção em METROS, homografia não degenerada);
3. `HomographyTracker`: suaviza H ao longo do tempo, detecta CORTE DE CÂMERA
   (salto grande) e mantém a última H boa quando um quadro não tem pontos suficientes
   (close, replay), sinalizando que as posições desse quadro não são confiáveis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from pitchmind.models import PITCH_LENGTH as L, PITCH_WIDTH as W

_C = W / 2
# 32 keypoints (mesma ordem/convensão do Roboflow Sports soccer pitch config), em metros
PITCH_KEYPOINTS: dict[str, tuple[float, float]] = {
    "corner_tl": (0, 0), "corner_bl": (0, W),
    "pen_l_top": (0, _C - 20.16), "goal_l_top": (0, _C - 9.16), "goal_l_bot": (0, _C + 9.16),
    "pen_l_bot": (0, _C + 20.16),
    "goal_l_in_top": (5.5, _C - 9.16), "goal_l_in_bot": (5.5, _C + 9.16),
    "spot_l": (11, _C),
    "pen_l_in_top": (16.5, _C - 20.16), "pen_l_arc_top": (16.5, _C - 7.31), "pen_l_arc_bot": (16.5, _C + 7.31),
    "pen_l_in_bot": (16.5, _C + 20.16),
    "half_top": (L / 2, 0), "circle_top": (L / 2, _C - 9.15), "circle_bot": (L / 2, _C + 9.15),
    "half_bot": (L / 2, W),
    "pen_r_in_top": (L - 16.5, _C - 20.16), "pen_r_arc_top": (L - 16.5, _C - 7.31),
    "pen_r_arc_bot": (L - 16.5, _C + 7.31), "pen_r_in_bot": (L - 16.5, _C + 20.16),
    "spot_r": (L - 11, _C),
    "goal_r_in_top": (L - 5.5, _C - 9.16), "goal_r_in_bot": (L - 5.5, _C + 9.16),
    "corner_tr": (L, 0), "pen_r_top": (L, _C - 20.16), "goal_r_top": (L, _C - 9.16),
    "goal_r_bot": (L, _C + 9.16), "pen_r_bot": (L, _C + 20.16), "corner_br": (L, W),
    "circle_left": (L / 2 - 9.15, _C), "circle_right": (L / 2 + 9.15, _C),
}
KEYPOINT_NAMES = list(PITCH_KEYPOINTS)
TEMPLATE = np.array(list(PITCH_KEYPOINTS.values()), np.float32)


@dataclass
class Calibration:
    H: Optional[np.ndarray]          # imagem -> campo (metros)
    inliers: int
    reproj_m: float                  # erro médio de reprojeção dos inliers, em metros
    ok: bool
    reason: str = ""


def _project(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)


def estimate_homography(image_pts: np.ndarray, kp_ids: np.ndarray, conf: Optional[np.ndarray] = None,
                        min_conf: float = 0.5, min_inliers: int = 6, ransac_m: float = 1.5,
                        max_reproj_m: float = 1.0) -> Calibration:
    """image_pts (n,2) em pixels; kp_ids (n,) índices em KEYPOINT_NAMES; conf (n,) opcional."""
    image_pts = np.asarray(image_pts, np.float32).reshape(-1, 2)
    kp_ids = np.asarray(kp_ids, int).reshape(-1)
    if conf is not None:
        keep = np.asarray(conf) >= min_conf
        image_pts, kp_ids = image_pts[keep], kp_ids[keep]
    if len(image_pts) < 4:
        return Calibration(None, len(image_pts), float("inf"), False, "menos de 4 keypoints")
    pitch = TEMPLATE[kp_ids]
    # pontos colineares (ex.: só a linha do meio-campo) não determinam a homografia
    sv = np.linalg.svd(pitch - pitch.mean(axis=0), compute_uv=False)
    if sv[-1] / np.sqrt(len(pitch)) < 2.0:  # dispersão < 2 m fora da reta principal
        return Calibration(None, len(pitch), float("inf"), False, "keypoints colineares")
    H, mask = cv2.findHomography(image_pts, pitch, cv2.RANSAC, ransac_m)
    if H is None:
        return Calibration(None, 0, float("inf"), False, "RANSAC falhou")
    inl = mask.ravel().astype(bool)
    err = float(np.linalg.norm(_project(H, image_pts[inl]) - pitch[inl], axis=1).mean()) if inl.any() else np.inf
    det = np.linalg.det(H[:2, :2])
    if inl.sum() < min_inliers:
        return Calibration(H, int(inl.sum()), err, False, f"poucos inliers ({inl.sum()})")
    if not np.isfinite(det) or abs(det) < 1e-9:
        return Calibration(H, int(inl.sum()), err, False, "homografia degenerada")
    if err > max_reproj_m:
        return Calibration(H, int(inl.sum()), err, False, f"erro de reprojeção alto ({err:.2f} m)")
    return Calibration(H, int(inl.sum()), err, True)


class HomographyTracker:
    """Suavização temporal de H + detecção de corte de câmera."""

    def __init__(self, alpha: float = 0.6, cut_m: float = 6.0, max_stale: int = 25,
                 probe: Optional[np.ndarray] = None, frame_size: tuple[int, int] = (1280, 720)):
        self.alpha, self.cut_m, self.max_stale = alpha, cut_m, max_stale
        w, h = frame_size
        self.probe = probe if probe is not None else np.float32(
            [[w * x, h * y] for x in (0.2, 0.5, 0.8) for y in (0.4, 0.7, 0.95)])
        self.H: Optional[np.ndarray] = None
        self.stale = 0
        self.cuts = 0

    def update(self, cal: Calibration) -> tuple[Optional[np.ndarray], bool]:
        """Retorna (H a usar, confiável?)."""
        if not cal.ok:
            self.stale += 1
            return self.H, self.H is not None and self.stale <= self.max_stale
        if self.H is None:
            self.H, self.stale = cal.H / cal.H[2, 2], 0
            return self.H, True
        a, b = _project(self.H, self.probe), _project(cal.H, self.probe)
        jump = float(np.linalg.norm(a - b, axis=1).mean())
        Hn = cal.H / cal.H[2, 2]
        if jump > self.cut_m:  # corte de câmera: não mistura com a H anterior
            self.cuts += 1
            self.H = Hn
        else:  # mesmo plano: média exponencial (reduz tremor dos keypoints)
            self.H = self.alpha * Hn + (1 - self.alpha) * self.H
            self.H /= self.H[2, 2]
        self.stale = 0
        return self.H, True


class KeypointDetector:  # pragma: no cover - requer pesos treinados
    """Adaptador para um modelo YOLOv8-pose treinado nos 32 keypoints do campo.

    Ex.: o modelo `football-field-detection` do Roboflow Sports (exportado para .pt).
    """

    def __init__(self, weights: str, min_conf: float = 0.5):
        from ultralytics import YOLO
        self.model, self.min_conf = YOLO(weights), min_conf

    def __call__(self, frame: np.ndarray) -> Calibration:
        r = self.model(frame, verbose=False)[0]
        if r.keypoints is None or len(r.keypoints) == 0:
            return Calibration(None, 0, float("inf"), False, "sem campo visível")
        xy = r.keypoints.xy[0].cpu().numpy()
        cf = r.keypoints.conf[0].cpu().numpy() if r.keypoints.conf is not None else np.ones(len(xy))
        ids = np.arange(len(xy))
        vis = (xy[:, 0] > 1) & (xy[:, 1] > 1)
        return estimate_homography(xy[vis], ids[vis], cf[vis], self.min_conf)


def synthetic_camera(seed: int = 0, frame_size=(1280, 720)) -> np.ndarray:
    """H campo->imagem plausível de transmissão (câmera alta na lateral), para testes."""
    rng = np.random.default_rng(seed)
    w, h = frame_size
    cx = rng.uniform(25, 80)       # para onde a câmera aponta (x do campo)
    span = rng.uniform(45, 70)     # zoom: metros visíveis na largura
    src = np.float32([[cx - span / 2, 0], [cx + span / 2, 0], [cx + span / 1.4, W], [cx - span / 1.4, W]])
    dst = np.float32([[w * 0.12, h * 0.18], [w * 0.88, h * 0.18], [w * 1.05, h * 1.0], [w * -0.05, h * 1.0]])
    return cv2.getPerspectiveTransform(src, dst)
