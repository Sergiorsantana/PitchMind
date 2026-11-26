"""Extract: vídeo real -> FrameState no plano do campo.

Pipeline: YOLO (ultralytics, opcional) -> separação de times por cor da
camisa (KMeans em HSV, sem GPU) -> homografia imagem->campo.

A homografia vem de 4+ pontos correspondentes (pixel -> metro) informados
em JSON, ex.: {"image": [[x,y],...], "pitch": [[X,Y],...]}.
Sem homografia, usa uma aproximação linear (câmera tática aberta).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np
from sklearn.cluster import KMeans

from pitchmind.models import FrameState, PITCH_LENGTH, PITCH_WIDTH


class Homography:
    def __init__(self, H: Optional[np.ndarray] = None, frame_size: tuple[int, int] | None = None):
        self.H = H
        self.frame_size = frame_size

    @classmethod
    def from_json(cls, path: str | Path) -> "Homography":
        d = json.loads(Path(path).read_text())
        H, _ = cv2.findHomography(np.float32(d["image"]), np.float32(d["pitch"]))
        return cls(H)

    def to_pitch(self, pts: np.ndarray) -> np.ndarray:
        if len(pts) == 0:
            return pts.reshape(0, 2)
        if self.H is not None:
            return cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float32), self.H).reshape(-1, 2)
        w, h = self.frame_size
        return np.c_[pts[:, 0] / w * PITCH_LENGTH, pts[:, 1] / h * PITCH_WIDTH]


def jersey_features(frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Média HSV do tronco (metade superior central da bbox)."""
    feats = []
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    for x1, y1, x2, y2 in boxes.astype(int):
        h, w = y2 - y1, x2 - x1
        crop = hsv[y1 + h // 5: y1 + h // 2, x1 + w // 4: x2 - w // 4]
        feats.append(crop.reshape(-1, 3).mean(axis=0) if crop.size else np.zeros(3))
    return np.array(feats)


class TeamClassifier:
    """Ajusta KMeans(k=2) nas cores das camisas dos primeiros frames."""

    def __init__(self):
        self.km: Optional[KMeans] = None

    def fit(self, feats: np.ndarray) -> None:
        self.km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(feats)

    def predict(self, feats: np.ndarray) -> np.ndarray:
        assert self.km is not None, "chame fit() primeiro"
        return self.km.predict(feats)


def read_video(path: str | Path, every_n: int = 5,
               model_name: str = "yolov8n.pt",
               homography: Optional[Homography] = None,
               keep_images: bool = True) -> Iterator[FrameState]:
    """Lê o vídeo e produz FrameStates. Requer `pip install ultralytics`."""
    try:
        from ultralytics import YOLO
    except ImportError as e:  # pragma: no cover - depende de extra opcional
        raise SystemExit("Modo vídeo requer: pip install 'pitchmind[video]'") from e

    model = YOLO(model_name)
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    homography = homography or Homography(frame_size=(w, h))
    teams = TeamClassifier()
    warmup: list[np.ndarray] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % every_n:
            idx += 1
            continue
        res = model(frame, verbose=False)[0]
        cls = res.boxes.cls.cpu().numpy().astype(int)
        xyxy = res.boxes.xyxy.cpu().numpy()
        person, ball = xyxy[cls == 0], xyxy[cls == 32]  # COCO: 0=person, 32=sports ball
        feats = jersey_features(frame, person)
        if teams.km is None:
            warmup.append(feats)
            if sum(len(f) for f in warmup) >= 60:
                teams.fit(np.vstack(warmup))
            idx += 1
            continue
        labels = teams.predict(feats) if len(feats) else np.array([], int)
        feet = np.c_[(person[:, 0] + person[:, 2]) / 2, person[:, 3]]  # ponto do pé
        pitch = homography.to_pitch(feet)
        ball_pt = None
        if len(ball):
            b = ball[0]
            ball_pt = homography.to_pitch(np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]]))[0]
        yield FrameState(t=idx / fps, players_a=pitch[labels == 0], players_b=pitch[labels == 1],
                         ball=ball_pt, image=frame if keep_images else None)
        idx += 1
    cap.release()
