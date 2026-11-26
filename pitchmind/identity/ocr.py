"""Leitura real de números de camisa -> Observations para o `JerseyFuser`.

- `crop_quality`: qualidade do recorte (tamanho, nitidez via variância do Laplaciano,
  contraste e proporção), usada para PONDERAR cada leitura na fusão;
- `JerseyReader`: protocolo de qualquer leitor (PARSeq, OCR genérico, VLM);
- `ParseqReader`: PARSeq (scene-text recognition) via torch.hub, opcional;
- `observations_from_track`: transforma as caixas de um track em observações.
"""
from __future__ import annotations

import re
from typing import Optional, Protocol

import cv2
import numpy as np

from pitchmind.identity.jersey import Observation


class JerseyReader(Protocol):
    def read(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        """Para cada recorte BGR retorna (texto lido, confiança). Texto vazio = ilegível."""


def torso_crop(frame: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Recorte do tronco/costas onde fica o número (≈ 15%–60% da altura da caixa)."""
    x1, y1, x2, y2 = box.astype(int)
    h = y2 - y1
    H, W = frame.shape[:2]
    y_a, y_b = max(0, y1 + int(0.15 * h)), min(H, y1 + int(0.60 * h))
    return frame[y_a:y_b, max(0, x1):min(W, x2)]


def crop_quality(crop: np.ndarray, min_h: int = 24, good_h: int = 80) -> float:
    """Qualidade ∈ [0,1]: tamanho × nitidez × contraste × proporção plausível."""
    if crop is None or crop.size == 0:
        return 0.0
    h, w = crop.shape[:2]
    size = np.clip((h - min_h) / (good_h - min_h), 0, 1)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    sharp = np.clip(cv2.Laplacian(gray, cv2.CV_64F).var() / 300.0, 0, 1)
    contrast = np.clip(gray.std() / 50.0, 0, 1)
    aspect = 1.0 if 0.4 <= w / max(h, 1) <= 1.6 else 0.5  # muito largo/estreito = de lado/ocluído
    return float(size * (0.5 + 0.5 * sharp) * (0.6 + 0.4 * contrast) * aspect)


def normalize_read(text: str) -> str:
    """Mantém só dígitos; números de camisa têm 1–2 dígitos (sem zero à esquerda)."""
    d = re.sub(r"\D", "", text or "")
    if not d or len(d) > 2 or (len(d) == 2 and d[0] == "0"):
        return ""
    return d


class ParseqReader:
    """PARSeq (Bautista & Atienza, ECCV 2022) via torch.hub. Requer `pip install torch torchvision`."""

    def __init__(self, device: str = "cpu"):
        try:
            import torch
            from torchvision import transforms as T
        except ImportError as e:  # pragma: no cover
            raise SystemExit("ParseqReader requer: pip install torch torchvision") from e
        self.torch = torch
        self.model = torch.hub.load("baudm/parseq", "parseq", pretrained=True).eval().to(device)
        self.device = device
        self.tf = T.Compose([T.ToPILImage(), T.Resize((32, 128)), T.ToTensor(), T.Normalize(0.5, 0.5)])

    def read(self, crops):  # pragma: no cover - requer pesos baixados
        if not crops:
            return []
        x = self.torch.stack([self.tf(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops]).to(self.device)
        with self.torch.no_grad():
            probs = self.model(x).softmax(-1)
        texts, confs = self.model.tokenizer.decode(probs)
        return [(normalize_read(t), float(c.prod()) if len(c) else 0.0) for t, c in zip(texts, confs)]


def observations_from_track(frames: dict[int, np.ndarray], history: list[tuple[int, np.ndarray]],
                            reader: JerseyReader, fps: float = 25.0, min_quality: float = 0.05,
                            max_reads: int = 60, quality_fn=crop_quality) -> list[Observation]:
    """Converte o histórico de um track em observações, lendo só os recortes mais úteis.

    Seleciona até `max_reads` quadros de maior qualidade (o custo de OCR fica limitado
    mesmo em tracks longos) e devolve leituras com conf e qualidade para a fusão.
    """
    cands = []
    for fi, box in history:
        if fi not in frames:
            continue
        crop = torso_crop(frames[fi], box)
        q = quality_fn(crop)
        if q >= min_quality:
            cands.append((q, fi, crop))
    cands.sort(key=lambda c: -c[0])
    cands = cands[:max_reads]
    reads = reader.read([c for _, _, c in cands]) if cands else []
    return [Observation(t=fi / fps, read=normalize_read(txt), conf=conf, quality=q, crop_id=f"f{fi}")
            for (q, fi, _), (txt, conf) in zip(cands, reads)]
