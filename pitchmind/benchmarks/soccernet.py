"""Benchmarks SoccerNet.

1) Jersey Number Recognition (SoccerNet 2023/2024)
   Layout oficial (após `SoccerNetDownloader(...).downloadDataTask(task="jersey-2023", ...)`):

       <root>/<split>/images/<tracklet_id>/<frame>.jpg
       <root>/<split>/<split>_gt.json          {"<tracklet_id>": número, ...}   (-1 = ilegível/ausente)

   Métrica oficial: acurácia por TRACKLET (inclui acertar o -1).
   Aqui comparamos, com o MESMO leitor (PARSeq/OCR/qualquer `JerseyReader`):
     - voto por maioria das leituras por quadro;
     - fusão bayesiana do PitchMind (escalação aberta 1–99, abstenção -> -1).

2) Game State Reconstruction (SoccerNet GSR 2024)
   A métrica oficial (GS-HOTA) é calculada pelo toolkit `sn-gamestate`/TrackEval.
   `export_gsr_predictions` grava as predições do PitchMind no formato JSON de
   entrada desse toolkit, para que a avaliação seja feita pelo código oficial,
   e não por uma reimplementação nossa.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

from pitchmind.identity.jersey import JerseyFuser, UNKNOWN, majority_vote
from pitchmind.identity.ocr import JerseyReader, crop_quality, normalize_read
from pitchmind.identity.jersey import Observation

OPEN_ROSTER = [str(i) for i in range(1, 100)]


def _to_int(n: str) -> int:
    return -1 if n in (UNKNOWN, "") else int(n)


def load_jersey_split(root: str | Path, split: str = "test") -> tuple[dict[str, list[Path]], dict[str, int]]:
    root = Path(root) / split
    gt = {str(k): int(v) for k, v in json.loads((root / f"{split}_gt.json").read_text()).items()}
    imgs = {d.name: sorted(p for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
            for d in sorted((root / "images").iterdir()) if d.is_dir()}
    return imgs, gt


def run_jersey_benchmark(root: str | Path, reader: JerseyReader, split: str = "test",
                         limit: Optional[int] = None, max_reads: int = 40,
                         fuser: Optional[JerseyFuser] = None) -> dict[str, float]:
    """Avalia por tracklet. As imagens do SoccerNet já são recortes do jogador inteiro."""
    imgs, gt = load_jersey_split(root, split)
    fuser = fuser or JerseyFuser({"A": OPEN_ROSTER}, accept=0.80, margin=0.30)
    ids = [t for t in imgs if t in gt][:limit]
    hit_vote = hit_fus = answered = correct_answered = 0
    for tid in ids:
        crops = []
        for p in imgs[tid]:
            im = cv2.imread(str(p))
            if im is not None:
                crops.append((crop_quality(im, min_h=16, good_h=60), im))
        crops.sort(key=lambda c: -c[0])
        crops = crops[:max_reads]
        reads = reader.read([c for _, c in crops]) if crops else []
        obs = [Observation(t=float(i), read=normalize_read(txt), conf=conf, quality=q)
               for i, ((q, _), (txt, conf)) in enumerate(zip(crops, reads))]
        truth = gt[tid]
        vote = _to_int(majority_vote(obs))
        fused = _to_int(fuser.identify(0, "A", obs).number)
        hit_vote += vote == truth
        hit_fus += fused == truth
        if fused != -1:
            answered += 1
            correct_answered += fused == truth
    n = max(len(ids), 1)
    return {"tracklets": float(len(ids)), "accuracy_majority_vote": hit_vote / n,
            "accuracy_pitchmind_fusion": hit_fus / n,
            "precision_when_answered": correct_answered / max(answered, 1),
            "coverage": answered / n}


def make_toy_jersey_dataset(root: str | Path, n_tracklets: int = 12, seed: int = 0) -> Path:
    """Gera um mini-dataset no MESMO layout do SoccerNet (para testes e CI)."""
    rng = np.random.default_rng(seed)
    root = Path(root) / "test"
    gt = {}
    for t in range(n_tracklets):
        num = int(rng.choice([-1, 7, 8, 10, 18, 23]))
        gt[str(t)] = num
        d = root / "images" / str(t)
        d.mkdir(parents=True, exist_ok=True)
        for k in range(int(rng.integers(5, 12))):
            im = np.full((80, 40, 3), (40, 40, 200), np.uint8)
            if num != -1:
                cv2.putText(im, str(num), (4, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            # nome do arquivo carrega o número verdadeiro (só para o leitor falso dos testes)
            cv2.imwrite(str(d / f"{k:03d}_{num}.png"), im)
    (root / "test_gt.json").write_text(json.dumps(gt))
    return root.parent


# ------------------------------------------------------------------ GSR
def export_gsr_predictions(out_path: str | Path, video_id: str,
                           detections: Iterable[dict]) -> Path:
    """Grava predições no formato de submissão do SoccerNet GSR (sn-gamestate, ChallengeRules.md).

    Um JSON por vídeo (ex.: SNGS-116.json) com a chave única "predictions". Cada item de
    `detections`:
        {"image_id": str, "track_id": int, "bbox_image": [x, y, w, h], "confidence": float,
         "bbox_pitch": {"x_bottom_middle": X, "y_bottom_middle": Y},   # metros, origem no centro
         "role": "player"|"goalkeeper"|"referee"|"ball"|"other",
         "team": "left"|"right"|None, "jersey": "18"|None}
    """
    preds = []
    for i, d in enumerate(detections):
        preds.append({
            "id": str(i), "image_id": str(d["image_id"]), "video_id": str(video_id),
            "track_id": int(d["track_id"]), "category_id": 1.0, "supercategory": "object",
            "confidence": float(d.get("confidence", 1.0)),
            "bbox_image": dict(zip(("x", "y", "w", "h"), map(float, d["bbox_image"]))),
            "bbox_pitch": d.get("bbox_pitch"),
            "attributes": {"role": d.get("role", "player"), "jersey": d.get("jersey"), "team": d.get("team")},
        })
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"predictions": preds}))
    return out


def pitch_to_gsr(xy_m: np.ndarray) -> dict[str, float]:
    """Converte (x, y) do PitchMind (origem no canto) para o GSR (origem no centro do campo)."""
    from pitchmind.models import PITCH_LENGTH, PITCH_WIDTH
    return {"x_bottom_middle": float(xy_m[0] - PITCH_LENGTH / 2),
            "y_bottom_middle": float(xy_m[1] - PITCH_WIDTH / 2)}
