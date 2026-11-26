import json

import cv2
import numpy as np

from pitchmind.benchmarks.soccernet import (export_gsr_predictions, make_toy_jersey_dataset, pitch_to_gsr,
                                            run_jersey_benchmark)
from pitchmind.calibration.keypoints import (Calibration, HomographyTracker, TEMPLATE, estimate_homography,
                                             synthetic_camera)
from pitchmind.extract.simulator import simulate_match
from pitchmind.identity.jersey import JerseyFuser
from pitchmind.identity.ocr import crop_quality, normalize_read, observations_from_track
from pitchmind.tracking.sim import evaluate_tracker, simulate_tracking
from pitchmind.tracking.tracker import ByteTracker, TrackerConfig, iou_matrix
from pitchmind.transform.features import aggregate_possession_windows, aggregate_windows, phase_purity
from pitchmind.pipeline import run_pipeline


# ---------------- tracking
def test_iou_matrix():
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], float)
    np.testing.assert_allclose(iou_matrix(a, b), [[1.0, 1 / 3, 0.0]], atol=1e-6)


def test_reid_beats_iou_only():
    seq = simulate_tracking(seed=0)
    full = evaluate_tracker(ByteTracker(), seq)
    iou_only = evaluate_tracker(ByteTracker(embed_fn=None), seq)
    assert full["id_purity"] > 0.95 and full["id_purity"] > iou_only["id_purity"] + 0.1
    assert full["n_ids"] <= 16  # 14 jogadores reais


def test_tracks_feed_jersey_fuser():
    seq = simulate_tracking(seed=1, n_frames=60)
    tr = ByteTracker()
    for s in seq:
        tr.update(s["boxes"], s["scores"], s["frame"])
    track = max(tr.all_tracks(), key=lambda t: t.hits)
    frames = {i: s["frame"] for i, s in enumerate(seq)}

    class FakeReader:  # 1 em cada 3 leituras perde o "1" (18 -> 8)
        def read(self, crops):
            return [("8", 0.5) if i % 3 == 0 else ("18", 0.9) for i in range(len(crops))]

    # recortes da cena sintética são pequenos; qualidade fixa isola a lógica de integração
    obs = observations_from_track(frames, track.history, FakeReader(), fps=25, max_reads=20,
                                  quality_fn=lambda c: 0.7)
    assert len(obs) == 20 and all(o.crop_id.startswith("f") for o in obs)
    assert JerseyFuser({"A": ["8", "18"]}).identify(track.track_id, "A", obs).number == "18"
    assert 0 <= crop_quality(frames[0][:40, :20]) <= 1


def test_normalize_read():
    assert normalize_read("#18") == "18" and normalize_read("108") == "" and normalize_read("07") == ""


# ---------------- calibração
def _project(H, pts):
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)


def test_homography_from_noisy_keypoints_with_outliers():
    rng = np.random.default_rng(3)
    H_true = synthetic_camera(3)
    img = _project(H_true, TEMPLATE)
    vis = np.where((img[:, 0] > 0) & (img[:, 0] < 1280) & (img[:, 1] > 0) & (img[:, 1] < 720))[0]
    ids = vis.copy()
    ids[:2] = (ids[:2] + 5) % len(TEMPLATE)  # 2 keypoints com id errado
    cal = estimate_homography(img[vis] + rng.normal(0, 1.5, (len(vis), 2)), ids)
    assert cal.ok and cal.reproj_m < 1.0
    probe = np.float32([[30, 20], [50, 40], [70, 50]])
    back = _project(cal.H, _project(H_true, probe))
    assert np.abs(back - probe).max() < 1.0


def test_rejects_collinear_keypoints():
    ids = np.array([13, 14, 15, 16])  # só a linha do meio-campo
    cal = estimate_homography(TEMPLATE[ids] * 10, ids)
    assert not cal.ok and "colinear" in cal.reason


def test_homography_tracker_detects_camera_cut():
    ht = HomographyTracker()
    H1 = np.linalg.inv(synthetic_camera(0))
    H2 = np.linalg.inv(synthetic_camera(9))
    ht.update(Calibration(H1, 10, 0.2, True))
    ht.update(Calibration(H1 * 1.0001, 10, 0.2, True))
    assert ht.cuts == 0
    ht.update(Calibration(H2, 10, 0.2, True))
    assert ht.cuts == 1
    H, reliable = ht.update(Calibration(None, 0, np.inf, False))
    assert H is not None and reliable  # mantém a última H por alguns quadros


# ---------------- janelas por posse
def test_possession_windows_are_purer():
    fr = simulate_match(300, seed=7)
    fixed, poss = aggregate_windows(fr, 10), aggregate_possession_windows(fr)
    assert phase_purity(poss, fr) > phase_purity(fixed, fr)
    durs = [w.t_end - w.t_start for w in poss]
    assert max(durs) <= 12.5


def test_pipeline_possession_mode():
    store, _, rep = run_pipeline(seed=7, duration_s=300, windowing="possession")
    assert rep.n_windows > 10 and "A_possession" in store.df().columns


# ---------------- SoccerNet
def test_soccernet_jersey_benchmark_on_toy_layout(tmp_path):
    root = make_toy_jersey_dataset(tmp_path, n_tracklets=10)

    class Reader:  # leitor que nunca consegue ler nada
        def read(self, crops):
            return [("", 0.0) for _ in crops]

    res = run_jersey_benchmark(root, Reader())
    assert res["tracklets"] == 10 and 0 <= res["accuracy_pitchmind_fusion"] <= 1
    # sem leitura alguma, a fusão abstém (-1) e acerta exatamente os tracklets ilegíveis
    gt = json.loads((root / "test" / "test_gt.json").read_text())
    assert abs(res["accuracy_pitchmind_fusion"] - sum(v == -1 for v in gt.values()) / 10) < 1e-9


def test_gsr_export_format(tmp_path):
    out = export_gsr_predictions(tmp_path / "SNGS-116.json", "116", [{
        "image_id": "3116000001", "track_id": 1, "bbox_image": [477, 565, 51, 140],
        "bbox_pitch": pitch_to_gsr(np.array([82.7, 36.0])), "role": "player", "team": "left", "jersey": "33"}])
    d = json.loads(out.read_text())
    p = d["predictions"][0]
    assert list(d) == ["predictions"] and p["category_id"] == 1.0 and p["supercategory"] == "object"
    assert set(p["attributes"]) == {"role", "jersey", "team"}
    assert abs(p["bbox_pitch"]["x_bottom_middle"] - 30.2) < 1e-6
