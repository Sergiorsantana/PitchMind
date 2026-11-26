"""Orquestração ETL: Extract (tracking) -> Transform (métricas + eventos + legendas) -> Load (DuckDB)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from pitchmind.extract.simulator import simulate_match
from pitchmind.load.store import MatchStore
from pitchmind.models import FrameState, Window
from pitchmind.transform.events import describe, detect_events
from pitchmind.transform.features import aggregate_windows
from pitchmind.vlm.backends import OfflineBackend


@dataclass
class PipelineReport:
    n_frames: int = 0
    n_windows: int = 0
    timings: dict[str, float] = field(default_factory=dict)
    vlm_calls: int = 0


def run_pipeline(frames: list[FrameState] | None = None, window_s: float = 10.0, backend=None,
                 db_path: str | Path = ":memory:", caption_every: int = 1,
                 seed: int = 7, duration_s: float = 600) -> tuple[MatchStore, list[FrameState], PipelineReport]:
    rep = PipelineReport()
    backend = backend or OfflineBackend()

    t0 = time.perf_counter()
    frames = frames if frames is not None else simulate_match(duration_s=duration_s, seed=seed)
    rep.timings["extract"] = time.perf_counter() - t0
    rep.n_frames = len(frames)

    t0 = time.perf_counter()
    windows: list[Window] = aggregate_windows(frames, window_s)
    by_t = {round(f.t, 3): f for f in frames}
    for w in windows:
        w.events = detect_events(w)
        numeric = describe(w)
        img = by_t.get(round(w.keyframe_t or -1, 3))
        img = img.image if img is not None else None
        if backend.name != "offline" and w.window_id % caption_every == 0:
            try:
                w.caption = backend.caption(img, numeric)
                rep.vlm_calls += 1
            except Exception:  # pragma: no cover
                w.caption = numeric
        else:
            w.caption = numeric
    rep.timings["transform"] = time.perf_counter() - t0
    rep.n_windows = len(windows)

    t0 = time.perf_counter()
    store = MatchStore(db_path)
    store.load(windows, {"window_s": window_s, "n_frames": len(frames), "backend": backend.name})
    rep.timings["load"] = time.perf_counter() - t0
    return store, frames, rep
