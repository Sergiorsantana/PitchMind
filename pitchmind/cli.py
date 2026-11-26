"""CLI do PitchMind.

    pitchmind demo                          # pipeline simulado + perguntas de exemplo
    pitchmind ask "pressão alta do time A"  # uma pergunta
    pitchmind eval                          # precision@k do RAG + acurácia da fusão de camisas
    pitchmind video jogo.mp4 --homography h.json   # requer extra [video]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from pitchmind.identity.jersey import JerseyFuser, UNKNOWN, majority_vote, simulate_reads
from pitchmind.pipeline import run_pipeline
from pitchmind.rag.engine import AskEngine, evaluate
from pitchmind.vlm.backends import get_backend

EXAMPLES = [
    "Quando o time A fez pressão alta? top 3",
    "Momentos em que o time B ficou mais compacto",
    "Qual a posse média do time A nos primeiros 5 minutos?",
    "Compare a altura do bloco dos dois times",
    "Janelas com controle de campo do time A acima de 55",
    "Bola mais rápida que o normal",
]


def _print_answer(a, show_plan: bool = False):
    print(f"\n❓ {a.question}")
    if show_plan:
        print("   plano:", json.dumps({k: v for k, v in a.plan.items() if v}, ensure_ascii=False, default=str))
    print("  ", a.text.replace("\n", "\n   "))
    v = a.verification
    print(f"   {'✔' if v['ok'] else '✘'} verificação: {v['numbers_found']} números, lastro {v['grounded_ratio']:.0%}"
          + ("" if v["ok"] else f" — sem lastro: {v['unsupported']}"))


def jersey_benchmark(n_tracks: int = 400, seed: int = 0) -> dict[str, float]:
    """Compara voto por maioria (por frame) com a fusão bayesiana (+ VLM simulado nas ambiguidades)."""
    rng = np.random.default_rng(seed)
    roster = {"A": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "14", "16", "18", "19", "21", "23"]}
    stats = {k: 0 for k in ("vote_ok", "vote_err", "fus_ok", "fus_err", "vlm_ok", "vlm_err", "vlm_calls")}
    for i in range(n_tracks):
        true = roster["A"][rng.integers(len(roster["A"]))]
        obs = simulate_reads(true, n=int(rng.integers(6, 40)), rng=rng, hard=rng.random() < 0.5)

        def vlm(best, cands, _t=true):  # VLM simulado (90% de acerto entre os candidatos)
            stats["vlm_calls"] += 1
            if _t in cands and rng.random() < 0.9:
                return _t, 0.9
            return cands[rng.integers(len(cands))], 0.6

        v = majority_vote(obs)
        f = JerseyFuser(roster).identify(i, "A", obs)
        g = JerseyFuser(roster, vlm=vlm).identify(i, "A", obs)
        stats["vote_ok"] += v == true
        stats["vote_err"] += v not in (true, UNKNOWN)
        stats["fus_ok"] += f.number == true
        stats["fus_err"] += f.number not in (true, UNKNOWN)
        stats["vlm_ok"] += g.number == true
        stats["vlm_err"] += g.number not in (true, UNKNOWN)
    n = n_tracks
    return {
        "voto_maioria: acerto": stats["vote_ok"] / n,
        "voto_maioria: ERRO (id errada)": stats["vote_err"] / n,
        "fusao: acerto": stats["fus_ok"] / n,
        "fusao: ERRO (id errada)": stats["fus_err"] / n,
        "fusao: abstencao": 1 - (stats["fus_ok"] + stats["fus_err"]) / n,
        "fusao+VLM: acerto": stats["vlm_ok"] / n,
        "fusao+VLM: ERRO (id errada)": stats["vlm_err"] / n,
        "fusao+VLM: % tracks que chamaram VLM": stats["vlm_calls"] / n,
    }


def tracking_benchmark(seeds=(0, 1, 2)) -> dict[str, str]:
    from pitchmind.tracking.sim import evaluate_tracker, simulate_tracking
    from pitchmind.tracking.tracker import ByteTracker, TrackerConfig
    variants = {
        "ingenuo (só alta conf., sem memória)": lambda: ByteTracker(TrackerConfig(low_thresh=0.5, max_lost=1),
                                                                    embed_fn=None),
        "ByteTrack (2 estágios, só IoU)": lambda: ByteTracker(embed_fn=None),
        "ByteTrack + ReID (PitchMind)": lambda: ByteTracker(),
    }
    out = {}
    for name, make in variants.items():
        rs = [evaluate_tracker(make(), simulate_tracking(seed=s)) for s in seeds]
        out[name] = (f"pureza de ID {np.mean([r['id_purity'] for r in rs]):.3f} · "
                     f"IDs criados {np.mean([r['n_ids'] for r in rs]):.0f} (real: 14) · "
                     f"trocas {np.mean([r['id_switches'] for r in rs]):.0f}")
    return out


def calibration_benchmark(n: int = 50) -> dict[str, str]:
    import cv2
    from pitchmind.calibration.keypoints import TEMPLATE, estimate_homography, synthetic_camera
    errs, ok, rejected = [], 0, 0
    for s in range(n):
        rng = np.random.default_rng(s)
        H = synthetic_camera(s)
        img = cv2.perspectiveTransform(TEMPLATE.reshape(-1, 1, 2), H).reshape(-1, 2)
        vis = np.where((img[:, 0] > 0) & (img[:, 0] < 1280) & (img[:, 1] > 0) & (img[:, 1] < 720))[0]
        pts, ids = img[vis] + rng.normal(0, 2, (len(vis), 2)), vis.copy()
        k = max(1, len(ids) // 5)
        bad = rng.choice(len(ids), k, replace=False)
        ids[bad] = rng.integers(0, len(TEMPLATE), k)
        cal = estimate_homography(pts, ids)
        if not cal.ok:
            rejected += 1
            continue
        ok += 1
        g = np.c_[rng.uniform(0, 105, 200), rng.uniform(0, 68, 200)].astype(np.float32)
        gi = cv2.perspectiveTransform(g.reshape(-1, 1, 2), H).reshape(-1, 2)
        m = (gi[:, 0] > 0) & (gi[:, 0] < 1280) & (gi[:, 1] > 0) & (gi[:, 1] < 720)
        back = cv2.perspectiveTransform(gi[m].reshape(-1, 1, 2), cal.H).reshape(-1, 2)
        errs.append(float(np.linalg.norm(back - g[m], axis=1).mean()))
    return {"calibradas": f"{ok}/{n} (recusadas por evidência insuficiente: {rejected})",
            "erro médio no campo": f"{np.mean(errs):.2f} m (pior caso {np.max(errs):.2f} m)"}


def windowing_benchmark(seeds=(7, 1, 2)) -> dict[str, str]:
    from pitchmind.extract.simulator import simulate_match
    from pitchmind.transform.features import aggregate_possession_windows, aggregate_windows, phase_purity
    out = {}
    for name, fn in (("fixas 10 s", lambda f: aggregate_windows(f, 10)),
                     ("adaptativas por posse", aggregate_possession_windows)):
        pur, n = [], []
        for s in seeds:
            fr = simulate_match(600, seed=s)
            w = fn(fr)
            pur.append(phase_purity(w, fr))
            n.append(len(w))
        out[name] = f"pureza de fase {np.mean(pur):.3f} · {np.mean(n):.0f} janelas"
    return out


def main(argv=None):
    p = argparse.ArgumentParser(prog="pitchmind")
    p.add_argument("--backend", default="auto", choices=["auto", "offline", "claude"])
    p.add_argument("--seed", type=int, default=7)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo")
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("--narrate", action="store_true")
    sub.add_parser("eval")
    an = sub.add_parser("animate", help="gera GIF do radar ao vivo")
    an.add_argument("--start", type=float, default=240.0)
    an.add_argument("--seconds", type=float, default=80.0)
    an.add_argument("--out", default="docs/img/live.gif")
    dm = sub.add_parser("demos", help="gera os vídeos de demonstração (MP4 + GIF) de cada modo")
    dm.add_argument("--out", default="docs/media")
    v = sub.add_parser("video")
    v.add_argument("path")
    v.add_argument("--homography")
    v.add_argument("--question", default="momentos de pressão alta")
    v.add_argument("--field-model", help="pesos YOLOv8-pose dos keypoints do campo (calibração automática)")
    bj = sub.add_parser("bench-jersey", help="SoccerNet Jersey Number Recognition")
    bj.add_argument("root", help="pasta do dataset jersey-2023 do SoccerNet")
    bj.add_argument("--split", default="test")
    bj.add_argument("--limit", type=int)
    p.add_argument("--windowing", default="fixed", choices=["fixed", "possession"],
                   help="janelas fixas de 10 s ou adaptativas por posse")
    args = p.parse_args(argv)

    if args.cmd == "demos":
        from pitchmind.render.demos import make_all
        for path in make_all(args.out):
            print(f"  {path}  ({path.stat().st_size / 1e6:.1f} MB)")
        return

    backend = get_backend(args.backend)
    if args.cmd == "video":
        from pitchmind.extract.video import Homography, read_video
        H = Homography.from_json(args.homography) if args.homography else None
        frames = list(read_video(args.path, homography=H, field_model=args.field_model))
        store, _, rep = run_pipeline(frames, backend=backend, windowing=args.windowing)
    elif args.cmd == "bench-jersey":
        from pitchmind.benchmarks.soccernet import run_jersey_benchmark
        from pitchmind.identity.ocr import ParseqReader
        res = run_jersey_benchmark(args.root, ParseqReader(), args.split, args.limit)
        for k, val in res.items():
            print(f"  {k:28s} {val:.3f}")
        return
    else:
        store, frames, rep = run_pipeline(backend=backend, seed=args.seed, windowing=args.windowing)
    print(f"ETL: {rep.n_frames} frames -> {rep.n_windows} janelas | "
          + " ".join(f"{k}={v * 1000:.0f}ms" for k, v in rep.timings.items()) + f" | backend={backend.name}")

    if args.cmd == "animate":
        from pitchmind.render.animate import animate
        nums = {"A": ["1", "2", "4", "3", "6", "5", "8", "10", "7", "9", "11"],
                "B": ["1", "2", "4", "3", "6", "5", "8", "10", "7", "9", "11"]}
        path = animate(frames, store.windows(), args.out, args.start, args.seconds, numbers=nums)
        print(f"GIF salvo em {path} ({path.stat().st_size / 1e6:.1f} MB)")
        return

    eng = AskEngine(store, backend, use_llm=backend.name != "offline")
    if args.cmd == "demo":
        for q in EXAMPLES:
            _print_answer(eng.ask(q), show_plan=True)
    elif args.cmd == "ask":
        _print_answer(eng.ask(args.question, narrate=args.narrate), show_plan=True)
    elif args.cmd == "video":
        _print_answer(eng.ask(args.question), show_plan=True)
    elif args.cmd == "eval":
        print("\nRAG quantitativo — precision@5 por fase (ground truth do simulador):")
        for k, val in evaluate(store, AskEngine(store)).items():
            print(f"  {k:14s} {val:.2f}")
        print("\nRastreamento (ByteTrack) — 3 cenas sintéticas com oclusões e cruzamentos:")
        for k, val in tracking_benchmark().items():
            print(f"  {k:40s} {val}")
        print("\nCalibração automática por keypoints — 50 câmeras sintéticas, 20% de keypoints errados:")
        for k, val in calibration_benchmark().items():
            print(f"  {k:40s} {val}")
        print("\nJanelas fixas (10 s) vs adaptativas por posse — 3 partidas simuladas:")
        for k, val in windowing_benchmark().items():
            print(f"  {k:40s} {val}")
        print("\nFusão temporal de camisas vs voto por maioria (400 trajetórias simuladas):")
        for k, val in jersey_benchmark().items():
            print(f"  {k:40s} {val:6.1%}")


if __name__ == "__main__":
    main()
