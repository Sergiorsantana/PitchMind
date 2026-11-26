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
    v = sub.add_parser("video")
    v.add_argument("path")
    v.add_argument("--homography")
    v.add_argument("--question", default="momentos de pressão alta")
    args = p.parse_args(argv)

    backend = get_backend(args.backend)
    if args.cmd == "video":
        from pitchmind.extract.video import Homography, read_video
        H = Homography.from_json(args.homography) if args.homography else None
        frames = list(read_video(args.path, homography=H))
        store, _, rep = run_pipeline(frames, backend=backend)
    else:
        store, frames, rep = run_pipeline(backend=backend, seed=args.seed)
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
        print("\nFusão temporal de camisas vs voto por maioria (400 trajetórias simuladas):")
        for k, val in jersey_benchmark().items():
            print(f"  {k:40s} {val:6.1%}")


if __name__ == "__main__":
    main()
