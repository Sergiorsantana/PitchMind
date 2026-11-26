"""Motor de perguntas: planejar -> recuperar -> responder -> verificar."""
from __future__ import annotations

import numpy as np

from pitchmind.load.store import MatchStore
from pitchmind.models import Answer, Evidence
from pitchmind.rag.planner import LLMPlanner, QuantQuery, RulePlanner
from pitchmind.rag.retriever import apply_plan, retrieve
from pitchmind.rag.verifier import verify
from pitchmind.transform.features import METRIC_DOCS


def mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def _fmt_ev(e: Evidence) -> str:
    w = e.window
    return f"[{mmss(w.t_start)}–{mmss(w.t_end)}] " + "; ".join(e.reasons or [w.caption[:80]])


class AskEngine:
    def __init__(self, store: MatchStore, backend=None, use_llm: bool = False):
        self.store = store
        self.backend = backend
        self.planner = LLMPlanner(backend) if (use_llm and backend is not None and backend.name != "offline") \
            else RulePlanner()

    # ---------- respostas determinísticas (fonte da verdade) ----------
    def _aggregate(self, q: QuantQuery) -> tuple[str, dict[str, float]]:
        df, _ = apply_plan(self.store, QuantQuery(time_range=q.time_range, events=q.events, team=q.team))
        if q.events:
            n, total = len(df), len(self.store.df())
            secs = float((df["t_end"] - df["t_start"]).sum()) if n else 0.0
            ev = ", ".join(q.events)
            return (f"{n} de {total} janelas contêm {ev} (≈{secs / 60:.1f} min).",
                    {"n": n, "total": total, "min": secs / 60})
        metric = q.rank_metric or (q.filters[0].metric if q.filters else None) or f"{q.team or 'A'}_possession"
        if metric not in df.columns or df.empty:
            return "Não há dados suficientes para essa agregação.", {}
        v = df[metric].dropna()
        desc, unit = METRIC_DOCS.get(metric, (metric, ""))
        return (f"{desc}: média {v.mean():.1f} {unit} (mín {v.min():.1f}, máx {v.max():.1f}) em {len(v)} janelas.",
                {"mean": v.mean(), "min": v.min(), "max": v.max(), "n": len(v)})

    def _compare(self, q: QuantQuery) -> tuple[str, dict[str, float]]:
        df, _ = apply_plan(self.store, QuantQuery(time_range=q.time_range))
        base = (q.rank_metric or "A_block_height").split("_", 1)[1]
        out, extra = [], {}
        for t in "AB":
            col = f"{t}_{base}"
            if col in df.columns:
                v = float(df[col].mean())
                extra[t] = v
                out.append(f"time {t} {v:.1f}")
        unit = METRIC_DOCS.get(f"A_{base}", ("", ""))[1]
        return f"Comparação de {base} ({unit}): " + " vs ".join(out) + ".", extra

    def _moments(self, q: QuantQuery, ev: list[Evidence]) -> str:
        if not ev:
            return "Nenhum momento satisfaz todos os critérios medidos. (Prefiro não inventar.)"
        lines = [f"Encontrei {len(ev)} momento(s):"] + [f"• {_fmt_ev(e)}" for e in ev]
        return "\n".join(lines)

    # ---------- API ----------
    def ask(self, question: str, narrate: bool = False) -> Answer:
        q = self.planner.plan(question)
        ev = retrieve(self.store, q)
        # números citados na própria pergunta (limiares, top-k, minutos) também têm lastro
        extra: dict[str, float] = {f"q{i}": f.value for i, f in enumerate(q.filters)}
        if q.intent == "aggregate":
            text, agg = self._aggregate(q)
            extra.update(agg)
        elif q.intent == "compare":
            text, agg = self._compare(q)
            extra.update(agg)
        else:
            text = self._moments(q, ev)

        if narrate and self.backend is not None and self.backend.name != "offline":
            ctx = text + "\n\nEvidências:\n" + "\n".join(_fmt_ev(e) + " | " + e.window.caption for e in ev)
            try:
                draft = self.backend.text(
                    "Você é um analista de futebol. Responda em português, em até 4 frases, usando SOMENTE os "
                    "números presentes no contexto. Não invente números.",
                    f"Pergunta: {question}\n\nContexto medido:\n{ctx}")
                check = verify(draft, ev, extra)
                if check["ok"]:
                    text = draft
                else:
                    text += f"\n\n(narrativa do LLM descartada: números sem lastro {check['unsupported']})"
            except Exception as e:  # pragma: no cover - rede
                text += f"\n\n(narração indisponível: {e})"

        return Answer(question=question, text=text, evidence=ev, plan=q.to_dict(),
                      verification=verify(text, ev, extra))


def evaluate(store: MatchStore, engine: AskEngine, k: int = 5) -> dict[str, float]:
    """precision@k por fase usando o ground truth do simulador."""
    probes = {
        "a_high_press": "momentos de pressão alta do time A",
        "b_high_press": "momentos de pressão alta do time B",
        "a_low_block": "quando o time A ficou em bloco baixo",
        "a_counter": "contra-ataques do time A",
        "a_build_up": "construção de jogo do time A",
        "b_build_up": "saída de bola do time B",
    }
    res = {}
    for phase, qtxt in probes.items():
        ans = engine.ask(qtxt + f" top {k}")
        hits = [e.window.true_phase == phase for e in ans.evidence]
        res[phase] = float(np.mean(hits)) if hits else 0.0
    res["mean"] = float(np.mean(list(res.values())))
    return res
