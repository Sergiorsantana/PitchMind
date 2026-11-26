"""Query planner: pergunta em linguagem natural -> plano quantitativo (QuantQuery).

O plano é um objeto estruturado e auditável (filtros numéricos, ranking,
eventos, intervalo de tempo). Dois planners com o mesmo contrato:
- `RulePlanner`  : regex PT/EN, determinístico, offline
- `LLMPlanner`   : Claude com saída JSON estruturada (fallback para regras)
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field, asdict
from typing import Optional

from pitchmind.transform.features import METRIC_DOCS

EVENT_KINDS = ("high_press", "low_block", "counter", "build_up")


@dataclass
class Filter:
    metric: str
    op: str          # > >= < <=
    value: float
    relative: bool = False  # True => value em desvios-padrão (z-score)


@dataclass
class QuantQuery:
    intent: str = "find_moments"          # find_moments | aggregate | compare
    team: Optional[str] = None
    events: list[str] = field(default_factory=list)
    filters: list[Filter] = field(default_factory=list)
    rank_metric: Optional[str] = None
    rank_order: str = "desc"
    time_range: Optional[tuple[float, float]] = None
    semantic: str = ""
    top_k: int = 5

    def to_dict(self) -> dict:
        return asdict(self)


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if not unicodedata.combining(c))


_EVENT_WORDS = {
    "high_press": ["pressao alta", "pressionou alto", "pressiona alto", "pressionando alto", "high press",
                   "pressing", "gegenpress", "marcacao alta", "marcacao pressao", "pressao"],
    "low_block": ["bloco baixo", "retranca", "recuad", "low block", "fechad", "parked the bus", "estacion"],
    "counter": ["contra-ataque", "contra ataque", "contragolpe", "counter", "transicao rapida", "transicao ofensiva"],
    "build_up": ["construcao", "saida de bola", "build-up", "build up", "buildup", "circulacao", "troca de passes",
                 "posse paciente"],
}

# (palavras, métrica sem time, ordem)  — ordem dos itens importa (mais específico primeiro)
_RANK_WORDS = [
    (["mais compact", "most compact", "menos espac", "tighter", "mais junto"], "compactness", "asc"),
    (["menos compact", "least compact", "mais espac", "mais espalhad"], "compactness", "desc"),
    (["linha defensiva mais alta", "linha mais alta", "highest defensive line", "linha adiantada"], "def_line", "desc"),
    (["linha defensiva mais baixa", "linha mais baixa", "lowest defensive line"], "def_line", "asc"),
    (["bloco mais alto", "mais adiantad", "highest block", "mais alto"], "block_height", "desc"),
    (["bloco mais baixo", "mais recuad", "lowest block", "mais baixo"], "block_height", "asc"),
    (["mais larg", "widest", "mais aberto", "amplitude"], "width", "desc"),
    (["mais estreit", "narrowest"], "width", "asc"),
    (["mais jogadores perto da bola", "mais pressao", "mais intens", "most pressure", "pressers"], "pressers", "desc"),
    (["controle de campo", "controle territorial", "pitch control", "dominio territorial", "mais espaco"],
     "pitch_control", "desc"),
    (["field tilt", "terco final", "final third", "mais perto do gol"], "field_tilt", "desc"),
    (["mais posse", "most possession", "dominou a posse"], "possession", "desc"),
    (["mais rapid", "fastest", "bola mais rapida", "velocidade"], "ball_speed", "desc"),
]

_METRIC_ALIASES = {
    "posse": "possession", "possession": "possession",
    "bloco": "block_height", "block": "block_height",
    "linha": "def_line", "line": "def_line",
    "largura": "width", "width": "width",
    "compactacao": "compactness", "area": "compactness", "compactness": "compactness",
    "controle": "pitch_control", "control": "pitch_control",
    "velocidade": "ball_speed", "speed": "ball_speed",
    "pressionadores": "pressers", "jogadores perto": "pressers", "pressers": "pressers",
    "tilt": "field_tilt",
}
_OPS = {
    "acima de": ">", "maior que": ">", "mais de": ">", "superior a": ">", "above": ">", "over": ">",
    "more than": ">", "greater than": ">", ">": ">", "pelo menos": ">=", "at least": ">=", ">=": ">=",
    "abaixo de": "<", "menor que": "<", "menos de": "<", "inferior a": "<", "below": "<", "under": "<",
    "less than": "<", "<": "<", "no maximo": "<=", "at most": "<=", "<=": "<=",
}
_TEAMLESS = {"ball_speed", "ball_progression_A"}


def _team(q: str) -> Optional[str]:
    m = re.search(r"\b(?:time|team|equipe|equipa)\s+([ab])\b", q)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b(?:do|o|da|pelo)\s+([ab])\b", q)
    return m.group(1).upper() if m else None


def _with_team(metric: str, team: Optional[str]) -> str:
    if metric in _TEAMLESS:
        return metric
    return f"{team or 'A'}_{metric}"


class RulePlanner:
    name = "rules"

    def plan(self, question: str) -> QuantQuery:
        q = _norm(question)
        qq = QuantQuery(semantic=question)
        qq.team = _team(q)

        if re.search(r"\b(compar|versus|vs\.?|contra o outro|diferenca entre)", q):
            qq.intent = "compare"
        elif re.search(r"\b(media|medio|quanto|quantas|quantos|average|how much|how many|percentual|total)\b", q):
            qq.intent = "aggregate"

        for kind, words in _EVENT_WORDS.items():
            if any(w in q for w in words):
                teams = [qq.team] if qq.team else ["A", "B"]
                qq.events += [f"{t}_{kind}" for t in teams]
                if kind == "high_press":  # "pressão" genérica não deve capturar outros eventos
                    break

        for words, metric, order in _RANK_WORDS:
            if any(w in q for w in words):
                qq.rank_metric, qq.rank_order = _with_team(metric, qq.team), order
                break

        op_re = "|".join(sorted(map(re.escape, _OPS), key=len, reverse=True))
        al_re = "|".join(sorted(map(re.escape, _METRIC_ALIASES), key=len, reverse=True))
        for m in re.finditer(rf"({al_re})[^\d<>]{{0,25}}?({op_re})\s*(\d+(?:[.,]\d+)?)", q):
            metric = _with_team(_METRIC_ALIASES[m.group(1)], qq.team)
            qq.filters.append(Filter(metric, _OPS[m.group(2)], float(m.group(3).replace(",", "."))))

        if re.search(r"(acima do normal|que o normal|do normal|atipic|anormal|anomal|fora do padrao|unusual|outlier|than usual)", q) and qq.rank_metric:
            op = ">" if qq.rank_order == "desc" else "<"
            qq.filters.append(Filter(qq.rank_metric, op, 1.5 if op == ">" else -1.5, relative=True))

        if m := re.search(r"(?:primeiros|first)\s+(\d+)\s*(?:min|minutos|minutes)", q):
            qq.time_range = (0.0, float(m.group(1)) * 60)
        elif m := re.search(r"(?:ultimos|last)\s+(\d+)\s*(?:min|minutos|minutes)", q):
            qq.time_range = (-float(m.group(1)) * 60, float("inf"))  # negativo => relativo ao fim
        elif m := re.search(r"entre\s+(?:o\s+)?(?:minuto\s+)?(\d+)\s+e\s+(?:o\s+)?(?:minuto\s+)?(\d+)", q):
            qq.time_range = (float(m.group(1)) * 60, float(m.group(2)) * 60)
        elif m := re.search(r"(?:apos|depois d[eo]s?|after)\s+(?:o\s+)?(?:minuto\s+)?(\d+)", q):
            qq.time_range = (float(m.group(1)) * 60, float("inf"))

        if m := re.search(r"\btop\s*(\d+)|\b(\d+)\s+(?:momentos|janelas|lances|moments|windows)", q):
            qq.top_k = int(m.group(1) or m.group(2))
        return qq


PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": ["find_moments", "aggregate", "compare"]},
        "team": {"type": ["string", "null"], "enum": ["A", "B", None]},
        "events": {"type": "array", "items": {"type": "string"}},
        "filters": {"type": "array", "items": {
            "type": "object",
            "properties": {"metric": {"type": "string"}, "op": {"type": "string", "enum": [">", ">=", "<", "<="]},
                           "value": {"type": "number"}, "relative": {"type": "boolean"}},
            "required": ["metric", "op", "value", "relative"], "additionalProperties": False}},
        "rank_metric": {"type": ["string", "null"]},
        "rank_order": {"type": "string", "enum": ["asc", "desc"]},
        "time_start_s": {"type": ["number", "null"]},
        "time_end_s": {"type": ["number", "null"]},
        "top_k": {"type": "integer"},
    },
    "required": ["intent", "team", "events", "filters", "rank_metric", "rank_order",
                 "time_start_s", "time_end_s", "top_k"],
    "additionalProperties": False,
}


class LLMPlanner:
    name = "llm"

    def __init__(self, backend, fallback: RulePlanner | None = None):
        self.backend = backend
        self.fallback = fallback or RulePlanner()

    def plan(self, question: str) -> QuantQuery:
        metrics = "\n".join(f"- {k}: {d} ({u})" for k, (d, u) in METRIC_DOCS.items())
        events = ", ".join(f"{t}_{k}" for t in "AB" for k in EVENT_KINDS)
        system = ("Você converte perguntas sobre uma partida de futebol em um plano de consulta quantitativo. "
                  "Use APENAS os nomes de métricas e eventos listados. relative=true significa valor em desvios-padrão.")
        prompt = f"Métricas:\n{metrics}\n\nEventos: {events}\n\nPergunta: {question}"
        try:
            d = self.backend.json(system, prompt, PLAN_SCHEMA)
        except Exception:
            return self.fallback.plan(question)
        valid_events = {f"{t}_{k}" for t in "AB" for k in EVENT_KINDS}
        tr = None
        if d.get("time_start_s") is not None or d.get("time_end_s") is not None:
            tr = (d.get("time_start_s") or 0.0, d.get("time_end_s") or float("inf"))
        return QuantQuery(
            intent=d["intent"], team=d["team"],
            events=[e for e in d["events"] if e in valid_events],
            filters=[Filter(**f) for f in d["filters"] if f["metric"] in METRIC_DOCS],
            rank_metric=d["rank_metric"] if d["rank_metric"] in METRIC_DOCS else None,
            rank_order=d["rank_order"], time_range=tr, semantic=question, top_k=max(1, min(d["top_k"], 20)),
        )
