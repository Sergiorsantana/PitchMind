"""Verificador numérico: a inovação central do PitchMind.

Todo número que aparece na resposta final precisa ser RASTREÁVEL até um
valor medido em alguma janela de evidência (ou em uma agregação dela).
Números não suportados são marcados — e, se forem muitos, a resposta é
substituída pela versão determinística (gerada só a partir dos dados).

Isso transforma "o VLM/LLM disse" em "o tracking mediu", com auditoria.
"""
from __future__ import annotations

import re

from pitchmind.models import Evidence

_NUM = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)?)(?![\w])")


def supported_values(evidence: list[Evidence], extra: dict[str, float] | None = None) -> list[float]:
    vals: list[float] = []
    for e in evidence:
        w = e.window
        vals += [w.t_start, w.t_end, w.t_start / 60, w.t_end / 60, float(w.window_id)]
        vals += list(w.metrics.values())
    vals += list((extra or {}).values())
    vals += [float(len(evidence))] + [float(i) for i in range(1, 12)]  # contagens/ordinais triviais
    return vals


def _matches(x: float, vals: list[float], abs_tol: float | None = None, decimals: int = 0) -> bool:
    """x tem lastro se algum valor medido, arredondado na precisão em que x foi escrito, é igual a x."""
    tol = abs_tol if abs_tol is not None else 0.5 * 10 ** (-decimals) + 1e-9
    return any(abs(x - v) <= tol for v in vals)


def verify(text: str, evidence: list[Evidence], extra: dict[str, float] | None = None) -> dict:
    vals = supported_values(evidence, extra)
    found, unsupported = [], []
    # ignora timestamps no formato mm:ss (verificados pelas janelas) convertendo para segundos
    for m in re.finditer(r"(\d+):(\d{2})", text):
        s = int(m.group(1)) * 60 + int(m.group(2))
        found.append(m.group(0))
        if not _matches(float(s), vals, abs_tol=10):
            unsupported.append(m.group(0))
    clean = re.sub(r"\d+:\d{2}", " ", text)
    for m in _NUM.finditer(clean):
        raw = m.group(1).replace(",", ".")
        x, dec = float(raw), len(raw.split(".")[1]) if "." in raw else 0
        found.append(m.group(1))
        if not _matches(x, vals, decimals=dec):
            unsupported.append(m.group(1))
    n = len(found)
    return {
        "numbers_found": n,
        "unsupported": unsupported,
        "grounded_ratio": 1.0 if n == 0 else round(1 - len(unsupported) / n, 3),
        "ok": len(unsupported) == 0,
    }
