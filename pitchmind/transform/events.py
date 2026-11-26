"""Detecção de eventos táticos por regras explícitas e auditáveis.

As regras são deliberadamente simples e com limiares documentados:
o objetivo é que toda afirmação do sistema seja rastreável até números.
"""
from __future__ import annotations

from pitchmind.models import Window

EVENT_DOCS = {
    "{t}_high_press": "time {t} sem a bola, bloco alto (>= 52 m), linha defensiva >= 38 m e >= 2.5 jogadores a menos de 10 m da bola",
    "{t}_low_block": "time {t} sem a bola, bloco baixo (<= 30 m) e mais compacto que o rival",
    "{t}_counter": "time {t} com a bola, bola rápida (>= 10 m/s)",
    "{t}_build_up": "time {t} com a bola, circulação lenta (< 10 m/s) e mais espaçado que o rival",
}


def detect_events(w: Window) -> list[str]:
    m, ev = w.metrics, []
    speed = m.get("ball_speed", 0.0)
    for t, o in (("A", "B"), ("B", "A")):
        poss = m.get(f"{t}_possession", 50.0)
        bh = m.get(f"{t}_block_height", 50.0)
        if poss < 40 and bh >= 52 and m.get(f"{t}_def_line", 0) >= 38 and m.get(f"{t}_pressers", 0) >= 2.5:
            ev.append(f"{t}_high_press")
        if poss < 40 and bh <= 30 and m.get(f"{t}_compactness", 1e9) < m.get(f"{o}_compactness", 0):
            ev.append(f"{t}_low_block")
        if poss > 60 and speed >= 10.0:
            ev.append(f"{t}_counter")
        if poss > 60 and speed < 10.0 and m.get(f"{t}_compactness", 0) > m.get(f"{o}_compactness", 1e9):
            ev.append(f"{t}_build_up")
    return ev


def describe(w: Window) -> str:
    """Legenda textual determinística a partir dos números (fallback sem VLM)."""
    m = w.metrics
    parts = [f"{w.t_start:.0f}s–{w.t_end:.0f}s."]
    if "A_possession" in m:
        parts.append(f"Posse A {m['A_possession']:.0f}% / B {m['B_possession']:.0f}%.")
    for t in ("A", "B"):
        if f"{t}_block_height" in m:
            parts.append(f"Time {t}: bloco a {m[f'{t}_block_height']:.0f} m, "
                         f"compactação {m.get(f'{t}_compactness', float('nan')):.0f} m², "
                         f"{m.get(f'{t}_pressers', 0):.1f} jogadores perto da bola.")
    labels = {
        "high_press": "pressão alta", "low_block": "bloco baixo",
        "counter": "contra-ataque", "build_up": "construção de jogo",
    }
    for e in w.events:
        t, kind = e.split("_", 1)
        parts.append(f"Evento: {labels.get(kind, kind)} do time {t}.")
    return " ".join(parts)
