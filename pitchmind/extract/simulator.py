"""Simulador tático de partida.

Gera trajetórias de 22 jogadores + bola com fases táticas rotuladas
(pressão alta, bloco baixo, contra-ataque, posse construída). Serve para:

1. rodar o pipeline completo sem GPU nem vídeo;
2. AVALIAR o RAG quantitativo com ground truth (precision@k por fase).
"""
from __future__ import annotations

import numpy as np

from pitchmind.models import FrameState, PITCH_LENGTH, PITCH_WIDTH

# formação 4-3-3 base (time A atacando para a direita), em fração do campo
_BASE_433 = np.array([
    [0.05, 0.50],                                              # goleiro
    [0.22, 0.15], [0.20, 0.38], [0.20, 0.62], [0.22, 0.85],    # defesa
    [0.38, 0.30], [0.36, 0.50], [0.38, 0.70],                  # meio
    [0.52, 0.18], [0.55, 0.50], [0.52, 0.82],                  # ataque
])

# fases: (nome, time com a bola, deslocamento do bloco em x, compressão, posição da bola)
PHASES = {
    "a_high_press":   dict(owner="B", shift_a=+0.28, shift_b=-0.05, comp_a=0.65, comp_b=1.0, ball=(0.80, 0.5)),
    "a_low_block":    dict(owner="B", shift_a=-0.10, shift_b=+0.25, comp_a=0.60, comp_b=1.1, ball=(0.25, 0.5)),
    "a_counter":      dict(owner="A", shift_a=+0.15, shift_b=+0.20, comp_a=1.20, comp_b=1.0, ball=(0.55, 0.5), fast=True),
    "a_build_up":     dict(owner="A", shift_a=+0.05, shift_b=+0.05, comp_a=1.25, comp_b=0.85, ball=(0.35, 0.5)),
    "b_high_press":   dict(owner="A", shift_a=+0.05, shift_b=+0.28, comp_a=1.0, comp_b=0.65, ball=(0.20, 0.5)),
    "b_build_up":     dict(owner="B", shift_a=-0.05, shift_b=-0.05, comp_a=0.85, comp_b=1.25, ball=(0.65, 0.5)),
}


def _formation(team: str, shift: float, comp: float, t: float, rng: np.random.Generator) -> np.ndarray:
    f = _BASE_433.copy()
    # compressão em torno do centro do bloco de linha (exceto goleiro)
    center = f[1:].mean(axis=0)
    f[1:] = center + (f[1:] - center) * comp
    f[1:, 0] += shift
    f[0, 0] = min(f[0, 0] + max(shift, 0) * 0.25, 0.18)
    # oscilação natural + ruído de tracking
    f += 0.008 * np.sin(t * 0.7 + np.arange(11)[:, None] * 1.3)
    f += rng.normal(0, 0.004, f.shape)
    if team == "B":  # espelha: B ataca para a esquerda
        f[:, 0] = 1.0 - f[:, 0]
    f = np.clip(f, 0.005, 0.995)
    return f * [PITCH_LENGTH, PITCH_WIDTH]


def default_script(duration_s: float = 600, seed: int = 7) -> list[tuple[float, float, str]]:
    """Sequência de fases (inicio, fim, fase) com durações aleatórias de 25-60s."""
    rng = np.random.default_rng(seed)
    names = list(PHASES)
    script, t, last = [], 0.0, None
    while t < duration_s:
        choices = [n for n in names if n != last]
        ph = choices[rng.integers(len(choices))]
        d = float(rng.uniform(25, 60))
        script.append((t, min(t + d, duration_s), ph))
        t += d
        last = ph
    return script


def simulate_match(duration_s: float = 600, fps: float = 5.0, seed: int = 7,
                   script: list[tuple[float, float, str]] | None = None) -> list[FrameState]:
    rng = np.random.default_rng(seed)
    script = script or default_script(duration_s, seed)
    frames: list[FrameState] = []
    prev_a = prev_b = None
    for i in range(int(duration_s * fps)):
        t = i / fps
        phase = next((p for s, e, p in script if s <= t < e), script[-1][2])
        cfg = PHASES[phase]
        a = _formation("A", cfg["shift_a"], cfg["comp_a"], t, rng)
        b = _formation("B", cfg["shift_b"], cfg["comp_b"], t, rng)
        # suavização temporal (transições entre fases não são instantâneas)
        alpha = 0.35 if cfg.get("fast") else 0.15
        if prev_a is not None:
            a = prev_a + alpha * (a - prev_a)
            b = prev_b + alpha * (b - prev_b)
        prev_a, prev_b = a, b
        bx, by = cfg["ball"]
        amp = 0.25 if cfg.get("fast") else 0.08
        ball = np.array([
            np.clip(bx + amp * np.sin(t * (0.8 if cfg.get("fast") else 0.5)), 0.02, 0.98) * PITCH_LENGTH,
            np.clip(by + 0.3 * np.sin(t * 0.4 + 1.0), 0.05, 0.95) * PITCH_WIDTH,
        ])
        # aproxima o portador da bola (time com posse) da bola
        owner = a if cfg["owner"] == "A" else b
        k = int(np.argmin(np.linalg.norm(owner - ball, axis=1)))
        owner[k] = ball + rng.normal(0, 0.6, 2)
        # na pressão alta, os 3 jogadores mais próximos do time sem bola cercam o portador
        if phase.endswith("high_press"):
            presser = a if phase.startswith("a_") else b
            near = np.argsort(np.linalg.norm(presser - ball, axis=1))[:3]
            presser[near] = ball + rng.normal(0, 3.0, (3, 2))
        # 4% dos frames sem bola detectada (oclusão), como em vídeo real
        frames.append(FrameState(t=t, players_a=a.copy(), players_b=b.copy(),
                                 ball=None if rng.random() < 0.04 else ball, phase=phase))
    return frames
