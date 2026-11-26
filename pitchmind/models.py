"""Estruturas de dados centrais do PitchMind.

Coordenadas sempre em METROS no sistema do campo:
x ∈ [0, 105] (gol do time A em x=0, time A ataca para a direita), y ∈ [0, 68].
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
TEAMS = ("A", "B")


@dataclass
class FrameState:
    """Estado de um frame já projetado no plano do campo."""

    t: float                      # segundos desde o início
    players_a: np.ndarray         # (n, 2) posições do time A
    players_b: np.ndarray         # (n, 2) posições do time B
    ball: Optional[np.ndarray]    # (2,) ou None quando a bola não foi detectada
    phase: Optional[str] = None   # rótulo verdadeiro (somente no simulador) para avaliação
    image: Optional[np.ndarray] = None  # frame BGR opcional, usado pelo VLM

    def team(self, name: str) -> np.ndarray:
        return self.players_a if name == "A" else self.players_b


@dataclass
class Window:
    """Janela temporal agregada (a unidade de recuperação do RAG)."""

    window_id: int
    t_start: float
    t_end: float
    metrics: dict[str, float] = field(default_factory=dict)
    events: list[str] = field(default_factory=list)
    caption: str = ""
    true_phase: Optional[str] = None
    keyframe_t: Optional[float] = None

    def to_row(self) -> dict:
        row = {k: v for k, v in asdict(self).items() if k != "metrics"}
        row["events"] = ",".join(self.events)
        row.update(self.metrics)
        return row


@dataclass
class Evidence:
    window: Window
    score: float
    reasons: list[str]


@dataclass
class Answer:
    question: str
    text: str
    evidence: list[Evidence]
    plan: dict
    verification: dict
