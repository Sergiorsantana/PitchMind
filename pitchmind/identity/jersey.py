"""Identidade de jogadores: fusão temporal bayesiana de números de camisa.

Problema (do post original): em um frame isolado, "18" vira "8" por baixa
resolução, ângulo ou oclusão. Em vez de decidir por frame:

1. cada leitura (OCR/detector/VLM) é uma OBSERVAÇÃO ruidosa com qualidade q ∈ [0,1];
2. um modelo de erro explícito P(leitura | número real) cobre dígito perdido
   ("18"→"8"), dígito trocado ("6"↔"8", "1"↔"7") e leitura espúria;
3. a evidência é acumulada ao longo da trajetória, restrita à ESCALAÇÃO;
4. os k quadros mais informativos são enviados ao VLM só quando há ambiguidade;
5. uma atribuição global (Hungarian) impede dois tracks simultâneos com o mesmo número;
6. sem evidência suficiente, a identidade fica DESCONHECIDA (abstenção explícita).
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

UNKNOWN = "?"
# pares visualmente confundíveis (simétrico)
_CONFUSABLE = {("6", "8"), ("3", "8"), ("1", "7"), ("5", "6"), ("0", "8"), ("4", "1"), ("9", "8")}
_CONFUSABLE |= {(b, a) for a, b in _CONFUSABLE}


@dataclass
class Observation:
    t: float
    read: str            # texto lido ("8", "18", "" quando ilegível)
    conf: float          # confiança do leitor [0,1]
    quality: float       # qualidade do recorte [0,1] (tamanho, nitidez, oclusão, pose frontal)
    crop_id: Optional[str] = None


@dataclass
class TrackIdentity:
    track_id: int
    team: str
    posterior: dict[str, float]
    number: str = UNKNOWN
    vlm_used: bool = False
    best_frames: list[Observation] = field(default_factory=list)
    t_span: tuple[float, float] = (0.0, 0.0)

    @property
    def confidence(self) -> float:
        return self.posterior.get(self.number, 0.0)


def read_likelihood(read: str, true: str, p_ok: float = 0.80) -> float:
    """P(leitura | número real) — modelo de erro de leitura de camisa."""
    if read == "":
        return 1.0  # ilegível: não informa
    if read == true:
        return p_ok
    if len(true) == 2 and read in (true[0], true[1]):
        # perdeu um dígito (oclusão / dobra da camisa). O da direita é mais comum de sobrar visível
        return 0.10 if read == true[1] else 0.06
    if len(read) == len(true):
        diff = [(a, b) for a, b in zip(read, true) if a != b]
        if len(diff) == 1:
            return 0.06 if diff[0] in _CONFUSABLE else 0.01
    return 0.004  # leitura espúria


def informativeness(o: Observation) -> float:
    """Quão útil é um quadro para desambiguar (usado para escolher frames para o VLM)."""
    return o.quality * (0.5 + 0.5 * o.conf) * (1.15 if len(o.read) == 2 else 1.0)


class JerseyFuser:
    def __init__(self, roster: dict[str, list[str]], prior_unknown: float = 0.05,
                 accept: float = 0.85, margin: float = 0.30,
                 vlm: Optional[Callable[[list[Observation], list[str]], Optional[tuple[str, float]]]] = None,
                 k_frames: int = 3):
        self.roster = roster
        self.prior_unknown = prior_unknown
        self.accept = accept
        self.margin = margin
        self.vlm = vlm
        self.k_frames = k_frames

    def posterior(self, obs: list[Observation], team: str) -> dict[str, float]:
        cands = self.roster[team]
        logp = {c: math.log((1 - self.prior_unknown) / len(cands)) for c in cands}
        # "desconhecido": jogador fora da escalação/leitura inútil — verossimilhança plana
        logp[UNKNOWN] = math.log(self.prior_unknown)
        for o in obs:
            w = max(0.0, min(1.0, o.quality * o.conf))  # tempera: frames ruins pesam menos
            if w == 0 or o.read == "":
                continue
            for c in cands:
                logp[c] += w * math.log(read_likelihood(o.read, c))
            logp[UNKNOWN] += w * math.log(0.02)
        mx = max(logp.values())
        z = sum(math.exp(v - mx) for v in logp.values())
        return {k: math.exp(v - mx) / z for k, v in sorted(logp.items(), key=lambda kv: -kv[1])}

    def _decide(self, post: dict[str, float]) -> str:
        items = sorted(post.items(), key=lambda kv: -kv[1])
        top, p1 = items[0]
        p2 = items[1][1] if len(items) > 1 else 0.0
        return top if (top != UNKNOWN and p1 >= self.accept and p1 - p2 >= self.margin) else UNKNOWN

    def identify(self, track_id: int, team: str, obs: list[Observation]) -> TrackIdentity:
        post = self.posterior(obs, team)
        best = sorted(obs, key=informativeness, reverse=True)[: self.k_frames]
        ident = TrackIdentity(track_id, team, post, self._decide(post), best_frames=best,
                              t_span=(min((o.t for o in obs), default=0), max((o.t for o in obs), default=0)))
        if ident.number == UNKNOWN and self.vlm is not None and best:
            cands = [c for c, _ in sorted(post.items(), key=lambda kv: -kv[1]) if c != UNKNOWN][:4]
            out = self.vlm(best, cands)
            if out is not None:
                num, conf = out
                vlm_obs = [Observation(t=-1, read=num, conf=conf, quality=1.0)] * 2  # VLM vale ~2 quadros perfeitos
                post = self.posterior(obs + vlm_obs, team)
                ident.posterior, ident.number, ident.vlm_used = post, self._decide(post), True
        return ident

    def assign(self, idents: list[TrackIdentity]) -> list[TrackIdentity]:
        """Restrição global: tracks que coexistem no tempo não podem ter o mesmo número."""
        for team in {i.team for i in idents}:
            tr = [i for i in idents if i.team == team]
            cands = self.roster[team]
            # colunas: números da escalação + um "desconhecido" por track
            cost = np.full((len(tr), len(cands) + len(tr)), 0.0)
            for r, i in enumerate(tr):
                for c, num in enumerate(cands):
                    cost[r, c] = -math.log(max(i.posterior.get(num, 0.0), 1e-9))
                cost[r, len(cands):] = 1e3
                cost[r, len(cands) + r] = -math.log(max(1 - self.accept, 1e-9))
            # tracks que não se sobrepõem no tempo poderiam repetir número (substituição/re-ID);
            # aqui assumimos clipe curto em que todos coexistem.
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                i = tr[r]
                new = cands[c] if c < len(cands) else UNKNOWN
                if new != UNKNOWN and (i.posterior.get(new, 0) < self.accept * 0.75):
                    new = UNKNOWN
                i.number = new
        return idents


def majority_vote(obs: list[Observation]) -> str:
    """Baseline ingênuo: moda das leituras por frame."""
    reads = [o.read for o in obs if o.read]
    return Counter(reads).most_common(1)[0][0] if reads else UNKNOWN  # empate: primeira leitura vista


# ------------------------------------------------------------------ simulação
def simulate_reads(true: str, n: int, rng: np.random.Generator, hard: bool = False) -> list[Observation]:
    """Gera leituras ruidosas realistas de um número de camisa ao longo de uma trajetória."""
    out = []
    for k in range(n):
        q = float(np.clip(rng.beta(2, 3) if hard else rng.beta(3, 2), 0.05, 1))
        r = rng.random()
        p_drop = 0.45 if hard else 0.25
        if r < 0.30 * (1 - q):
            read = ""
        elif len(true) == 2 and r < 0.30 * (1 - q) + p_drop * (1 - q):
            read = true[1] if rng.random() < 0.7 else true[0]
        elif r < 0.30 * (1 - q) + p_drop * (1 - q) + 0.12 * (1 - q):
            pool = sorted(b for a, b in _CONFUSABLE if a == true[-1]) or ["0"]
            read = true[:-1] + pool[rng.integers(len(pool))]
        else:
            read = true
        out.append(Observation(t=k * 0.2, read=read, conf=float(np.clip(q + rng.normal(0, 0.1), 0.05, 1)),
                               quality=q))
    return out
