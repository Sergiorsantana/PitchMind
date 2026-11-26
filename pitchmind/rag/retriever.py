"""Retriever híbrido: filtros numéricos exatos + ranking por métrica + similaridade textual.

Diferente de um RAG "só vetorial", a recuperação aqui é majoritariamente
QUANTITATIVA: o plano (QuantQuery) vira predicados sobre colunas numéricas.
A similaridade textual (legendas do VLM) entra só como desempate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pitchmind.load.store import MatchStore
from pitchmind.models import Evidence
from pitchmind.rag.planner import QuantQuery

_OPS = {">": np.greater, ">=": np.greater_equal, "<": np.less, "<=": np.less_equal}


def _resolve_time(df: pd.DataFrame, tr) -> pd.Series:
    if tr is None:
        return pd.Series(True, index=df.index)
    start, end = tr
    if start < 0:  # "últimos N minutos"
        start, end = df["t_end"].max() + start, float("inf")
    return (df["t_start"] >= start) & (df["t_end"] <= end)


def apply_plan(store: MatchStore, q: QuantQuery) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retorna (janelas filtradas e ordenadas, estatísticas usadas)."""
    df = store.df()
    stats = store.stats()
    mask = _resolve_time(df, q.time_range)
    reasons = {i: [] for i in df.index}

    for f in q.filters:
        if f.metric not in df.columns:
            continue
        col = df[f.metric]
        if f.relative:
            mu, sd = stats.loc[f.metric, "mean"], stats.loc[f.metric, "std"] or 1.0
            col = (col - mu) / sd
        mask &= _OPS[f.op](col.fillna(np.nan), f.value).fillna(False)
        unit = "σ" if f.relative else ""
        for i in df.index:
            reasons[i].append(f"{f.metric}={col[i]:.2f}{unit} {f.op} {f.value}{unit}")

    if q.events:
        has = df["events"].fillna("").apply(lambda s: any(e in s.split(",") for e in q.events))
        mask &= has
        for i in df.index:
            hit = [e for e in q.events if e in (df.at[i, "events"] or "").split(",")]
            if hit:
                reasons[i].append("evento: " + ", ".join(hit))

    out = df[mask].copy()
    out["score"] = 0.0
    if q.rank_metric and q.rank_metric in out.columns:
        mu, sd = stats.loc[q.rank_metric, "mean"], stats.loc[q.rank_metric, "std"] or 1.0
        z = (out[q.rank_metric] - mu) / sd
        out["score"] = z if q.rank_order == "desc" else -z
    elif q.events:
        out["score"] = 1.0
    sem = store.semantic_scores(q.semantic)
    if len(sem):
        out["score"] += 0.05 * sem[out.index]  # desempate semântico
    out = out.sort_values("score", ascending=False)
    out["reasons"] = [reasons[i] for i in out.index]
    return out, stats


def retrieve(store: MatchStore, q: QuantQuery) -> list[Evidence]:
    df, _ = apply_plan(store, q)
    wins = {w.window_id: w for w in store.windows()}
    ev = []
    for _, r in df.head(q.top_k).iterrows():
        w = wins[int(r["window_id"])]
        reasons = list(r["reasons"])
        if q.rank_metric and q.rank_metric in w.metrics and not any(x.startswith(q.rank_metric + "=") for x in reasons):
            reasons.append(f"{q.rank_metric}={w.metrics[q.rank_metric]:.2f}")
        ev.append(Evidence(window=w, score=float(r["score"]), reasons=reasons))
    return ev
