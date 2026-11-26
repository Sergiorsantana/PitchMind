"""Load: persiste janelas em DuckDB (camada analítica) + índice textual.

- tabela `windows`: uma linha por janela, uma coluna por métrica (consultável em SQL)
- índice TF-IDF (char n-grams) sobre legendas do VLM -> busca semântica leve, sem GPU
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from pitchmind.models import Window


class MatchStore:
    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        self.con = duckdb.connect(self.path)
        self._vec: TfidfVectorizer | None = None
        self._mat = None

    # ---------- escrita ----------
    def load(self, windows: list[Window], match_meta: dict | None = None) -> None:
        df = pd.DataFrame([w.to_row() for w in windows])
        self.con.execute("DROP TABLE IF EXISTS windows")
        self.con.register("_df", df)
        self.con.execute("CREATE TABLE windows AS SELECT * FROM _df")
        self.con.unregister("_df")
        self.con.execute("CREATE OR REPLACE TABLE meta (k VARCHAR, v VARCHAR)")
        for k, v in (match_meta or {}).items():
            self.con.execute("INSERT INTO meta VALUES (?, ?)", [k, json.dumps(v)])
        self._build_index()

    def _build_index(self) -> None:
        caps = self.df()["caption"].fillna("").tolist()
        if any(caps):
            self._vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
            self._mat = self._vec.fit_transform(caps)

    # ---------- leitura ----------
    def df(self) -> pd.DataFrame:
        return self.con.execute("SELECT * FROM windows ORDER BY window_id").df()

    def metric_columns(self) -> list[str]:
        skip = {"window_id", "t_start", "t_end", "events", "caption", "true_phase", "keyframe_t"}
        return [c for c in self.df().columns if c not in skip]

    def stats(self) -> pd.DataFrame:
        """Média / desvio de cada métrica — base dos z-scores ('acima do normal')."""
        return self.df()[self.metric_columns()].agg(["mean", "std", "min", "max"]).T

    def sql(self, query: str) -> pd.DataFrame:
        return self.con.execute(query).df()

    def semantic_scores(self, text: str) -> np.ndarray:
        n = len(self.df())
        if not text or self._vec is None:
            return np.zeros(n)
        q = self._vec.transform([text])
        return (self._mat @ q.T).toarray().ravel()

    def windows(self) -> list[Window]:
        out = []
        cols = self.metric_columns()
        for r in self.df().to_dict("records"):
            metrics = {c: float(r[c]) for c in cols if r[c] is not None and not math.isnan(r[c])}
            out.append(Window(window_id=int(r["window_id"]), t_start=r["t_start"], t_end=r["t_end"],
                              metrics=metrics, events=[e for e in (r["events"] or "").split(",") if e],
                              caption=r["caption"] or "", true_phase=r.get("true_phase"),
                              keyframe_t=r.get("keyframe_t")))
        return out
