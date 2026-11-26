"""PitchMind — app Streamlit.

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pitchmind.cli import EXAMPLES, jersey_benchmark  # noqa: E402
from pitchmind.identity.jersey import JerseyFuser, majority_vote, simulate_reads  # noqa: E402
from pitchmind.pipeline import run_pipeline  # noqa: E402
from pitchmind.rag.engine import AskEngine, evaluate, mmss  # noqa: E402
from pitchmind.render.radar import radar  # noqa: E402
from pitchmind.transform.features import METRIC_DOCS  # noqa: E402
from pitchmind.vlm.backends import get_backend  # noqa: E402

st.set_page_config(page_title="PitchMind", page_icon="⚽", layout="wide")


@st.cache_resource(show_spinner="Rodando ETL (extract → transform → load)…")
def build(seed: int, backend_name: str, windowing: str = "fixed"):
    backend = get_backend(backend_name)
    store, frames, rep = run_pipeline(backend=backend, seed=seed, windowing=windowing)
    return store, frames, rep, backend


with st.sidebar:
    st.title("⚽ PitchMind")
    st.caption("Pergunte à partida. Toda resposta tem lastro numérico.")
    seed = st.number_input("Seed da partida simulada", 0, 999, 7)
    backend_name = st.selectbox("Backend VLM/LLM", ["offline", "auto", "claude"],
                                help="'claude' requer ANTHROPIC_API_KEY. 'offline' é determinístico.")
    windowing = st.radio("Janelas", ["fixed", "possession"], horizontal=True,
                         format_func=lambda x: "fixas 10 s" if x == "fixed" else "por posse")
    narrate = st.toggle("Narração por LLM (verificada)", value=False, disabled=backend_name == "offline")

store, frames, rep, backend = build(int(seed), backend_name, windowing)
eng = AskEngine(store, backend, use_llm=backend.name != "offline")
fps = len(frames) / (frames[-1].t + 1e-9)

tab_ask, tab_radar, tab_id, tab_data = st.tabs(["🔎 Perguntar", "📡 Radar", "🔢 Identidade (camisas)", "🗄️ Dados & avaliação"])

with tab_ask:
    c1, c2 = st.columns([3, 1])
    q = c1.text_input("Pergunta em linguagem natural", EXAMPLES[0])
    c2.write("")
    pick = c2.selectbox("ou um exemplo", ["—"] + EXAMPLES, label_visibility="collapsed")
    if pick != "—":
        q = pick
    ans = eng.ask(q, narrate=narrate)
    v = ans.verification
    m1, m2, m3 = st.columns(3)
    m1.metric("Evidências", len(ans.evidence))
    m2.metric("Números na resposta", v["numbers_found"])
    m3.metric("Lastro numérico", f"{v['grounded_ratio']:.0%}", delta="verificado" if v["ok"] else "falhou",
              delta_color="normal" if v["ok"] else "inverse")
    st.markdown("**Resposta**")
    st.code(ans.text, language=None)
    with st.expander("Plano de consulta (auditável)"):
        st.json({k: val for k, val in ans.plan.items() if val not in (None, [], "")})
    if ans.evidence:
        cols = st.columns(min(3, len(ans.evidence)))
        for col, e in zip(cols, ans.evidence[:3]):
            w = e.window
            i = int(round((w.keyframe_t or w.t_start) * fps))
            col.pyplot(radar(frames[min(i, len(frames) - 1)], f"{mmss(w.t_start)}–{mmss(w.t_end)}"))
            col.caption(" · ".join(e.reasons))

with tab_radar:
    t = st.slider("Tempo (s)", 0.0, float(frames[-1].t), 260.0, step=1.0)
    f = frames[int(t * fps)]
    st.pyplot(radar(f, f"{mmss(t)}  ·  fase real: {f.phase}"))

with tab_id:
    st.markdown("""**Problema:** em um frame, o **18** vira **8**. A PitchMind acumula evidência ao longo da
trajetória com um modelo de erro de leitura, restringe à escalação, chama o VLM **só** nos casos ambíguos
(com os quadros mais informativos) e **se abstém** quando não há evidência.""")
    c1, c2, c3 = st.columns(3)
    true = c1.selectbox("Número real", ["18", "8", "16", "6", "21", "1"])
    n = c2.slider("Quadros na trajetória", 3, 40, 40)
    hard = c3.toggle("Condição difícil (zoom/oclusão)", True)
    roster = {"A": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "14", "16", "18", "19", "21", "23"]}
    obs = simulate_reads(true, n, np.random.default_rng(n * 31 + int(hard)), hard=hard)
    ident = JerseyFuser(roster).identify(0, "A", obs)
    st.write(f"Leituras por frame: `{' '.join(o.read or '∅' for o in obs)}`")
    a, b = st.columns(2)
    a.metric("Voto por maioria", majority_vote(obs))
    b.metric("Fusão bayesiana", ident.number, f"confiança {max(ident.posterior.values()):.0%}")
    st.bar_chart(pd.Series(ident.posterior).head(6))
    if st.button("Rodar benchmark (400 trajetórias)"):
        st.table(pd.Series(jersey_benchmark()).map("{:.1%}".format).rename("valor"))

with tab_data:
    st.markdown(f"ETL: **{rep.n_frames}** frames → **{rep.n_windows}** janelas · "
                + " · ".join(f"{k} {val * 1000:.0f} ms" for k, val in rep.timings.items()))
    sql = st.text_area("SQL (DuckDB) sobre a tabela `windows`",
                       "SELECT window_id, t_start, events, A_block_height, B_block_height, A_possession\n"
                       "FROM windows WHERE events <> '' ORDER BY A_block_height DESC LIMIT 10")
    try:
        st.dataframe(store.sql(sql), width="stretch")
    except Exception as e:
        st.error(str(e))
    with st.expander("Dicionário de métricas"):
        st.table(pd.DataFrame(METRIC_DOCS, index=["descrição", "unidade"]).T)
    if st.button("Avaliar RAG (precision@5 com ground truth)"):
        st.table(pd.Series(evaluate(store, AskEngine(store))).map("{:.2f}".format).rename("precision@5"))
