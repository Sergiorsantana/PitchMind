import numpy as np
import pytest

from pitchmind.identity.jersey import JerseyFuser, Observation, UNKNOWN, read_likelihood
from pitchmind.pipeline import run_pipeline
from pitchmind.rag.engine import AskEngine, evaluate
from pitchmind.rag.planner import RulePlanner
from pitchmind.rag.verifier import verify

ROSTER = {"A": ["1", "6", "8", "10", "16", "18"], "B": ["2", "8", "9", "18"]}


@pytest.fixture(scope="module")
def store():
    s, _, _ = run_pipeline(seed=7, duration_s=600)
    return s


def test_etl_shapes(store):
    df = store.df()
    assert len(df) == 60
    assert {"A_block_height", "B_compactness", "A_pitch_control", "events", "caption"} <= set(df.columns)
    assert (df["A_pitch_control"] + df["B_pitch_control"]).round(6).eq(100).all()


def test_planner_parses_pt():
    q = RulePlanner().plan("Momentos de pressão alta do time B nos primeiros 5 minutos top 3")
    assert q.events == ["B_high_press"] and q.time_range == (0.0, 300.0) and q.top_k == 3
    q = RulePlanner().plan("controle de campo do time A acima de 55")
    assert q.filters[0].metric == "A_pitch_control" and q.filters[0].op == ">" and q.filters[0].value == 55
    q = RulePlanner().plan("bola mais rápida que o normal")
    assert q.filters[0].relative


def test_rag_precision(store):
    res = evaluate(store, AskEngine(store))
    assert res["mean"] >= 0.8


def test_answers_are_grounded(store):
    eng = AskEngine(store)
    for q in ["Quando o time A fez pressão alta?", "Compare a altura do bloco dos dois times",
              "Qual a posse média do time A?"]:
        assert eng.ask(q).verification["ok"], q


def test_verifier_flags_hallucinated_numbers(store):
    ans = AskEngine(store).ask("Quando o time A fez pressão alta? top 2")
    bad = ans.text + " O time A teve 87.3% de posse e 9 finalizações."
    v = verify(bad, ans.evidence)
    assert not v["ok"] and "87.3" in v["unsupported"]


def test_abstains_when_nothing_matches(store):
    ans = AskEngine(store).ask("controle de campo do time A acima de 99")
    assert ans.evidence == [] and "não inventar" in ans.text


def test_likelihood_models_dropped_digit():
    assert read_likelihood("8", "18") > read_likelihood("8", "16") > 0
    assert read_likelihood("18", "18") > read_likelihood("8", "18")


def test_fusion_recovers_18_from_mostly_8_reads():
    # 6 leituras "8" de baixa qualidade + 3 "18" nítidas -> deve ser 18, não 8
    obs = [Observation(i, "8", 0.5, 0.35) for i in range(6)] + [Observation(10 + i, "18", 0.95, 0.9) for i in range(3)]
    ident = JerseyFuser(ROSTER).identify(1, "A", obs)
    assert ident.number == "18"


def test_fusion_abstains_without_evidence():
    obs = [Observation(i, "", 0.1, 0.1) for i in range(5)] + [Observation(9, "7", 0.3, 0.3)]
    assert JerseyFuser(ROSTER).identify(1, "A", obs).number == UNKNOWN


def test_vlm_called_only_on_ambiguity():
    calls = []
    def vlm(best, cands):
        calls.append(cands)
        return "18", 0.95
    clear = [Observation(i, "10", 0.95, 0.95) for i in range(8)]
    amb = [Observation(i, "8", 0.6, 0.5) for i in range(3)] + [Observation(5, "18", 0.6, 0.5)]
    f = JerseyFuser(ROSTER, vlm=vlm)
    assert f.identify(1, "A", clear).number == "10" and not calls
    r = f.identify(2, "A", amb)
    assert calls and "18" in calls[0] and r.vlm_used and r.number == "18"


def test_global_assignment_no_duplicate_numbers():
    f = JerseyFuser(ROSTER)
    strong = [Observation(i, "8", 0.95, 0.95) for i in range(10)]
    weak = [Observation(i, "8", 0.9, 0.9) for i in range(4)]
    ids = f.assign([f.identify(1, "B", strong), f.identify(2, "B", weak)])
    nums = [i.number for i in ids if i.number != UNKNOWN]
    assert len(nums) == len(set(nums)) and ids[0].number == "8"


@pytest.mark.parametrize("seed", [1, 2])
def test_build_up_not_confused_with_attack_vs_low_block(seed):
    s, _, _ = run_pipeline(seed=seed, duration_s=600)
    res = evaluate(s, AskEngine(s))
    assert res["b_build_up"] >= 0.8 and res["mean"] >= 0.9
