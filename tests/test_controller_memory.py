import pytest


@pytest.mark.parametrize("score", [-0.1, 1.2, float("nan"), float("inf")])
@pytest.mark.parametrize("current", [0, 1])
def test_curriculum_holds_for_invalid_scores(score, current):
    from agg.controller.curriculum import CurriculumGate

    decision = CurriculumGate({0: 0.8, 1: 0.7}).evaluate(current, {0: score, 1: 0.9})
    assert decision.action == "hold"
    assert decision.missing_stages == (0,)


@pytest.mark.parametrize("score", [0.0, 1.0])
def test_curriculum_accepts_score_boundaries(score):
    from agg.controller.curriculum import CurriculumGate

    assert CurriculumGate({0: score}).evaluate(0, {0: score}).action == "complete"


@pytest.mark.parametrize(
    "invalid_ids,expected", [(("z",), "composition_failure"), (("a",), "bad_knowledge")]
)
def test_retrieval_only_relevant_invalid_ids_suppress_composition(invalid_ids, expected):
    from agg.controller.memory import RetrievalEvidence, diagnose_retrieval

    evidence = RetrievalEvidence(("a", "b"), ("a", "b"), invalid_ids, composition_success=False)
    assert diagnose_retrieval(evidence) == (expected,)


def test_curriculum_backtracks_to_shallowest_and_holds_for_missing():
    from agg.controller.curriculum import CurriculumGate

    gate = CurriculumGate({0: 0.8, 1: 0.75, 2: 0.7})
    decision = gate.evaluate(2, {0: 0.7, 1: 0.6, 2: 0.9})
    assert decision.action == "backtrack" and decision.stage == 0
    assert gate.evaluate(2, {1: 0.9, 2: 0.9}).action == "hold"
    assert gate.evaluate(1, {0: 0.9, 1: 0.9}).action == "advance"


def test_overlapping_abstractions_exclusions_and_roundtrip():
    from agg.controller.memory import Abstraction, AbstractionRegistry

    registry = AbstractionRegistry(activation_threshold=0.7)
    registry.add(Abstraction("global", "rule", "block", confidence=0.9))
    registry.add(
        Abstraction(
            "subset",
            "specific",
            "block",
            conditions={"kind": "x"},
            exclusions={"exception": True},
            confidence=0.9,
        )
    )
    assert {a.id for a in registry.retrieve("block", {"kind": "x"})} == {"global", "subset"}
    assert [a.id for a in registry.retrieve("block", {"kind": "x", "exception": True})] == [
        "global"
    ]
    assert registry.retrieve("other", {}) == []
    assert AbstractionRegistry.from_dict(registry.to_dict()).to_dict() == registry.to_dict()
    registry.record_applicability("subset", should_apply=False, selected=True)
    assert registry.get("subset").false_applicability_rate == 1
    registry.retire("global")
    assert registry.retrieve("block", {}) == []


def test_retrieval_separates_selection_knowledge_and_composition():
    from agg.controller.memory import RetrievalEvidence, diagnose_retrieval

    miss = diagnose_retrieval(RetrievalEvidence(correct_ids=("a",), retrieved_ids=()))
    assert "miss" in miss and "bad_knowledge" not in miss
    wrong = diagnose_retrieval(RetrievalEvidence(correct_ids=("a",), retrieved_ids=("b",)))
    assert "false_positive" in wrong
    bad = diagnose_retrieval(
        RetrievalEvidence(correct_ids=("a",), retrieved_ids=("a",), invalid_ids=("a",))
    )
    assert "bad_knowledge" in bad
    composition = diagnose_retrieval(
        RetrievalEvidence(
            correct_ids=("a", "b"), retrieved_ids=("a", "b"), composition_success=False
        )
    )
    assert "composition_failure" in composition and "miss" not in composition
    assert "uncovered_query" in diagnose_retrieval(RetrievalEvidence((), ()))


def test_composition_prioritizes_dependencies_without_all_pairs():
    from agg.controller.memory import CompositionResult, prioritize_compositions

    cases = prioritize_compositions([("a", "b")], {("a", "c"): 10, ("b", "c"): 1}, budget=2)
    assert cases == [("a", "b"), ("a", "c")]
    result = CompositionResult(
        ("a", "b"),
        {"a": True, "b": True},
        False,
        interface_failures=("shape",),
        order_dependent=True,
    )
    assert result.composition_failure


def test_retention_curve_half_life_and_transfer_separation():
    from agg.controller.curriculum import RetentionTracker

    tracker = RetentionTracker()
    assert tracker.summary("unmeasured")["retained_accuracy"] is None
    for step in range(9):
        tracker.record("exact", step, 0.8 * 2 ** (-step / 4))
        tracker.record("far", step, 0.9)
    assert tracker.summary("exact")["half_life"] == pytest.approx(4)
    assert tracker.summary("far")["half_life"] is None
    assert tracker.summary("exact")["forgetting"] == pytest.approx(0.6)
