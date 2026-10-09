import pytest
import torch

from agg.controller.memory import Abstraction, AbstractionRegistry, TransferEvidence
from agg.tasks.causal import oracle
from agg.tasks.lessons import extract_lessons, lesson_examples
from agg.tasks.protocol import make_protocol


def record():
    return Abstraction(
        "lesson",
        "query value determines answer",
        "retrieval",
        source_episodes=("source:0",),
        status="experimental",
        provenance={"source_block": 0, "operation": "value_cycle"},
    )


def evidence(block, **changes):
    values = dict(
        future_block=block,
        standalone=0.2,
        marginal=0.1,
        marginal_lower=0.05,
        source="future-selection:1",
        feasible=True,
        oracle_valid=True,
    )
    return TransferEvidence(**{**values, **changes})


def test_registry_requires_lagged_verified_utility_and_can_retire():
    registry = AbstractionRegistry()
    registry.add(record())
    assert registry.retrieve("retrieval", {}) == []
    with pytest.raises(ValueError, match="later"):
        registry.record_transfer("lesson", evidence(0))
    registry.record_transfer("lesson", evidence(1))
    assert registry.get("lesson").status == "active"
    assert registry.get("lesson").transfer_utility == 0.1
    assert registry.get("lesson").supporting_episodes == ("future-selection:1",)
    with pytest.raises(ValueError, match="increase"):
        registry.record_transfer("lesson", evidence(1))
    registry.record_transfer("lesson", evidence(2, marginal=-0.1, marginal_lower=-0.2))
    assert registry.get("lesson").status == "retired"
    assert registry.get("lesson").contradicting_episodes == ("future-selection:1",)
    loaded = AbstractionRegistry.from_dict(registry.to_dict())
    assert loaded.get("lesson") == registry.get("lesson")


@pytest.mark.parametrize(
    "changes",
    [
        dict(oracle_valid=False),
        dict(feasible=False),
        dict(marginal_lower=None),
        dict(standalone=0.0),
    ],
)
def test_positive_source_or_unverified_transfer_cannot_activate(changes):
    registry = AbstractionRegistry()
    registry.add(record())
    registry.record_transfer("lesson", evidence(1, **changes))
    assert registry.get("lesson").status != "active"
    assert registry.retrieve("retrieval", {}) == []


def test_templates_are_grounded_and_negative_labels_remain_invalid():
    protocol = make_protocol("retrieval", seed=91, samples=16)
    split = protocol.development().train
    lessons = extract_lessons(
        split,
        protocol.metadata,
        source_block=0,
        episode_prefix="block-0",
        predictions=split.y.tolist(),
        seed=2,
    )
    assert {a.provenance["operation"] for a in lessons} == {"value_cycle", "irrelevant_permutation"}
    for lesson in lessons:
        assert lesson.status == "experimental"
        assert lesson.source_episodes
        transformed, details = lesson_examples(
            split, protocol.metadata, lesson.provenance["operation"], seed=2
        )
        assert torch.equal(oracle(transformed.x, protocol.metadata), transformed.y)
        assert details["oracle_agreement"] == 1
        assert len(transformed.y) == len(split.y)
    wrong, details = lesson_examples(
        split, protocol.metadata, "value_cycle", seed=2, control="incorrect"
    )
    assert details["oracle_agreement"] == 0
    assert not torch.equal(oracle(wrong.x, protocol.metadata), wrong.y)
    with pytest.raises(ValueError, match="scope"):
        lesson_examples(split, protocol.metadata, "operand_swap", seed=2)
