import json

import pytest

from agg.experiments.blocks import Block, BlockConfig, audit_blocks, run_blocks
from agg.experiments.independent import file_hash


def test_real_block_learning_is_lagged_budget_matched_and_frozen(tmp_path, monkeypatch):
    from agg.tasks.protocol import EvaluationProtocol

    evaluation = EvaluationProtocol.evaluation
    protected_allowed = False

    def guarded(protocol, role):
        assert protected_allowed or role == "selection"
        return evaluation(protocol, role)

    monkeypatch.setattr(EvaluationProtocol, "evaluation", guarded)
    config = BlockConfig(
        data_seed=97,
        model_seed=5,
        steps=1,
        batch_size=16,
        blocks=(
            Block("retrieval", length=8, distance=2, density=0.2),
            Block("retrieval", length=10, distance=3, density=0.8),
            Block("modular"),
        ),
    )
    run = tmp_path / "blocks"
    result = run_blocks(config, run)
    assert result["schema_version"] == "agg.block-learning/1"
    assert len(result["blocks"]) == 3
    assert result["blocks"][0]["active_lesson_ids"] == []
    assert result["blocks"][1]["active_lesson_ids"] == []  # future utility has not arrived yet
    assert result["blocks"][2]["active_lesson_ids"] == []  # family shift forbids retrieval lessons
    for block in result["blocks"]:
        budgets = [block["lineages"][name]["budget"] for name in ("baseline", "summary", "replay")]
        assert budgets[0] == budgets[1] == budgets[2]
    assert set(result["blocks"][1]["controls"]) == {
        "incorrect",
        "irrelevant",
        "contradictory",
        "obsolete",
        "random",
        "out_of_scope",
    }
    assert result["blocks"][1]["controls"]["out_of_scope"]["status"] == "failed"
    assert result["blocks"][1]["controls"]["irrelevant"]["oracle_valid"] is True
    assert result["blocks"][1]["controls"]["irrelevant"]["knowledge_eligible"] is False
    registry = json.loads((run / "abstractions.json").read_text())
    for lesson in registry["records"]:
        for update in lesson["provenance"].get("transfer_history", []):
            assert update["future_block"] > lesson["provenance"]["source_block"]
            assert update["role"] == "selection"
    before = file_hash(run / "selected-state.pt")
    protected_allowed = True
    audit = audit_blocks(run, seed=2)
    assert len(audit["blocks"]) == 3
    assert audit["blocks"][1]["lineages"]["baseline"]["ood"]["score"] is not None
    assert audit["final_test"] is None
    assert set(audit["blocks"][2]["retention"]) == {"0", "1", "2"}
    assert file_hash(run / "selected-state.pt") == before
    with pytest.raises(ValueError, match="already"):
        audit_blocks(run)


def test_block_configuration_preserves_structural_roles():
    with pytest.raises(ValueError, match="modulus"):
        BlockConfig(blocks=(Block("modular", modulus=17), Block("modular", modulus=19)))
    with pytest.raises(ValueError):
        BlockConfig(steps=0)


def test_hierarchy_query_token_has_stable_role_across_depths():
    from agg.experiments.blocks import _data, _encoding, _protocols

    protocols = _protocols(
        BlockConfig(
            data_seed=101, blocks=(Block("hierarchy", depth=4), Block("hierarchy", depth=5))
        )
    )
    offsets, vocab, classes, length, queries = _encoding(protocols)
    encoded = [
        _data(p, offsets["hierarchy"], length, vocab, classes, queries["hierarchy"])
        for p in protocols
    ]
    queries = {int(d.train.x[0, -1]) for d in encoded}
    assert len(queries) == 1
    query = queries.pop()
    for protocol in protocols:
        from agg.experiments.blocks import _encode

        for role in ("selection", "audit", "final"):
            private = protocol.evaluation(role)
            for name in ("id", "ood"):
                split = _encode(
                    getattr(private, name), offsets["hierarchy"], length, query_token=query
                )
                assert (split.x[:, -1] == query).all()
                assert not (split.x[:, :-1] == query).any()


def test_active_lesson_marginal_comparison_removes_it_from_reference(tmp_path, monkeypatch):
    from dataclasses import replace

    from agg.controller.memory import AbstractionRegistry

    original = AbstractionRegistry.record_transfer

    def fixture_activation(registry, identity, evidence):
        # Exercise the application path; synthetic positive evidence is confined
        # to this disposable unit fixture and is not a research result.
        evidence = replace(
            evidence,
            standalone=0.2,
            marginal=0.1,
            marginal_lower=0.05,
            feasible=True,
            oracle_valid=True,
            source="unit-fixture-positive",
        )
        return original(registry, identity, evidence)

    monkeypatch.setattr(AbstractionRegistry, "record_transfer", fixture_activation)
    config = BlockConfig(
        steps=1, blocks=(Block(density=0.2), Block(density=0.5), Block(density=0.8))
    )
    result = run_blocks(config, tmp_path / "active")
    assert result["blocks"][2]["active_lesson_ids"]
    for transfer in result["blocks"][2]["transfer"]:
        if transfer["lesson_id"] in result["blocks"][2]["active_lesson_ids"]:
            assert transfer["marginal_reference"].startswith("without:")
            record = result["blocks"][2]["attempts"][transfer["marginal_reference"]]
            assert transfer["operation"] not in record["operations"]
