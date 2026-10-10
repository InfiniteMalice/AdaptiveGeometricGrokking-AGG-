import json
from dataclasses import replace

import pytest

from agg.experiments.config import ExperimentConfig, Features
from agg.experiments.independent import report_run
from agg.experiments.runner import run_experiment
from agg.training import TrainConfig


def test_continued_control_is_frozen_and_independently_paired(tmp_path):
    config = ExperimentConfig(
        independent_evaluation=True,
        data_seed=211,
        dimensions=(4,),
        training=TrainConfig(steps=2, eval_every=1, width=8, heads=2, layers=1, batch_size=1000),
        candidate_steps=1,
        features=replace(Features.baseline(), consolidation=True, distillation=True),
    )
    run_experiment(config, tmp_path / "run")
    root = tmp_path / "run"
    manifest = json.loads((root / "selection-frozen.json").read_text())
    assert "continued-state.pt" in manifest["artifacts"]
    assert "continued-optimizer.pt" in manifest["artifacts"]
    cost = json.loads((root / "continued-cost.json").read_text())
    assert cost["training_updates"] == 1
    assert cost["training_examples"] < 1000
    assert cost["training_tokens"] == 3 * cost["training_examples"]
    report = report_run(root)
    assert report["continued"]["ood"]["score"] is not None
    assert report["selected_vs_continued"]["ood"]["effect"] == pytest.approx(
        report["selected"]["ood"]["score"] - report["continued"]["ood"]["score"]
    )
    (root / "continued-state.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="changed"):
        report_run(
            root,
            role="final",
            authorization="unit-fixture-only",
            manifest_sha256=__import__("hashlib")
            .sha256((root / "selection-frozen.json").read_bytes())
            .hexdigest(),
        )
    assert not (root / "final-attempt.json").exists()
