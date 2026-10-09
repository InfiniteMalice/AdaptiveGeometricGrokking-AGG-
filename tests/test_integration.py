import json
from dataclasses import asdict

import pytest

from agg.experiments.independent import file_hash
from agg.experiments.integration import (
    IntegrationConfig,
    build_bundle,
    prepare_release,
    release_final,
    run_integration,
)


def test_small_integrated_study_bundle_and_explicit_one_use_release(tmp_path):
    config = IntegrationConfig(
        tasks=("modular",),
        seeds=(23,),
        steps=2,
        candidate_steps=1,
        samples=16,
        width=8,
        eval_every=1,
    )
    study = tmp_path / "study"
    report = run_integration(config, study)
    assert report["config"] == asdict(config)
    assert report["cells"][0]["status"] == "complete"
    assert report["cells"][0]["budget"]["training_updates"] == 4
    assert report["cells"][0]["forward_calls"]["audit"]["calls"] > 0
    assert not list(study.rglob("final-attempt.json"))
    bundle = tmp_path / "bundle"
    manifest = build_bundle({"integrated": study}, bundle)
    assert manifest["sources"]["integrated"]["files"]
    root = study / report["cells"][0]["id"]
    release = prepare_release([root], tmp_path / "release.json")
    assert release["runs"][0]["manifest_sha256"] == file_hash(root / "selection-frozen.json")
    with pytest.raises(ValueError, match="authorization"):
        release_final(tmp_path / "release.json", authorization="", digest="bad")
    with pytest.raises(ValueError, match="digest"):
        release_final(tmp_path / "release.json", authorization="unit-fixture-only", digest="bad")
    assert not (root / "final-attempt.json").exists()
    final = release_final(
        tmp_path / "release.json",
        authorization="unit-fixture-only",
        digest=file_hash(tmp_path / "release.json"),
    )
    assert final["runs"][0]["status"] == "complete"
    with pytest.raises(ValueError, match="already"):
        release_final(
            tmp_path / "release.json",
            authorization="unit-fixture-only",
            digest=file_hash(tmp_path / "release.json"),
        )
    # Copying an old bundle cannot conceal changed source evidence.
    (root / "continued-cost.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        build_bundle({"integrated": study}, tmp_path / "changed")


def test_release_preflights_all_runs_before_any_final_access(tmp_path, monkeypatch):
    from agg.experiments import integration

    roots = []
    for i in range(2):
        root = tmp_path / str(i)
        root.mkdir()
        (root / "experiment.json").write_text("{}")
        (root / "selection-frozen.json").write_text(
            json.dumps(
                {
                    "schema_version": "agg.frozen-selection/1",
                    "selection_closed": True,
                    "artifacts": {"experiment.json": file_hash(root / "experiment.json")},
                }
            )
        )
        roots.append(root)
    prepare_release(roots, tmp_path / "release.json")
    (roots[1] / "experiment.json").write_text("tamper")

    def forbidden(*args, **kwargs):
        pytest.fail("final inference before all source validation")

    monkeypatch.setattr(integration, "report_run", forbidden)
    with pytest.raises(ValueError, match="changed"):
        release_final(
            tmp_path / "release.json",
            authorization="unit-fixture-only",
            digest=file_hash(tmp_path / "release.json"),
        )
    assert not (tmp_path / "release-attempt.json").exists()
