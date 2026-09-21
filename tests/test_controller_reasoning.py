import json
import math
from dataclasses import replace

import pytest
import torch


def test_reasoning_schema_and_v1_migration():
    from agg.telemetry.controller import Observation, Reasoning

    item = Observation(3, reasoning=Reasoning(predictive_entropy=2, trajectory_cosine=-1))
    assert item.schema_version == "agg.controller/2"
    assert Observation.from_dict(json.loads(json.dumps(item.to_dict()))) == item
    old = {"step": 2, "schema_version": "agg.controller/1", "performance": {"reward": 4}}
    loaded = Observation.from_dict(old)
    assert loaded.reasoning == Reasoning()
    assert loaded.performance.reward == 4
    assert loaded.schema_version == "agg.controller/2"
    assert loaded.metrics() == {"performance.reward": 4}
    with pytest.raises(ValueError):
        Observation.from_dict({**old, "reasoning": {"predictive_entropy": 1}})
    with pytest.raises(ValueError):
        Observation.from_dict({**old, "schema_version": "agg.controller/99"})
    assert "reasoning" not in Observation(0, schema_version="agg.controller/1").to_dict()


@pytest.mark.parametrize(
    "field",
    [
        "representation_dispersion",
        "state_displacement",
        "predictive_entropy",
        "trajectory_cosine",
        "active_evidence_fraction",
        "evidence_turnover",
        "reasoning_progress",
    ],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True])
def test_reasoning_rejects_nonfinite_or_boolean(field, value):
    from agg.telemetry.controller import Reasoning

    with pytest.raises(ValueError):
        Reasoning(**{field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("representation_dispersion", -1),
        ("state_displacement", -1),
        ("predictive_entropy", -1),
        ("trajectory_cosine", 1.1),
        ("trajectory_cosine", -1.1),
        ("active_evidence_fraction", 1.1),
        ("evidence_turnover", -0.1),
    ],
)
def test_reasoning_domains(field, value):
    from agg.telemetry.controller import Reasoning

    with pytest.raises(ValueError):
        Reasoning(**{field: value})
    assert Reasoning(predictive_entropy=100, state_displacement=100, reasoning_progress=-7)


def test_extraction_formulas_and_missing_direction():
    from agg.telemetry.reasoning import (
        predictive_entropy,
        representation_dispersion,
        state_displacement,
        trajectory_cosine,
    )

    assert representation_dispersion(torch.tensor([[0.0, 0.0], [2.0, 0.0]])) == 1
    assert representation_dispersion(torch.ones(1, 3)) == 0
    assert state_displacement(torch.zeros(2), torch.tensor([3.0, 4.0])) == 5
    assert trajectory_cosine(torch.tensor([1.0, 0.0]), torch.tensor([-2.0, 0.0])) == -1
    assert trajectory_cosine(torch.zeros(2), torch.ones(2)) is None
    assert (
        trajectory_cosine(
            torch.tensor([1e300, 0.0], dtype=torch.float64),
            torch.tensor([1e300, 0.0], dtype=torch.float64),
        )
        == 1
    )
    assert predictive_entropy(torch.zeros(2, 4)) == pytest.approx(math.log(4))
    assert predictive_entropy(torch.tensor([1e300, -1e300], dtype=torch.float64)) == 0
    assert predictive_entropy(torch.tensor([10000.0, 10000.0])) == pytest.approx(math.log(2))
    assert predictive_entropy(torch.tensor([1.0])) == 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_extraction_rejects_nonfinite(bad):
    from agg.telemetry.reasoning import (
        predictive_entropy,
        representation_dispersion,
        state_displacement,
        trajectory_cosine,
    )

    for call in (
        lambda: predictive_entropy(torch.tensor([bad])),
        lambda: representation_dispersion(torch.tensor([[bad]])),
        lambda: state_displacement(torch.tensor([bad]), torch.ones(1)),
        lambda: trajectory_cosine(torch.ones(1), torch.tensor([bad])),
    ):
        with pytest.raises(ValueError):
            call()


def test_extraction_shapes_no_autograd_or_input_mutation():
    from agg.telemetry.reasoning import (
        predictive_entropy,
        representation_dispersion,
        state_displacement,
        trajectory_cosine,
    )

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    before = x.detach().clone()
    assert isinstance(representation_dispersion(x), float)
    assert isinstance(predictive_entropy(x), float)
    assert x.grad is None and torch.equal(before, x)
    for call in (
        lambda: representation_dispersion(torch.ones(2)),
        lambda: representation_dispersion(torch.empty(0, 3)),
        lambda: predictive_entropy(torch.empty(2, 0)),
        lambda: predictive_entropy(torch.tensor(1.0)),
        lambda: state_displacement(torch.ones(2), torch.ones(3)),
        lambda: trajectory_cosine(torch.ones(2, 2), torch.ones(2, 2)),
    ):
        with pytest.raises(ValueError):
            call()


@pytest.mark.parametrize("magnitude", [1e150, 1e180])
def test_large_identical_representations_have_exact_zero_dispersion(magnitude):
    from agg.telemetry.reasoning import representation_dispersion

    same = torch.tensor([[magnitude, 3 * magnitude]], dtype=torch.float64).repeat(100, 1)
    assert representation_dispersion(same) == 0


def test_reasoning_temporal_missingness_proxy_and_reward_independence():
    from agg.controller.temporal import TemporalTelemetry
    from agg.telemetry.controller import Observation, Performance, Reasoning

    temporal = TemporalTelemetry()
    for step in range(6):
        item = Observation(
            step,
            reasoning=Reasoning(state_displacement=step * step),
            performance=Performance(reward=9),
        )
        summary = temporal.update(item)["reasoning.state_displacement"]
    assert summary.second_derivative == pytest.approx(2)
    assert item.performance.reward == 9
    assert "reasoning.state_displacement" not in temporal.update(
        replace(item, step=6, reasoning=Reasoning())
    )
    result = temporal.update(replace(item, step=7, proxy_metrics=("reasoning.state_displacement",)))
    assert result["reasoning.state_displacement"].samples == 1
