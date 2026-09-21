from dataclasses import asdict

from agg.telemetry.controller import Observation, Reasoning


def contexts():
    from agg.controller.routing import ControlContext

    return tuple(
        ControlContext.from_observation(
            Observation(
                t,
                reasoning=Reasoning(
                    predictive_entropy=t, state_displacement=t + 1, trajectory_cosine=-1 + t / 2
                ),
            )
        )
        for t in range(4)
    )


def test_control_zeroing_and_dynamics_disabled_preserve_observations():
    from agg.controller.reasoning_ablations import absolute_state_only, zero_coordinates
    from agg.controller.routing import ControlContext

    original = contexts()[2]
    before = asdict(original)
    zeroed = zero_coordinates(original, ("predictive_entropy",))
    assert zeroed.state.predictive_entropy == 0
    assert dict(zeroed.features)["reasoning.predictive_entropy"] == 0
    assert asdict(original) == before
    assert zeroed.intervention
    no_dynamics = absolute_state_only(original)
    assert no_dynamics.state.state_displacement is None
    assert no_dynamics.state.trajectory_cosine is None
    assert no_dynamics.state.predictive_entropy == 2
    missing = ControlContext.from_observation(Observation(0))
    assert zero_coordinates(missing, ("predictive_entropy",)).state.predictive_entropy is None


def test_permutation_and_scramble_are_seeded_and_preserve_domains():
    from agg.controller.reasoning_ablations import permute_coordinates, scramble_temporal

    controls = contexts()
    shuffled = permute_coordinates(controls, ("predictive_entropy",), seed=2)
    assert shuffled == permute_coordinates(controls, ("predictive_entropy",), seed=2)
    assert sorted(c.state.predictive_entropy for c in shuffled) == [0, 1, 2, 3]
    assert [c.state.state_displacement for c in shuffled] == [1, 2, 3, 4]
    assert [c.state.predictive_entropy for c in shuffled] != [0, 1, 2, 3]
    scrambled = scramble_temporal(controls, seed=2)
    assert [c.state for c in scrambled] != [c.state for c in controls]
    assert all(c.intervention for c in scrambled)


def test_unknown_coordinate_is_rejected():
    import pytest

    from agg.controller.reasoning_ablations import zero_coordinates

    with pytest.raises(ValueError):
        zero_coordinates(contexts()[0], ("reward",))
