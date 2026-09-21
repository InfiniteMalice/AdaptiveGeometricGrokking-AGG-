import json
from dataclasses import replace

import pytest

from agg.controller.events import EventLog
from agg.controller.memory import Abstraction, AbstractionRegistry
from agg.ledger import Ledger
from agg.telemetry.controller import Observation, Reasoning


def setup_router(tmp_path, **config):
    from agg.controller.routing import AuditedEvidenceRouter, ReferenceRouter, RoutingConfig

    registry = AbstractionRegistry()
    for identity in ("a", "b", "bad", "unknown"):
        registry.add(Abstraction(identity, "rule " + identity, "task", confidence=0.9))
    registry.add(Abstraction("retired", "old rule", "task", confidence=0.9, status="retired"))
    ledger = Ledger(tmp_path / "events.jsonl")
    harness = AuditedEvidenceRouter(
        registry, ReferenceRouter(RoutingConfig(**config)), EventLog(ledger, "test")
    )
    return registry, ledger, harness


def route(harness, state=0.0, step=0, **kwargs):
    from agg.controller.routing import EvidenceValidity

    return harness.route(
        Observation(step, reasoning=Reasoning(predictive_entropy=state)),
        block="task",
        context={},
        validity=EvidenceValidity(("a", "b", "retired"), ("bad",), ("missing",), "external-v1"),
        profiles={
            "a": {"reasoning.predictive_entropy": 0.0},
            "b": {"reasoning.predictive_entropy": 4.0},
        },
        **kwargs,
    )


def test_state_changes_activation_not_knowledge_and_all_drops_are_audited(tmp_path):
    registry, ledger, harness = setup_router(tmp_path)
    before = registry.to_dict()
    first = route(harness)
    second = route(harness, state=4, step=1)
    assert first.selected_ids == ("a",)
    assert second.selected_ids == ("b",)
    assert first.eligible_ids == ("a", "b")
    assert first.dropped_ids == ("b",)
    payload = ledger.read()[0]["payload"]
    assert payload["invalid_ids"] == ["bad"]
    assert payload["unknown_ids"] == ["unknown"]
    assert payload["unavailable_ids"] == ["missing"]
    assert payload["ineligible_ids"] == ["retired"]
    assert len(payload["evidence_refs"]) == 5
    assert payload["observation"]["reasoning"]["predictive_entropy"] == 0
    assert payload["profiles"]["a"]["reasoning.predictive_entropy"] == 0
    assert registry.to_dict() == before
    harness.record_downstream(second, {"action": "observe_more"}, proposal_id="test-proposal")
    assert ledger.read()[-1]["payload"]["routing_id"] == second.id
    assert ledger.read()[-1]["proposal_id"] == "test-proposal"
    # Mutating a decoded audit view cannot rewrite the immutable decision.
    first.to_dict()["selected_ids"].clear()
    assert first.selected_ids == ("a",)


def test_deterministic_ties_missing_state_and_no_evidence(tmp_path):
    from agg.controller.routing import EvidenceValidity

    _, _, harness = setup_router(tmp_path)
    assert route(harness, state=2).selected_ids == ("a",)
    missing = harness.route(
        Observation(1),
        block="task",
        context={},
        validity=EvidenceValidity(("a",), source="host"),
        profiles={},
    )
    assert missing.selected_ids == () and missing.dropped_ids == ("a",)
    empty = harness.route(
        Observation(2),
        block="task",
        context={},
        validity=EvidenceValidity(source="host"),
        profiles={},
    )
    assert empty.selected_ids == () and empty.eligible_ids == ()


def test_invalid_selection_and_missing_scores_fail_closed(tmp_path):
    from agg.controller.routing import RoutingSelection

    _, ledger, harness = setup_router(tmp_path)

    class BadRouter:
        def configuration(self):
            return {"kind": "test-invalid-selector"}

        def select(self, control, eligible):
            return RoutingSelection(("bad",), (), ())

    harness.router = BadRouter()
    with pytest.raises(ValueError, match="eligible"):
        route(harness)
    assert ledger.read()[-1]["payload"]["status"] == "rejected"


def test_audit_failure_never_exposes_selection(tmp_path):
    _, _, harness = setup_router(tmp_path)
    directory = tmp_path / "directory"
    directory.mkdir()
    harness.events = EventLog(Ledger(directory), "broken-sink")
    with pytest.raises(OSError):
        route(harness)


@pytest.mark.parametrize("mode", ["random", "state_independent", "fixed", "full_history"])
def test_routing_ablation_baselines(tmp_path, mode):
    _, _, harness = setup_router(tmp_path, mode=mode, fixed_ids=("b",), seed=7)
    first = route(harness)
    replay = route(harness, state=4)
    assert replay.selected_ids == first.selected_ids
    if mode == "full_history":
        assert first.selected_ids == ("a", "b")
    elif mode == "fixed":
        assert first.selected_ids == ("b",)


def test_temporal_context_and_proxy_are_audited(tmp_path):
    from agg.controller.routing import ControlContext, FeatureScale, RoutingConfig
    from agg.controller.temporal import TemporalTelemetry

    temporal = TemporalTelemetry()
    for step in range(6):
        obs = Observation(
            step,
            reasoning=Reasoning(predictive_entropy=float(step * step)),
            proxy_metrics=("reasoning.predictive_entropy",),
        )
        summaries = temporal.update(obs)
    control = ControlContext.from_observation(obs, summaries)
    values = dict(control.features)
    assert values["reasoning.predictive_entropy.second_derivative"] == pytest.approx(2)
    assert values["reasoning.predictive_entropy.first_derivative"] == 9
    config = RoutingConfig(features=(FeatureScale("reasoning.predictive_entropy", 2),))
    assert config.features[0].scale == 2
    with pytest.raises(ValueError):
        RoutingConfig(features=(FeatureScale("reasoning.predictive_entropy", 0),))
    assert control.proxy_metrics == obs.proxy_metrics


def test_turnover_oscillation_and_shrinking_uncertain_support(tmp_path):
    from agg.controller.routing_diagnostics import diagnose_routing

    _, _, harness = setup_router(tmp_path)
    a, b, a2 = route(harness), route(harness, state=4, step=1), route(harness, step=2)
    diagnostic = diagnose_routing(a2, b, a)
    assert diagnostic.evidence_turnover == 1
    assert "retrieval_oscillation" in diagnostic.findings
    assert diagnostic.reactivated_ids == ("a",)
    assert diagnose_routing(a).evidence_turnover is None
    assert diagnose_routing(a).active_evidence_fraction == 0.5
    assert all("deception" not in finding for finding in diagnostic.findings)


def test_validity_domains_and_finite_profiles(tmp_path):
    from agg.controller.routing import EvidenceValidity

    with pytest.raises(ValueError):
        EvidenceValidity(("a",), ("a",), source="host")
    with pytest.raises(ValueError):
        EvidenceValidity(("a",), source="")
    _, _, harness = setup_router(tmp_path)
    with pytest.raises(ValueError):
        harness.route(
            Observation(0),
            block="task",
            context={},
            validity=EvidenceValidity(("a",), source="host"),
            profiles={"a": {"reasoning.predictive_entropy": float("nan")}},
        )


def test_counterfactual_control_preserves_observed_state(tmp_path):
    from agg.controller.routing import ControlContext

    _, ledger, harness = setup_router(tmp_path)
    control = ControlContext.from_observation(
        Observation(0, reasoning=Reasoning(predictive_entropy=4))
    )
    decision = route(harness, control=replace(control, intervention="counterfactual entropy"))
    assert decision.selected_ids == ("b",)
    record = json.loads(json.dumps(ledger.read()[-1]["payload"]))
    assert record["observation"]["reasoning"]["predictive_entropy"] == 0
    assert record["control"]["state"]["predictive_entropy"] == 4
    assert record["control"]["intervention"] == "counterfactual entropy"


def test_rejected_nonfinite_router_score_still_has_audit(tmp_path):
    from agg.controller.routing import RoutingSelection

    _, ledger, harness = setup_router(tmp_path)

    class NonfiniteRouter:
        def configuration(self):
            return {"kind": "nonfinite-test"}

        def select(self, control, eligible):
            return RoutingSelection(
                ("a",), (("a", float("nan")), ("b", 0.0)), (("a", "test"), ("b", "test"))
            )

    harness.router = NonfiniteRouter()
    with pytest.raises(ValueError):
        route(harness)
    assert ledger.read()[-1]["payload"]["status"] == "rejected"
    assert "nan" in ledger.read()[-1]["payload"]["invalid_selection_repr"]


def test_shrink_redirection_stopping_and_semantic_disagreement(tmp_path):
    from agg.controller.routing import EvidenceValidity, ReferenceRouter, RoutingConfig
    from agg.controller.routing_diagnostics import diagnose_routing

    _, _, harness = setup_router(tmp_path, mode="full_history")
    earlier = route(harness, state=0)
    harness.router = ReferenceRouter(RoutingConfig(max_active=1))
    current = harness.route(
        Observation(
            1, reasoning=Reasoning(predictive_entropy=2, trajectory_cosine=-1, reasoning_progress=1)
        ),
        block="task",
        context={},
        validity=EvidenceValidity(("a", "b"), source="host"),
        profiles={
            "a": {"reasoning.predictive_entropy": 0},
            "b": {"reasoning.predictive_entropy": 4},
        },
    )
    findings = diagnose_routing(current, earlier, semantic_continuity=1, stopping=True).findings
    assert "uncertainty_rising_support_shrinking" in findings
    assert "trajectory_redirection_with_removal" in findings
    assert "semantic_internal_continuity_disagreement" in findings
    assert "possible_premature_stopping" in findings
    assert diagnose_routing(current, earlier).telemetry(Reasoning()).evidence_turnover == 0.5


def test_v1_observation_routing_diagnostics(tmp_path):
    from agg.controller.routing import EvidenceValidity
    from agg.controller.routing_diagnostics import diagnose_routing

    _, _, harness = setup_router(tmp_path)
    result = harness.route(
        Observation(0, schema_version="agg.controller/1"),
        block="task",
        context={},
        validity=EvidenceValidity(source="host"),
        profiles={},
    )
    assert diagnose_routing(result).active_evidence_fraction is None


def test_causal_zeroing_changes_route_and_preserves_original_audit(tmp_path):
    from agg.controller.reasoning_ablations import counterfactual_replay, zero_coordinates
    from agg.controller.routing import ControlContext, EvidenceValidity

    _, _, harness = setup_router(tmp_path)
    observation = Observation(0, reasoning=Reasoning(predictive_entropy=4))
    profiles = {"a": {"reasoning.predictive_entropy": 0}, "b": {"reasoning.predictive_entropy": 4}}
    kwargs = {
        "block": "task",
        "context": {},
        "validity": EvidenceValidity(("a", "b"), source="host"),
        "profiles": profiles,
    }
    observed = harness.route(observation, **kwargs)
    perturbed = zero_coordinates(
        ControlContext.from_observation(observation), ("predictive_entropy",)
    )
    replay = counterfactual_replay(harness, observation, perturbed, **kwargs)
    assert observed.selected_ids == ("b",) and replay.selected_ids == ("a",)
    assert replay.to_dict()["observation"]["reasoning"]["predictive_entropy"] == 4


def test_mutable_plugin_output_cannot_escape_validation(tmp_path):
    from agg.controller.routing import RoutingSelection

    _, ledger, harness = setup_router(tmp_path)

    class MutableRouter:
        selected = ["a"]

        def configuration(self):
            return {"kind": "mutable-test"}

        def select(self, control, eligible):
            return RoutingSelection(
                self.selected, (("a", 1), ("b", 0)), (("a", "test"), ("b", "test"))
            )

    harness.router = MutableRouter()
    with pytest.raises(ValueError, match="immutable tuple"):
        route(harness)
    assert ledger.read()[-1]["payload"]["status"] == "rejected"


def test_control_context_owns_immutable_feature_pairs():
    from agg.controller.routing import ControlContext

    features = [["reasoning.predictive_entropy", 1]]
    proxies = ["reasoning.predictive_entropy"]
    control = ControlContext(Reasoning(predictive_entropy=1), features, proxies)
    features[0][1] = 9
    proxies.clear()
    assert control.features == (("reasoning.predictive_entropy", 1),)
    assert control.proxy_metrics == ("reasoning.predictive_entropy",)


def test_random_routing_seed_changes_selection_and_budget_zero_is_valid(tmp_path):
    from agg.controller.routing import ReferenceRouter, RoutingConfig

    _, _, harness = setup_router(tmp_path, mode="random", seed=1)
    assert route(harness).selected_ids == ("b",)
    harness.router = ReferenceRouter(RoutingConfig(mode="random", seed=7))
    assert route(harness).selected_ids == ("a",)
    harness.router = ReferenceRouter(RoutingConfig(max_active=0))
    assert route(harness).selected_ids == ()
    harness.router = ReferenceRouter(RoutingConfig(minimum_score=0.9))
    assert route(harness, state=2).selected_ids == ()


def test_selector_cannot_change_ids_between_validation_and_audit(tmp_path):
    _, ledger, harness = setup_router(tmp_path)

    class ChangingSelection:
        reads = 0
        scores = (("a", 1.0), ("b", 0.0))
        rationale = (("a", "test"), ("b", "test"))

        @property
        def selected_ids(self):
            self.reads += 1
            return ("a",) if self.reads == 1 else ("bad",)

    class ChangingRouter:
        def configuration(self):
            return {"kind": "changing-test"}

        def select(self, control, eligible):
            return ChangingSelection()

    harness.router = ChangingRouter()
    with pytest.raises(ValueError, match="RoutingSelection"):
        route(harness)
    assert ledger.read()[-1]["payload"]["status"] == "rejected"
