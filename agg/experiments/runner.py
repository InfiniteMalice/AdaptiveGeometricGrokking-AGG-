import copy
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
from torch import nn

from agg.consolidation import InterventionPolicy, Proposal, state_hash, trial
from agg.evaluation.model import make_evaluator
from agg.gating import ScalarGate
from agg.geometry import AdaptedModel, GeometryAdapter
from agg.ledger import Ledger
from agg.models import TinyTransformer
from agg.probes import retrieval_crossing
from agg.storage import benchmark_codecs, choose_storage, decode_ternary, encode_ternary
from agg.tasks import TaskData, hierarchy, modular_addition, retrieval
from agg.telemetry import TelemetryCollector, TelemetryHistory
from agg.training import evaluate, seed_all, train

from .candidates import fit_candidate, proposals
from .config import ExperimentConfig, Features
from .diagnostics import coding_proxy, execution_demo, horizon_sweep, observe


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def make_data(config: ExperimentConfig) -> TaskData:
    if config.independent_evaluation:
        return evaluation_protocol(config).development()
    data_seed = config.training.seed if config.data_seed is None else config.data_seed
    if config.task == "modular":
        return modular_addition(config.modulus, seed=data_seed)
    if config.task == "hierarchy":
        return hierarchy(depth=config.tree_depth, seed=data_seed)
    return retrieval(
        samples=config.samples,
        length=config.context_length,
        distance=config.distance,
        density=config.density,
        hard=config.hard_distractors,
        seed=data_seed,
    )


def evaluation_protocol(config: ExperimentConfig):
    from agg.tasks.protocol import make_protocol

    if not config.independent_evaluation or config.data_seed is None:
        raise ValueError("independent evaluation and data_seed must be configured")
    return make_protocol(
        config.task,
        seed=config.data_seed,
        modulus=config.modulus,
        depth=config.tree_depth,
        samples=config.samples,
        length=config.context_length,
        distance=config.distance,
        density=config.density,
        hard=config.hard_distractors,
        selection_samples=config.selection_samples,
    )


def _gate_ablation(model: nn.Module, data: TaskData) -> list[dict[str, Any]]:
    from contextlib import ExitStack

    gates = [module for module in model.modules() if isinstance(module, ScalarGate)]
    records: list[dict[str, Any]] = []
    if not gates:
        return records
    for value in (0.0, 0.25, 0.5, 0.75, 1.0):
        with ExitStack() as stack:
            for gate in gates:
                stack.enter_context(gate.force(value))
            records.append(
                {"gate": value, "id": evaluate(model, data.id), "ood": evaluate(model, data.ood)}
            )
    return records


def _storage_demonstration(output: Path, objective: str) -> dict[str, Any]:
    tiny = torch.ones(1, 1)
    sparse = torch.zeros(32, 32)
    sparse[0, 0], sparse[2, 1] = 1, -1
    illustrative = {}
    for name, tensor in (("tiny", tiny), ("sparse", sparse)):
        records = benchmark_codecs(tensor)
        illustrative[name] = {
            "records": [asdict(r) for r in records],
            "selected_bytes": choose_storage(records, "bytes").codec,
            "selected_latency": choose_storage(records, "latency").codec,
            "scope": "illustrative exact ternary fixture, not a compressed model",
        }
    return {"objective": objective, "illustrative": illustrative, "accepted_model_tensors": []}


def run_experiment(config: ExperimentConfig, output: Path) -> dict[str, Any]:
    """Run a seeded trajectory, then independent candidate trials at its checkpoint.

    ID/OOD splits here are development validation sets used for candidate selection.
    Confirmatory claims require new seeds and untouched test sets; see docs.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "experiment.json", asdict(config))
    seed_all(config.training.seed)
    data = make_data(config)
    model: nn.Module = TinyTransformer(
        data.vocab_size,
        data.classes,
        width=config.training.width,
        layers=config.training.layers,
        heads=config.training.heads,
        max_length=data.train.x.shape[1],
    )
    if config.adapt_during_training:
        if not (config.features.geometry or config.features.dimension or config.features.gating):
            raise ValueError("adapt_during_training requires an adapter mechanism")
        if not isinstance(model, TinyTransformer):
            raise TypeError("initial adapter requires the baseline Transformer")
        model = AdaptedModel(
            model,
            GeometryAdapter(
                config.training.width,
                config.dimensions[0] if config.features.dimension else config.training.width,
                config.geometries[0] if config.features.geometry else "euclidean",
                config.gate if config.features.gating else 1.0,
                config.learned_gate if config.features.gating else False,
                config.curvature,
                config.learned_curvature,
            ),
        )
    telemetry = TelemetryHistory(output / "telemetry.jsonl")
    phase_rows = []
    triggers = []
    policy = InterventionPolicy(config.training.steps)

    def on_checkpoint(step: int, current: nn.Module, metrics: dict[str, Any]) -> None:
        if config.independent_evaluation:
            from .checkpoints import save_model

            save_model(current, output / "training" / f"inference-{step}.pt")
        raw: dict[str, Any] = {"task": metrics["train"], "id": metrics["id"], "ood": metrics["ood"]}
        hidden = None
        if config.features.telemetry:
            mode = config.telemetry_mode
            if mode != "task":
                observed, hidden = observe(
                    current,
                    data,
                    attribution=mode in {"full", "attribution"},
                    memorization=mode in {"full", "attribution"},
                    gates=mode in {"full", "gates_horizon"},
                )
                raw.update(observed)
                if "retrieval" in raw:
                    phase_rows.append(
                        {
                            "step": step,
                            "M": raw["memorization"]["score"],
                            "R": raw["retrieval"]["score"],
                            "G": raw["generalization"]["score"],
                        }
                    )
                if mode in {"full", "gates_horizon"}:
                    raw["horizon"] = horizon_sweep(current, data)
                allowed = {
                    "geometry": {"task", "id", "ood"},
                    "geometry_derivatives": {"task", "id", "ood"},
                    "attribution": {
                        "task",
                        "id",
                        "ood",
                        "attribution",
                        "memorization",
                        "retrieval",
                        "generalization",
                    },
                    "gates_horizon": {"task", "id", "ood", "gates", "horizon"},
                }
                if mode in allowed:
                    raw = {key: value for key, value in raw.items() if key in allowed[mode]}
                if mode in {"attribution", "gates_horizon"}:
                    hidden = None
            if config.features.epiplexity:
                raw["complexity"] = coding_proxy(current, data, config.training.seed)
            snapshots = TelemetryCollector().collect(
                step,
                f"checkpoint-{step}",
                raw,
                hidden,
                derivatives=mode in {"full", "geometry_derivatives"},
            )
            for snapshot in snapshots:
                telemetry.append(snapshot)
        if config.features.consolidation:
            event = policy.propose(step, raw)
            if event:
                triggers.append(event)

    trained = train(data, config.training, output / "training", on_checkpoint, model)
    model = trained.model
    model.eval()
    with torch.no_grad():
        _, baseline_hidden = model(data.id.x, return_hidden=True)
        reference_hidden = baseline_hidden[-1][:, -1].detach().clone()
    evaluator = make_evaluator(model, data)

    anchor = evaluator(model)
    recorder = None
    if config.independent_evaluation:
        from .checkpoints import save_model
        from .independent import CandidateRecorder

        save_model(model, output / "baseline-state.pt")
        recorder = CandidateRecorder(
            output, parent_checkpoint=output / "training" / f"checkpoint-{config.training.steps}.pt"
        )
    retained = model
    ledger = Ledger(output / "ledger.jsonl")
    candidate_gates: list[dict[str, Any]] = []
    accepted_tensors: list[tuple[str, torch.Tensor]] = []
    precision_map: dict[str, str] = {}
    if config.features.consolidation:
        write_json(output / "triggers.json", triggers)
        # Equal additional-update comparator for distillation and representation fitting.
        if config.features.distillation or config.features.geometry or config.features.dimension:
            control = fit_candidate(copy.deepcopy(model), model, data, config, False)
            write_json(output / "continued_baseline.json", asdict(evaluator(control)))

        def measured(candidate: nn.Module) -> dict[str, Any]:
            mode = config.telemetry_mode
            if mode == "task":
                return {"id": evaluate(candidate, data.id), "ood": evaluate(candidate, data.ood)}
            observations, hidden = observe(
                candidate,
                data,
                attribution=mode in {"full", "attribution"},
                memorization=mode in {"full", "attribution"},
                gates=mode in {"full", "gates_horizon"},
            )
            if mode == "full":
                from .diagnostics import representation_diagnostics

                observations.update(
                    representation_diagnostics(hidden, reference_hidden=[reference_hidden])
                )
            elif mode in {"geometry", "geometry_derivatives"}:
                snapshots = TelemetryCollector().collect(
                    config.training.steps,
                    "candidate",
                    {},
                    hidden,
                    derivatives=mode == "geometry_derivatives",
                )
                observations = {
                    "geometry": [s.raw["geometry"] for s in snapshots],
                    "dimension": [s.raw["dimension"] for s in snapshots],
                    "derivatives": [s.derived for s in snapshots],
                }
            if mode in {"full", "gates_horizon"}:
                observations["horizon"] = horizon_sweep(candidate, data)
            if config.features.epiplexity:
                observations["complexity"] = coding_proxy(candidate, data, config.training.seed)
            return observations

        def execute(proposal: Proposal, base: nn.Module):
            result = trial(
                base,
                proposal,
                evaluator,
                config.constraints,
                ledger,
                run_id=output.name,
                step=config.training.steps,
                reference=anchor,
                observer=measured if config.features.telemetry else None,
                recorder=recorder,
                selection_gain_floor=config.selection_gain_floor,
            )
            if proposal.kind == "geometry" and result.accepted and config.features.gating:
                candidate_gates.extend(
                    {"candidate": proposal.state, **record}
                    for record in _gate_ablation(result.model, data)
                )
            return result

        def stage_proposals(base: nn.Module, stage: ExperimentConfig):
            return proposals(
                base,
                data,
                stage,
                optimizer_observer=recorder.record_optimizer if recorder is not None else None,
            )

        # Every stage compares to the original capability anchor. Within a stage,
        # independent candidates share the same input checkpoint and random state.
        if config.features.distillation:
            stage = replace(config, features=replace(Features.baseline(), distillation=True))
            for proposal in stage_proposals(retained, stage):
                result = execute(proposal, retained)
                retained = result.model
        if config.features.geometry or config.features.dimension or config.features.gating:
            stage = replace(
                config,
                features=replace(
                    config.features, pruning=False, quantization=False, distillation=False
                ),
            )
            candidates = []
            for proposal in stage_proposals(retained, stage):
                result = execute(proposal, retained)
                if result.accepted:
                    candidates.append((proposal.state["dimension"], result))
            if candidates:
                # Choose the smallest feasible bottleneck; physical model size is
                # evaluated separately and is not claimed to shrink at this stage.
                retained = min(
                    candidates,
                    key=lambda item: (
                        item[0],
                        -item[1].after.ood_accuracy,
                        -item[1].after.id_accuracy,
                    ),
                )[1].model
        if config.features.pruning:
            stage = replace(config, features=replace(Features.baseline(), pruning=True))
            for proposal in stage_proposals(retained, stage):
                retained = execute(proposal, retained).model
        if config.features.quantization:
            stage = replace(config, features=replace(Features.baseline(), quantization=True))
            grouped: dict[str, list[Proposal]] = {}
            for proposal in stage_proposals(retained, stage):
                grouped.setdefault(proposal.component, []).append(proposal)
            precision_order = {
                "FP32": 32.0,
                "FP16": 16.0,
                "BF16": 16.0,
                "INT8": 8.0,
                "INT4": 4.0,
                "ternary": 1.585,
            }
            for component, group in grouped.items():
                feasible = []
                for proposal in group:
                    result = execute(proposal, retained)
                    if result.accepted:
                        key = "weight" if proposal.kind == "precision" else "activation"
                        precision = proposal.state[key]
                        feasible.append((precision_order[precision], precision, result))
                if feasible:
                    _, precision, result = min(feasible, key=lambda item: item[0])
                    retained = result.model
                    precision_map[component] = precision
                else:
                    precision_map[component] = "FP32"
            # Activation wrappers prefix parameter names. Resolve the underlying
            # jointly validated weights rather than serializing isolated trials.
            weights = retained
            from .candidates import ActivationPrecisionModel

            while isinstance(weights, ActivationPrecisionModel):
                weights = weights.base
            for name, parameter in weights.named_parameters():
                if precision_map.get(name) == "ternary":
                    accepted_tensors.append((name, parameter.detach()))
            write_json(output / "precision-map.json", precision_map)
        # Include a fixed checkpoint gate intervention even when all fitted adapters fail.
        if config.features.gating and not candidate_gates:
            root = model.base if isinstance(model, AdaptedModel) else model
            if not isinstance(root, TinyTransformer):
                raise TypeError("gate probe requires TinyTransformer")
            probe = AdaptedModel(
                copy.deepcopy(root),
                GeometryAdapter(config.training.width, config.dimensions[0], gate=config.gate),
            )
            candidate_gates = [
                {"unaccepted_probe": True, **record} for record in _gate_ablation(probe, data)
            ]
    if config.features.gating:
        write_json(output / "gate-ablation.json", _gate_ablation(model, data) + candidate_gates)
    if config.features.storage:
        storage = _storage_demonstration(output, config.storage_objective)
        for index, (name, tensor) in enumerate(accepted_tensors):
            scale = float(tensor.abs().max())
            symbols = tensor.sign().to(torch.int8)
            records = benchmark_codecs(symbols)
            selected = choose_storage(records, config.storage_objective)
            filename = f"accepted-tensor-{index}.aggt"
            blob = encode_ternary(symbols, selected.codec)
            (output / filename).write_bytes(blob)
            reconstructed = (
                decode_ternary((output / filename).read_bytes()).to(tensor.dtype) * scale
            )
            if not torch.equal(reconstructed, tensor.cpu()):
                raise RuntimeError("selected physical encoding changed accepted numerical weights")
            dense = next(record for record in records if record.codec == "dense")
            fingerprint = state_hash(retained)
            ledger.append(
                {
                    "schema_version": "1.0",
                    "run_id": output.name,
                    "checkpoint": config.training.steps,
                    "component": name,
                    "intervention_type": "storage",
                    "accepted": True,
                    "reason": "selected measured objective; exact decoded tensor equality verified",
                    "proposed_state": {
                        "filename": filename,
                        "scale": scale,
                        "objective": config.storage_objective,
                    },
                    "state_before": fingerprint,
                    "state_retained": fingerprint,
                    "telemetry_before": {"storage": asdict(dense)},
                    "telemetry_after": {"storage": asdict(selected)},
                    "storage_format": selected.codec,
                    "zero_density": selected.zero_density,
                    "serialized_byte_delta": selected.serialized_bytes - dense.serialized_bytes,
                    "latency_delta": selected.matmul_seconds - dense.matmul_seconds,
                    "hardware": selected.hardware,
                    "id_delta": 0.0,
                    "ood_delta": 0.0,
                    "mechanism_delta": 0.0,
                    "parameter_delta": 0,
                    "sparsity_delta": 0.0,
                    "precision_delta": None,
                    "execution_evidence": "exact tensor round trip",
                    "scope": "codec tensor envelope; per-tensor scale kept in storage.json",
                }
            )
            storage["accepted_model_tensors"].append(
                {
                    "component": name,
                    "scale": scale,
                    "filename": filename,
                    "selected": selected.codec,
                    "records": [asdict(record) for record in records],
                    "scope": "jointly validated model tensor; lossless symbol packing",
                }
            )
        write_json(output / "storage.json", storage)
    f = config.features
    if f.executable or f.verified or f.teacher or f.process:
        write_json(
            output / "execution.json",
            execution_demo(
                executable=f.executable,
                verified=f.verified,
                teacher=f.teacher,
                process=f.process,
                gamma=config.gamma,
            ),
        )
    if f.epiplexity and not f.telemetry:
        write_json(output / "complexity.json", coding_proxy(retained, data, config.training.seed))
    crossing = None
    if phase_rows:
        crossing = asdict(
            retrieval_crossing(
                [r["step"] for r in phase_rows],
                [r["M"] for r in phase_rows],
                [r["R"] for r in phase_rows],
                margin=0.05,
                sustain=3,
            )
        )
        write_json(
            output / "phase.json",
            {
                "observations": phase_rows,
                "crossing": crossing,
                "proxy": True,
                "warning": "kNN agreement may fail to separate mechanisms",
            },
        )
    from .checkpoints import save_model

    save_model(retained, output / "selected-state.pt")
    summary = {
        "task": config.task,
        "seed": config.training.seed,
        "baseline": asdict(anchor),
        "selected": asdict(evaluator(retained)),
        "interventions": len(ledger.read()),
        "accepted": sum(r["accepted"] for r in ledger.read()),
        "crossing": crossing,
        "cost_definition": "active parameter count, not deployment bytes",
        "scientific_status": "software smoke/research run; hypotheses unestablished",
    }
    write_json(output / "summary.json", summary)
    if config.independent_evaluation:
        from .independent import freeze_run

        freeze_run(output, evaluation_protocol(config).manifest())
    return summary
