import pytest
import torch

from agg.compression import PrecisionConfig, fake_quantize, low_rank, magnitude_prune
from agg.credit import assign_credit, discounted_returns
from agg.storage import benchmark_codecs, choose_storage, decode_ternary, encode_ternary
from agg.supervision import EvidencePolicy, Signal, ToyEnvironment, combine_evidence


def test_execution_distinguishes_intent_failure_and_observation():
    env = ToyEnvironment(3)
    first = env.execute("add", 2, intent="reach 100", expected_state=100)
    assert first.pre_state == 3 and first.post_state == 5
    assert not first.verified and first.error is None
    second = env.execute("divide", 0, intent="divide", expected_state=5)
    assert second.error and second.post_state == 5 and not second.verified
    assert first.post_hash == second.pre_hash
    assert [e.order for e in env.trace] == [0, 1]
    assert first.verifier_provenance and first.verifier_confidence == 1


def test_strong_verified_signal_dominates_many_teachers():
    signals = [Signal("external_outcome", -1, 1, 0, "test")]
    signals += [Signal("teacher", 1, 1, 0, "teacher")] * 100
    combined = combine_evidence(signals)
    assert combined.value < -0.5 and combined.conflicts
    assert combined.primary_source == "external_outcome"
    assert combine_evidence(signals, EvidencePolicy(auxiliary_bound=0)).value == -1
    with pytest.raises(ValueError):
        Signal("teacher", float("nan"), 1, 0, "bad")


def test_discount_endpoints_and_trace():
    assert discounted_returns([1, 2, 3], 0) == [1, 2, 3]
    assert discounted_returns([1, 2, 3], 1) == [6, 5, 3]
    signals = [Signal("external_outcome", 1, 1, 2, "verifier")]
    records = assign_credit(signals, 3, gamma=0.5)
    assert [r.credit for r in records] == [0.25, 0.5, 1]
    assert records[0].gamma == 0.5
    with pytest.raises(ValueError):
        discounted_returns([1], 1.1)


@pytest.mark.parametrize("precision", ["FP32", "FP16", "BF16", "INT8", "INT4", "ternary"])
def test_quantization_and_proposals(precision):
    x = torch.tensor([[-2.0, -0.1], [0.3, 1.0]])
    before = x.clone()
    q = fake_quantize(x, precision)
    assert q.dtype == x.dtype and torch.isfinite(q).all()
    assert torch.equal(x, before)
    assert torch.count_nonzero(magnitude_prune(x, 0.5)) == 2
    assert torch.linalg.matrix_rank(low_rank(x, 1)) == 1
    assert PrecisionConfig(weight=precision).activation == "FP32"
    if precision == "ternary":
        assert q.unique().numel() <= 3


@pytest.mark.parametrize("codec", ["fixed_ternary", "bitcos_like", "dense"])
@pytest.mark.parametrize("shape", [(0,), (), (3, 7), (100,)])
def test_codec_lossless_and_validated(codec, shape):
    x = torch.randint(-1, 2, shape)
    blob = encode_ternary(x, codec)
    assert torch.equal(decode_ternary(blob), x.to(torch.int8))
    assert len(blob) > 0
    broken = blob[:-1]
    with pytest.raises(ValueError):
        decode_ternary(broken)
    with pytest.raises(ValueError):
        encode_ternary(torch.tensor([2]), codec)


def test_benchmark_measures_metadata_and_objective():
    x = torch.zeros(32, 32)
    records = benchmark_codecs(x, repeats=2)
    assert len(records) == 3
    assert all(r.serialized_bytes == len(encode_ternary(x, r.codec)) for r in records)
    assert all(r.decode_seconds > 0 and r.matmul_seconds > 0 for r in records)
    assert all(r.hardware["device"] == "cpu" and r.zero_density == 1 for r in records)
    assert choose_storage(records, "bytes").codec == "bitcos_like"
    assert choose_storage(records, "latency").matmul_seconds == min(
        r.matmul_seconds for r in records
    )


def test_dense_wins_for_tiny_tensor_and_invalid_measurements_fail():
    records = benchmark_codecs(torch.zeros(1, 1), repeats=1)
    assert choose_storage(records, "bytes").codec == "dense"
    with pytest.raises(ValueError):
        choose_storage(records, "imaginary")


def test_corrupt_payload_with_recomputed_checksum_is_still_rejected():
    import hashlib
    import json
    import struct

    blob = encode_ternary(torch.tensor([1]), "fixed_ternary")
    header_length = struct.unpack("<I", blob[5:9])[0]
    body = bytearray(blob[:-32])
    body[9 + header_length] = 255
    with pytest.raises(ValueError, match="base-3"):
        decode_ternary(bytes(body) + hashlib.sha256(body).digest())
    metadata = json.loads(blob[9 : 9 + header_length])
    metadata["shape"] = [100]
    header = json.dumps(metadata).encode()
    body = bytearray(b"AGGT1" + struct.pack("<I", len(header)) + header + b"\x02")
    with pytest.raises(ValueError, match="shape"):
        decode_ternary(bytes(body) + hashlib.sha256(body).digest())


def test_precision_invalid_numerics_and_proposal_endpoints():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert torch.equal(magnitude_prune(x, 0), x)
    assert not torch.count_nonzero(magnitude_prune(x, 1))
    assert not torch.count_nonzero(low_rank(x, 0))
    with pytest.raises(ValueError):
        fake_quantize(torch.tensor([float("nan")]), "INT4")
    with pytest.raises(ValueError):
        fake_quantize(torch.tensor([1e10]), "FP16")


def test_integer_environment_rejects_noninteger_states_and_arguments():
    with pytest.raises(ValueError):
        ToyEnvironment(1.5)
    env = ToyEnvironment()
    with pytest.raises(ValueError):
        env.execute("add", 0.5, intent="add", expected_state=1)
    assert not env.trace and env.state == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16, torch.float64])
@pytest.mark.parametrize("precision", ["INT8", "INT4", "ternary"])
def test_quantization_preserves_finite_subnormal_endpoints(dtype, precision):
    tiny = torch.nextafter(torch.zeros((), dtype=dtype), torch.ones((), dtype=dtype))
    x = torch.stack((-tiny, torch.zeros_like(tiny), tiny))
    result = fake_quantize(x, precision)
    assert result.dtype == dtype
    assert torch.isfinite(result).all()
    assert torch.equal(result, x)


def test_quantization_accepts_gradients_without_scalar_conversion_warning():
    import warnings

    x = torch.tensor([0.0, 1e-7], dtype=torch.float16, requires_grad=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = fake_quantize(x, "INT8")
    assert torch.isfinite(result).all()
    assert result.requires_grad
