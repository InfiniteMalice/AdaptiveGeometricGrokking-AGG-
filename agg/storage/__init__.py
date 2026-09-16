"""CPU reference codecs, with metadata-inclusive costs and measured dense fallback.

BITCOS-like means bitmap plus packed signs; this is not a BITCOS-compatible kernel.
Every serialized object includes a versioned JSON header and SHA256 corruption check.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import statistics
import struct
import time
from dataclasses import dataclass

import torch
from torch import Tensor

CODECS = ("fixed_ternary", "bitcos_like", "dense")
MAGIC = b"AGGT1"


def _pack_bits(bits: list[int]) -> bytes:
    result = bytearray((len(bits) + 7) // 8)
    for index, bit in enumerate(bits):
        result[index // 8] |= bit << (index % 8)
    return bytes(result)


def _unpack_bits(payload: bytes, count: int) -> list[int]:
    if len(payload) != (count + 7) // 8:
        raise ValueError("Invalid bitmap length")
    if count % 8 and payload[-1] >> (count % 8):
        raise ValueError("Noncanonical bitmap padding")
    return [(payload[i // 8] >> (i % 8)) & 1 for i in range(count)]


def encode_ternary(tensor: Tensor, codec: str = "fixed_ternary") -> bytes:
    if codec not in CODECS:
        raise ValueError("Unknown codec")
    values = tensor.detach().cpu().reshape(-1)
    if not bool(((values == -1) | (values == 0) | (values == 1)).all()):
        raise ValueError("Ternary symbols must be exactly -1, 0 or 1")
    symbols: list[int] = values.to(torch.int8).tolist()
    if codec == "dense":
        payload = bytes(value + 1 for value in symbols)
    elif codec == "fixed_ternary":
        packed = bytearray()
        for start in range(0, len(symbols), 5):
            packed.append(sum((s + 1) * 3**i for i, s in enumerate(symbols[start : start + 5])))
        payload = bytes(packed)
    else:
        payload = _pack_bits([int(s != 0) for s in symbols])
        payload += _pack_bits([int(s > 0) for s in symbols if s != 0])
    metadata = {
        "codec": codec,
        "shape": list(tensor.shape),
        "count": len(symbols),
        "payload_bytes": len(payload),
        "version": 1,
    }
    header = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("ascii")
    body = MAGIC + struct.pack("<I", len(header)) + header + payload
    return body + hashlib.sha256(body).digest()


def decode_ternary(blob: bytes) -> Tensor:
    if len(blob) < 41 or blob[:5] != MAGIC:
        raise ValueError("Invalid ternary envelope")
    body, digest = blob[:-32], blob[-32:]
    if hashlib.sha256(body).digest() != digest:
        raise ValueError("Corrupt ternary checksum")
    header_length = struct.unpack("<I", body[5:9])[0]
    if header_length > len(body) - 9:
        raise ValueError("Truncated header")
    try:
        metadata = json.loads(body[9 : 9 + header_length])
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("Invalid metadata") from exc
    if not isinstance(metadata, dict) or set(metadata) != {
        "codec",
        "shape",
        "count",
        "payload_bytes",
        "version",
    }:
        raise ValueError("Invalid metadata fields")
    shape = metadata["shape"]
    count = metadata["count"]
    codec = metadata["codec"]
    if (
        metadata["version"] != 1
        or codec not in CODECS
        or not isinstance(shape, list)
        or not all(type(d) is int and 0 <= d <= 2**31 for d in shape)
        or len(shape) > 64
        or type(count) is not int
        or math.prod(shape) != count
    ):
        raise ValueError("Invalid shape, count or codec")
    payload = body[9 + header_length :]
    if len(payload) != metadata["payload_bytes"]:
        raise ValueError("Invalid payload length")
    symbols: list[int] = []
    if codec == "dense":
        if len(payload) != count or any(s > 2 for s in payload):
            raise ValueError("Invalid dense symbols")
        symbols = [s - 1 for s in payload]
    elif codec == "fixed_ternary":
        if len(payload) != (count + 4) // 5 or any(s >= 243 for s in payload):
            raise ValueError("Invalid base-3 payload")
        for value in payload:
            for _ in range(min(5, count - len(symbols))):
                symbols.append(value % 3 - 1)
                value //= 3
            if value:
                raise ValueError("Noncanonical base-3 padding")
    else:
        bitmap_size = (count + 7) // 8
        present = _unpack_bits(payload[:bitmap_size], count)
        signs = iter(_unpack_bits(payload[bitmap_size:], sum(present)))
        symbols = [2 * next(signs) - 1 if bit else 0 for bit in present]
    return torch.tensor(symbols, dtype=torch.int8).reshape(shape)


@dataclass(frozen=True)
class StorageBenchmark:
    codec: str
    shape: tuple[int, ...]
    zero_density: float
    theoretical_bits_per_weight: float
    serialized_bytes: int
    encode_seconds: float
    decode_seconds: float
    matmul_seconds: float
    throughput_weights_per_second: float
    hardware: dict[str, str]
    kernel: str = "Python decode + torch CPU FP32 dense matmul"
    measured: bool = True


def benchmark_codecs(tensor: Tensor, repeats: int = 5) -> list[StorageBenchmark]:
    """Median CPU wall time after warmup; matmul includes decode and dtype conversion.

    Uses one fixed all-ones RHS vector. No sparse-kernel speedup or full-training
    end-to-end performance is inferred. Encode/decode overhead includes validation.
    """
    if tensor.ndim != 2 or tensor.numel() == 0 or repeats < 1:
        raise ValueError("Benchmark requires a nonempty matrix and positive repeats")
    matrix = tensor.detach().cpu()
    rhs = torch.ones(matrix.shape[1], 1)
    zero_density = float((matrix == 0).float().mean())
    hardware = {
        "device": "cpu",
        "model": platform.processor() or platform.machine(),
        "architecture": platform.machine(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_threads": str(torch.get_num_threads()),
    }
    records = []
    for codec in CODECS:
        blob = encode_ternary(matrix, codec)
        decoded = decode_ternary(blob)
        if not torch.equal(decoded, matrix.to(torch.int8)):
            raise RuntimeError("Codec failed pre-benchmark round trip")
        decoded.float() @ rhs
        timings: list[list[float]] = [[], [], []]
        for _ in range(repeats):
            start = time.perf_counter()
            encode_ternary(matrix, codec)
            timings[0].append(time.perf_counter() - start)
            start = time.perf_counter()
            decode_ternary(blob)
            timings[1].append(time.perf_counter() - start)
            start = time.perf_counter()
            decode_ternary(blob).float() @ rhs
            timings[2].append(time.perf_counter() - start)
        encode, decode, matmul = (statistics.median(t) for t in timings)
        bits = {"dense": 8.0, "fixed_ternary": 8 / 5, "bitcos_like": 2 - zero_density}[codec]
        records.append(
            StorageBenchmark(
                codec,
                tuple(matrix.shape),
                zero_density,
                bits,
                len(blob),
                encode,
                decode,
                matmul,
                matrix.numel() / matmul,
                dict(hardware),
            )
        )
    return records


def choose_storage(records: list[StorageBenchmark], objective: str = "bytes") -> StorageBenchmark:
    """Choose solely from measured candidates under the explicit objective."""
    if not records or objective not in ("bytes", "latency", "decode"):
        raise ValueError("Measured candidates and bytes/latency/decode objective required")
    if any(
        not r.measured
        or not math.isfinite(r.matmul_seconds)
        or r.matmul_seconds <= 0
        or not math.isfinite(r.decode_seconds)
        or r.decode_seconds <= 0
        or r.serialized_bytes < 0
        for r in records
    ):
        raise ValueError("Invalid measurements")
    return min(
        records,
        key=lambda r: {
            "bytes": float(r.serialized_bytes),
            "latency": r.matmul_seconds,
            "decode": r.decode_seconds,
        }[objective],
    )
