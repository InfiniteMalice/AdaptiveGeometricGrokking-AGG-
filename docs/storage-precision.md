# Logical precision and physical storage

Numerical acceptance precedes physical encoding. `PrecisionConfig` distinguishes weight, activation, compute and accumulator precision. `fake_quantize` simulates rounding into ordinary dense floating tensors; it does not install integer kernels or reduce serialized tensor size. Compute and accumulator fields describe targets; they do not emulate every low-precision accumulation step.

Weight candidates include FP32, FP16, BF16, INT8, INT4 and scaled ternary. Floating casts are restored to the original tensor dtype after rounding. Integer simulation is symmetric per tensor; ternary simulation uses a maximum-absolute scale and a threshold. Activation candidates round the final hidden representation before classification. Every accepted precision should be reported by component, and sensitive components may remain FP32.

`magnitude_prune` and `low_rank` return dense tensors. Zero entries and reduced numerical rank alone do not reduce allocated parameter storage. Candidate acceptance must separately check ID/OOD capability and mechanism constraints. Storage cost reduction is only mandatory where `require_cost_reduction` is configured.

## Reference ternary codecs

`encode_ternary` accepts exact symbols `{-1,0,1}`. `decode_ternary` returns int8 symbols. Scaled quantized weights require separately retained scale metadata before the decoded symbols reconstruct the numerical weights.

| Codec | Payload | Nominal payload bits/weight |
|---|---|---|
| `dense` | One byte per ternary symbol | 8 |
| `fixed_ternary` | Five base-three symbols per byte | 8/5, before final-byte padding |
| `bitcos_like` | Presence bitmap and packed signs of nonzero entries | `2 - zero_density`, before padding |

All codecs include a versioned JSON header, shape/count information, magic bytes and SHA256 integrity checksum. `serialized_bytes` counts the complete envelope. Header lengths and padding can dominate small tensors. The nominal crossover of payload-only formulae is not a selection rule for complete files or measured latency.

The name **BITCOS-like** denotes the local bitmap-plus-signs idea. Exact compatibility with any external BITCOS format or kernel is unverified. The `dense` codec is a byte-symbol storage reference, not a claim to native FP32 model storage.

`benchmark_codecs` warms up and records median CPU wall times for encoding, decoding, and decode-plus-FP32-dense-matmul with one fixed all-ones RHS. It records hardware/platform/software, shape, zero density and measured throughput. `choose_storage` can minimize measured bytes, latency or decode time; the experiment configuration exposes bytes/latency. Results depend on size, CPU, thread count and implementation overhead.

No accelerated sparse/ternary kernel, device memory-traffic measurement, GPU benchmark or whole-training throughput improvement is implied. For deployment comparisons, include scale metadata, surrounding model storage and end-to-end invocation overhead. Keep tiny cases where dense encoding wins and larger sparse cases where bitmap encoding wins; a valid negative result is that sparse encoding loses under the chosen objective.
