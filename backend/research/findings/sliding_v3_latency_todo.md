# V3 latency research and execution checklist

This is a follow-up to `sliding_v3_implementation.md`. Historical results there
remain historical; compare new optimizations within the new paired experiment.

## Measured outcome (L4, batch 1,024)

Artifact: `artifacts/sliding_v3/encoder_profile/result.json`; Chrome trace alongside
it. Official prepared training prefix, 73,728 events, 8 warmup and 64 measured
batches per arm, resident token bank, seeded weights, inference only. The final
artifact's package source hashes all matched the local Python sources. GC is
disabled consistently for timed regions. These are batch completion times, not
event arrival-to-alert latency; include batching delay separately in deployment.

| Arm | Median ms | p95 ms | p99 ms | Interpretation |
|---|---:|---:|---:|---|
| Existing corrected V3 | 23.996 | 34.989 | 44.115 | Paired baseline |
| Frozen FP32 lookup | 21.384 | 22.383 | 24.221 | Recommended opt-in; same weights |
| Lookup + static short read | 22.376 | 23.548 | 26.189 | No gain over lookup in this run |
| Lookup + static + temporal graphs | 20.781 | 24.340 | 222.470 | Tail regression; experimental only |
| Lookup + attention-only short read | 20.149 | 21.199 | 22.783 | Changes architecture; retrain |
| Lookup + no short read contribution | 19.585 | 20.530 | 21.774 | Changes architecture; retrain |
| Lookup + causal attention + attention-only short read | 19.988 | 20.546 | 20.927 | Changes architecture/parameters; retrain |

Frozen lookup reduces median latency by 10.9% and p99 by 45.1% in this paired
run. It precomputes 1,849 identities into 473,344 bytes of FP32 storage. Maximum
absolute logit error across all chronological batches is 1.94e-7 for lookup,
static reads and temporal graphs. This is floating-point equivalence, not bitwise
equivalence or a claim that every decision at an exact threshold is unchanged.

Separate encoder CUDA-event instrumentation across the stream found median
2.815 ms and median 136 unique identities per batch (excluding the profiler-heavy
batch and warmups from the timing median). Repeated identities do not make the
uncached encoder free. These spans include host-launch gaps; they are not summed
kernel compute. The profiled baseline batch launched 1,005 GPU kernels. A trace
has profiling overhead, so its idle fraction must not be treated as the production
idle fraction. The remaining overhead includes planner sorting/indexing, counters,
affine-scan launches and state updates; changing only CNN FLOPs cannot remove it.

Field-count p99 is 6; stored value-token p99 is 51 **including ordered bigrams**.
These describe this prefix only, not uncapped raw vocabulary or full training
coverage. The previously suggested 16-token cap is therefore not a measured safe
replacement. Vocabulary and staging work remains listed below.

An earlier paired run measured lower absolute times (~17 ms baseline), so do not
mix runs to claim a larger speedup. The graph arm also had a major tail outlier in
that run. Investigate allocation/capture/runtime tails before enabling it in a
latency-sensitive deployment. Static reads are workload-dependent and stay opt-in.

Validation: 86 passed, one pre-existing archived-fixture test deselected. New tests
cover frozen-cache outputs/invalidation/gradient bypass, mixed-chunk static-read
outputs and gradients, causal attention padding/future independence, and graph
CPU/training fallback. The remote benchmark asserts GPU output parity for all
72 chronological batches for each execution-only arm. No full accuracy ablation
has been run in this follow-up.

## Completed implementation

- [x] Inspect actual encoding and temporal paths instead of assuming costs from MACs.
- [x] Frozen inference identity table, FP32 by default, chunked preparation, explicit
  opt-in (`--frozen-eval`). Training and eval-with-grad still encode differentiably.
  Weight updates, checkpoint loading, device/dtype changes and training invalidate
  the table. It is runtime state, not an extra learned parameter/checkpoint tensor.
- [x] Static short-memory read (`static_short_read`): remove the data-dependent
  nonzero/active-index/scatter path. Mask empty histories; retain the original path
  for comparison. Static padding can do extra work when most histories are empty.
- [x] Inference-only CUDA graph capture for the temporal region (`temporal_graphs`),
  with at most two shape entries, eager fallback and independent output storage.
  Requires static short reads. The planner remains eager.
- [x] Attention-only compressed-ring readout (`short_readout="attention"`) and
  disabled readout (`"none"`) for retraining experiments. The latter still maintains
  rings: it is a contribution ablation, not complete removal of memory bookkeeping.
- [x] Full causal SDPA temporal alternative (`temporal="attention"`), with causal
  and padding masks, finite empty rows, and future-independence testing.
- [x] Paired real-prefix latency benchmark, CUDA trace, encoder identity counts,
  field/subtoken percentiles, and encoder CUDA-event spans separate from timing.
- [x] Three-seed accuracy experiment driver (`python -m sliding_node_mem_v3.ablate`).

## Remaining experiments, not claimed fixed

- [ ] Run full official-split retraining across three seeds; report AP, calibrated
  node precision/recall/F1, latency and memory. The seeded-weight prefix benchmark
  cannot establish any accuracy improvement or even non-regression at a threshold.
- [ ] Factor relation out of offline identities; this changes normalization and
  needs a new token format/checkpoint plus retraining. Removing a field changes
  every attention weight's softmax denominator, despite field-local attention logits.
- [ ] Evaluate multiscale affine memory with an explicit memory/parameter budget.
  The existing fast/slow channels are not a five-timescale bank.
- [ ] Measure full benign-training vocabulary and collisions before introducing
  path normalization or OOV embeddings. Hash occupancy alone cannot detect novel
  strings colliding with occupied buckets. Normalizing paths can erase evidence.
- [ ] Whole-forward CUDA graph: requires bounded static planner shapes and removal
  of dynamic unique/nonzero/allocation operations. Partial temporal capture does
  not solve these remaining planner costs.
- [ ] Compare memmap vs resident staging in the new profiler; current paired arms
  all use a resident bank. Do not attribute their difference to memmap savings.

## Research conclusions

The encoder already calls MultiheadAttention with `need_weights=False`, which
uses optimized SDPA when available ([PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)).
Replacing that call alone is not a new optimization. Current token fields and
subtokens are padded to bank-wide fixed widths; unique identity row count and
planner/short-read sizes are the dynamic dimensions. Do not silently truncate to
prefix p99: that is not a training-only coverage measurement.

[CUDA graph guidance](https://docs.pytorch.org/docs/stable/notes/cuda) supports
capturing safe regions when dynamic control flow, shapes or CPU synchronization
prevent whole-network capture. The implementation follows that narrower approach.

[Linear attention](https://proceedings.mlr.press/v119/katharopoulos20a.html) changes
the attention formulation to obtain linear sequence complexity. At nine slots,
this does not establish a speed advantage over full SDPA; measure actual kernels.
[S5](https://arxiv.org/abs/2208.04933) provides evidence for parallel scans in state
space models, not evidence that a lossy state will preserve this detector's F1.
A GRU is sequential across updates to the same entity; collapsing messages is an
aggregation-policy choice, not an unavoidable mathematical property of a GRU.

Parameter counts and theoretical peak FLOPs are not timings. A fitted batch-time
intercept is not a measurement of GPU idle time. Kernel durations, CUDA-event spans
(which may include launch gaps), and synchronized wall time must be kept separate.
