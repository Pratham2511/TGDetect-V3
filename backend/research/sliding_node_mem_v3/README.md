# Tensor-ring TGNN V3

V3 implements device-side active and short history rings, fixed-size count-min
rarity, offline token-ID memmaps, and the requested detection experiments.
There is no Python loop over events in `forward_batch`. Offline compilation
still performs JSON parsing, tokenization and negative sampling once.

## Execution and equivalence

The logical active history is eight slots per endpoint. Ring rows hold event ID,
time, endpoints, token identity, compression and selection metadata. Short history
stores eight compressed 64-dimensional events per endpoint. A dense device-side
node-to-slot map assigns compact storage only to observed nodes; storage grows
by capacity blocks rather than allocating every declared node's rings at startup.
The initial reserve is 16,384 nodes, configurable through `initial_capacity`.

The planner sorts endpoint occurrences within the batch and performs segmented
reads/writes. It preserves repeated endpoints, self-edges and equal-time ingestion
order. Reading only the pre-batch ring would miss earlier events in the same
batch. Short-memory reads also precede the current event's eviction writes.
Legacy sticky selection is propagated to the other endpoint even when it is
quiet. Compression is copied into rings; there are no `Record` or `TokenRef`
objects and no periodic token-block compaction.

Active window records are deduplicated and re-encoded with current weights.
The encoder accepts integer identities, pools already-tokenized rows, and makes
one learned encoder pass per unique identity. Memmap mode requires one batched
identity transfer plus token staging; resident token-bank mode keeps that bank
on the GPU. Neither mode performs online JSON serialization/tokenization.
GPU `unique`, sorting and variable-sized gathers still have launch and
synchronization costs; this is not a zero-overhead or CPU-free pipeline.

`profile="compatible"` preserves the existing V2 compatible architecture and
uses legacy single-hash rows. For exact counter equivalence on small fixtures,
use `exact_counts=True`: its dense table reproduces unsaturated per-node counts.
This does not reproduce LRU eviction after the old 262,144-entry limit. The
default count-min table is approximate: collisions can suppress rarity, and it
has no LRU eviction. It must not be called numerically equivalent to the old
counter. The default table is four rows of 262,144 int64 counters (8 MiB).

## Detection changes

The corrected profile uses:

- Four active-history slots reserved per endpoint, with deduplication and spare
  slots filled by recency.
- Two causal provenance GAT layers. The second is independent by default;
  `tied_gat=True` shares weights as an explicit capacity ablation.
- Two independently salted hashes summed from the same 5,303-by-64 embedding
  table, plus position-tagged adjacent subtokens for value order.
- Benign-path Word2Vec skip-gram/negative-sampling pretraining, followed by
  fitting the learned word vectors into the shared hash-sum table. This trains
  its own corpus vectors; it does not import ORTHRUS's original weights or
  implement fastText. [SGNS reference](https://arxiv.org/abs/1310.4546).
- Embedding learning rate 1e-4 versus 1e-5 for other parameters.
- A bilinear endpoint-pair head with sampled same-type destination negatives.
  All observed training pairs and identical destination attributes are excluded;
  ambiguous/absent negatives are masked. Unobserved real pairs can still be
  false negatives. Pair plausibility concerns endpoint attributes, not a separate
  learned embedding for every node ID.
- Relation CE + 0.1 pair loss + 0.01 contextual compression loss, with two-batch
  chronological unrolling. Inference calibrates relation CE plus 0.1 pair anomaly
  together, rather than training a head whose score is discarded.
- Explicit novelty/local-rarity admission into the short ring, decided once
  when observed. A differentiable novelty gate controls long-memory writes;
  repeated future top-half selections do not accumulate retention.
- The affine endpoint scan, dilated causal CNN, contextual compression and
  elapsed-time fast/slow decay (one-hour/one-day half-lives by default).

Short storage uses the newest eight admitted events; it is not the old V2
age-priority/exception-quota policy. The original GAT already has causal
shared-entity edges, rather than only star edges. Two layers expand propagation
along those edges; they do not turn an eight-event history into ORTHRUS's
20-neighbor entity graph or connect every disconnected historical component.

The default compatible model has 764,202 parameters. Corrected mode has 785,327:
two hashes add none, but the second GAT adds 17,028 and the pair head adds 4,097.
With tied GAT weights and a pair head, the count is 768,299.

Because both hashes use a summed shared table, swapping their bucket positions
also gives identical embeddings. Under independent uniform hashes, the full
collision probability is approximately 2/(B-1)^2, not exactly 1/B^2. Partial
bucket overlap and text truncation remain possible.

## Run

Compile once from the checked official manifest, then train on a GPU host:

```bash
PYTHONPATH=research python -m sliding_node_mem_v3.prepare \
  --manifest /path/to/official/prepared/manifest.json \
  --output artifacts/sliding_v3_prepared

PYTHONPATH=research python -m sliding_node_mem_v3.run \
  --prepared artifacts/sliding_v3_prepared \
  --output artifacts/sliding_v3_training --device cuda \
  --epochs 5 --batch 1024 --unroll-batches 2 --resident-tokens
```

The runner automatically pretrains embeddings unless `--initialization` supplies
a matching artifact or `--random-init` explicitly requests an ablation. The
pretraining corpus contains paths only from nodes observed in benign training;
validation/test text does not enter that corpus or negative-sampling candidates.
`prepare --compatible` compiles legacy rows for parity experiments.

Training writes source copies, best weights, resumable state, binary edge scores
and metrics. Resume checks input, initialization, source and configuration hashes.
The first validation third selects the best benign relation-CE checkpoint; the
remaining thirds fit relation calibration and the alert threshold. Alert scores
preserve extreme evidence and have no top-30 cap. The empirical threshold budget
is per node-window and does not guarantee a whole-stream false-positive rate.

```bash
PYTHONPATH=research python -m pytest -q research/test_sliding_v3.py
PYTHONPATH=research modal run research/modal_sliding_v3_benchmark.py
```

## Interpretation

See [implementation report](../findings/sliding_v3_implementation.md) for measured
latency, tests and remaining limitations. The real-data benchmark uses prepared
CADETS training prefixes and seeded weights, with eight warmup and 64 measured
batches. It includes encoding, ID staging, state updates and maintenance; offline
compilation is timed separately. It excludes input ingestion, arrival waiting,
optimizer steps and alert scoring. Synthetic CUDA training checks gradients and
snapshot replay; it is not a held-out accuracy experiment.

Full corrected-model CADETS retraining/F1 evaluation has not been completed.
Small GPU batches can be slower than V2, and ring storage can use more memory
than V2 on a short prefix. Slot-map storage is proportional to declared nodes;
ring storage is proportional to reserved/observed nodes. Capacity growth and
full multi-day state still need operational profiling. Two-hash encoding,
count-min rarity, deeper GAT, pretraining, the pair objective and novelty retention
need separate ablations before attributing accuracy gains to any one change.
# Latency follow-up

Frozen evaluation can use an FP32 identity lookup without changing learned weights:
pass `--frozen-eval` to the training runner, or call
`model.eval(); model.event_encoder.precompute()` before `torch.no_grad()` inference.
Preparation costs and table memory must be included in deployment sizing. At one
million identities FP32 is 256 MB; explicitly selecting FP16 halves this but is a
precision ablation. Normal training, parameter updates, state loading and device
changes invalidate the table; do not mutate parameters through `.data`.

`--model-options '{"static_short_read":true}'` enables masked fixed-batch short
reads. `"temporal_graphs":true` additionally captures the temporal region on CUDA;
it remains experimental and does not capture the dynamic planner. Graph caches
are bounded to two shapes and bypassed during training or gradient-enabled eval.

Architecture ablations require retraining: `"short_readout":"attention"` skips
the second temporal pass; `"short_readout":"none"` disables the read contribution
while retaining ring maintenance; `"temporal":"attention"` replaces the CNN with
one causal SDPA layer. Defaults preserve the existing architecture.

Run the paired real-prefix benchmark with
`modal run research/modal_sliding_v3_profile.py`. Run official-split accuracy
ablations with `PYTHONPATH=research python -m sliding_node_mem_v3.ablate
--prepared PATH --output NEW_PATH --device cuda` (default three seeds, five epochs).
The latter is a full training experiment, not a completed accuracy result.

See [research checklist](../findings/sliding_v3_latency_todo.md) for research sources,
completed work, limitations and remaining experiments.

