# Tensor-ring TGNN implementation, explanations and validation

Implementation: [V3 README](../sliding_node_mem_v3/README.md),
[model](../sliding_node_mem_v3/model.py), [ring planner](../sliding_node_mem_v3/rings.py),
[offline tokens](../sliding_node_mem_v3/tokens.py),
[preparation](../sliding_node_mem_v3/prepare.py),
[pretraining](../sliding_node_mem_v3/pretrain.py),
[training/scoring](../sliding_node_mem_v3/run.py).

## Implemented changes

| Request | Implementation |
|---|---|
| Replace deques/Records with GPU history rings | Eight logical slots per endpoint, GPU node-to-slot map, vectorized chronological planning including repeated batch endpoints |
| Replace TokenRef short queues | Eight compressed 64-dimensional slots per endpoint; gather/deduplicate on device; no global compaction |
| Bound rarity memory | Four-row count-min-style table, vectorized within-batch count prefixes; exact dense counter option for unsaturated parity fixtures |
| Offline tokenization | Ragged token-ID memmap, event identity index, optional resident GPU bank; one encoder pass per unique identity |
| Keep current-weight window encoding | Active observed event identities deduplicated and re-encoded each batch |
| Endpoint quota | Four active history slots per endpoint, deduplicate, then fill unused slots by recency |
| Deeper graph processing | Two independent causal GAT layers by default, with tied weights available as an ablation |
| Two hashes without larger token table | Independent salted hashes summed from the same embedding table; ordered adjacent subtokens retained |
| Pretrained token table | Word2Vec SGNS on paths of benign training nodes, followed by fitting vectors into the shared hash table |
| Higher embedding learning rate | 1e-4 for embeddings; 1e-5 for other model weights |
| Pair objective | Bilinear endpoint-attribute plausibility head; same-type corrupted destinations; known training pairs and identical attribute identities excluded |
| Remove repeated hard top-half retention | One novelty/rarity storage decision at observation; differentiable novelty gate for long-memory writes |
| Preserve core temporal architecture | Affine endpoint scan, dilated causal CNN, batched windows and elapsed-time fast/slow memory decay |

Default compatible count: 764,202 parameters. Corrected: 785,327. Two hashes
add no parameters; the second GAT adds 17,028 and the pair head adds 4,097.

## What equivalence means here

Execution rewrites have sequential/gradient equivalence tests against the current
V2 compatible implementation using legacy token rows and collision-free counters.
Tests include self-edges, equal timestamps, different batch boundaries, optimizer
updates and a busy endpoint selecting an event while its other endpoint is quiet.
Selected flags must reach both ring copies to preserve shared Record semantics.

Count-min is approximate and does not reproduce LRU eviction. Two hashes,
pretraining, balanced histories, a second GAT, pair loss and novelty retention
change the detector. They require retraining and held-out evaluation, and must
not be called numerically equivalent to the existing detector.

The original graph layer is already a causal shared-entity event graph, not
strictly a star. The CNN and current-event aggregation already mix information
across the window. Two GAT layers allow another propagation step on its existing
edges; they do not supply the same neighborhood coverage as ORTHRUS's entity
graph with 20 recent neighbors. The supplied 63%/99.6% figures have not been
independently established for this rewritten model.

For a summed shared table, both ordered hash matches and swapped hash matches
give identical embeddings. Under uniform independent hashes, the full collision
probability is 2/(B-1)^2 - 1/(B-1)^3, about 7.1e-8 at B=5,303. Partial overlap,
truncation and correlations remain; this is not a guarantee of collision-free
features.

## Why use a key–value encoder?

Here "key" means an attribute name, and "value" means its contents. For example,
`src.path=/usr/bin/bash`, `dst.path=/etc/passwd`, and `dst.remote_port=443` are
separate typed fields. Field names are namespaced by source/destination. Shared
subtoken embeddings represent textual values; kind embeddings distinguish paths,
IP addresses, numbers and missing values; a numeric projection represents signed
log magnitudes. The encoder combines each field name with its value/type, then
attention pools the unordered field set into a 64-dimensional event vector.
Adjacent subtokens preserve local value order. This supports varying attributes
without a separate learned encoder for each event schema. These attribute keys
are distinct from the attention layer's mathematical key/value projections.

The repository's ORTHRUS reference receives an already computed 128-dimensional
Word2Vec attribute feature plus a three-dimensional node-type feature per
endpoint, then projects source/destination features for its Graph Transformer.
Word2Vec learns textual co-occurrence patterns; the type feature explicitly
identifies an entity class. The TGNN encoder instead learns how field names,
typed values, roles and missingness combine for an event. Word2Vec is a text
embedding objective, while the key–value encoder is the structure that combines
fields. They can be used together: V3 pretrains the token table with the SGNS
objective, then fine-tunes the field encoder for graph tasks.

The implemented pretraining is based on
[Word2Vec's skip-gram/negative-sampling objective](https://arxiv.org/abs/1310.4546).
It trains its own benign-path corpus and fits vectors into summed hash buckets;
it is not ORTHRUS's exact trained feature file or a fastText implementation.

## Which components caused latency?

| Component in old sliding path | Why it costs time | V3 treatment |
|---|---|---|
| History deques, Record creation and per-event dictionaries | Thousands of Python operations and object accesses around each batch | Integer metadata and segmented device-side ring reads/writes |
| Short queues and TokenRef gathering | Build unions, deduplicate/sort per event, construct many gather indices and retain sparse backing batches | Device-side fixed-width merges and compressed ring rows |
| Online JSON/tokenization | Serialize attributes, flatten fields, parse paths/IPs and hash text repeatedly | Compile once into identity-indexed token memmaps |
| OrderedDict rarity updates | Per-endpoint Python lookups, insertion/moves and LRU bookkeeping | Fixed GPU counters with batched chronological updates |
| CPU/GPU index transfers and scalar decisions | Repeated small allocations, transfers and synchronization | Device-side metadata/ring planning; resident bank avoids token staging |
| Compaction | Periodic traversal and repacking causes latency spikes | Ring rows own their compressed values; no periodic TokenRef compaction |
| Encoder, GAT, CNN and short-context network | Real learned computation still runs, sometimes with repeated identities and two window passes | Deduplicate identity encoding; keep dense CNN and affine scan; extra GAT is an explicit cost |

The earlier L4 profile of the upgrade recorded about 0.49 seconds across four
1,024-event forwards, including 78 torch.tensor calls (~84 ms) and compressed
gathering (~65 ms). One operator profile showed ~2.6 ms self CUDA time. These
nested measurements are not additive or comparable to full training wall time,
but identify large host overhead around the learned GPU computation.

The new planner still launches sorting, unique, prefix/gather and counter kernels.
Launch costs and dynamic-size synchronization have a fixed per-batch component,
which explains the small-batch regression. Memmap mode still stages token arrays
from CPU; resident mode uses additional GPU memory. Batch arrival waiting is not
measured by these forward benchmarks.

## Final L4 benchmark

Real CADETS training prefixes, seeded weights, eight warmup and 64 measured
batches per size. The same prepared queries are compared against current V2
compatible mode. Timing includes encoding, metadata transfer, forward, state
update and detach/maintenance. Offline compilation is separate; ingestion,
optimizer steps, alert scoring and arrival waiting are excluded. No trained
accuracy result is inferred from these prefixes.

| Events/batch | V2 median ms | Ring compatible, resident ms | Ring corrected, resident ms |
|---:|---:|---:|---:|
| 32 | 11.413 | 20.725 | 21.964 |
| 256 | 25.245 | 21.346 | 23.458 |
| 1024 | 65.246 | 22.763 | 24.253 |

At batch 1,024, compatible p95/p99 improved from 84.483/131.974 to
23.891/25.509 ms. Throughput improved from 14,023 to 40,776 events/s.
Corrected p95/p99 were 25.173/25.934 ms and throughput was 38,268 events/s.
Memmap-compatible median/p99 were 23.812/29.548 ms.

Peak allocated GPU bytes at batch 1,024: V2 102,529,024; compatible resident
275,646,976; corrected resident 280,411,136. This is a latency gain with a
short-prefix memory tradeoff. Dense allocation for all 2,682,632 declared nodes
would take about 14 GB for rings alone; compact slot assignment avoids that
startup allocation. Long-stream growth and reserve-capacity increases still need
measurement. The fixed node-to-slot map alone uses about 21.5 MB.

For the 73,728-event batch-1,024 prefix, the compiled bank contained 1,849
identities and took about 2.8 seconds per encoding configuration to build. That
high reuse is workload-specific; arbitrary attributes may have one identity per
event. The artifact `artifacts/sliding_v3/gpu_latency.json` records exact measured
source hashes, every batch duration, initialization/timing scope and GPU metadata.
Earlier artifacts preserve dense and intermediate measurements.

## Validation and remaining limits

The final combined suite passed **80 tests**, with one legacy test deselected
because its archived fixture
`artifacts/sliding_node_mem_e3/performance_before/model.py` is missing.

V3 tests cover online/offline encoder outputs and gradients, dual-hash resident
parity, sequential inference/gradient parity against V2, count-min collisions,
batch/causality invariance, balanced slots, no sticky corrected retention,
cross-batch credit, sparse allocation/growth, elapsed-time scan parity,
pretraining/negative preparation, and interrupted/resumed training.
The Modal run also passed exact-counter parity on four real GPU batches and
synthetic CUDA unrolling with pair/memory gradients and snapshot replay.

The corrected model has not completed a full CADETS retraining and held-out F1
comparison. There is no established accuracy/F1 gain yet. The pair head learns
attribute-pair plausibility, so two distinct IDs with identical attributes remain
indistinguishable to that head. Negative sampling, memory/retention settings and
each accuracy change need separate ablations. The checkpoint still minimizes
benign relation CE, which is a proxy for anomaly detection. Empirical calibration
does not guarantee a whole-stream false-alert rate under distribution shift.
Small-batch latency and short-prefix memory regressions must be included in any
deployment decision.
