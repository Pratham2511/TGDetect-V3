# Modal V3 versus ORTHRUS comparison

## Configuration

Latency: Modal L4, batch sizes 32/256/1024, three rotated-order repetitions per
arm, eight warmup plus 64 measured batches per repetition. Inputs are identical
chronological CADETS training prefixes. Seeded weights; inference mode. Includes
prepared input transfers, model computation and state maintenance. Excludes
offline feature preparation, scoring/calibration, and arrival-to-batch waiting.
ORTHRUS uses its prepared Word2Vec/type features; V3 uses prepared typed KV tokens.

Accuracy: separate L4 jobs, five epochs, seed 0, batch 1024, full official prepared
CADETS split. V3 uses its corrected architecture, pretrained hash embeddings,
two-batch gradient unroll, embedding LR 1e-4 and other LR 1e-5. ORTHRUS retains the
authors' architecture and Adam LR 1e-5. This is a system comparison, not identical
objectives, update counts or parameter budgets. V3 has 785,327 parameters versus
ORTHRUS's 764,202.

Prepared data counts are 3,526,541 train events, 105,481 validation events and
1,905,942 test events. ORTHRUS's existing Word2Vec features were built over all
database node attributes (transductive, per the pinned release and manifest).
V3's new SGNS initialization uses only paths from nodes observed in training.
Thus feature pretraining scope also differs; this is not a controlled encoder
ablation. The official partition is reused without modifying its labels or days.

Both accuracy arms select the checkpoint by benign relation CE on the first
validation third, then use remaining validation events for the same calibrated
node-evidence scoring procedure. ORTHRUS uses relation CE; V3 additionally uses
the learned pair anomaly score. Neither selects checkpoints or thresholds from
test labels. Report unmatched ground-truth nodes separately. This evaluation does
not reproduce ORTHRUS's released top-30 clustering rule or dependency tracing.

The tested V3 retains both CNN passes and dynamic short reads. CUDA graphs,
attention-only readout and causal-attention replacement are **not enabled**.
Frozen FP32 encoding is enabled for V3 evaluation.

## Completed latency results

Median of three per-run medians, milliseconds:

| Batch | ORTHRUS | V3 | V3 frozen lookup |
|---|---:|---:|---:|
| 32 | 4.286 | 21.987 | 20.196 |
| 256 | 4.617 | 22.905 | 20.510 |
| 1024 | 5.437 | 23.554 | 21.068 |

At batch 1024, median per-run throughput was 170,723 events/s for ORTHRUS,
39,594 for V3, and 44,093 for frozen V3. The throughput statistic uses total time
across measured batches, so it is not necessarily batch size divided by median
latency. Median per-run p99 was 6.183, 24.107 and 22.801 ms respectively.

Frozen lookup reduces V3's median latency by 10.6%, but frozen V3 is still 3.87x
slower than ORTHRUS here. Prefix peak allocated GPU memory was 3,603 MiB for
ORTHRUS versus 268 MiB for frozen V3. ORTHRUS reserves dense structures for the
declared node universe; V3 allocates rings for encountered nodes. This is a prefix
memory result, not a claim about peak memory on the full dataset.

Accuracy results are pending until `accuracy.json` exists with both complete arms.

## Preparation correction during this run

The first V3 attempt built SQLite files directly on the Modal volume and was
superseded before training. A scratch-storage retry was also profiled before
training: 1,168 of 1,508 samples were at the negative-pair SQLite lookup. The final
attempt uses an exact in-memory packed-integer set for the 2,108,847 known training
pairs and builds artifacts on scratch before copying them to the volume. RNG
order and sampled negatives are unchanged; a 1,000-query sequential-equivalence
test against SQLite passed, along with all 21 V3 tests. This is an offline
preparation fix and does not change the tested neural architecture. Both earlier
V3 attempts were cancelled before any optimization steps; ORTHRUS completed and
its result was saved before cancelling the original app's V3 call.

## Architecture diagram

The diagram source is [v3_tested_architecture.mmd](v3_tested_architecture.mmd).
The frozen table is an inference execution optimization of the trained KV encoder,
not an additional learned layer. Current relation is masked from the query and
used as the reconstruction target; observed relations appear in past-event inputs.
Event writes/evictions respect causal ordering. The history quota has up to four
slots per endpoint, with recency filling spare slots; self-edges are deduplicated.

## Artifacts

- `artifacts/v3_vs_orthrus/latency.json`: completed raw latency repetitions.
- `artifacts/v3_vs_orthrus/calls.json`: accuracy function call IDs.
- `artifacts/v3_vs_orthrus/accuracy.json`: written only once both training and
  held-out scoring jobs finish successfully.
- Modal volume output: `/study/v3_vs_orthrus_20261010`.
