# Experimental TGDetect sliding upgrade

This package is isolated from `sliding_node_mem` and the original saved checkpoints. It retains a shared typed key/value encoder, a single 128-dimensional provenance line-graph GAT, causal CNNs, selected/rare compressed short memory, and long node memory. Trainable parameter count remains **764,202**, matching the existing ORTHRUS reference. Runtime state and CPU caches are excluded from that count.

## Findings and changes

| Finding in original code | Upgrade | Evidence / limitation |
|---|---|---|
| Value subtokens are averaged; paths with rearranged components have identical value embeddings | Position-weighted subtoken pooling, preserving the beginning and end when long values are truncated | Path ordering and field-set permutation tests pass. This changes representations and needs training before accuracy can be judged. |
| Embedding every padded subtoken produces a large dense intermediate | `embedding_bag` over real subtokens | Reduces padded embedding work; CPU and GPU have different performance characteristics. |
| Current queries and observed history variants repeatedly serialize the same endpoint attributes | Bounded endpoint-query cache, canonical-identity cache, and token-ID cache | Learned embeddings are recomputed each forward; cached objects never contain learned embeddings. |
| Global rarity counts stop treating an action as rare after it occurs elsewhere | Bounded entity/action/type rarity counts | New-entity rarity test passes; useful rare behavior is retained locally. |
| Hard exclusion sets admission to zero for unselected events | 0.05 admission floor | Gradients can continue training long-memory writes; tuning remains necessary. |
| Compressed tokens lose training credit across detached batch boundaries | Small immediate compression-consistency loss (0.01) | Tests verify finite nonzero compression gradients. This is an experimental regularizer, not proof that overfitting is solved. |
| Reconstruction training loss falls while test PR-AUC remains poor | 0.1 dropout on the detection input and gradient clipping at norm 1 | Regularization is implemented; overfitting is a hypothesis until validation curves/ablations exist. |
| Per-event Python sets/sorts plan neighborhoods | NumPy vectorized planning for many overlapping windows | Randomized self-edge and sequential-equivalence tests pass. |
| CPU query preparation blocks the ordered GPU consumer | A bounded producer thread prefetches the next batch and pretokenizes query/observed variants | Stateful inference remains in order; graph loading and deterministic preprocessing can overlap GPU computation. |
| Second CNN processes all-empty short-memory windows | Gather only nonempty compressed windows and scatter their results | Batched and single-event inference agree. |
| Runtime memory allocates all 2.68 million global entities | Compact entity-to-row mapping and geometric table growth | Tested prefix uses 4,194,304 bytes instead of 1,373,507,584 bytes; stream snapshot/recovery tested. Full-stream memory grows with observed entities. |
| Maximum raw node loss favors noisy events and high-degree nodes; a maximum validation threshold is sensitive to a single outlier | Additional relation-tail calibration, top-3 evidence, degree correction and fixed benign quantile | Original max-loss and top-30 metrics remain available as ablations. First chronological half of benign validation fits relation distributions; second half sets the 0.9999 node-score quantile. No attack labels fit calibration. These are heuristic detection choices with no false-positive guarantee under dataset shift. |

## Parallel processing

The model processes many overlapping 8/9-event windows in a batch. The affine endpoint scan preserves pre-event memory reads. Arbitrarily shuffling windows or updating shared entity memory from independent workers would change causal behavior. This implementation parallelizes independent tensor work and overlaps deterministic CPU preparation with the ordered consumer.

Changing batch size changes optimizer-step frequency and truncated gradient boundaries. Faster throughput at a larger batch cannot be assumed to preserve training quality. The benchmark measures 512, 1024 and 2048; select based on measured GPU timing and held-out benign validation.

## Validation and timing

Local tests:

```bash
PYTHONPATH=research /home/mahesh/main_env/bin/python -m pytest research/test_sliding_node_mem.py research/test_sliding_upgrade.py -q
```

Real CADETS CPU training-prefix measurements are in `artifacts/sliding_node_mem_upgrade/smoke_cpu.json`. They use 4,096 warmup events and 8,192 measured events, with three repetitions and reversed implementation order on one repetition. Timed work includes synchronous query preparation, forward, backward and Adam. File loading and threaded pipeline overlap are outside this timing; the pipeline has separate tests. Disposable weights and smoke losses are excluded from research accuracy results.

GPU smoke command:

```bash
MODAL_PROFILE=fukatkaamsathi PYTHONPATH=research /home/mahesh/main_env/bin/python -m modal run research/modal_sliding_upgrade.py
```

Two attempted GPU smoke launches failed to connect to Modal; an API connectivity probe returned HTTP 503. GPU speed and full CADETS precision/recall/F1/PR-AUC of the upgrade are therefore unverified. No full training was launched in this upgrade run.

Isolated trainer:

```bash
PYTHONPATH=research /home/mahesh/main_env/bin/python -m sliding_node_mem_upgrade.run \
  --manifest /path/to/complete/local/manifest.json \
  --output /path/to/new/upgrade-results --device cuda --epochs 13 --batch 1024
```

The trainer validates dataset checksums and split coverage, trains only on benign relation targets, records its separate source/protocol signature, and reports raw and calibrated detection metrics. It rejects incompatible resumes and does not alter an old experiment to adopt these architecture changes.

## Research sources

- [ORTHRUS paper](https://www.usenix.org/system/files/usenixsecurity25-jiang-baoxiang.pdf): node-level provenance detection, rare positives, and attribution-quality evaluation. Higher ROC-AUC alone does not establish useful attack alerts.
- [Official ORTHRUS code](https://github.com/ubc-provenance/orthrus): reference architecture and detection pipeline.
- [PyTorch Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html): measuring CPU/GPU operators rather than inferring bottlenecks from GPU memory occupancy.
- [PyTorch transfer guidance](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html): nonblocking transfers and the cost of pinning memory in the main thread. This implementation does not claim pageable-memory copies automatically overlap computation.
