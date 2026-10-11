# Sliding-window + node-memory V2

Two explicit profiles separate execution improvements from detection experiments.
The original models and saved experiments remain reproducible.

## Compatible profile

Use `TGNNSlidingNodeMem(..., profile="compatible")` with the original upgrade's
model `state_dict`. Initialize with the same dimensions and reset stream state
before replay. Old runtime snapshots are not V2 snapshots. Keep the existing
training schedule and scoring policy when making a parity comparison.

```python
from sliding_node_mem_v2 import TGNNSlidingNodeMem

model = TGNNSlidingNodeMem(n_nodes, profile="compatible")
model.load_state_dict(original_model.state_dict())
model.eval()
model.reset_state()
# Replay chronological prepared query batches under torch.no_grad().
```

This profile preserves the dense CNN, GAT, learned dimensions and 764,202 default
parameters. Bulk metadata transfers, shared-block gathers, a CPU-planned exact
affine scan, bounded frozen-embedding caching, and periodic token-block compaction
reduce execution overhead. Small CPU batches can use an equivalent linear CNN
implementation; CUDA keeps convolutions. Numerical parity is tested within
floating-point tolerances, not bitwise equality.

The cache is disabled during training/gradient recording and invalidated when
weights change, load, or move device. Prepared attributes must be immutable.
Compaction repacks at most eight backing blocks after every 32 detach calls;
retain that maintenance cost in
latency measurements. `compact_every=0` disables it explicitly.

## Corrected profile

`profile="corrected"` changes model semantics and requires retraining:

- Reserve active history slots for both endpoints, then fill unused slots.
- Decide admission once at observation; give regular and exceptional retained
  events separate quotas and evict by age-decayed priority.
- Encode position-tagged adjacent value subtokens alongside unigrams.
- Remove forced memory writes, write fused context, and apply elapsed-time
  fast/slow memory decay (default half-lives one hour/one day).
- Train compressed representations with a normalized contextual target.

The V2 runner additionally retains gradients across two chronological batches
per optimizer update, selects the checkpoint by benign validation loss, and
uses disjoint validation portions for relation calibration and alert thresholds.
It preserves isolated extreme evidence, aggregates repeated evidence in fixed
time windows, and removes the top-30 alert cap. These runner changes apply even
with `--profile compatible`; that flag alone does not recreate the old experiment.

```bash
PYTHONPATH=research /home/mahesh/main_env/bin/python -m sliding_node_mem_v2.run \
  --manifest /path/to/prepared/manifest.json \
  --output artifacts/sliding_v2_experiment --device cuda \
  --epochs 5 --batch 1024 --unroll-batches 2 --profile corrected
```

The manifest must identify official CADETS E3 preprocessing, the complete
chronological splits, and checksums for all consumed input files. Output is under
`TGNN_Sliding_Node_Mem_V2/`: best weights, resumable training state, source copies,
binary edge scores and final metrics. Resume requires matching sources/config.

## Validation and limits

```bash
PYTHONPATH=research /home/mahesh/main_env/bin/python -m pytest -q research/test_sliding_v2.py
PYTHONPATH=research /home/mahesh/main_env/bin/python -m sliding_node_mem_v2.benchmark \
  --warmup 8 --repeats 64 --output artifacts/sliding_v2/cpu_latency.json
PYTHONPATH=research /home/mahesh/main_env/bin/modal run research/modal_sliding_v2_benchmark.py
```

The Modal command uses the existing prepared-data volume for a bounded L4
benchmark; it performs no optimizer updates. Its timing includes forward,
encoding, state updates and detach/compaction, but excludes input preparation,
file I/O, alert scoring and batch-arrival waiting. Per-event amortized timing is
throughput accounting, not individual online response latency.

The corrected profile has not completed a real-data retraining/evaluation run.
Do not claim improved F1 or accuracy from structural tests or latency benchmarks.
It can be slower, especially with mostly new attributes; use the compatible
profile for the demonstrated execution optimization. Two-batch unrolling also
increases training activation memory and changes optimizer update frequency.

Token hashing/truncation remain lossy; relation prediction remains a proxy task;
global state still grows with distinct observed entities. Compaction frees sparse
backing blocks but is not a global memory cap; finding eligible blocks still
scans live state. Decay constants and quotas need
ablation. The empirical false-alert budget is per node-window, not a guarantee
for the maximum over an entire stream or under distribution shift. Insufficient
calibration support produces an infinite threshold and `supported=false`.
