# Recorded V3 and ORTHRUS results

These are copied measurements, not new runs. Both recorded runs use CADETS E3, five training epochs, seed 0, batch size 1,024 and an NVIDIA L4. Neural architectures differ; the matched detector uses benign validation calibration and node-level evidence.

| Model | Precision | Recall | F1 | Average precision | ROC AUC | Parameters |
|---|---:|---:|---:|---:|---:|---:|
| V3 | 0.00419360 | 0.35294118 | 0.00828872 | 0.00779703 | 0.89176153 | 785,327 |
| ORTHRUS | 0.00175185 | 0.25000000 | 0.00347933 | 0.00233889 | 0.70950612 | 764,202 |

Metric values above are fractions, not percentages. Low absolute precision and single-seed coverage limit conclusions; these results do not establish general superiority across datasets.

- `v3_vs_orthrus/V3_accuracy.json`: full V3 configuration, source hashes, epoch history and measured test metrics.
- `v3_vs_orthrus/ORTHRUS_accuracy.json`: complete recorded matched ORTHRUS comparison result.
- `v3_vs_orthrus/latency.json`: measured prefix inference timings, GPU details, batch sizes, repetitions and experiment scope. This benchmark is distinct from end-to-end application latency and trained-checkpoint accuracy.
- `v3_vs_orthrus/source_manifest.json`: original data provenance; historical cloud paths are preserved.
- `sliding_v3/`: V3 development/profiling results; filenames identify earlier configurations, not additional final-model scores.
- `dashboard_validation/`: real-data integration/browser evidence from dashboard development.

No V4 or unrelated baseline experiment-result directories were copied. Original result files are unedited.
