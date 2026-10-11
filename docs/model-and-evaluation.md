# Model and evaluation

The supplied checkpoint is the original five-epoch, seed-0 V3 trained on the prepared DARPA TC CADETS E3 corpus. Its SHA-256 is:

```text
78d935ac58e54f34e3e9b89fbd23b9ca4aec95965e52b1d7712a4dfec458a345
```

The corpus includes **3,526,541 train**, **105,481 validation**, and **1,905,942 test** events, totaling **5,537,964**. Recorded result artifacts retain the exact trained configuration, source hashes, thresholds, epoch history, and measurements.

## Held-out node results

| Metric | Fraction | Percentage |
| --- | ---: | ---: |
| Precision | 0.00419360 | 0.4194% |
| Recall | 0.35294118 | 35.2941% |
| F1 | 0.00828872 | 0.8289% |
| Average precision | 0.00779703 | 0.7797% |
| ROC AUC | 0.89176153 | 89.1762% |

Confusion counts: **24 TP, 5,699 FP, 44 FN, 262,386 TN**. Scores cover 268,153 held-out nodes; 5,723 exceed the validation-selected threshold. These metrics are copied from [V3_accuracy.json](../backend/results/v3_vs_orthrus/V3_accuracy.json), not newly measured training outcomes.

Low absolute precision and single-seed evaluation limit conclusions. ROC AUC alone does not establish useful alert quality. The [matched ORTHRUS comparison](../backend/results/README.md) documents its protocol and coverage; it does not claim general superiority across datasets.

## Scoring and interpretation

Event losses are unbounded anomaly evidence rather than probabilities. The original relation-specific validation calibration is retained. The strongest three incident evidence values in each 900-second window are aggregated as `max(largest, sum(top3)/sqrt(3))`; the maximum window score determines node evidence. The threshold is supplied by the backend artifact.

Node benchmark labels are contextual membership annotations. They are not attack labels for individual events. Paths connect observed activity and are investigative hypotheses, not causal proof or invented attack stages. Evidence explanations trace the scoring rule; they do not provide feature-level causal attribution or SHAP values.

## Replay and uploads

`POST /api/cadets/replay` loads the verified checkpoint, resets state, and replays a chronological split prefix with the original 1,024-event batch boundaries. It compares losses against saved GPU evaluation scores. Setup time is reported separately from CPU batch time. These timings are not the recorded NVIDIA L4 benchmark or per-event arrival latency.

Training events have no saved evaluation score and remain visibly unscored. Uploaded telemetry uses the same checkpoint and calibration, but cross-domain predictions are unvalidated. No new accuracy claim is made for an uploaded dataset.

UI changes and repository organization do not improve these metrics or change the model. Original results and source hashes are preserved under `backend/results/`, `backend/artifacts/`, and `docs/provenance/`.
