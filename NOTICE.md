# Attribution notices

## TGDetect upstream

The backend, model implementation, prepared dataset, original interface foundations,
and recorded research/comparison artifacts derive from:

- Repository: https://github.com/MaheshMarathe005/TGDetect
- Base commit: `d83f2eab2c84394a6e9677fd9545dc15470bd573`
- Original frontend MIT copyright notice: `2025 TGDetect Contributors`

This repository introduces a reorganized application layout, a redesigned frontend,
and repository documentation/tooling. It preserves model and research source bytes,
trained weights, calibration, recorded measurements, and their original provenance.

## ORTHRUS

Official reference implementation and paper artifact:

- Repository: https://github.com/ubc-provenance/orthrus
- Included source snapshot: `d26613684b83d12e5d1f4bc51283b7de96124776`
- Original project title: *ORTHRUS: Achieving High Quality of Attribution in
  Provenance-based Intrusion Detection Systems*

Original source, configuration, references, and comparison records are retained
under `backend/research/` and `backend/results/`. The included snapshot is not a
newly trained ORTHRUS checkpoint or a claim of reproduction of every paper result.

## Data and trained artifacts

The release bundle contains the original CADETS E3 prepared corpus, trained V3
checkpoint, token bank, saved scores, and investigation index. Its original source
is the `v3-dashboard-2026-10-11` release of MaheshMarathe005/TGDetect. The archive
is mirrored without changing file bytes. SHA-256 records are in `DATA_BUNDLE.json`
and `DATA_MANIFEST.json`; historical experiment manifests retain their original
paths as provenance.

## Instrument Sans

Instrument Sans by Instrument, obtained from the official Google Fonts repository:

- https://github.com/Instrument/instrument-sans
- https://github.com/google/fonts/tree/main/ofl/instrumentsans

Normal/italic variable font files are self-hosted under SIL OFL 1.1. The original
license and font notices are included next to the binaries in `frontend/src/app/fonts/`.
