# Contributing to TGDetect

Thanks for contributing. Start with the [README](README.md) and [setup guide](docs/setup.md). Use a bug report or feature request to explain a substantial change before implementing it. Small fixes can go directly to a pull request.

## Development

1. Fork this repository and create a descriptive branch.
2. Install the pinned backend dependencies and frontend lockfile dependencies.
3. Install the verified data bundle if your change needs real-data integration tests.
4. Run the backend with `python scripts/run_backend.py --reload` and the frontend with `npm run dev` from `frontend/`.

## Checks

```sh
python -m pytest backend/research/test_sliding_v3.py \
  backend/research/test_orthrus_universal.py -q
cd frontend
npm run lint
npm run build
```

For API, graph, upload, or inference changes, also run:

```sh
python -m pytest backend/research/test_cadets_dashboard.py \
  backend/research/test_graph_uploads.py -q
```

Verify both themes, navigation, small-screen layout, graph controls, error states, and reduced-motion behavior when changing the interface. Keep readable contrast and accessible names for controls.

The source inventory is deliberate: after finishing your changes, run `python scripts/update_manifest.py`, then `python scripts/verify_package.py --code-only`, and include the manifest update in the pull request.

## Model integrity

The frozen research implementation is tied to the supplied checkpoint's source hashes. UI and packaging changes must preserve those bytes, weights, calibration, scoring, and original recorded results. Intentional model work needs its own versioned artifact and provenance; do not weaken checks to load a mismatched model.

Use API-provided model metadata. Distinguish predictions from confirmed attacks, node labels from event labels, and chronological candidate chains from causal proof. Do not insert fabricated measurements, hardcoded success results, or new accuracy claims into the interface or documentation.

## Pull requests

Explain the concrete problem, resulting behavior, and verification. Include screenshots in both themes for visible changes. Keep changes focused and preserve attribution and component license notices. Follow the [Code of Conduct](CODE_OF_CONDUCT.md).

Do not commit `.env` files, credentials, local uploads, dependencies, build output, or corpus/checkpoint binaries. Use sanitized examples and the release-data installer. See [SECURITY.md](SECURITY.md) for private vulnerability reporting.
