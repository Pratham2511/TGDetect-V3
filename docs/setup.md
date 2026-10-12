# Setup and troubleshooting

Run commands from the repository root unless a command explicitly enters another directory. Start with the [README quick start](../README.md#quick-start).

## Runtime and environment

The local validation environment uses Python 3.13.5, Node.js 20.19.2, PyTorch 2.11.0+cpu, and the frontend package lock. Python packages are pinned in `backend/requirements.txt`; the recorded upstream environment is in `backend/environment-record.json`.

For CPU operation, install the pinned Torch CPU wheel first, then the requirements. A compatible CUDA wheel may be selected for separate research work. The default dashboard replay remains a local CPU workflow.

## Downloaded data

`DATA_BUNDLE.json` pins the release URL, compressed size, archive SHA-256, and upstream source. `DATA_MANIFEST.json` pins all 18 extracted files, including the checkpoint, prepared corpus, tokens, saved scores, and SQLite investigation database. The original archive's internal `dashboard/data/` paths are mapped into `backend/dashboard/data/` without modifying file bytes.

The compressed archive is approximately 262 MiB; extracted files total approximately 1.53 GiB. Downloads are cached in `.cache/`, which is ignored by Git. Uploads live separately in `backend/dashboard/data/graph_uploads/` and are never included in the release or source manifest.

Offline installation:

```sh
python scripts/download_local_data.py --archive /path/to/cadets-v3-local-data.tar.gz
python scripts/verify_package.py
```

The installer rejects unexpected files, symbolic links, path traversal, duplicate entries, and checksum mismatches. Keep the backend stopped while installing or replacing artifacts. Restart it after rebuilding an investigation index.

## Ports and development

```sh
python scripts/run_backend.py --host 127.0.0.1 --port 8001 --reload
```

Set `TGDETECT_API_ORIGIN=http://127.0.0.1:8001` in `frontend/.env.local`, then restart Next.js. The frontend proxy uses the origin at build/runtime configuration time; rebuild production output after changing it.

Frontend development:

```sh
cd frontend
npm run dev
```

The frontend opens the landing page at `http://localhost:3000/`. Use **Open workspace** or visit `/dashboard` for the investigation application. `npm run dev` and `npm run build` build the landing assets automatically. After editing `frontend/landing/` during development, run `npm run build:landing` from `frontend/` to refresh them. No separate landing server is required.

Use `Ctrl+C` in each service terminal to stop it. Avoid running a development server and a production build against the same `.next/` directory simultaneously.

## Common problems

| Symptom | Action |
| --- | --- |
| Port 3000 or 8000 is occupied | Stop the other service or select another port and adjust the frontend API origin. |
| Landing assets are missing | Run `npm run build:landing` from `frontend/`, or rebuild with `npm run build`. Keep generated `public/landing/` beside the Next.js output when deploying. |
| API proxy cannot connect | Start the backend, check `http://127.0.0.1:8000/api/cadets/summary`, and check the proxy origin. |
| CADETS index unavailable | Run the verified data installer. It includes the prepared investigation index. |
| Checkpoint/source hash mismatch | Restore the pinned research files and data bundle; do not disable validation or modify trained source to bypass it. |
| Replay returns 409 | Another replay owns the model runtime. Wait for it to finish and retry. |
| An upload is rejected | Convert it to the supported endpoint-event schema in the [investigation guide](investigation.md). |
| `/legacy` returns missing-artifact errors | That interface needs independent Sysmon/DEDALE artifacts and is outside the supplied CADETS package. |

## Optional research tools

Research command from `backend/`:

```sh
PYTHONPATH=research python -m sliding_node_mem_v3.run --help
```

Cloud scripts require a configured Modal account and their original volume names. They are retained for reproducibility and are not launched by normal setup. An ORTHRUS trained checkpoint is not bundled; its official source and recorded comparison outputs are included.

## Hosting boundary

Default servers bind to loopback for local use. Authentication, authorization, TLS, rate limiting, and deployment-specific upload retention must be configured before exposing the service to other users. The current research checkpoint's detection limits remain applicable after deployment.
