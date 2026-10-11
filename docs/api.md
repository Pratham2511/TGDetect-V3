# API reference

The local backend exposes OpenAPI at `http://127.0.0.1:8000/docs`. The frontend forwards `/api/` to it through a same-origin proxy.

| Route | Purpose |
| --- | --- |
| `GET /api/cadets/summary` | Corpus counts, metrics, calibration, trained configuration, and limitations. |
| `GET /api/cadets/nodes` | Paginated entity alert queue. |
| `GET /api/cadets/events` | Paginated chronological events; supports split, node, and relation filters. |
| `GET /api/cadets/events/{split}/{index}` | One observed event and exact endpoint attributes. |
| `GET /api/cadets/nodes/{id}/explanation` | Contributing evidence and scoring-rule reconstruction. |
| `GET /api/cadets/nodes/{id}/chain` | Bounded observed temporal neighborhood. |
| `GET /api/cadets/nodes/{id}/construction` | Ranked directed chronological candidate paths. |
| `GET /api/cadets/detection-graph` | Connected graph with prediction classes, entity search, and bounded hops. |
| `POST /api/cadets/replay` | Real checkpoint CPU prefix replay and saved-loss comparison. |
| `GET /api/graph-datasets` | Available benchmark/upload datasets. |
| `POST /api/graph-datasets` | Submit a supported telemetry upload for background scoring. |
| `GET /api/graph-datasets/{id}` | Dataset status, progress, and failure information. |
| `GET /api/graph-datasets/{id}/entities` | Dataset-specific searchable entity options. |
| `GET /api/graph-datasets/{id}/graph` | Connected graph from that dataset. |

Replay example:

```sh
curl --fail http://127.0.0.1:8000/api/cadets/summary
curl --fail -H 'Content-Type: application/json' \
  -d '{"split":"test","batches":1}' \
  http://127.0.0.1:8000/api/cadets/replay
```

Replay accepts 1–4 batches. Concurrent replay requests receive 409. Missing/mismatched model artifacts return 503 instead of guessed results. Request limits and parameter validation are visible in OpenAPI.

See the [investigation guide](investigation.md) for upload schemas, limits, graph controls, and interpretation.
