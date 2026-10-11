# Investigation guide

## Navigate the workspace

Use Overview for corpus and checkpoint status, Detection graph for connected activity, Investigate alerts for evidence and temporal paths, Event explorer for exact records, and Model & validation for configuration and replay.

The initial theme is black. Theme controls switch to Solarized Light and retain the preference. Both modes use the same Instrument Sans family. Reduced-motion preferences stop decorative background animation.

## Detection graph

Select the original CADETS dataset or an uploaded dataset. Search for an IP, host, process, file, or other available entity. Entity options come from actual dataset attributes. Choose a starting prediction class, select 1–3 hops, and inspect connected context. Both prediction classes remain visible around a filtered seed.

Nodes use circular type glyphs and labels; directed links represent observed relationships. Red/green indicate predicted malicious/benign entities, with a distinct selected state. These are predictions rather than confirmed attacks. Repeated observations may be bundled; inspection resolves an actual evidence event.

Use zoom, fit, focus, reset, neighborhood selection, and JSON export. Display bounds are explicit: sampled graph counts do not describe corpus-wide proportions. Exported evidence preserves interpretation limits.

## Alerts and chains

Select a scored entity to inspect its strongest contributing evidence against the original validation threshold. Chain exploration and directed path construction follow observed endpoints and time order. Candidate path controls support forward replay, reverse backtracing, pause, speed, manual steps, and a scrubber.

Directed paths require consecutive destination/source matches and strictly increasing event order; paths do not invent stages. Ground truth does not seed, expand, or rank them. A path is a hypothesis for review, not proof of causal information flow.

## Upload endpoint events

Supported formats: JSON arrays, JSONL/NDJSON, and CSV. Uploads are bounded to **50 MB** and **100,000 events**. One background scoring job runs at a time. Ready datasets remain in local storage across reloads.

Required fields:

| Field | Accepted values |
| --- | --- |
| `timestamp` or `time` | ISO 8601, or Unix seconds/milliseconds/microseconds/nanoseconds. |
| `relation` | CONNECT, EXECUTE, OPEN, READ, RECVFROM, RECVMSG, SENDMSG, SENDTO, WRITE, CLONE; optionally prefixed `EVENT_`, or IDs 0–9. |
| `src` and `dst` | Endpoint IDs/IP strings or objects with identifiers and observed attributes. Source/destination/target aliases are supported. |

Endpoint attributes may include `id`, `type`, `remote_ip`, `host`, `path`, and `cmd_line`. Flat CSV columns such as `src_ip`, `dst_ip`, `src_host`, `dst_host`, endpoint type/path/command fields are also supported.

```json
[
  {
    "timestamp": "2026-10-11T10:00:00Z",
    "relation": "CONNECT",
    "src": {
      "id": "process-1",
      "host": "workstation-01",
      "type": "subject",
      "cmd_line": "curl"
    },
    "dst": {
      "id": "network-1",
      "remote_ip": "192.0.2.10",
      "type": "netflow"
    }
  }
]
```

This example illustrates the format; it is not benchmark evidence. Raw CDM envelopes and arbitrary vendor-specific schemas need conversion and are rejected rather than guessed.

The upload is normalized, stably sorted chronologically, and scored by the verified checkpoint from fresh state. Original calibration and threshold determine predictions. Supplied scores, alerts, and ground-truth labels are ignored. Event-time attributes prevent later observations from entering earlier inputs.

Uploads remain separate from the benchmark in ignored `backend/dashboard/data/graph_uploads/`. V3 was trained on CADETS; predictions in other domains are unvalidated. See [model limitations](model-and-evaluation.md).
