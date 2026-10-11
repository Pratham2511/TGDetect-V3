"""Artifact-backed dashboard, exact checkpoint loading and causal streaming inference."""
import asyncio
import hashlib
import json
import threading
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from dashboard.runtime.node_memory_models import make_mem_model as make_old
from dashboard.runtime.node_memory_v3 import make_mem_model as make_v3
from dashboard.runtime.windows_apt_models import make_model as make_gnn

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'dashboard/data'
ARTIFACTS = ROOT / 'research/artifacts'
torch.set_num_threads(4)
app = FastAPI(title='TG-Detect', version='1.0.0')
dataset = None
metadata = None
metadata_stamp = None
preprocessing = None
preprocessing_stamp = None
sessions = {}
session_lock = threading.RLock()
sync_lock = threading.Lock()
data_lock = threading.RLock()


def load_data():
    global dataset
    with data_lock:
        if dataset is None:
            path = DATA / 'prepared.npz'
            if not path.exists():
                raise HTTPException(503, 'Prepared telemetry dataset has not been downloaded')
            with np.load(path, allow_pickle=False) as a:
                dataset = {key: a[key] for key in a.files}
    return dataset


def load_preprocessing():
    global preprocessing, preprocessing_stamp
    path = DATA / 'preprocessing.json'
    if path.exists() and preprocessing_stamp != path.stat().st_mtime_ns:
        preprocessing = json.loads(path.read_text())
        audit = json.loads((DATA / 'audit.json').read_text())
        if preprocessing['prepared_sha256'] != audit['prepared_sha256']:
            raise HTTPException(503, 'Preprocessing belongs to a different dataset')
        preprocessing_stamp = path.stat().st_mtime_ns
    return preprocessing


def event_metadata():
    global metadata, metadata_stamp
    path = DATA / 'event_metadata.json'
    with data_lock:
        if path.exists() and metadata_stamp != path.stat().st_mtime_ns:
            metadata = json.loads(path.read_text())
            metadata_stamp = path.stat().st_mtime_ns
    return metadata


def registry():
    groups = [('v3', ARTIFACTS / 'dedale_host_mem_v3/results_v3'),
              ('previous', ARTIFACTS / 'dedale_host_mem_v3/results_previous'),
              ('matched', ARTIFACTS / 'dedale_host_mem_matched/results')]
    output = []
    for run, folder in groups:
        for path in sorted(folder.glob('*_seed[0-9].json')):
            row = json.loads(path.read_text())
            if row.get('status') != 'complete':
                continue
            arm, seed = row['arm'], row['seed']
            if run == 'previous' and arm == 'TGNN_Node_Mem_Sel':
                continue  # old batch-dependent memory is available in comparison metrics only
            checkpoint = (folder / f'{arm}_seed{seed}.pt' if run == 'matched'
                          else DATA / f'models/{arm}_seed{seed}.pt')
            output.append(dict(id=f'{run}:{arm}:{seed}', run=run, arm=arm, seed=seed,
                               label=arm.replace('TGNN_Node_Mem_', 'Memory ').replace('TGNN_Q_A_C', 'QAC').replace('_', ' '),
                               checkpoint_available=checkpoint.exists(), checkpoint_path=str(checkpoint),
                               threshold=row['val_threshold'], validation_pr_ap=row['best_val_pr_ap'],
                               metrics=row['test_val_threshold'], epochs=row['epochs_trained'],
                               checkpoint_sha256=row.get('checkpoint_sha256'),
                               protocol=row.get('protocol', row.get('training_protocol', 'Original experiment')),
                               feature_dim=72, memory='Node_Mem' in arm,
                               parameter_count=row.get('parameter_count', row.get('active_parameters'))))
    return output


def get_model(model_id):
    rows = [row for row in registry() if row['id'] == model_id]
    if not rows:
        raise HTTPException(404, 'Unknown model identifier')
    return rows[0]


def default_model():
    rows = registry()
    candidates = [r for r in rows if r['run'] == 'matched' and r['arm'] == 'TGNN_Node_Mem_V3']
    if not candidates:
        candidates = [r for r in rows if r['run'] == 'v3']
    if not candidates:
        raise HTTPException(503, 'No trained V3 artifact is available')
    # Deployment choice uses validation AP only, never test metrics.
    return max(candidates, key=lambda r: r['validation_pr_ap'])


def checkpoint_model(row):
    path = Path(row['checkpoint_path'])
    if not path.exists():
        raise HTTPException(503, 'Checkpoint download is incomplete')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if row['checkpoint_sha256'] and digest != row['checkpoint_sha256']:
        raise HTTPException(503, 'Checkpoint checksum mismatch')
    ck = torch.load(path, map_location='cpu', weights_only=True)
    if ck['arm'] != row['arm'] or ck['seed'] != row['seed'] or ck['input_dim'] != 72:
        raise HTTPException(503, 'Checkpoint metadata does not match the model registry')
    nodes = int(load_data()['n_nodes'])
    if row['arm'] == 'GNN_H':
        model = make_gnn('GNN_H', 72)
    elif row['run'] == 'previous':
        model = make_old(row['arm'], 72, nodes)
    else:
        model = make_v3(row['arm'], 72, nodes)
    model.load_state_dict(ck['state_dict'], strict=True)
    model.eval()
    return model, digest


def saved_predictions(row):
    folder = (ARTIFACTS / 'dedale_host_mem_matched/results' if row['run'] == 'matched'
              else ARTIFACTS / f'dedale_host_mem_v3/results_{"v3" if row["run"] == "v3" else "previous"}')
    path = folder / f'{row["arm"]}_seed{row["seed"]}_predictions.npz'
    if not path.exists():
        raise HTTPException(503, 'Prediction artifact is still syncing')
    a = np.load(path, allow_pickle=False)
    d = load_data()
    if not np.array_equal(a['ids'], d['ids']) or not np.array_equal(a['y'], d['y']):
        raise HTTPException(503, 'Prediction artifact does not match telemetry')
    return a['probability']


def iso(epoch):
    return datetime.fromtimestamp(float(epoch), timezone.utc).isoformat()


def serialize_event(i, score, threshold):
    d, meta = load_data(), event_metadata()
    extra = meta[i] if meta else {}
    return dict(index=int(i), id=str(d['ids'][i]), timestamp=iso(d['times'][i]),
                src=int(d['src'][i]), dst=int(d['dst'][i]), score=float(score),
                alert=bool(score >= threshold), label=int(d['y'][i]), **extra)


@app.get('/api/health')
def health():
    return dict(status='ok', dataset_available=(DATA / 'prepared.npz').exists(),
                raw_ingestion_ready=load_preprocessing() is not None)


@app.get('/api/bootstrap')
def bootstrap():
    d = load_data()
    audit = json.loads((DATA / 'audit.json').read_text())
    rows = registry()
    for row in rows:
        row.pop('checkpoint_path')
    return dict(models=rows, default_model=default_model()['id'], dataset=audit,
                train_end=int(d['train_end']), val_end=int(d['val_end']), events=len(d['y']),
                raw_ingestion_ready=load_preprocessing() is not None,
                source='Recorded DEDALE Sysmon telemetry',
                feature_schema=dict(name='dedale-sysmon72-v1', dimensions=72,
                                    categorical_buckets=64, numeric=['log_message_length', 'log_event_code',
                                    'hour_sin', 'hour_cos', 'has_event_data', 'log_field_count',
                                    'log_command_length', 'log_parent_command_length']))


@app.get('/api/window')
def window(model_id: str | None = None, start: int | None = None, count: int = 600,
           threshold: float | None = None, host: int | None = None):
    d = load_data()
    row = get_model(model_id) if model_id else default_model()
    p = saved_predictions(row)
    start = int(d['val_end']) if start is None else start
    if count < 1 or count > 5000 or start < 0 or start >= len(p):
        raise HTTPException(422, 'Window is outside dataset bounds; count must be 1–5000')
    threshold = row['threshold'] if threshold is None else threshold
    if not 0 <= threshold <= 1:
        raise HTTPException(422, 'Threshold must be between 0 and 1')
    indices = np.arange(start, min(start + count, len(p)))
    if host is not None:
        indices = indices[d['dst'][indices] == host]
    events = [serialize_event(i, p[i], threshold) for i in indices]
    nodes, edges = {}, {}
    for e in events:
        image = e.get('image', '').replace('\\', '/').split('/')[-1]
        for node, kind in [(e['src'], 'process'), (e['dst'], 'host')]:
            if node not in nodes:
                nodes[node] = dict(id=str(node), kind=kind, label=(e.get('host') or f'Host {node}') if kind == 'host' else image or f'Process {node}',
                                   score=0., events=0, alerts=0)
            nodes[node]['score'] = max(nodes[node]['score'], e['score'])
            nodes[node]['events'] += 1
            nodes[node]['alerts'] += int(e['alert'])
        edge = (e['src'], e['dst'])
        if edge not in edges:
            edges[edge] = dict(id=f'{edge[0]}-{edge[1]}', source=str(edge[0]), target=str(edge[1]), events=0, alerts=0)
        edges[edge]['events'] += 1
        edges[edge]['alerts'] += int(e['alert'])
    # Bounded, deterministic graph. All hosts and most active/alerting processes.
    ordered = sorted(nodes.values(), key=lambda n: (n['kind'] == 'host', n['alerts'], n['events']), reverse=True)
    shown = ordered[:80]
    kept = {n['id'] for n in shown}
    graph = dict(nodes=shown, edges=[e for e in edges.values() if e['source'] in kept and e['target'] in kept],
                 total_nodes=len(nodes), hidden_nodes=max(0, len(nodes) - len(shown)))
    timeline = []
    for group in np.array_split(indices, min(30, len(indices))):
        if len(group):
            timeline.append(dict(timestamp=iso(d['times'][group[0]]), events=len(group),
                                 alerts=int((p[group] >= threshold).sum()), mean_score=float(p[group].mean())))
    return dict(model_id=row['id'], threshold=threshold, threshold_mode='validation-selected' if threshold == row['threshold'] else 'exploratory',
                source='Saved checkpoint predictions · recorded telemetry', start=start,
                next_start=min(start + count, len(p)), end=len(p), events=events, graph=graph, timeline=timeline,
                summary=dict(events=len(events), alerts=sum(e['alert'] for e in events), nodes=len(nodes),
                             mean_score=float(np.mean([e['score'] for e in events])) if events else 0))


@app.get('/api/events/{index}')
def event(index: int, model_id: str | None = None):
    d = load_data()
    if not 0 <= index < len(d['y']):
        raise HTTPException(404, 'Unknown event')
    row = get_model(model_id) if model_id else default_model()
    p = saved_predictions(row)
    previous = d['ego'][index][d['valid'][index]][1:]
    return dict(event=serialize_event(index, p[index], row['threshold']),
                normalized_features=d['x'][index].tolist(),
                history=[serialize_event(int(i), p[int(i)], row['threshold']) for i in previous],
                history_ages=d['ages'][index][d['valid'][index]][1:].tolist(),
                feature_schema='dedale-sysmon72-v1', feature_space='training-standardized',
                note='Historical neighbors are actual model inputs. Admission/read gates require a live inference session; scores are not causal explanations.')


@app.get('/api/comparison')
def comparison():
    matched = ARTIFACTS / 'dedale_host_mem_matched/results'
    old = ARTIFACTS / 'dedale_host_mem_v3/verified_comparison.json'
    results = [json.loads(p.read_text()) for p in sorted(matched.glob('*_seed[0-9].json'))]
    progress = [json.loads(p.read_text()) for p in sorted(matched.glob('*_progress.json'))]
    verification = [json.loads(p.read_text()) for p in sorted(matched.glob('*_matched_verification.json'))]
    return dict(protocol='Same-seed QAC warm start; 80 additional epochs; equal optimizer steps and replay schedule',
                results=results, progress=progress, verification=verification,
                complete=len(results) == 9 and len(verification) == 3,
                previous=json.loads(old.read_text()) if old.exists() else None)


class SessionRequest(BaseModel):
    model_id: str
    source: str = 'recorded'
    warm_prefix: int = Field(default=0, ge=0, le=153056)


@app.post('/api/sessions')
def create_session(request: SessionRequest):
    if request.source not in ('recorded', 'raw'):
        raise HTTPException(422, 'Source must be recorded or raw')
    if request.source == 'raw' and load_preprocessing() is None:
        raise HTTPException(503, 'Raw logs require the verified training scaler; its recovery is still running')
    row = get_model(request.model_id)
    model, digest = checkpoint_model(row)
    sid = uuid.uuid4().hex
    with session_lock:
        if len(sessions) >= 8:
            raise HTTPException(429, 'Close an existing inference session first')
        session = dict(model=model, row=row, checksum=digest, source=request.source,
                       position=0, status='warming' if request.warm_prefix else 'ready',
                       lock=threading.RLock(), raw_history=defaultdict(lambda: deque(maxlen=4)),
                       pending=[], pending_time=None, last_timestamp=None, nodes={}, last_result=None)
        if request.source == 'raw':
            session['nodes'] = {token: int(k) for k, token in load_preprocessing()['entities'].items()}
        sessions[sid] = session
    if request.warm_prefix:
        def warm():
            try:
                with session['lock']:
                    if request.source != 'recorded':
                        raise ValueError('Warm prefix only applies to recorded sessions')
                    while session['position'] < request.warm_prefix:
                        infer_recorded(session, min(512, request.warm_prefix - session['position']))
                    session['status'] = 'ready'
            except Exception as exc:
                session['status'], session['error'] = 'error', str(exc)
        threading.Thread(target=warm, daemon=True).start()
    return dict(id=sid, status=session['status'], checkpoint_sha256=digest,
                model_id=row['id'], threshold=row['threshold'])


def get_session(sid):
    with session_lock:
        if sid not in sessions:
            raise HTTPException(404, 'Unknown or closed session')
        return sessions[sid]


def aux_values(aux, i):
    if not isinstance(aux, dict):
        return {}
    return {key: value[i].detach().cpu().tolist() for key, value in aux.items()
            if isinstance(value, torch.Tensor) and value.ndim > 0}


def infer_recorded(session, count):
    d = load_data()
    lo, hi = session['position'], min(session['position'] + count, len(d['y']))
    if lo == hi:
        return dict(events=[], finished=True, position=hi)
    started = time.perf_counter()
    with torch.no_grad():
        logits, aux = session['model'].forward_events(
            torch.from_numpy(d['x'][d['ego'][lo:hi]]), torch.from_numpy(d['valid'][lo:hi]),
            torch.from_numpy(d['ages'][lo:hi]), d['src'][lo:hi], d['dst'][lo:hi], d['times'][lo:hi])
        score = logits.softmax(-1)[:, 1].cpu().numpy()
        session['model'].detach_state()
    threshold = session['row']['threshold']
    output = [dict(**serialize_event(lo+i, p, threshold), memory=aux_values(aux, i)) for i, p in enumerate(score)]
    session['position'] = hi
    result = dict(events=output, position=hi, finished=hi == len(d['y']),
                  inference_ms=(time.perf_counter() - started) * 1000,
                  source='Live checkpoint inference on recorded telemetry', checkpoint_sha256=session['checksum'])
    session['last_result'] = result
    return result


@app.get('/api/sessions/{sid}')
def session_status(sid: str):
    s = get_session(sid)
    return dict(id=sid, status=s['status'], position=s['position'], source=s['source'],
                model_id=s['row']['id'], error=s.get('error'), last_result=s['last_result'])


@app.post('/api/sessions/{sid}/next')
def next_batch(sid: str, count: int = 64):
    s = get_session(sid)
    if s['source'] != 'recorded':
        raise HTTPException(422, 'Use the raw event endpoint for this session')
    if s['status'] != 'ready':
        raise HTTPException(409, f'Session is {s["status"]}')
    if not 1 <= count <= 512:
        raise HTTPException(422, 'Count must be 1–512')
    with s['lock']:
        return infer_recorded(s, count)


@app.delete('/api/sessions/{sid}')
def delete_session(sid: str):
    with session_lock:
        if sid not in sessions:
            raise HTTPException(404, 'Unknown session')
        del sessions[sid]
    return dict(status='closed')


def infer_raw(s, events):
    from dashboard.runtime.preprocessing import features, node_ids, event_key, _epoch
    prep = load_preprocessing()
    mean, std = np.asarray(prep['mean'], dtype=np.float32), np.asarray(prep['std'], dtype=np.float32)
    # Validate the entire batch before modifying stream state.
    parsed = []
    last = s['last_timestamp']
    for e in events:
        if not isinstance(e, dict):
            raise HTTPException(422, 'Every raw event must be a JSON object')
        t = _epoch(e.get('@timestamp'))
        if t is None or (last is not None and t < last):
            raise HTTPException(422, 'Valid @timestamp values must arrive in nondecreasing order')
        host = (e.get('host') or {}).get('name') or (e.get('agent') or {}).get('name') or (e.get('winlog') or {}).get('computer_name')
        if not host or not (e.get('winlog') or e.get('event')):
            raise HTTPException(422, 'Expected Sysmon/Winlogbeat fields and a host identity')
        s_token, d_token = node_ids(e)
        x = ((features(e) - mean) / std).astype(np.float32)
        if not np.isfinite(x).all():
            raise HTTPException(422, 'Nonfinite feature values')
        parsed.append((e, t, s_token, d_token, x))
        last = t
    results = []
    started = time.perf_counter()
    model = s['model']
    for e, t, st, dt, x in parsed:
        for token in (st, dt):
            if token not in s['nodes']:
                s['nodes'][token] = len(s['nodes'])
        src, dst = s['nodes'][st], s['nodes'][dt]
        size = len(s['nodes'])
        if getattr(model, 'use_memory', False) and size > model.n_nodes:
            model.n_nodes = size
            if model.Mem is not None:
                model.Mem = F_pad_rows(model.Mem, size)
                model.Reference = F_pad_rows(model.Reference, size)
        if s['pending_time'] is not None and t != s['pending_time']:
            for ps, pd, pt, px, pi in s['pending']:
                for node in {ps, pd}:
                    s['raw_history'][node].append((pt, px, pi))
            s['pending'] = []
        prior = {i: (pt, px) for node in {src, dst} for pt, px, i in s['raw_history'][node]}
        recent = sorted(prior, reverse=True)[:4]
        wx = np.zeros((1, 5, 72), dtype=np.float32)
        wx[0, 0] = x
        mask = np.zeros((1, 5), dtype=bool)
        mask[0, :1+len(recent)] = True
        ages = np.zeros((1, 5), dtype=np.float32)
        for j, i in enumerate(recent, 1):
            pt, px = prior[i]
            wx[0, j] = px
            ages[0, j] = np.log1p(t - pt)
        with torch.no_grad():
            logits, aux = model.forward_events(torch.from_numpy(wx), torch.from_numpy(mask),
                                              torch.from_numpy(ages), [src], [dst], [t])
            score = float(logits.softmax(-1)[0, 1])
            model.detach_state()
        index = s['position']
        s['pending'].append((src, dst, t, x, index))
        s['pending_time'], s['last_timestamp'], s['position'] = t, t, index + 1
        ed = (e.get('winlog') or {}).get('event_data') or {}
        results.append(dict(index=index, id='|'.join(map(str, event_key(e))), timestamp=iso(t),
                            src=src, dst=dst, score=score, alert=score >= s['row']['threshold'],
                            label=None, image=ed.get('Image') or '', host=dt.split('|')[0],
                            memory=aux_values(aux, 0), normalized_features=x.tolist()))
    result = dict(events=results, position=s['position'], inference_ms=(time.perf_counter()-started)*1000,
                  checkpoint_sha256=s['checksum'], source='Live checkpoint inference on supplied raw Sysmon logs')
    s['last_result'] = result
    return result


def F_pad_rows(table, size):
    return torch.cat((table, table.new_zeros(size - len(table), table.shape[1])))


@app.post('/api/sessions/{sid}/events')
async def raw_events(sid: str, request: Request):
    s = get_session(sid)
    if s['source'] != 'raw':
        raise HTTPException(422, 'Create a raw-log session first')
    if int(request.headers.get('content-length', '0')) > 5_000_000:
        raise HTTPException(413, 'Upload limit is 5 MB per request')
    try:
        body = await request.body()
        if len(body) > 5_000_000:
            raise HTTPException(413, 'Upload limit is 5 MB per request')
        text = body.decode('utf-8')
        parsed = json.loads(text) if text.lstrip().startswith('[') else [json.loads(line) for line in text.splitlines() if line.strip()]
    except (ValueError, UnicodeError):
        raise HTTPException(422, 'Expected a JSON array or JSONL Sysmon events')
    if not isinstance(parsed, list) or not 1 <= len(parsed) <= 2000:
        raise HTTPException(422, 'Send 1–2000 chronological events per request')
    def run():
        with s['lock']:
            return infer_raw(s, parsed)
    return await asyncio.to_thread(run)


def sync_artifacts():
    import modal
    if not sync_lock.acquire(blocking=False):
        return dict(status='already_syncing')
    try:
        volume = modal.Volume.from_name('tg-detect-dedale-v1')
        target = ARTIFACTS / 'dedale_host_mem_matched/results'
        target.mkdir(parents=True, exist_ok=True)
        count = 0
        for entry in volume.listdir('/results/dedale_host_mem_matched'):
            name = Path(entry.path).name
            if name.endswith('.json'):
                raw = b''.join(volume.read_file(entry.path))
                result = json.loads(raw)
                if name.endswith('_progress.json') or name.endswith('_verification.json'):
                    (target / name).write_bytes(raw)
                elif result.get('status') == 'complete':
                    stem = name[:-5]
                    for suffix in ('.pt', '_predictions.npz'):
                        local = target / (stem + suffix)
                        if not local.exists():
                            temp = local.with_suffix(local.suffix + '.tmp')
                            temp.write_bytes(b''.join(volume.read_file(f'/results/dedale_host_mem_matched/{stem}{suffix}')))
                            temp.replace(local)
                    # Publish registry metadata only after both model and predictions exist.
                    (target / name).write_bytes(raw)
                count += 1
        for name in ('preprocessing.json', 'event_metadata.json'):
            if not (DATA / name).exists():
                try:
                    raw = b''.join(volume.read_file(f'/prepared/dedale_host_mem/{name}'))
                    json.loads(raw)
                    (DATA / name).write_bytes(raw)
                except (modal.exception.NotFoundError, FileNotFoundError):
                    pass
        return dict(status='synced', json_files=count)
    finally:
        sync_lock.release()


@app.post('/api/sync')
async def sync():
    try:
        return await asyncio.to_thread(sync_artifacts)
    except Exception as exc:
        raise HTTPException(502, f'Modal synchronization failed: {type(exc).__name__}: {exc}')


FRONTEND = ROOT / 'dashboard/frontend/dist'
if FRONTEND.exists():
    app.mount('/assets', StaticFiles(directory=FRONTEND / 'assets'), name='assets')


@app.get('/')
def index(request: Request):
    if not (FRONTEND / 'index.html').exists():
        return RedirectResponse(str(request.url.replace(port=3000)), status_code=307)
    return FileResponse(FRONTEND / 'index.html')


from dashboard.teammate_api import register as register_teammate_api
register_teammate_api(app)

from dashboard.cadets import router as cadets_router
app.include_router(cadets_router)

from dashboard.graph_datasets import router as graph_datasets_router
app.include_router(graph_datasets_router)
