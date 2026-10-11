"""Persistent uploaded-log graphs, scored by the verified V3 checkpoint."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import io
import ipaddress
import json
import math
import os
from pathlib import Path
import sqlite3
import threading
import time
import uuid

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request
from dashboard.cadets import (DATA, RESEARCH, CHECKPOINT_SHA, EVENT_DTYPE, EDGE_DTYPE,
                             RELATIONS, digest, research_imports, result)

router = APIRouter(prefix='/api/graph-datasets', tags=['Uploaded detection graphs'])
STORE = DATA.parent / 'graph_uploads'
WORKER = ThreadPoolExecutor(max_workers=1, thread_name_prefix='v3-upload')
BUSY = threading.Lock()
MAX_BYTES = 50_000_000
MAX_EVENTS = 100_000


def folder(dataset):
    if len(dataset) != 32 or any(c not in '0123456789abcdef' for c in dataset):
        raise HTTPException(404, 'Dataset not found')
    path = STORE / dataset
    if not (path / 'status.json').exists():
        raise HTTPException(404, 'Dataset not found')
    return path


def status(path):
    data = json.loads((path / 'status.json').read_text())
    if data.get('status') in ('queued', 'processing') and data.get('worker_pid'):
        try:
            os.kill(data['worker_pid'], 0)
        except ProcessLookupError:
            data.update(status='failed', stage='Interrupted', error='Backend stopped before scoring finished. Upload the file again.', progress=0)
    return data


def update(path, **values):
    previous = status(path) if (path / 'status.json').exists() else {}
    previous.update(values)
    temporary = path / 'status.tmp'
    temporary.write_text(json.dumps(previous))
    temporary.replace(path / 'status.json')
    return previous


def builtin():
    return dict(id='cadets', name='DARPA CADETS E3 · recorded test predictions', status='ready', events=1905942,
                nodes=268153, source='Original held-out test', progress=1)


@contextmanager
def connection(dataset):
    path = DATA / 'investigation.sqlite' if dataset == 'cadets' else folder(dataset) / 'graph.sqlite'
    if dataset != 'cadets' and status(folder(dataset))['status'] != 'ready':
        raise HTTPException(409, 'Dataset scoring has not completed')
    db = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    try:
        yield db
    finally:
        db.close()


def first(row, *keys):
    for key in keys:
        value = row.get(key)
        if value is not None and value != '':
            return value
    return None


def timestamp(value):
    try:
        number = float(value)
        # Explicit conventional unix units, normalized to V3 seconds.
        if number > 1e17: number /= 1e9
        elif number > 1e14: number /= 1e6
        elif number > 1e11: number /= 1e3
    except (ValueError, TypeError):
        parsed = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        if parsed.tzinfo is None: parsed = parsed.replace(tzinfo=timezone.utc)
        number = parsed.timestamp()
    if not math.isfinite(number):
        raise ValueError('Timestamp must be finite')
    return number


def endpoint(row, role):
    aliases = ('src', 'source') if role == 'src' else ('dst', 'destination', 'target')
    raw = first(row, *aliases)
    attrs = dict(raw) if isinstance(raw, dict) else {}
    supplied = row.get(role + '_attributes')
    if isinstance(supplied, dict): attrs.update(supplied)
    for output, names in {
        'remote_ip': [f'{a}_ip' for a in aliases],
        'host': [f'{a}_host' for a in aliases] + [f'{a}_hostname' for a in aliases],
        'path': [f'{a}_path' for a in aliases],
        'cmd_line': [f'{a}_cmd_line' for a in aliases],
        'type': [f'{a}_type' for a in aliases],
    }.items():
        value = first(row, *names)
        if value is not None: attrs[output] = value
    if attrs.get('ip') and not attrs.get('remote_ip'): attrs['remote_ip'] = attrs['ip']
    identity = first(attrs, 'id', 'uuid', 'entity_id')
    if identity is None and raw is not None and not isinstance(raw, dict): identity = str(raw)
    if identity is not None:
        try:
            ipaddress.ip_address(str(identity))
            attrs.setdefault('remote_ip', str(identity))
        except ValueError: pass
    identity = identity or first(attrs, 'remote_ip', 'host', 'hostname', 'path', 'cmd_line', 'name')
    if identity is None:
        raise ValueError(f'Missing {role} endpoint: provide {role}, {role}_ip, or a structured {role} object')
    attrs.setdefault('type', 'netflow' if attrs.get('remote_ip') else 'subject' if attrs.get('cmd_line') else 'file' if attrs.get('path') else 'host')
    attrs.setdefault('name', str(identity))
    # Explicit IDs identify entities. Attribute-only records share stable identities.
    return str(identity), attrs


def parse_logs(body, name):
    text = body.decode('utf-8-sig')
    if name.lower().endswith('.csv'):
        raw = list(csv.DictReader(io.StringIO(text)))
    elif text.lstrip().startswith('['):
        raw = json.loads(text)
    elif text.lstrip().startswith('{'):
        try:
            value = json.loads(text)
            raw = value.get('events', [value]) if isinstance(value, dict) else value
        except json.JSONDecodeError:
            raw = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        raise ValueError('Upload JSON, JSONL, or CSV endpoint-event records')
    if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_EVENTS:
        raise ValueError(f'Upload 1–{MAX_EVENTS:,} events per dataset')
    entities, records = {}, []
    for index, row in enumerate(raw):
        try:
            if not isinstance(row, dict): raise ValueError('Each event must be an object')
            if isinstance(row.get('attributes'), dict):
                row = {**row['attributes'], **row}
            relation = first(row, 'relation', 'event_type', 'action', 'type')
            if isinstance(relation, int) or str(relation).isdigit():
                relation = int(relation)
                if not 0 <= relation < len(RELATIONS): raise ValueError('Relation ID must be 0–9')
            else:
                relation = str(relation).upper().removeprefix('EVENT_')
                if relation not in RELATIONS: raise ValueError('Supported relations: ' + ', '.join(RELATIONS))
                relation = RELATIONS.index(relation)
            when = timestamp(first(row, 'time', 'timestamp', '@timestamp', 'timestampNanos'))
            ids, observed = [], {}
            for role in ('src', 'dst'):
                key, attrs = endpoint(row, role)
                observed[role + '_attributes'] = dict(attrs)
                # Host-scoped process/file IDs avoid cross-host collisions.
                key = str(attrs.get('host') or attrs.get('hostname') or '') + '|' + key
                if key not in entities:
                    entities[key] = dict(id=len(entities), attributes=attrs)
                else:
                    entities[key]['attributes'].update(attrs)
                ids.append(entities[key]['id'])
            records.append(dict(src=ids[0], dst=ids[1], time=when, relation=relation, original_index=index, **observed))
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f'Event {index + 1}: {exc}') from exc
    records.sort(key=lambda e: (e['time'], e['original_index']))
    return list(entities.values()), records


def score_upload(path, entities, events):
    """Fresh model state, original tokenization/calibration; never supplied labels."""
    try:
        import torch
        import torch.nn.functional as F
        research_imports()
        from sliding_node_mem_v3.tokens import TokenWriter, TokenBank
        from sliding_node_mem_v3.model import TGNNSlidingNodeMem
        from orthrus_comparison.protocol import reconstruction_query
        from sliding_node_mem_upgrade.scoring import RelationCalibrator
        from sliding_node_mem_v2.scoring import NodeEvidence
        started = time.perf_counter()
        update(path, status='processing', stage='Tokenizing uploaded entities', progress=.05)
        if digest(DATA / 'best.pt') != CHECKPOINT_SHA: raise RuntimeError('Checkpoint integrity failure')
        saved = torch.load(DATA / 'best.pt', map_location='cpu', weights_only=False)
        for name, expected in saved['signature']['sources'].items():
            if digest(RESEARCH / name) != expected: raise RuntimeError('Trained model source mismatch: ' + name)
        config = dict(saved['signature']['model'])
        token_config = config.pop('token_config')
        for key in ('upgrade', 'version'): config.pop(key)
        writer = TokenWriter(path / 'tokens', **{k:token_config[k] for k in ('buckets','max_fields','max_subtokens','two_hash','ordered')})
        with (path / 'normalized.jsonl').open('w') as stream:
            for event in events: stream.write(json.dumps(event) + '\n')
        rows = np.zeros(len(events), dtype=EVENT_DTYPE)
        signatures = {}
        for i, event in enumerate(events):
            u, v = event['src'], event['dst']
            query = reconstruction_query(event['src_attributes'], event['dst_attributes'])
            source, destination = query['attributes']['src'], query['attributes']['dst']
            sig = (event['relation'], str(source.get('type')), str(destination.get('type')))
            signatures.setdefault(sig, len(signatures))
            for k in ('src', 'dst', 'time', 'relation'): rows[k][i] = event[k]
            rows['signature'][i] = signatures[sig]
            rows['query'][i] = writer.add(query)
            rows['observed'][i] = writer.add(dict(query, type=str(event['relation'])))
            rows['src_identity'][i] = writer.add({'attributes':source})
            rows['dst_identity'][i] = writer.add({'attributes':destination})
            rows['negative_identity'][i] = rows['dst_identity'][i]
        writer.close()
        config.update(n_nodes=len(entities), signatures=max(1, len(signatures)))
        model = TGNNSlidingNodeMem(bank=TokenBank(path / 'tokens'), **config)
        model.load_state_dict(saved['model'], strict=True)
        torch.set_num_threads(4)
        model.eval(); model.event_encoder.precompute(); model.reset_state()
        # Fit exactly the validation calibration segment used by published V3 artifacts.
        val = np.memmap(DATA / 'val_edges.bin', dtype=EDGE_DTYPE, mode='r')
        remaining = val[len(val)//3:]
        calibrator = RelationCalibrator().fit(remaining['relation'][:len(remaining)//2], remaining['loss'][:len(remaining)//2])
        evidence = NodeEvidence()
        losses = np.empty(len(events), dtype=np.float32)
        values = np.empty(len(events), dtype=np.float64)
        with torch.inference_mode():
            for lo in range(0, len(rows), 1024):
                block = rows[lo:lo+1024]
                batch = {k:torch.from_numpy(np.array(block[k])) for k in EVENT_DTYPE.names}
                logits, aux = model.forward_batch(batch)
                loss = (F.cross_entropy(logits, batch['relation'], reduction='none') + .1 * aux['pair_anomaly']).numpy()
                end = lo + len(block)
                losses[lo:end] = loss
                values[lo:end] = calibrator.transform(block['relation'], loss)
                evidence.update(block['src'], block['dst'], block['time'], values[lo:end])
                evidence.drain_completed(); model.detach_state()
                update(path, stage=f'Scoring {end:,} / {len(rows):,} events', progress=.15 + .8 * end/len(rows))
        scores = evidence.scores(); threshold = result()['threshold']
        db = sqlite3.connect(path / 'graph.sqlite')
        db.executescript('CREATE TABLE nodes(id INTEGER PRIMARY KEY,score REAL,alert INTEGER,benchmark_member INTEGER,attrs TEXT); CREATE TABLE events(split TEXT,idx INTEGER,src INTEGER,dst INTEGER,time REAL,relation INTEGER,loss REAL,evidence REAL,PRIMARY KEY(split,idx)); CREATE INDEX events_src ON events(src,time); CREATE INDEX events_dst ON events(dst,time); CREATE INDEX nodes_risk ON nodes(alert,score DESC);')
        db.executemany('INSERT INTO nodes VALUES(?,?,?,?,?)', [(n['id'],scores[n['id']],int(scores[n['id']] > threshold),None,json.dumps(n['attributes'])) for n in entities])
        db.executemany('INSERT INTO events VALUES(?,?,?,?,?,?,?,?)', [('upload',i,e['src'],e['dst'],e['time'],e['relation'],float(losses[i]),float(values[i])) for i,e in enumerate(events)])
        db.commit(); db.close()
        update(path, status='ready', stage='Graph ready', progress=1, events=len(events), nodes=len(entities),
               alerts=sum(s > threshold for s in scores.values()), elapsed_seconds=time.perf_counter()-started,
               checkpoint_sha256=CHECKPOINT_SHA, threshold=threshold,
               scoring='Frozen CADETS V3 checkpoint, fresh chronological state, 1024-event batches, original validation calibration. Predictions on other log domains have not been validated. Host/name metadata identifies graph entities; model inputs use trained endpoint fields only.')
    except Exception as exc:
        update(path, status='failed', stage='Scoring failed', error=str(exc), progress=0)
    finally:
        BUSY.release()


@router.get('')
def datasets():
    STORE.mkdir(parents=True, exist_ok=True)
    uploaded = [status(p.parent) for p in STORE.glob('*/status.json')]
    return dict(items=[builtin()] + sorted(uploaded, key=lambda d:d.get('created',0), reverse=True))


@router.post('', status_code=202)
async def upload(request: Request, name: str = Query('uploaded-logs.json', max_length=180)):
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_BYTES: raise HTTPException(413, 'Upload limit is 50 MB')
        chunks.append(chunk)
    try:
        entities, events = await asyncio.to_thread(parse_logs, b''.join(chunks), name)
    except (ValueError, UnicodeError, csv.Error) as exc:
        raise HTTPException(422, str(exc)) from exc
    if not BUSY.acquire(blocking=False): raise HTTPException(409, 'Another upload is being scored; wait for it to finish')
    try:
        identifier = uuid.uuid4().hex
        path = STORE / identifier; path.mkdir(parents=True)
        record = update(path, id=identifier, name=Path(name).name, status='queued', stage='Queued for V3 scoring', progress=0,
                        created=time.time(), worker_pid=os.getpid(), events=len(events), nodes=len(entities))
        WORKER.submit(score_upload, path, entities, events)
        return record
    except Exception:
        BUSY.release()
        raise


@router.get('/{dataset}')
def dataset_status(dataset: str):
    return builtin() if dataset == 'cadets' else status(folder(dataset))


FIELDS = {
    'ip': "json_extract(attrs,'$.remote_ip')",
    'host': "coalesce(json_extract(attrs,'$.host'),json_extract(attrs,'$.hostname'))",
    'process': "CASE WHEN json_extract(attrs,'$.type') IN ('subject','process') THEN coalesce(json_extract(attrs,'$.cmd_line'),json_extract(attrs,'$.name'),'process #'||id) END",
    'file': "CASE WHEN json_extract(attrs,'$.type')='file' THEN coalesce(json_extract(attrs,'$.path'),json_extract(attrs,'$.name'),'file #'||id) END",
    'entity': "coalesce(json_extract(attrs,'$.name'),json_extract(attrs,'$.cmd_line'),json_extract(attrs,'$.path'),'entity #'||id)",
}


@router.get('/{dataset}/entities')
def entity_options(dataset: str, kind: str = Query('ip', pattern='^(ip|host|process|file|entity)$'),
                   q: str = Query('', max_length=200), prediction: str = Query('all', pattern='^(all|malicious|benign)$')):
    expr = FIELDS[kind]
    condition = '' if prediction == 'all' else ' AND alert=' + ('1' if prediction == 'malicious' else '0')
    with connection(dataset) as db:
        rows = db.execute(f"SELECT {expr} AS label,count(*) AS count,max(alert) AS alert FROM nodes WHERE {expr} IS NOT NULL AND {expr} LIKE ? {condition} GROUP BY {expr} ORDER BY max(score) DESC,label LIMIT 100", ('%'+q+'%',)).fetchall()
        available = [key for key, field in FIELDS.items() if db.execute(f'SELECT 1 FROM nodes WHERE {field} IS NOT NULL LIMIT 1').fetchone()]
    return dict(items=[dict(r) for r in rows], available_kinds=available, truncated=len(rows)==100)


def node_record(row):
    n = dict(row); n['attributes'] = json.loads(n.pop('attrs'))
    n['alert'] = bool(n['alert'])
    n['benchmark_member'] = None if n['benchmark_member'] is None else bool(n['benchmark_member'])
    n['ip'] = n['attributes'].get('remote_ip')
    n['prediction'] = 'predicted_malicious' if n['alert'] else 'predicted_benign'
    return n


@router.get('/{dataset}/graph')
def graph(dataset: str, node: int | None = Query(None, ge=0), kind: str = Query('ip', pattern='^(ip|host|process|file|entity)$'),
          value: str = Query('', max_length=1000), prediction: str = Query('all', pattern='^(all|malicious|benign)$'),
          limit: int = Query(24, ge=8, le=60), hops: int = Query(2, ge=1, le=3)):
    with connection(dataset) as db:
        expr = FIELDS[kind]
        where, args = [], []
        if node is not None: where.append('id=?'); args.append(node)
        elif value: where.append(f'{expr}=?'); args.append(value)
        if prediction != 'all' and node is None:
            where.append('alert=?'); args.append(int(prediction=='malicious'))
        clause = ' WHERE ' + ' AND '.join(where) if where else ''
        roots = db.execute('SELECT * FROM nodes'+clause+' ORDER BY score DESC,id LIMIT 9', args).fetchall()
        roots_truncated = len(roots) > 8
        roots = roots[:8]
        if node is not None and not roots: raise HTTPException(404, 'Entity not found in this dataset')
        # No selector: one high-score root, with its actual neighborhood.
        if node is None and not value: roots = roots[:1]; roots_truncated = False
        nodes = {r['id']:node_record(r) for r in roots}; edges = {}; frontier = list(nodes)
        truncated = roots_truncated
        split = 'test' if dataset == 'cadets' else 'upload'
        for _ in range(hops):
            following = []
            for current in frontier:
                rows = db.execute("SELECT * FROM (SELECT * FROM events WHERE split=? AND src=? ORDER BY time DESC LIMIT 48) UNION SELECT * FROM (SELECT * FROM events WHERE split=? AND dst=? ORDER BY time DESC LIMIT 48) ORDER BY time DESC,idx DESC LIMIT 64",(split,current,split,current)).fetchall()
                truncated |= len(rows)==64
                for row in rows:
                    other = row['dst'] if row['src']==current else row['src']
                    if other not in nodes:
                        if len(nodes)>=limit: truncated=True; continue
                        found = db.execute('SELECT * FROM nodes WHERE id=?',(other,)).fetchone()
                        if found is None: continue
                        nodes[other]=node_record(found); following.append(other)
                    if len(edges)>=120: truncated=True; continue
                    e=dict(row); e.update(id=f'{dataset}:{e["idx"]}',relation_name=RELATIONS[e['relation']])
                    edges[e['idx']]=e
            frontier=following
        items=list(nodes.values())
    meta = dataset_status(dataset)
    return dict(dataset=meta, nodes=items, events=sorted(edges.values(),key=lambda e:(e['time'],e['idx'])), seeds=[r['id'] for r in roots],
                threshold=result()['threshold'], truncated=truncated,
                counts=dict(predicted_malicious=sum(n['alert'] for n in items),predicted_benign=sum(not n['alert'] for n in items)),
                scope='Connected observed neighborhoods of the selected entities. Class selection filters starting entities; connected context keeps both prediction classes. Display bounds may omit other connections.',
                ip_scope='Only recorded IP addresses are displayed. Colors are model predictions, not confirmed attacks.')

