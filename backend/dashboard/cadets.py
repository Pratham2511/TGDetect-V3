"""CADETS V3 investigation API. Separate from the legacy 72D Sysmon runtime."""
from collections import defaultdict
from contextlib import contextmanager
from functools import lru_cache
import hashlib
import heapq
import json
import math
from pathlib import Path
import sqlite3
import sys
import threading
import time

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'dashboard/data/cadets_v3'
RESEARCH = ROOT / 'research'
CHECKPOINT_SHA = '78d935ac58e54f34e3e9b89fbd23b9ca4aec95965e52b1d7712a4dfec458a345'
EVENT_DTYPE = np.dtype([(k, '<i8') for k in ('src', 'dst', 'relation', 'query', 'observed', 'signature', 'src_identity', 'dst_identity', 'negative_identity')] + [('time', '<f8'), ('negative_valid', '?')])
EDGE_DTYPE = np.dtype([('src', '<i8'), ('dst', '<i8'), ('time', '<f8'), ('relation', '<i4'), ('loss', '<f4')])
RELATIONS = ['CONNECT', 'EXECUTE', 'OPEN', 'READ', 'RECVFROM', 'RECVMSG', 'SENDMSG', 'SENDTO', 'WRITE', 'CLONE']
router = APIRouter(prefix='/api/cadets', tags=['CADETS V3'])


def research_imports():
    if str(RESEARCH) not in sys.path:
        sys.path.insert(0, str(RESEARCH))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def result():
    return json.loads((DATA / 'result.json').read_text())


@contextmanager
def database():
    if not (DATA / 'investigation.sqlite').exists():
        raise HTTPException(503, 'CADETS index unavailable. Run python -m dashboard.scripts.prepare_cadets.')
    db = sqlite3.connect(f'file:{DATA / "investigation.sqlite"}?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    try:
        yield db
    finally:
        db.close()


def node_info(db, node):
    row = db.execute('SELECT * FROM nodes WHERE id=?', (node,)).fetchone()
    if row is None:
        raise HTTPException(404, 'Node not present in the test scoring universe')
    out = dict(row)
    out['attributes'] = json.loads(out.pop('attrs'))
    out['alert'] = bool(out['alert'])
    out['benchmark_member'] = bool(out['benchmark_member'])
    return out


def event_info(row):
    d = dict(row)
    d['id'] = f'{d["split"]}:{d["idx"]}'
    d['relation_name'] = RELATIONS[d['relation']] if 0 <= d['relation'] < len(RELATIONS) else str(d['relation'])
    return d


@router.get('/summary')
def summary():
    with database() as db:
        meta = {r['key']: json.loads(r['value']) for r in db.execute('SELECT * FROM metadata')}
    r = result()
    return dict(dataset='DARPA TC CADETS E3', model='TGNN Sliding Window V3', **meta,
        metrics=r['metrics'], threshold=r['threshold'], epochs=r['signature']['epochs'], batch_size=r['signature']['batch'],
        seed=r['signature']['seed'], parameter_count=r['parameter_count'], checkpoint_epoch=r['checkpoint_epoch'],
        validation_ce=r['validation_ce'], test_ce=r['test_ce'], model_config=r['signature']['model'],
        checkpoint_sha256=CHECKPOINT_SHA, unseen_ground_truth=r['unseen_ground_truth'],
        score_scope='Held-out test node metrics. Event losses are unbounded anomaly evidence, not probabilities.',
        limitations=['Benchmark labels describe nodes, not individual attack events.',
            'Connections in a reconstructed chain are investigative hypotheses, not causal proof.',
            'Evidence explanations trace the scoring rule; they are not feature-level causal attribution.',
            'The current checkpoint has low alert precision; analyst verification is required.'])


@router.get('/nodes')
def nodes(limit: int = Query(30, ge=1, le=100), offset: int = Query(0, ge=0), alerts_only: bool = True):
    with database() as db:
        where = 'WHERE alert=1' if alerts_only else ''
        total = db.execute(f'SELECT count(*) FROM nodes {where}').fetchone()[0]
        ids = db.execute(f'SELECT id FROM nodes {where} ORDER BY score DESC,id LIMIT ? OFFSET ?', (limit, offset))
        return dict(total=total, items=[node_info(db, row['id']) for row in ids], offset=offset)


@router.get('/events')
def events(split: str = Query('test', pattern='^(train|val|test)$'), node: int | None = Query(None, ge=0),
           relation: int | None = Query(None, ge=0, le=9), offset: int = Query(0, ge=0, le=5537964), limit: int = Query(30, ge=1, le=100)):
    where, args = ['split=?'], [split]
    if node is not None:
        where.append('(src=? OR dst=?)'); args.extend([node, node])
    if relation is not None:
        where.append('relation=?'); args.append(relation)
    with database() as db:
        if node is not None and relation is None:
            total = db.execute('SELECT count(*) FROM events WHERE split=? AND src=?', (split,node)).fetchone()[0]
            total += db.execute('SELECT count(*) FROM events WHERE split=? AND dst=?', (split,node)).fetchone()[0]
            total -= db.execute('SELECT count(*) FROM events WHERE split=? AND src=? AND dst=?', (split,node,node)).fetchone()[0]
            indices = db.execute('SELECT idx FROM events WHERE split=? AND src=? UNION SELECT idx FROM events WHERE split=? AND dst=? ORDER BY idx LIMIT ? OFFSET ?',
                                 (split,node,split,node,limit,offset)).fetchall()
            if not indices:
                return dict(total=total,items=[],offset=offset,split=split)
            placeholders = ','.join('?' for _ in indices)
            rows = db.execute(f'SELECT * FROM events WHERE split=? AND idx IN ({placeholders}) ORDER BY idx', [split]+[row['idx'] for row in indices])
            return dict(total=total,items=[event_info(row) for row in rows],offset=offset,split=split)
        condition = ' AND '.join(where)
        selection = 'SELECT * FROM events WHERE ' + condition
        if node is not None:
            # A generic OR can make SQLite scan the whole split. Two indexed
            # endpoint reads plus UNION also deduplicate self interactions.
            suffix = ' AND relation=?' if relation is not None else ''
            selection = ('SELECT * FROM events WHERE split=? AND src=?' + suffix +
                         ' UNION SELECT * FROM events WHERE split=? AND dst=?' + suffix)
            endpoint_args = [split, node] + ([] if relation is None else [relation])
            args = endpoint_args + endpoint_args
        if node is None and relation is None:
            # Counts are verified at index creation; avoid scanning a million
            # rows every time the user requests a 20-row page.
            total = json.loads(db.execute("SELECT value FROM metadata WHERE key='events'").fetchone()[0])[split]
        else:
            total = db.execute('SELECT count(*) FROM (' + selection + ')', args).fetchone()[0]
        rows = db.execute(selection + ' ORDER BY idx LIMIT ? OFFSET ?', args + [limit, offset])
        return dict(total=total, items=[event_info(row) for row in rows], offset=offset, split=split)


@router.get('/detection-graph')
def detection_graph(node: int | None = Query(None, ge=0), ip: str = Query('', max_length=64),
                    limit: int = Query(32, ge=8, le=60)):
    """A separate, bounded graph of real model decisions and network addresses."""
    with database() as db:
        if node is not None:
            seeds = [node_info(db, node)]
        else:
            seeds = []
            # Include both prediction classes in the default overview. These
            # are separate observed neighborhoods, never fabricated links.
            for alert in (1, 0):
                rows = db.execute("SELECT id FROM nodes WHERE alert=? AND json_extract(attrs,'$.remote_ip') IS NOT NULL AND json_extract(attrs,'$.remote_ip') LIKE ? ORDER BY score DESC,id LIMIT 2",
                                  (alert, '%' + ip + '%')).fetchall()
                seeds.extend(node_info(db,r['id']) for r in rows)
        entities = {n['id']:n for n in seeds}
        edges = {}
        frontier = [n['id'] for n in seeds]
        pruned = False
        for depth in range(2):
            following = []
            for current in frontier:
                rows = db.execute("SELECT * FROM (SELECT * FROM events WHERE split='test' AND src=? ORDER BY time DESC LIMIT 32) UNION SELECT * FROM (SELECT * FROM events WHERE split='test' AND dst=? ORDER BY time DESC LIMIT 32) ORDER BY time DESC,idx DESC LIMIT 48",(current,current)).fetchall()
                for row in rows:
                    other = row['dst'] if row['src']==current else row['src']
                    if other not in entities:
                        if len(entities)>=limit:
                            pruned=True
                            continue
                        entities[other]=node_info(db,other)
                        following.append(other)
                    if len(edges)<120:
                        edges[row['idx']]=event_info(row)
                    else:
                        pruned=True
            frontier=following[:8]
            pruned |= len(following)>8
        for n in entities.values():
            n['ip'] = n['attributes'].get('remote_ip')
            n['prediction'] = 'predicted_malicious' if n['alert'] else 'predicted_benign'
        items=list(entities.values())
        return dict(nodes=items,events=sorted(edges.values(),key=lambda e:(e['time'],e['idx'])),
            seeds=[n['id'] for n in seeds],threshold=result()['threshold'],truncated=pruned,
            counts=dict(predicted_malicious=sum(n['alert'] for n in items),predicted_benign=sum(not n['alert'] for n in items)),
            scope='Bounded recent-event neighborhoods of selected test entities. Colors show full-test V3 node predictions, not event labels or confirmed attacks.',
            ip_scope='Only recorded remote_ip attributes are shown as IP addresses. Process/file nodes without IP metadata are not assigned invented addresses.')


@router.get('/nodes/{node}/explanation')
@lru_cache(maxsize=128)
def explanation(node: int):
    with database() as db:
        info = node_info(db, node)
        windows = defaultdict(list)
        for row in db.execute("SELECT * FROM events WHERE split='test' AND src=? UNION SELECT * FROM events WHERE split='test' AND dst=? ORDER BY idx", (node, node)):
            heap = windows[math.floor(row['time'] / 900)]
            item = (row['evidence'], row['idx'])
            if len(heap) < 3:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
        def score(heap):
            return max(max((x[0] for x in heap), default=0), sum(x[0] for x in heap) / math.sqrt(3))
        window, heap = max(windows.items(), key=lambda pair: score(pair[1]))
        evidence = [event_info(db.execute("SELECT * FROM events WHERE split='test' AND idx=?", (idx,)).fetchone()) for _, idx in sorted(heap, reverse=True)]
        return dict(node=info, window_start=window * 900, window_seconds=900, contributors=evidence,
            single_event_channel=max(x[0] for x in heap), repeated_event_channel=sum(x[0] for x in heap) / math.sqrt(3),
            reconstructed_score=score(heap), threshold=result()['threshold'],
            rule='max over 15-minute windows of max(largest event evidence, sum(top 3 evidence) / sqrt(3))',
            meaning='Exact evidence attribution for the node alert. Relation-specific empirical calibration is fitted on validation only.',
            limitations='This explains the scoring calculation, not why a feature caused the model prediction.')


@router.get('/events/{split}/{index}')
def event_detail(split: str, index: int):
    if split not in ('train', 'val', 'test') or index < 0:
        raise HTTPException(422, 'A valid split and nonnegative event index are required')
    with database() as db:
        row = db.execute('SELECT * FROM events WHERE split=? AND idx=?', (split, index)).fetchone()
        if row is None:
            raise HTTPException(404, 'Event not found')
        out = event_info(row)
    prepared = np.memmap(DATA/'prepared'/(split+'.bin'), dtype=EVENT_DTYPE, mode='r')[index]
    with sqlite3.connect(f'file:{DATA / "prepared/tokens/identities.sqlite"}?mode=ro', uri=True) as identities:
        for role in ('src', 'dst'):
            identity = identities.execute('SELECT identity FROM identities WHERE idx=?', (int(prepared[role+'_identity']),)).fetchone()
            out[role+'_attributes'] = json.loads(identity[0])['attributes']
    out['explanation'] = 'Combined loss = relation cross-entropy + 0.1 × pair anomaly. Calibrated evidence measures deviation from the validation reference for this relation.'
    return out


@router.get('/nodes/{node}/chain')
def chain(node: int, hops: int = Query(2, ge=1, le=3), limit: int = Query(60, ge=1, le=120), seconds: int = Query(900, ge=1, le=3600)):
    exp = explanation(node)
    anchor = exp['contributors'][0]['time']
    # Expand chronological predecessor/successor paths. Reaching a node at a
    # time only permits subsequent edges in that search direction.
    found = {}
    truncated = False
    with database() as db:
        for direction in (-1, 1):
            direction_count = 0
            direction_limit = max(1, limit // 2)
            frontier = {node: anchor}
            visited = dict(frontier)
            for _ in range(hops):
                next_frontier = {}
                for current, arrival in frontier.items():
                    operator, ordering = ('<=', 'DESC') if direction < 0 else ('>=', 'ASC')
                    clause = f"split='test' AND {{endpoint}}=? AND time BETWEEN ? AND ? AND time {operator} ?"
                    args = (current, anchor-seconds, anchor+seconds, arrival)
                    rows = db.execute(f"SELECT * FROM events WHERE {clause.format(endpoint='src')} UNION SELECT * FROM events WHERE {clause.format(endpoint='dst')} ORDER BY time {ordering},idx {ordering} LIMIT ?",
                        args + args + (direction_limit+1,)).fetchall()
                    truncated |= len(rows) > direction_limit
                    for row in rows[:direction_limit]:
                        if len(found) >= limit or direction_count >= direction_limit:
                            truncated = True
                            break
                        if row['idx'] not in found:
                            found[row['idx']] = event_info(row)
                            direction_count += 1
                        other = row['dst'] if row['src'] == current else row['src']
                        when = row['time']
                        better = other not in visited or (when < visited[other] if direction > 0 else when > visited[other])
                        if better:
                            visited[other] = when; next_frontier[other] = when
                frontier = next_frontier
        entities = []
        ids = {node} | {e[k] for e in found.values() for k in ('src', 'dst')}
        for n in sorted(ids):
            entities.append(node_info(db, n))
    return dict(seed=node, anchor=anchor, hops=hops, seconds=seconds, truncated=truncated,
        nodes=entities, events=sorted(found.values(), key=lambda e: (e['time'], e['idx'])),
        strategy='Bidirectional temporal reachability on observed endpoints; arrows retain recorded event direction.',
        interpretation='Candidate attack-chain reconstruction. Temporal connectivity alone does not establish an attack or causal information flow.',
        labels_used=False)


@router.get('/nodes/{node}/construction')
def construct_chain(node: int, anchor_index: int | None = Query(None, ge=0),
                    direction: str = Query('both', pattern='^(both|backward|forward)$'),
                    max_steps: int = Query(5, ge=1, le=7), limit: int = Query(6, ge=1, le=12),
                    seconds: int = Query(900, ge=1, le=3600)):
    """Bounded directed, chronological event paths through a selected event.

    Endpoint/time compatibility supports an investigative hypothesis, not proof
    of causal information flow. Neither search nor ranking accepts truth labels.
    """
    exp = explanation(node)
    anchor_index = exp['contributors'][0]['idx'] if anchor_index is None else anchor_index
    with database() as db:
        row = db.execute("SELECT * FROM events WHERE split='test' AND idx=?", (anchor_index,)).fetchone()
        if row is None or node not in (row['src'], row['dst']):
            raise HTTPException(422, 'Anchor must be an observed test event incident to the selected entity')
        anchor_event = event_info(row)
        anchor_time = anchor_event['time']
        beam = [[anchor_event]]
        completed = []
        candidate_cache = {}
        pruned = False

        def strength(path):
            return sum(e['evidence'] for e in path)

        for _ in range(max_steps-1):
            expanded = []
            for path in beam:
                visited = {e[k] for e in path for k in ('src','dst')}
                additions = []
                for before in (True, False):
                    if (before and direction == 'forward') or (not before and direction == 'backward'):
                        continue
                    edge = path[0] if before else path[-1]
                    endpoint = edge['src'] if before else edge['dst']
                    key = (before, endpoint, edge['time'], edge['idx'])
                    if key not in candidate_cache:
                        col,op,order = ('dst','<','DESC') if before else ('src','>','ASC')
                        candidates = db.execute(f"SELECT * FROM events WHERE split='test' AND {col}=? AND time BETWEEN ? AND ? AND (time {op} ? OR (time=? AND idx {op} ?)) ORDER BY evidence DESC,time {order},idx {order} LIMIT 25",
                            (endpoint,anchor_time-seconds,anchor_time+seconds,edge['time'],edge['time'],edge['idx'])).fetchall()
                        pruned |= len(candidates) == 25
                        candidate_cache[key] = [event_info(e) for e in candidates[:24]]
                    eligible = [e for e in candidate_cache[key] if (e['src'] if before else e['dst']) not in visited]
                    pruned |= len(eligible) > 3
                    for e in eligible[:3]:
                        additions.append([e]+path if before else path+[e])
                if additions:
                    expanded.extend(additions)
                else:
                    completed.append(path)
            unique = {tuple(e['idx'] for e in p):p for p in expanded}
            candidates = sorted(unique.values(), key=lambda p:(-strength(p),tuple(e['idx'] for e in p)))
            pruned |= len(candidates) > limit
            beam = candidates[:limit]
            if not beam:
                break
        completed.extend(beam)
        unique = {tuple(e['idx'] for e in p):p for p in completed}
        ranked = sorted(unique.values(),key=lambda p:(-strength(p),tuple(e['idx'] for e in p)))[:limit]
        paths = [dict(id=f'path-{i+1}', events=p, evidence_sum=strength(p),
                      anchor_step=next(j for j,e in enumerate(p) if e['idx']==anchor_index)) for i,p in enumerate(ranked)]
        ids = {e[k] for p in ranked for e in p for k in ('src','dst')}
        entities = [node_info(db,n) for n in sorted(ids)]
    return dict(seed=node, anchor=anchor_event, direction=direction, max_steps=max_steps, seconds=seconds,
                paths=paths, nodes=entities, search_pruned=pruned, labels_used=False,
                ranking='Sum of calibrated event evidence; a ranking aid, not an attack probability.',
                rule='Consecutive edges must share destination/source; (timestamp, event index) strictly increases. No repeated entity is added.',
                interpretation='Candidate attack chains constructed from recorded directed events. Endpoint and time compatibility do not prove an attack or causal flow.')


class ReplayRequest(BaseModel):
    split: str = Field('test', pattern='^(train|val|test)$')
    batches: int = Field(1, ge=1, le=4)


class V3Runtime:
    """Bounded isolated replay; reset state per request, never random access."""
    def __init__(self):
        self.lock = threading.Lock()
        self.model = None

    def load(self):
        if self.model is not None:
            return
        import torch
        research_imports()
        from sliding_node_mem_v3.tokens import TokenBank
        from sliding_node_mem_v3.model import TGNNSlidingNodeMem
        if digest(DATA / 'best.pt') != CHECKPOINT_SHA:
            raise RuntimeError('V3 checkpoint integrity check failed')
        saved = torch.load(DATA / 'best.pt', map_location='cpu', weights_only=False)
        # Fail closed if the adapter would silently run changed model code.
        for name, expected in saved['signature']['sources'].items():
            if digest(RESEARCH / name) != expected:
                raise RuntimeError('Trained V3 source mismatch: ' + name)
        config = dict(saved['signature']['model'])
        for key in ('upgrade', 'version', 'token_config'):
            config.pop(key)
        bank = TokenBank(DATA / 'prepared/tokens')
        if bank.config != saved['signature']['model']['token_config']:
            raise RuntimeError('V3 token bank does not match checkpoint')
        model = TGNNSlidingNodeMem(bank=bank, **config)
        model.load_state_dict(saved['model'], strict=True)
        model.eval()
        torch.set_num_threads(4)
        model.event_encoder.precompute()
        self.model = model

    def replay(self, request):
        import torch
        import torch.nn.functional as F
        if not self.lock.acquire(blocking=False):
            raise HTTPException(409, 'A local V3 replay is already running')
        try:
            started = time.perf_counter(); self.load(); setup = time.perf_counter()-started
            self.model.reset_state()
            rows = np.memmap(DATA / 'prepared' / (request.split+'.bin'), dtype=EVENT_DTYPE, mode='r')[:1024*request.batches]
            output = []; timings = []
            with torch.no_grad():
                for lo in range(0, len(rows), 1024):
                    block = rows[lo:lo+1024]
                    batch = {k: torch.from_numpy(np.array(block[k])) for k in EVENT_DTYPE.names}
                    start = time.perf_counter()
                    logits, aux = self.model.forward_batch(batch)
                    ce = F.cross_entropy(logits, batch['relation'], reduction='none')
                    pair = aux['pair_anomaly']
                    losses = (ce + .1*pair).numpy()
                    timings.append((time.perf_counter()-start)*1000)
                    self.model.detach_state()
                    for i in range(len(block)):
                        output.append(dict(idx=lo+i, relation_ce=float(ce[i]), pair_anomaly=float(pair[i]), combined_loss=float(losses[i])))
            comparison = None
            if request.split != 'train':
                saved = np.memmap(DATA/(request.split+'_edges.bin'), dtype=EDGE_DTYPE, mode='r')[:len(rows)]
                comparison = float(np.max(np.abs(saved['loss']-np.array([x['combined_loss'] for x in output]))))
            return dict(events=len(rows), device='CPU', batch_size=1024, setup_seconds=setup, batch_ms=timings,
                max_absolute_error_vs_saved_gpu_scores=comparison, scores=output[:20],
                scope='Chronological split-prefix replay from empty state, same 1024-event batch boundaries as evaluation. Setup excluded from batch timings.')
        finally:
            if self.model is not None:
                self.model.reset_state()
            self.lock.release()


runtime = V3Runtime()


@router.post('/replay')
def replay(request: ReplayRequest):
    try:
        return runtime.replay(request)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(503, 'V3 replay unavailable: ' + str(exc)) from exc
