"""Teammate dashboard contracts backed by our DEDALE artifacts and checkpoints.

Graph edges are recorded process/host incidence, not inferred network flows.
Entity-time groups are investigative clusters, not proof of causal attack paths.
"""
import json
import time
import uuid
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import HTTPException, Request
from starlette.concurrency import run_in_threadpool


def register(app):
    from dashboard import backend as b

    store = b.DATA / 'ui_datasets'
    store.mkdir(exist_ok=True)
    base_id = 'dedale-host-memory'

    def model(request):
        return b.get_model(request.query_params['model_id']) if request.query_params.get('model_id') else b.default_model()

    @lru_cache(maxsize=8)
    def recorded(stamp):
        d, meta = b.load_data(), b.event_metadata()
        out = []
        for i in range(len(d['y'])):
            extra = meta[i] if meta else {}
            out.append(dict(event_id=f'{i}:{d["ids"][i]}', ts=float(d['times'][i]),
                            src_id=f'process:{int(d["src"][i])}', src_type='PROCESS',
                            dst_id=f'host:{int(d["dst"][i])}', dst_type='HOST',
                            relation='OBSERVED_ON', label=int(d['y'][i]), tactics=[],
                            apt_stage=None, source_tag=extra.get('host') or 'DEDALE',
                            chain_id=None, causal_parent=None,
                            attrs=dict(index=i, recorded_id=str(d['ids'][i]), **extra)))
        return out

    def dataset_meta():
        d = b.load_data()
        return dict(id=base_id, name='DEDALE Sysmon — recorded benchmark', kind='dedale',
                    source=str(b.DATA / 'prepared.npz'), metadata_dir=None,
                    size_bytes=(b.DATA / 'prepared.npz').stat().st_size,
                    estimated_events=len(d['y']), num_raw_events=len(d['y']), created_at=None,
                    last_job_id=f'job-{base_id}', tags=['dedale', 'sysmon', 'chronological'],
                    provenance='Recorded DEDALE telemetry; prepared benchmark includes benign sampling',
                    time_span_s=float(d['times'][-1] - d['times'][0]), is_sample=True,
                    description='72 trained features; labels are benchmark ground truth, scores are saved checkpoint predictions')

    def datasets():
        return [dataset_meta()] + [json.loads(p.read_text())['dataset'] for p in sorted(store.glob('*.json'))]

    def select(request):
        did = request.query_params.get('dataset_id') or base_id
        if did == base_id:
            path = b.DATA / 'event_metadata.json'
            with b.data_lock:
                return dataset_meta(), recorded(path.stat().st_mtime_ns if path.exists() else 0)
        path = store / f'{did}.json'
        if not path.is_file() or path.parent.resolve() != store.resolve():
            raise HTTPException(404, 'Unknown dataset')
        item = json.loads(path.read_text())
        return item['dataset'], item['events']

    def scored(request):
        ds, ev = select(request)
        row = model(request)
        if ds['id'] != base_id:
            return ds, ev  # Uploaded scores retain the checkpoint used to produce them.
        probs = b.saved_predictions(row)
        return ds, [dict(e, attrs=dict(e['attrs'], probability=float(probs[i]),
                                      prediction=int(probs[i] >= row['threshold']),
                                      threshold=row['threshold'], model_id=row['id'])) for i, e in enumerate(ev)]

    def graph(events):
        nodes = {}
        for e in events:
            for direction, key, typ in [('out', 'src_id', 'src_type'), ('in', 'dst_id', 'dst_type')]:
                nid = e[key]
                n = nodes.setdefault(nid, dict(node_id=nid, node_type=e[typ], first_seen_ts=e['ts'],
                                              last_seen_ts=e['ts'], out_degree=0, in_degree=0, malicious_events=0))
                n[f'{direction}_degree'] += 1
                n['last_seen_ts'] = max(n['last_seen_ts'], e['ts'])
                n['first_seen_ts'] = min(n['first_seen_ts'], e['ts'])
                n['malicious_events'] += int(e['label'] == 1)
        return list(nodes.values())

    def chains(events):
        groups = defaultdict(list)
        for e in events:
            if e['label'] == 1:
                groups[e['src_id']].append(e)
        output = []
        for node, rows in groups.items():
            cluster = []
            def emit():
                if not cluster:
                    return
                ns = sorted({x[k] for x in cluster for k in ('src_id', 'dst_id')})
                output.append(dict(chain_id=f'entity:{node}:{cluster[0]["event_id"].split(":")[0]}',
                                   strategy='entity_time', num_events=len(cluster), num_nodes=len(ns),
                                   start_ts=cluster[0]['ts'], end_ts=cluster[-1]['ts'],
                                   duration_s=cluster[-1]['ts']-cluster[0]['ts'], tactic_sequence=[],
                                   stage_sequence=[], relation_sequence=['OBSERVED_ON'], nodes=ns,
                                   event_ids=[x['event_id'] for x in cluster]))
            for e in sorted(rows, key=lambda x: x['ts']):
                if cluster and e['ts'] - cluster[-1]['ts'] > 300:
                    emit(); cluster = []
                cluster.append(e)
            emit()
        return sorted(output, key=lambda x: x['start_ts'])

    def stats(ds, ev):
        ns, cs = graph(ev), chains(ev)
        positives = sum(e['label'] == 1 for e in ev)
        known = sum(e['label'] is not None for e in ev)
        lo, hi = (min(e['ts'] for e in ev), max(e['ts'] for e in ev)) if ev else (0, 0)
        return dict(dataset=ds['name'], input=ds['source'], elapsed_s=0,
                    normalization=dict(seen=len(ev), accepted=len(ev), rejected=0, reject_reasons={}),
                    labeling=dict(mode='parser', total_events=len(ev), seed_indicator_hits=0,
                                  propagated_events=0, malicious_events=positives, benign_events=known-positives,
                                  malicious_ratio=positives/known if known else 0, top_reasons={}),
                    graph=dict(total_events=len(ev), total_nodes=len(ns), total_edges=len(ev),
                               benign_events=known-positives, malicious_events=positives,
                               unlabeled_events=len(ev)-known,
                               node_types=dict(Counter(n['node_type'] for n in ns)),
                               relation_types=dict(Counter(e['relation'] for e in ev)),
                               source_tags=dict(Counter(e['source_tag'] for e in ev)), tactics={},
                               earliest_timestamp=lo, latest_timestamp=hi, timestamp_span_s=hi-lo,
                               out_of_order_events=sum(ev[i]['ts'] < ev[i-1]['ts'] for i in range(1,len(ev)))),
                    attacks=dict(malicious_events_tracked=positives, malicious_events_dropped=0,
                                 total_chains=len(cs), chains_by_strategy=dict(chain_id=0,causal_parent=0,entity_time=len(cs)),
                                 events_per_chain_histogram=dict(Counter(str(c['num_events']) for c in cs)),
                                 ungrouped_malicious_events=0, dangling_causal_parents=0),
                    outputs=dict(events=ds['source'],edges=ds['source'],nodes=ds['source'],chains='',subgraphs_written=0))

    @lru_cache(maxsize=24)
    def summary(mid):
        row = b.get_model(mid)
        net, digest = b.checkpoint_model(row)
        count = sum(p.numel() for p in net.parameters())
        config = dict(in_channels=72, edge_dim=0, hidden_channels=128, out_channels=128,
                      num_gnn_layers=len(net.layers), num_rnn_layers=int(getattr(net,'history_gru',None) is not None),
                      dropout=float(net.layers[0].dropout.p),
                      node_types=2, num_relations=1, learning_rate=1e-4,
                      gnn_type='Causal GAT', rnn_type='History GRU over each event window',
                      has_node_classifier=False,has_snapshot_classifier=False,has_edge_classifier=True,target='edge')
        layers = [dict(name=name,type=type(module).__name__) for name,module in net.named_children()]
        return dict(architecture=row['arm'], model_name=row['label'], architecture_type='Causal event ego-graph',
                    target='edge',gnn_operator='Graph attention on strictly earlier incident events',
                    temporal_aggregator='History GRU + attention + causal CNN; '+('64D fast + 64D slow memory per endpoint' if row['memory'] else 'no persistent node memory'),
                    output_heads=['event_classifier'],loss='Weighted cross entropy',has_attention=True,
                    has_transformer=False,has_llm=False,total_parameters=count,trainable_parameters=count,
                    non_trainable_parameters=0,bn_running_stats=0,
                    total_state_dict_elements=sum(x.numel() for x in net.state_dict().values()),
                    checkpoint_path=row['checkpoint_path'],checkpoint_sha256=digest,config=config,layers=layers)

    def models():
        rows = [r for r in b.registry() if r['checkpoint_available'] and r['run']=='matched']
        default = b.default_model()['id']
        rows.sort(key=lambda r: (r['id'] != default, r['arm'],r['seed']))
        return [dict(id=r['id'],name=f'{r["label"]} · seed {r["seed"]}',
                     description=f'{r["arm"]}; {r["protocol"]}; 72D Sysmon features',target='edge',
                     dataset_id=base_id,dataset_name='DEDALE Sysmon',checkpoint=Path(r['checkpoint_path']).name,
                     checkpoint_path=r['checkpoint_path'],trainable_parameters=summary(r['id'])['total_parameters'],
                     total_parameters=summary(r['id'])['total_parameters'],in_channels=72,edge_dim=0,
                     output_heads=['event_classifier'],has_edge_classifier=True,evaluation_status='evaluated',
                     evaluation_run_id=r['id'],metrics=dict(roc_auc=r['metrics']['roc_auc'],pr_auc=r['metrics']['pr_ap'],
                     f1=r['metrics']['f1'],precision=r['metrics']['precision'],recall=r['metrics']['recall'],
                     accuracy=r['metrics']['accuracy'],samples=sum(r['metrics'][k] for k in ('tn','fp','fn','tp')),
                     positives=r['metrics']['tp']+r['metrics']['fn'])) for r in rows]

    def result(row):
        folder = b.ARTIFACTS / 'dedale_host_mem_matched/results'
        return json.loads((folder/f'{row["arm"]}_seed{row["seed"]}.json').read_text())

    def evaluation(row, split):
        if split not in ('train','val','test'):
            raise HTTPException(422,'Unknown split')
        from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,average_precision_score,roc_auc_score,confusion_matrix
        d,p = b.load_data(),b.saved_predictions(row)
        lo,hi = {'train':(0,int(d['train_end'])),'val':(int(d['train_end']),int(d['val_end'])),
                 'test':(int(d['val_end']),len(p))}[split]
        y,prob = d['y'][lo:hi],p[lo:hi]; pred = prob >= row['threshold']
        tn,fp,fn,tp = confusion_matrix(y,pred,labels=[0,1]).ravel()
        metrics = dict(threshold=row['threshold'],num_samples=len(y),num_positive=int(y.sum()),num_negative=int((y==0).sum()),
                       accuracy=float(accuracy_score(y,pred)),precision=float(precision_score(y,pred,zero_division=0)),
                       recall=float(recall_score(y,pred,zero_division=0)),f1=float(f1_score(y,pred,zero_division=0)),
                       auc_pr=float(average_precision_score(y,prob)),auc_roc=float(roc_auc_score(y,prob)) if len(np.unique(y))==2 else None,
                       confusion_matrix=dict(tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp)))
        return dict(id=f'{row["id"]}:{split}',training_run_id=row['id'],checkpoint_path=row['checkpoint_path'],
                    split=split,target='edge',metrics=metrics,predictions_path=str(b.ARTIFACTS/'dedale_host_mem_matched/results'/f'{row["arm"]}_seed{row["seed"]}_predictions.npz'),started_at=None,ended_at=None)

    def get(path, request):
        q=request.query_params
        if path.startswith('session/'): return b.session_status(path.split('/')[1])
        if path=='health': return dict(b.health(),service='TG-Detect',version='2.0',backend='Modal artifact-backed FastAPI')
        if path=='models': return models()
        if path=='comparison': return b.comparison()
        if path=='bootstrap': return b.bootstrap()
        if path=='datasets': return datasets()
        if path=='datasets/active': return dataset_meta()
        if path.startswith('datasets/'): return next((d for d in datasets() if d['id']==path.split('/')[1]),None)
        if path.startswith('model/'):
            row=model(request)
            if path=='model/config': return summary(row['id'])['config']
            if path=='model/summary': return summary(row['id'])
            if path in ('model/training','model/training/history'):
                r=result(row)
                history=[dict(epoch=h['epoch'],train_loss=h['loss'],loss=h['loss'],auc_pr=h.get('validation_pr_ap'),auc_roc=None,threshold=None,f1=None,accuracy=None,precision=None,recall=None) for h in r['history']]
                if path.endswith('/history'): return history
                return dict(id=row['id'],model_type=row['arm'],dataset_id=base_id,status='completed',
                            epochs_total=r['epochs_trained'],best_epoch=r['best_epoch'],best_metric='validation_pr_ap',
                            best_metric_value=r['best_val_pr_ap'],threshold=r['val_threshold'],history=history,
                            training_time_s=r['elapsed_seconds'],checkpoint_path=row['checkpoint_path'],
                            config=dict(batch_size=r['train_batch'],lr=r['learning_rate'],seed=r['seed'],
                                        split_mode='time',select_metric='validation_pr_ap',gnn_layers=2,rnn_layers=0,hidden_channels=128,out_channels=128))
            if path=='model/evaluation': return evaluation(row,q.get('split','test'))
            if path=='model/evaluation/runs': return [evaluation(r,'test') for r in b.registry() if r['run']=='matched']
            if path=='model/predictions':
                d,p=b.load_data(),b.saved_predictions(row); split=q.get('split','test')
                lo,hi={'train':(0,int(d['train_end'])),'val':(int(d['train_end']),int(d['val_end'])),'test':(int(d['val_end']),len(p))}.get(split,(0,0))
                return [dict(sequence=i,node_id=f'process:{int(d["src"][i])}',probability=float(p[i]),prediction=int(p[i]>=row['threshold']),ground_truth=int(d['y'][i]),snapshot_label=int(d['y'][i])) for i in range(lo,min(hi,lo+min(int(q.get('limit',100)),1000)))]
            if path.startswith('model/snapshots'):
                # Visualization aggregates, never advertised as training inputs.
                ds,ev=select(request); windows=[]
                for lo in range(0,len(ev),512):
                    block=ev[lo:lo+512]; ns=graph(block)
                    windows.append(dict(index=len(windows),sequence=len(windows),window_start_ts=block[0]['ts'],
                                        window_end_ts=block[-1]['ts'],num_nodes=len(ns),num_edges=len(block),
                                        num_malicious_nodes=sum(n['malicious_events']>0 for n in ns),
                                        num_malicious_edges=sum(e['label']==1 for e in block),
                                        num_benign_edges=sum(e['label']==0 for e in block),num_unlabeled_edges=sum(e['label'] is None for e in block),
                                        snapshot_label=int(any(e['label']==1 for e in block)),
                                        path=None))
                meta=dict(dataset=ds['name'],window_size_s=0,stride_s=0,num_snapshots=len(windows),
                          node_feature_mode='type_only',edge_feature_mode='relation_only',num_node_types=2,
                          num_relations=1,node_feature_dim=72,edge_feature_dim=0,earliest_ts=ev[0]['ts'],latest_ts=ev[-1]['ts'],
                          description='512-event visualization blocks; inference uses 5-event causal ego windows')
                if path.endswith('/meta'): return meta
                return dict(meta=meta,snapshots=windows[:max(1,min(1000,int(q.get('limit',100))))])
        ds,ev=select(request)
        if path=='overview':
            s=stats(ds,ev)
            return dict(dataset=ds['name'],dataset_id=ds['id'],dataset_kind=ds['kind'],provenance=ds.get('provenance',''),
                        **{k:s['graph'][k] for k in ('total_events','total_nodes','total_edges','benign_events','malicious_events','node_types','relation_types','source_tags','tactics')},
                        malicious_ratio=s['labeling']['malicious_ratio'],chain_count=s['attacks']['total_chains'],time_span_s=s['graph']['timestamp_span_s'],input=ds['source'],pipeline_status='completed')
        if path=='graph/stats': return stats(ds,ev)
        if path in ('graph/nodes','graph/edges'):
            only=q.get('malicious_only')=='true'; limit=max(1,min(int(q.get('limit',1000)),5000))
            if path.endswith('nodes'): return [n for n in graph(ev) if not only or n['malicious_events']>0][:limit]
            chosen=[e for e in ev if not only or e['label']==1][:limit]
            return [{k:e[k] for k in ('event_id','src_id','dst_id','relation','ts','label','chain_id','causal_parent','source_tag')} for e in chosen]
        if path=='events' or path.startswith('events/'):
            row=model(request)
            def score(rows):
                if ds['id']!=base_id: return rows
                probs=b.saved_predictions(row)
                return [dict(e,attrs=dict(e['attrs'],probability=float(probs[e['attrs']['index']]),
                                         prediction=int(probs[e['attrs']['index']]>=row['threshold']),
                                         threshold=row['threshold'],model_id=row['id'])) for e in rows]
            if path.startswith('events/'):
                found=next((e for e in ev if e['event_id']==path[7:]),None)
                return score([found])[0] if found else None
            selected=ev
            for key,field in [('labels','label'),('relations','relation'),('source_tags','source_tag'),('chain_ids','chain_id')]:
                if q.get(key): selected=[e for e in selected if str(e[field]) in q[key].split(',')]
            if q.get('node_types'): selected=[e for e in selected if e['src_type'] in q['node_types'].split(',') or e['dst_type'] in q['node_types'].split(',')]
            if q.get('node_id'): selected=[e for e in selected if q['node_id'] in (e['src_id'],e['dst_id'])]
            if q.get('search'): selected=[e for e in selected if q['search'].lower() in e['event_id'].lower()]
            if q.get('tactics'): selected=[] # No MITRE annotations in this source; never synthesize them.
            if q.get('start_ts'): selected=[e for e in selected if e['ts']>=float(q['start_ts'])]
            if q.get('end_ts'): selected=[e for e in selected if e['ts']<=float(q['end_ts'])]
            offset=max(0,int(q.get('offset',0))); limit=max(1,min(1000,int(q.get('limit',100))))
            return dict(items=score(selected[offset:offset+limit]),total=len(selected))
        if path=='chains' or path.startswith('chains/'):
            cs=chains(ev)
            if path=='chains':
                if q.get('strategy'): cs=[c for c in cs if c['strategy']==q['strategy']]
                for key,field,op in [('min_events','num_events','min'),('max_events','num_events','max'),('min_duration','duration_s','min'),('max_duration','duration_s','max')]:
                    if q.get(key): cs=[c for c in cs if (c[field]>=float(q[key]) if op=='min' else c[field]<=float(q[key]))]
                return cs
            parts=path.split('/'); c=next((c for c in cs if c['chain_id']==parts[1]),None)
            if not c: return None
            rows=[e for e in ev if e['event_id'] in set(c['event_ids'])]
            if len(parts)==2: return c
            if parts[2]=='events': return rows
            return dict(chain_id=c['chain_id'],strategy=c['strategy'],nodes=[dict(id=n['node_id'],node_type=n['node_type']) for n in graph(rows)],edges=rows)
        if path=='jobs':
            return [dict(id=f'job-{d["id"]}',dataset_id=d['id'],dataset_name=d['name'],status='completed',progress_pct=100,
                         description='Recorded benchmark loaded' if d['id']==base_id else 'Uploaded Sysmon scored by saved checkpoint',
                         started_at=d.get('created_at'),completed_at=d.get('created_at'),elapsed_s=0,
                         config=dict(dataset_kind='dedale',source_tag='DEDALE',out_dir=d['source'],graphs_out=d['source'],strategies=['entity_time'],label_mode='parser',label_propagation=False)) for d in datasets()]
        if path=='artifacts':
            rows=[('events',ev[:10],len(ev)),('edges',ev[:10],len(ev)),('nodes',graph(ev)[:10],len(graph(ev))),('chains_summary',chains(ev)[:10],len(chains(ev))),('graph_stats',[stats(ds,ev)],1)]
            return [dict(kind=k,path=f'/api/ui/artifacts/{k}/download?dataset_id={ds["id"]}',size_bytes=None,row_count=n,
                         schema=[dict(name=field,type=type(value).__name__,nullable=value is None) for field,value in (preview[0].items() if preview else [])],preview=preview) for k,preview,n in rows]
        if path.startswith('artifacts/') and path.endswith('/download'):
            from fastapi.responses import StreamingResponse
            kind=path.split('/')[1]
            if kind=='events': rows=ev
            elif kind=='edges': rows=({k:e[k] for k in ('event_id','src_id','dst_id','relation','ts','label','chain_id','causal_parent','source_tag')} for e in ev)
            elif kind=='nodes': rows=graph(ev)
            elif kind=='chains_summary': rows=chains(ev)
            elif kind=='graph_stats': rows=[stats(ds,ev)]
            else: raise HTTPException(404,'Unknown artifact')
            def stream():
                yield '['
                for i,row in enumerate(rows):
                    yield (',' if i else '')+json.dumps(row)
                yield ']'
            return StreamingResponse(stream(),media_type='application/json',headers={'Content-Disposition':f'attachment; filename="{ds["id"]}_{kind}.json"'})
        if path=='analytics/events':
            ts=np.array([e['ts'] for e in ev]); labels=np.array([e['label']==1 for e in ev]);unknown=np.array([e['label'] is None for e in ev]);bins=np.linspace(ts.min(),ts.max()+1,31) if len(ev) else np.arange(31)
            total,_=np.histogram(ts,bins); malicious,_=np.histogram(ts[labels],bins); unlabeled,_=np.histogram(ts[unknown],bins)
            return dict(timeline=[dict(bucket_start=float(bins[i]),bucket_end=float(bins[i+1]),total=int(total[i]),benign=int(total[i]-malicious[i]-unlabeled[i]),malicious=int(malicious[i]),unlabeled=int(unlabeled[i])) for i in range(30)],tactics={},source_tags=dict(Counter(e['source_tag'] for e in ev)))
        if path=='analytics/graph':
            ns=graph(ev); degrees=Counter(str(n['in_degree']+n['out_degree']) for n in ns)
            return dict(node_types=dict(Counter(n['node_type'] for n in ns)),relations=dict(Counter(e['relation'] for e in ev)),degree_distribution=[dict(range=k,count=v) for k,v in degrees.items()])
        if path=='analytics/attacks':
            cs=chains(ev)
            return dict(strategies=dict(entity_time=len(cs),chain_id=0,causal_parent=0),length_histogram=dict(Counter(str(c['num_events']) for c in cs)),durations=[dict(chain_id=c['chain_id'],strategy=c['strategy'],duration_s=c['duration_s']) for c in cs])
        if path=='analytics/datasets': return datasets()
        raise HTTPException(404,'Unsupported dashboard endpoint')

    @app.get('/api/ui/{path:path}')
    async def ui_get(path: str, request: Request):
        return await run_in_threadpool(get,path,request)

    @app.post('/api/ui/datasets/{dataset_id}/activate')
    def activate(dataset_id: str):
        ds=next((d for d in datasets() if d['id']==dataset_id),None)
        if not ds: raise HTTPException(404,'Unknown dataset')
        return dict(status='ok',active_dataset=dataset_id,dataset=ds)

    # Forward live inference routes to the existing isolated, checksum-verified sessions.
    @app.post('/api/ui/sessions')
    def session(body: b.SessionRequest):
        return b.create_session(body)

    @app.get('/api/ui/session/{sid}')
    def status(sid: str): return b.session_status(sid)

    @app.post('/api/ui/session/{sid}/next')
    def next_events(sid: str, count: int=64): return b.next_batch(sid,count)

    @app.delete('/api/ui/session/{sid}')
    def close(sid: str): return b.delete_session(sid)

    @app.post('/api/ui/sync')
    async def sync(): return await b.sync()

    uploads = b.DATA / 'ui_uploads'
    uploads.mkdir(exist_ok=True)

    def read_upload(token):
        # Only opaque upload tokens created here are accepted; never read user paths.
        if len(token)!=32 or any(c not in '0123456789abcdef' for c in token):
            raise HTTPException(422,'Invalid upload token')
        path=uploads/f'{token}.json'
        if not path.exists(): raise HTTPException(404,'Upload not found')
        return json.loads(path.read_text())

    @app.post('/api/ui/datasets/upload')
    async def upload(request: Request):
        from dashboard.runtime.preprocessing import _epoch, node_ids
        form=await request.form(); file=form.get('file')
        if file is None: raise HTTPException(422,'Select a JSON or JSONL file')
        raw=await file.read(20*1024*1024+1)
        if len(raw)>20*1024*1024: raise HTTPException(413,'Upload limit is 20 MB')
        errors=[]; rows=[]
        try:
            text=raw.decode('utf-8-sig')
            if text.lstrip().startswith('['): rows=json.loads(text)
            else:
                for lineno,line in enumerate(text.splitlines(),1):
                    if not line.strip(): continue
                    try: rows.append(json.loads(line))
                    except ValueError as exc: errors.append(dict(line=lineno,column=None,message=str(exc)))
        except (ValueError,UnicodeDecodeError) as exc:
            errors.append(dict(line=1,column=None,message=str(exc)))
        if not isinstance(rows,list): rows=[]; errors.append(dict(line=1,message='Expected JSON array or JSONL objects'))
        valid=[]; last=None; sample=[]
        for i,e in enumerate(rows,1):
            if not isinstance(e,dict): errors.append(dict(line=i,message='Expected an event object')); continue
            t=_epoch(e.get('@timestamp')); host=(e.get('host') or {}).get('name') or (e.get('agent') or {}).get('name') or (e.get('winlog') or {}).get('computer_name')
            if t is None or not host or not (e.get('winlog') or e.get('event')):
                errors.append(dict(line=i,message='Require @timestamp, host identity, and Winlogbeat/Sysmon event fields')); continue
            if last is not None and t<last:
                errors.append(dict(line=i,message='Timestamps must be in nondecreasing order')); continue
            last=t; valid.append(e)
            if len(sample)<5:
                src,dst=node_ids(e)
                sample.append(dict(ts=t,src_id=src,dst_id=dst,relation='OBSERVED_ON',label=None))
        if not valid and not errors: errors.append(dict(line=1,message='File contains no events'))
        token=uuid.uuid4().hex
        report=dict(status='invalid' if errors else 'valid_with_warnings',format='dedale',format_label='Sysmon / Winlogbeat JSON',
                    detected_format='JSON array' if raw.lstrip().startswith(b'[') else 'JSONL',total_rows_inspected=len(rows),
                    valid_rows=len(valid),invalid_rows=len(errors),error_count=len(errors),warning_count=1,
                    errors=errors[:100],warnings=[dict(message='Uploaded ground truth is unknown. Scores and alerts will be produced by the selected saved checkpoint.')],
                    columns_detected=sorted({k for e in valid[:100] for k in e}),missing_required_columns=[],
                    features_extracted=['64 hashed categorical features','8 training-scaled numeric features'],sample_events=sample,
                    saved_path=token if not errors else None,filename=file.filename,
                    file_info=dict(filename=file.filename,size_bytes=len(raw),readable=not bool(errors),archive_type=None))
        if not errors: (uploads/f'{token}.json').write_text(json.dumps(dict(events=valid,report=report)))
        return report

    @app.post('/api/ui/datasets/validate')
    async def validate(request: Request):
        body=await request.json()
        return read_upload(body.get('file_path',''))['report']

    def process_upload(body, request):
        from dashboard.runtime.preprocessing import node_ids, _epoch
        import re
        did=body.get('dataset_id','')
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',did) or did==base_id:
            raise HTTPException(422,'Use a unique dataset ID with letters, numbers, underscores or hyphens')
        dest=store/f'{did}.json'
        if dest.exists(): raise HTTPException(409,'Dataset ID already exists; choose a new ID')
        raw=read_upload(body.get('source_path',''))['events']
        row=model(request); started=time.time()
        created=b.create_session(b.SessionRequest(model_id=row['id'],source='raw'))
        sid=created['id']; s=b.get_session(sid)
        events=[]
        try:
            for lo in range(0,len(raw),256):
                result=b.infer_raw(s,raw[lo:lo+256])
                for e,sc in zip(raw[lo:lo+256],result['events']):
                    src,dst=node_ids(e)
                    events.append(dict(event_id=f'{len(events)}:{sc["id"]}',ts=_epoch(e['@timestamp']),
                                       src_id='process:'+src,src_type='PROCESS',dst_id='host:'+dst,dst_type='HOST',
                                       relation='OBSERVED_ON',label=None,tactics=[],apt_stage=None,source_tag=sc['host'],
                                       chain_id=None,causal_parent=None,
                                       attrs=dict(probability=sc['score'],prediction=int(sc['alert']),model_id=row['id'],
                                                  threshold=row['threshold'],image=sc['image'],memory=sc['memory'],
                                                  checkpoint_sha256=created['checkpoint_sha256'])))
            ds=dict(id=did,name=did.replace('_',' '),kind='dedale',source=str(dest),size_bytes=0,
                    estimated_events=len(events),num_raw_events=len(events),created_at=started,last_job_id=f'job-{did}',
                    tags=['sysmon','uploaded','checkpoint-inference'],provenance='User supplied telemetry; ground truth unknown',
                    time_span_s=events[-1]['ts']-events[0]['ts'],is_sample=False)
            item=dict(dataset=ds,events=events,model_id=row['id'])
            # Publish only after the entire chronological stream has succeeded.
            ds['size_bytes']=len(json.dumps(item).encode())
            dest.write_text(json.dumps(item))
            return dict(id=f'job-{did}',dataset_id=did,dataset_name=ds['name'],status='completed',progress_pct=100,
                        started_at=started,completed_at=time.time(),elapsed_s=time.time()-started,
                        description='Saved-checkpoint inference completed',config=dict(kind='dedale',dataset_kind='dedale',
                        source_tag='Sysmon',label_mode='parser',label_propagation=False,strategies=[],out_dir=str(dest),graphs_out=str(dest)))
        finally: b.delete_session(sid)

    @app.post('/api/ui/datasets/process')
    async def process(request: Request):
        body=await request.json()
        return await run_in_threadpool(process_upload,body,request)
