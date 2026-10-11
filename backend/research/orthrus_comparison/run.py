"""Matched reconstruction study on officially featurized CADETS E3 graphs.

Input manifest names the exact per-split TemporalData files, node attributes,
node ground truth and date coverage. No attack labels enter optimization or
checkpoint selection. Official baseline architecture is loaded verbatim.
"""
import argparse
from collections import defaultdict, deque
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score, roc_auc_score, matthews_corrcoef, confusion_matrix
from .official import OrthrusReference
from .models import TGNNV3Reconstruction, TGNNNodeMemV4KV
from .protocol import require_coverage, reconstruction_query, partition

ARMS = ('ORTHRUS_Reference','TGNN_Node_Mem_V3_Reconstruction','TGNN_Node_Mem_V4_KV')


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def files(manifest,split):
    # Trust only locally produced upstream artifacts; never deserialize unknown .pt files.
    for path in manifest['files'][split]:
        yield torch.load(path,map_location='cpu',weights_only=False)


def measure(scores, truth, prediction):
    ids=sorted(scores); s=np.array([scores[i] for i in ids]); y=np.array([int(i in truth) for i in ids])
    p=np.array([i in prediction for i in ids],dtype=int)
    tn,fp,fn,tp=map(int,confusion_matrix(y,p,labels=[0,1]).ravel())
    return dict(tn=tn,fp=fp,fn=fn,tp=tp,precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),
                f1=2*tp/max(1,2*tp+fp+fn),mcc=float(matthews_corrcoef(y,p)),
                pr_ap=float(average_precision_score(y,s)) if y.any() else None,
                roc_auc=float(roc_auc_score(y,s)) if len(np.unique(y))==2 else None)


def run(manifest_path, output, epochs=20, batch=1024, device='cpu', arm=ARMS[0], seed=0, on_checkpoint=None):
    torch.set_num_threads(4); torch.manual_seed(seed); np.random.seed(seed)
    manifest=json.loads(Path(manifest_path).read_text())
    require_coverage(manifest['days'])
    if manifest.get('dataset') != 'CADETS_E3' or not manifest.get('official_preprocessing'):
        raise ValueError('Official CADETS E3 preprocessing required')
    out=Path(output)/arm; out.mkdir(parents=True,exist_ok=True)
    for path, expected in manifest.get('file_sha256', {}).items():
        if sha256(path) != expected:
            raise ValueError(f'Dataset checksum mismatch: {path}')
    attrs=json.loads(Path(manifest['node_attributes']).read_text())
    truth=set(map(int,json.loads(Path(manifest['ground_truth_nodes']).read_text())))
    n_nodes=manifest['n_nodes']; official=arm==ARMS[0]; kv=arm==ARMS[2]
    if arm not in ARMS: raise ValueError(arm)
    model=(OrthrusReference(n_nodes,device=device) if official else
           TGNNNodeMemV4KV(n_nodes) if kv else TGNNV3Reconstruction(272,n_nodes)).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-5,weight_decay=1e-5)
    research=Path(__file__).resolve().parents[1]
    source_paths=list(Path(__file__).parent.glob('*.py'))
    source_paths += [research/p for p in ('node_memory_v3.py','node_memory_models.py','tgnn_components.py',
                                         'temporal_cnn.py','causal_tgn.py')]
    source_paths += list((research/'orthrus_official'/'src').rglob('*.py'))
    signature=dict(manifest_sha256=sha256(manifest_path),epochs=epochs,batch=batch,arm=arm,seed=seed,
                   source_hashes={str(p.relative_to(research)):sha256(p) for p in source_paths})
    journal=out/'resume.pt'; history=[]; first=0
    if journal.exists():
        state=torch.load(journal,map_location=device,weights_only=False)
        if signature!=state['signature']: raise ValueError('Resume source/protocol mismatch')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        history=state['history'];first=state['epoch']
        torch.set_rng_state(state['rng'].cpu())
        if device.startswith('cuda'): torch.cuda.set_rng_state_all(state['cuda_rng'])

    def stream(split, training=False, score_file=None):
        model.train(training);model.reset_state()
        history_by_node=defaultdict(lambda:deque(maxlen=4)); records={}; next_id=0
        scores={}; total=0.;count=0; last=-float('inf')
        full=SimpleNamespace(edge_type=torch.cat([g.msg[:,131:141] for g in files(manifest,split)])) if official else None
        handle=Path(score_file).open('w') if score_file else None
        try:
            for graph in files(manifest,split):
                for lo in range(0,len(graph.src),batch):
                    hi=min(lo+batch,len(graph.src))
                    s=graph.src[lo:hi];d=graph.dst[lo:hi];t=graph.t[lo:hi].double()/1e9
                    if len(t) and (t[0]<last or (t[1:]<t[:-1]).any()): raise ValueError('Unsorted graph stream')
                    if any(partition(float(v))!=('validation' if split=='val' else split) for v in t):
                        raise ValueError('Event in wrong paper partition')
                    last=float(t[-1]); msg=graph.msg[lo:hi]; targets=msg[:,131:141].argmax(-1).to(device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.set_grad_enabled(training):
                        if official:
                            losses=model(s.to(device),d.to(device),t.to(device),msg[:,:131].to(device),
                                         msg[:,141:].to(device),msg[:,131:141].to(device),full,inference=True)
                        else:
                            hs=[];age=[];valid=[]
                            for j,(u,v,when) in enumerate(zip(s.tolist(),d.tolist(),t.tolist())):
                                ids=sorted(set(history_by_node[u])|set(history_by_node[v]))
                                ids=[i for i in ids if records[i][0]<when][-4:]
                                # Native V3 encoder expects current at slot 0,
                                # then newest-to-oldest history. Its GRU/CNN
                                # reverse those history slots internally.
                                previous=[records[i] for i in reversed(ids)]
                                query=reconstruction_query(attrs.get(str(u),{}),attrs.get(str(v),{}))
                                qx=msg[j].clone();qx[131:141]=0
                                hs.append([query if kv else qx]+[r[1] if kv else r[2] for r in previous]+[None]*(4-len(ids)))
                                age.append([0.]+[when-r[0] for r in previous]+[0.]*(4-len(ids)))
                                valid.append([True]*(len(ids)+1)+[False]*(4-len(ids)))
                                observed=dict(query,type=str(int(msg[j,131:141].argmax())))
                                records[next_id]=(when,observed,msg[j].clone())
                                history_by_node[u].append(next_id)
                                if u!=v: history_by_node[v].append(next_id)
                                next_id+=1
                            ages=torch.tensor(age,dtype=torch.float32,device=device)
                            if kv: logits,_=model.forward_events(hs,ages,s,d,t)
                            else:
                                x=torch.stack([torch.stack([torch.zeros(272) if r is None else r for r in row]) for row in hs]).to(device)
                                logits,_=model.forward_events(x,torch.tensor(valid,device=device),ages,s,d,t)
                            losses=F.cross_entropy(logits,targets,reduction='none')
                        loss=losses.mean()
                        if not torch.isfinite(loss): raise RuntimeError('Nonfinite reconstruction loss')
                        if training:
                            loss.backward();optimizer.step()
                        if not official: model.detach_state()
                    total+=float(loss.detach())*len(s); count+=len(s)
                    for u,v,when,value in zip(s.tolist(),d.tolist(),t.tolist(),losses.detach().cpu().tolist()):
                        scores[u]=max(scores.get(u,0.),value);scores[v]=max(scores.get(v,0.),value)
                        if handle: handle.write(json.dumps(dict(srcnode=u,dstnode=v,t=when,loss=value))+'\n')
                    # Bound event record cache to references retained by endpoint histories.
                    if not official and next_id % (batch*32)<batch:
                        retained={i for q in history_by_node.values() for i in q}
                        records={i:r for i,r in records.items() if i in retained}
        finally:
            if handle:handle.close()
        if not count: raise ValueError('Empty split')
        return total/count,scores

    for epoch in range(first,epochs):
        started=time.perf_counter()
        loss,_=stream('train',True)
        history.append(dict(epoch=epoch+1,loss=loss,seconds=time.perf_counter()-started))
        payload=dict(signature=signature,epoch=epoch+1,history=history,model=model.state_dict(),
                     optimizer=optimizer.state_dict(),rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else [])
        torch.save(payload,out/'resume.tmp');(out/'resume.tmp').replace(journal)
        (out/'progress.json').write_text(json.dumps(history,indent=2))
        if on_checkpoint: on_checkpoint()
        print(arm,epoch+1,loss,flush=True)
    started=time.perf_counter()
    _,val=stream('val'); _,test=stream('test',score_file=out/'test_edges.jsonl')
    evaluation_seconds=time.perf_counter()-started
    threshold=max(val.values()); above={n for n,v in test.items() if v>threshold}
    # Official release clusters the top 30 nodes WITHOUT applying val threshold.
    selected=sorted(test,key=test.get)[-30:]
    if len(selected)<30: raise ValueError('Too few test nodes for released top-30 evaluation')
    cl=KMeans(n_clusters=2,random_state=0,n_init=10).fit(np.array([test[n] for n in selected]).reshape(-1,1))
    predicted={n for n,label in zip(selected,cl.labels_) if label==cl.cluster_centers_.argmax()}
    result=dict(signature=signature,status='complete',objective='benign relation cross entropy',
                checkpoint_selection='final epoch',threshold=threshold,
                threshold_only=measure(test,truth,above),released_top30_kmeans=measure(test,truth,predicted),
                predicted_nodes=sorted(predicted),threshold_nodes=sorted(above),
                parameter_count=sum(p.numel() for p in model.parameters()),
                training_seconds=sum(h['seconds'] for h in history),evaluation_seconds=evaluation_seconds,
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else None,
                node_scores=test,unseen_ground_truth=sorted(truth-set(test)),
                note='Detector comparison; official dependency tracing must be evaluated separately for full ORTHRUS QoA.')
    (out/'result.json').write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--device',default='cpu');p.add_argument('--batch',type=int,default=1024)
    p.add_argument('--seed',type=int,default=0)
    a=vars(p.parse_args());a['manifest_path']=a.pop('manifest');run(**a)
