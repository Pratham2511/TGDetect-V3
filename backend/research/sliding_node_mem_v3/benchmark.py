"""Prepared-input benchmark; preprocessing is timed separately from inference."""
import gc
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from sliding_node_mem_v2.model import TGNNSlidingNodeMem as V2
from sliding_node_mem_v2.runtime import stream_gc
from .tokens import TokenWriter,TokenBank
from .model import TGNNSlidingNodeMem


def pack_samples(samples,path,compatible):
    started=time.perf_counter();writer=TokenWriter(path,two_hash=not compatible,ordered=not compatible)
    signatures={};packed=[]
    for events,src,dst,times,relations in samples:
        columns={k:[] for k in ('query','observed','signature','src_identity','dst_identity')}
        for event,relation in zip(events,relations):
            attrs=event.get('attributes') or {};s=attrs.get('src',{});d=attrs.get('dst',{})
            key=(int(relation),str(s.get('type')),str(d.get('type')))
            if key not in signatures:signatures[key]=len(signatures)
            columns['signature'].append(signatures[key])
            columns['query'].append(writer.add(dict(event,type=None)))
            columns['observed'].append(writer.add(dict(event,type=str(int(relation)))))
            columns['src_identity'].append(writer.add({'attributes':s}))
            columns['dst_identity'].append(writer.add({'attributes':d}))
        columns.update(src=src,dst=dst,time=times,relation=relations,
                       negative_identity=columns['dst_identity'],negative_valid=[False]*len(src))
        packed.append({k:torch.as_tensor(v,dtype=torch.float64 if k=='time' else torch.bool if k=='negative_valid' else torch.long) for k,v in columns.items()})
    writer.close()
    return TokenBank(path),packed,len(signatures),time.perf_counter()-started


def synchronize(device):
    if str(device).startswith('cuda'):torch.cuda.synchronize()


def benchmark(samples,n_nodes,output,device='cpu',warmup=8,repeats=64):
    torch.set_num_threads(1)
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    legacy,batches,signatures,legacy_seconds=pack_samples(samples,root/'legacy_tokens',True)
    corrected,new_batches,_,new_seconds=pack_samples(samples,root/'corrected_tokens',False)
    if len(samples)<warmup+repeats:raise ValueError('Insufficient chronological batches')
    torch.manual_seed(42)
    reference=V2(n_nodes,profile='compatible').eval()
    weights={k:v.detach().clone() for k,v in reference.state_dict().items()};del reference
    measurements={}
    for name in ('v2_compatible','ring_compatible_memmap','ring_compatible_resident','ring_corrected_resident'):
        gc.collect();torch.manual_seed(42)
        if name=='v2_compatible':model=V2(n_nodes,profile='compatible')
        else:
            bank=corrected if 'corrected' in name else legacy
            if 'resident' in name:bank.to_device(device)
            model=TGNNSlidingNodeMem(n_nodes,bank,signatures=signatures,
                  profile='corrected' if 'corrected' in name else 'compatible')
        model=model.to(device).eval()
        if 'corrected' not in name:model.load_state_dict(weights,strict=False)
        if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats()
        durations=[]
        with stream_gc(),torch.no_grad():
            for i in range(warmup+repeats):
                synchronize(device);start=time.perf_counter()
                if name=='v2_compatible':model.forward_events(*samples[i])
                else:model.forward_batch((new_batches if 'corrected' in name else batches)[i])
                model.detach_state();synchronize(device)
                if i>=warmup:durations.append((time.perf_counter()-start)*1000)
        values=np.asarray(durations)
        measurements[name]=dict(p50_ms=float(np.median(values)),p95_ms=float(np.quantile(values,.95)),
            p99_ms=float(np.quantile(values,.99)),events_per_second=sum(len(s[1]) for s in samples[warmup:warmup+repeats])/values.sum()*1000,
            parameters=sum(p.numel() for p in model.parameters()),
            peak_cuda_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else None,
            batch_ms=durations)
        del model
        if str(device).startswith('cuda'):torch.cuda.empty_cache()
    # Strong GPU parity check uses exact counts; count-min is deliberately
    # approximate and must not be presented as numerically identical.
    a=V2(n_nodes,profile='compatible').to(device).eval();a.load_state_dict(weights)
    b=TGNNSlidingNodeMem(n_nodes,legacy,signatures=signatures,profile='compatible',exact_counts=True).to(device).eval()
    b.load_state_dict(weights,strict=False)
    with stream_gc(),torch.no_grad():
        for i in range(min(4,len(samples))):
            left=a.forward_events(*samples[i])[0];right=b.forward_batch(batches[i])[0]
            torch.testing.assert_close(left,right,rtol=3e-4,atol=3e-5)
            a.detach_state();b.detach_state()
    result=dict(device=str(device),gpu=torch.cuda.get_device_name() if str(device).startswith('cuda') else None,
        n_nodes=n_nodes,warmup=warmup,repeats=repeats,batch=len(samples[0][1]),torch=torch.__version__,
        parameters_note='Two hashes share the token table; independent second GAT and pair head add parameters',
        offline_seconds=dict(legacy=legacy_seconds,corrected=new_seconds),
        identities=dict(legacy=legacy.config['identities'],corrected=corrected.config['identities']),
        results=measurements,parity='Exact-count compatibility passed against V2 on four real chronological batches',
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
        scope='Seeded weights, prepared inputs including ID staging and state maintenance; excludes offline preparation from inference timing; no accuracy claim')
    (root/'result.json').write_text(json.dumps(result,indent=2));return result
