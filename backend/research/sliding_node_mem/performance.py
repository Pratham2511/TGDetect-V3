"""Disposable L4 comparison against the archived implementation, same data/weights."""
import cProfile
import gc
import importlib.util
import io
import json
import os
from pathlib import Path
import pstats
import sys
import time
import torch
from .model import TGNNSlidingNodeMem
from .encoding import token_row
from orthrus_comparison.protocol import reconstruction_query


def compare(manifest, attrs, warmup=32, measured=8):
    spec=importlib.util.spec_from_file_location('sliding_perf_reference',os.environ['SLIDING_REFERENCE_PATH'])
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    torch.manual_seed(0)
    old=module.TGNNSlidingNodeMem(manifest['n_nodes']).cuda()
    new=TGNNSlidingNodeMem(manifest['n_nodes']).cuda();new.load_state_dict(old.state_dict())
    models={'original':old,'optimized':new}
    optimizers={k:torch.optim.Adam(m.parameters(),lr=1e-5,weight_decay=1e-5) for k,m in models.items()}
    times={k:[] for k in models}; losses={k:[] for k in models}; profiles={}; count=0; steps=0
    gc_seconds={k:0. for k in models};gc_counts={k:0 for k in models};gc_started={};active=[None]
    def observe_gc(phase,info):
        generation=info['generation']
        if phase=='start':gc_started[generation]=(time.perf_counter(),active[0])
        elif generation in gc_started:
            started,name=gc_started.pop(generation)
            if name is not None:
                gc_seconds[name]+=time.perf_counter()-started;gc_counts[name]+=1
    gc.callbacks.append(observe_gc)
    for path in manifest['files']['train']:
        g=torch.load(path,map_location='cpu',weights_only=False)
        for lo in range(0,len(g.src),1024):
            hi=min(lo+1024,len(g.src));s=g.src[lo:hi].tolist();d=g.dst[lo:hi].tolist()
            t=(g.t[lo:hi].double()/1e9).tolist();r=g.msg[lo:hi,131:141].argmax(-1)
            events=[reconstruction_query(attrs.get(str(u),{}),attrs.get(str(v),{})) for u,v in zip(s,d)]
            predictions={}
            for name,m in models.items():
                active[0]=name
                torch.cuda.synchronize();started=time.perf_counter()
                profiler=cProfile.Profile() if steps==warmup else None
                if profiler:profiler.enable()
                gc_was_enabled=gc.isenabled()
                if name=='optimized':gc.disable()
                with torch.set_grad_enabled(steps>=warmup):
                    logits,_=m.forward_events(events,s,d,t,r)
                    loss=torch.nn.functional.cross_entropy(logits,r.cuda())
                    if steps>=warmup:
                        loss.backward();optimizers[name].step();optimizers[name].zero_grad(set_to_none=True)
                    m.detach_state()
                if gc_was_enabled:gc.enable()
                if profiler:
                    profiler.disable();output=io.StringIO()
                    pstats.Stats(profiler,stream=output).sort_stats('cumulative').print_stats(25)
                    profiles[name]=output.getvalue()
                torch.cuda.synchronize()
                if steps>=warmup:times[name].append(time.perf_counter()-started);losses[name].append(float(loss.detach()))
                predictions[name]=logits.detach()
            torch.testing.assert_close(predictions['original'],predictions['optimized'],rtol=1e-4,atol=1e-4)
            if steps>=warmup:count+=len(s)
            steps+=1
            if steps>=warmup+measured:break
        if steps>=warmup+measured:break
    result={'status':'passed','implementation':'performance-v4','training_events':count,'warmup_batches':warmup,
            'measured_batches':measured,'same_weights_data_and_optimizer':True,
            'gc_policy':'optimized hot path disables cyclic GC; reference counting remains active', 'arms':{}}
    for name,m in models.items():
        active[0]=name
        gc_was_enabled=gc.isenabled()
        if name=='optimized':gc.disable()
        torch.cuda.synchronize();started=time.perf_counter();snapshot=m.stream_state_dict();torch.cuda.synchronize()
        snapshot_seconds=time.perf_counter()-started
        path=Path('/tmp')/f'sliding_perf_{name}.pt';started=time.perf_counter();torch.save(snapshot,path)
        serialization_seconds=time.perf_counter()-started;size=path.stat().st_size;path.unlink();del snapshot
        if gc_was_enabled:gc.enable()
        result['arms'][name]=dict(seconds=sum(times[name]),events_per_second=count/sum(times[name]),
            snapshot_seconds=snapshot_seconds,serialization_seconds=serialization_seconds,snapshot_bytes=size,
            losses=losses[name],cpu_profile=profiles[name],parameters=sum(p.numel() for p in m.parameters()),
            cyclic_gc_seconds=gc_seconds[name],cyclic_gc_collections=gc_counts[name])
    result['training_speedup']=result['arms']['optimized']['events_per_second']/result['arms']['original']['events_per_second']
    result['snapshot_speedup']=result['arms']['original']['snapshot_seconds']/max(result['arms']['optimized']['snapshot_seconds'],1e-9)
    result['token_cache']=token_row.cache_info()._asdict()
    gc.callbacks.remove(observe_gc)
    return result
