"""Paired real-prefix inference ablations; separate instrumented CUDA profile."""
import json
import hashlib
import time
from pathlib import Path
import numpy as np
import torch
from .benchmark import pack_samples, synchronize
from .model import TGNNSlidingNodeMem
from sliding_node_mem_v2.runtime import stream_gc


def measure(samples,n_nodes,output,device='cuda',warmup=8):
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    bank,batches,signatures,offline=pack_samples(samples,root/'tokens',False)
    bank.to_device(device)
    arms={
        'baseline':{},
        'lookup':{'lookup':True},
        'lookup_static':{'lookup':True,'static_short_read':True},
        'lookup_graphs':{'lookup':True,'static_short_read':True,'temporal_graphs':True},
        'lookup_attention':{'lookup':True,'static_short_read':True,'short_readout':'attention'},
        'lookup_no_short_read':{'lookup':True,'static_short_read':True,'short_readout':'none'},
        'lookup_causal_attention':{'lookup':True,'static_short_read':True,'short_readout':'attention','temporal':'attention'},
    }
    results={};reference=[]
    for name,options in arms.items():
        options=dict(options);lookup=options.pop('lookup',False)
        torch.manual_seed(42)
        model=TGNNSlidingNodeMem(n_nodes,bank,signatures=signatures,**options).to(device).eval()
        synchronize(device);start=time.perf_counter()
        table_bytes=model.event_encoder.precompute() if lookup else 0
        synchronize(device);precompute=time.perf_counter()-start
        durations=[];max_error=0.
        with stream_gc(),torch.no_grad():
            for i,batch in enumerate(batches):
                synchronize(device);start=time.perf_counter()
                logits,_=model.forward_batch(batch);model.detach_state();synchronize(device)
                elapsed=1000*(time.perf_counter()-start)
                if i>=warmup:durations.append(elapsed)
                if name=='baseline':reference.append(logits.cpu())
                elif name in ('lookup','lookup_static','lookup_graphs'):
                    expected=reference[i].to(device)
                    torch.testing.assert_close(logits,expected,rtol=3e-4,atol=3e-5)
                    max_error=max(max_error,float((logits-expected).abs().max()))
        results[name]=dict(p50_ms=float(np.median(durations)),p95_ms=float(np.quantile(durations,.95)),
                           p99_ms=float(np.quantile(durations,.99)),batch_ms=durations,
                           table_bytes=table_bytes,precompute_seconds=precompute,
                           max_logit_error=max_error if name in ('lookup','lookup_static','lookup_graphs') else None)
        del model
    # Profile separately: instrumentation overhead must not contaminate timings.
    torch.manual_seed(42)
    model=TGNNSlidingNodeMem(n_nodes,bank,signatures=signatures).to(device).eval()
    spans=[];handles=[]
    encoder_rows=[];encoder_events=[]
    handles.append(model.event_encoder.register_forward_pre_hook(
        lambda module,args: encoder_rows.append(torch.unique(args[0]).numel())))
    if str(device).startswith('cuda'):
        def encoder_start(module,args):
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record();encoder_events.append((start,end))
        def encoder_end(module,args,result):encoder_events[-1][1].record()
        handles.extend([model.event_encoder.register_forward_pre_hook(encoder_start),
                        model.event_encoder.register_forward_hook(encoder_end)])
    for name,module in [('encoder',model.event_encoder),('gat1',model.gat),
                        ('gat2',model.gat_second),('cnn',model.cnn)]:
        if module is None:continue
        def before(module,args,label=name):
            span=torch.profiler.record_function(label);span.__enter__();spans.append(span)
        def after(module,args,result):spans.pop().__exit__(None,None,None)
        handles.extend([module.register_forward_pre_hook(before),module.register_forward_hook(after)])
    activities=[torch.profiler.ProfilerActivity.CPU]
    if str(device).startswith('cuda'):activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.no_grad():
        for batch in batches[:warmup]:model.forward_batch(batch);model.detach_state()
        with torch.profiler.profile(activities=activities,record_shapes=True) as prof:
            model.forward_batch(batches[warmup]);model.detach_state();synchronize(device)
        for batch in batches[warmup+1:]:model.forward_batch(batch);model.detach_state()
        synchronize(device)
    for handle in handles:handle.remove()
    prof.export_chrome_trace(str(root/'trace.json'))
    table=prof.key_averages().table(sort_by='self_cuda_time_total' if str(device).startswith('cuda') else 'self_cpu_time_total',row_limit=35)
    (root/'profile.txt').write_text(table)
    sizes=np.diff(bank.offsets)
    lengths=(bank.fields['word'][...,0]!=0).sum(-1)
    result=dict(results=results,profile=table,identities=bank.config['identities'],
                profiled_unique_identity_rows=encoder_rows[warmup],
                encoder_batches=[dict(batch=i,unique_rows=rows,
                    cuda_span_ms=encoder_events[i][0].elapsed_time(encoder_events[i][1]) if encoder_events else None)
                    for i,rows in enumerate(encoder_rows)],
                source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
                fields_p99=float(np.quantile(sizes,.99)),subtokens_including_bigrams_p99=float(np.quantile(lengths,.99)),
                offline_seconds=offline,batch=len(batches[0]['src']),warmup=warmup,
                scope='Seeded weights; real chronological prefix. Readout ablations require retraining; no accuracy claim.')
    (root/'result.json').write_text(json.dumps(result,indent=2))
    return result
