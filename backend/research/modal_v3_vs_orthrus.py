"""Fresh paired L4 inference comparison on identical CADETS event prefixes."""
from pathlib import Path
import modal
from modal_sliding_v2_benchmark import image as base_image,volume
image=base_image
if modal.is_local():
    root=Path(__file__).parent
    image=image.add_local_dir(root/'sliding_node_mem_v3','/root/research/sliding_node_mem_v3',ignore=['__pycache__'])
    image=image.add_local_file(root/'modal_sliding_v2_benchmark.py','/root/research/modal_sliding_v2_benchmark.py')
app=modal.App('tg-detect-v3-vs-orthrus')


@app.function(image=image,gpu='L4',cpu=4,memory=32768,timeout=1800,volumes={'/study':volume})
def latency():
    import gc,hashlib,itertools,json,time
    from types import SimpleNamespace
    import numpy as np
    import torch
    from sliding_node_mem_upgrade.pipeline import prepared_batches
    from sliding_node_mem_v3.benchmark import pack_samples
    from sliding_node_mem_v3.model import TGNNSlidingNodeMem
    from sliding_node_mem_v2.runtime import stream_gc
    from orthrus_comparison.official import OrthrusReference
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    manifest=json.loads(Path('/study/prepared/manifest.json').read_text())
    attrs=json.loads(Path(manifest['node_attributes']).read_text())
    results={}
    for size in (32,256,1024):
        items=list(itertools.islice(prepared_batches(manifest['files']['train'],attrs,batch=size,include_messages=True),72))
        samples=[(x[7],x[3],x[4],x[5],x[6]) for x in items]
        bank,batches,signatures,offline=pack_samples(samples,f'/tmp/prefix-{size}',False)
        bank.to_device('cuda')
        full=SimpleNamespace(edge_type=torch.cat([x[8][:,131:141] for x in items]))
        measured={name:[] for name in ('ORTHRUS','V3','V3_frozen')}
        for repeat in range(3):
            order=list(measured)
            order=order[repeat:]+order[:repeat]
            for name in order:
                gc.collect();torch.cuda.empty_cache();torch.manual_seed(42)
                model=(OrthrusReference(manifest['n_nodes'],device='cuda') if name=='ORTHRUS' else
                       TGNNSlidingNodeMem(manifest['n_nodes'],bank,signatures=signatures).cuda()).eval()
                torch.cuda.synchronize();start=time.perf_counter()
                table=model.event_encoder.precompute() if name=='V3_frozen' else 0
                torch.cuda.synchronize();setup=time.perf_counter()-start
                torch.cuda.reset_peak_memory_stats();durations=[]
                with stream_gc(),torch.no_grad():
                    for i,item in enumerate(items):
                        torch.cuda.synchronize();start=time.perf_counter()
                        if name=='ORTHRUS':
                            msg=item[8].cuda()
                            model(torch.tensor(item[3],device='cuda'),torch.tensor(item[4],device='cuda'),
                                  torch.tensor(item[5],dtype=torch.float64,device='cuda'),msg[:,:131],msg[:,141:],msg[:,131:141],full,inference=True)
                        else:
                            logits,aux=model.forward_batch(batches[i]);model.detach_state()
                        torch.cuda.synchronize()
                        if i>=8:durations.append(1000*(time.perf_counter()-start))
                measured[name].append(dict(round=repeat,p50_ms=float(np.median(durations)),p95_ms=float(np.quantile(durations,.95)),
                    p99_ms=float(np.quantile(durations,.99)),events_per_second=sum(len(x[3]) for x in items[8:])*1000/sum(durations),
                    peak_cuda_bytes=torch.cuda.max_memory_allocated(),parameters=sum(p.numel() for p in model.parameters()),
                    table_bytes=table,precompute_seconds=setup,batch_ms=durations))
                del model
        results[str(size)]=dict(arms=measured,identities=bank.config['identities'],offline_tokenization_seconds=offline)
        print('Completed batch',size,{k:[round(x['p50_ms'],3) for x in v] for k,v in measured.items()},flush=True)
        del bank,full,items,batches
    return dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,results=results,
        manifest_sha256=hashlib.sha256(Path('/study/prepared/manifest.json').read_bytes()).hexdigest(),
        scope='Inference only, random seeded weights; 3 rotated-order repetitions, 8 warmup +64 measured batches. Identical chronological events, prepared features including H2D and state updates. No accuracy claim.',
        v3_config=dict(profile='corrected',temporal='cnn',short_readout='cnn',static_short_read=False,temporal_graphs=False),
        source_hashes={str(p.relative_to('/root/research')):hashlib.sha256(p.read_bytes()).hexdigest()
                       for folder in ('sliding_node_mem_v3','orthrus_comparison','orthrus_official/src')
                       for p in (Path('/root/research')/folder).rglob('*.py')})


@app.local_entrypoint()
def main():
    import json
    result=latency.remote()
    out=Path(__file__).resolve().parents[1]/'artifacts/v3_vs_orthrus'
    out.mkdir(parents=True,exist_ok=True)
    (out/'latency.json').write_text(json.dumps(result,indent=2))
    print('Saved',out/'latency.json')
