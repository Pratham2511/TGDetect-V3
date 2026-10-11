"""Bounded read-only L4 latency check using the existing prepared CADETS prefix."""
from pathlib import Path
import modal

root = Path(__file__).parent
image = (modal.Image.debian_slim(python_version='3.10')
         .pip_install('torch==2.4.1','torch-geometric==2.5.3','numpy==1.26.4',
                      'scipy==1.10.1','scikit-learn==1.2.0','networkx==2.8.7',
                      'pytest==8.3.5','tqdm==4.67.1')
         .env({'PYTHONPATH':'/root/research','PYTHONHASHSEED':'0'}))
if modal.is_local():
    for name in ('sliding_node_mem_v2','sliding_node_mem_upgrade','sliding_node_mem',
                 'orthrus_comparison','orthrus_official'):
        image=image.add_local_dir(root/name,'/root/research/'+name,ignore=['__pycache__','.git'])
    for name in ('node_memory_v3.py','node_memory_models.py','tgnn_components.py',
                 'temporal_cnn.py','causal_tgn.py','pomsathit_baseline.py'):
        image=image.add_local_file(root/name,'/root/research/'+name)
app = modal.App('tg-detect-sliding-v2-latency')
volume = modal.Volume.from_name('tg-detect-sliding-node-mem-e3-v1')


@app.function(image=image,gpu='L4',cpu=4,memory=16384,timeout=600,volumes={'/study':volume})
def measure():
    import json
    import itertools
    import time
    import torch
    import cProfile, pstats, io
    from torch.profiler import profile,ProfilerActivity
    from sliding_node_mem_v2.benchmark import benchmark, synchronize
    from sliding_node_mem_v2.layers import TemporalCNN
    from sliding_node_mem_upgrade.pipeline import prepared_batches
    from sliding_node_mem_upgrade.model import TGNNSlidingNodeMem as Reference
    from sliding_node_mem_v2.model import TGNNSlidingNodeMem as Optimized
    from sliding_node_mem_v2.runtime import stream_gc
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    manifest=json.loads(Path('/study/prepared/manifest.json').read_text())
    attrs=json.loads(Path(manifest['node_attributes']).read_text())
    results={}
    for batch in (1,32,256,1024):
        samples=[(item[7],item[3],item[4],item[5],item[6]) for item in itertools.islice(
            prepared_batches(manifest['files']['train'],attrs,batch=batch),72)]
        results[str(batch)]=benchmark('cuda',(batch,),repeats=64,warmup=8,
            output=f'/tmp/latency-{batch}.json',input_batches=samples,n_nodes=manifest['n_nodes'])
        print('Completed batch',batch,flush=True)
    layers={}
    with torch.no_grad():
        for batch in (1,32,256,1024):
            x=torch.randn(batch,9,128,device='cuda'); mask=torch.ones(batch,9,dtype=torch.bool,device='cuda')
            conv=TemporalCNN(backend='conv').cuda();linear=TemporalCNN(backend='linear').cuda()
            linear.load_state_dict(conv.state_dict())
            torch.testing.assert_close(conv(x,mask),linear(x,mask),rtol=1e-4,atol=1e-5)
            layers[str(batch)]={}
            for name,m in [('conv',conv),('linear',linear)]:
                for _ in range(5):m(x,mask)
                synchronize('cuda');start=time.perf_counter()
                for _ in range(40):m(x,mask)
                synchronize('cuda')
                layers[str(batch)][name]=(time.perf_counter()-start)*1000/40
    torch.manual_seed(42)
    reference=Reference(manifest['n_nodes']).cuda().eval()
    compatible=Optimized(manifest['n_nodes'],profile='compatible').cuda().eval()
    compatible.load_state_dict(reference.state_dict())
    with stream_gc(),torch.no_grad():
        for data in samples[:4]:
            a=reference.forward_events(*data)[0];b=compatible.forward_events(*data)[0]
            torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-5)
            reference.detach_state();compatible.detach_state()
        cpu=cProfile.Profile();cpu.enable()
        for data in samples[4:8]:reference.forward_events(*data)
        synchronize('cuda');cpu.disable()
        buffer=io.StringIO();pstats.Stats(cpu,stream=buffer).sort_stats('cumulative').print_stats(30)
        with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
            reference.forward_events(*samples[8])
            synchronize('cuda')
    return dict(gpu=torch.cuda.get_device_name(),results=results,cnn_ms=layers,
                parity='compatible outputs passed on four chronological real batches',
                cpu_profile=buffer.getvalue(),operator_profile=prof.key_averages().table(sort_by='self_cpu_time_total',row_limit=30),
                cuda_profile=prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=30),
                scope='Read-only real training prefixes; no optimizer steps, no attack accuracy measurement')


@app.local_entrypoint()
def main():
    import json
    result=measure.remote()
    out=root.parent/'artifacts/sliding_v2/gpu_latency.json'
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
