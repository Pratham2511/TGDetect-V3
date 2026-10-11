"""Run the paired latency experiment on the existing prepared real data."""
from pathlib import Path
import modal
from modal_sliding_v2_benchmark import image as base_image,volume
image=base_image
if modal.is_local():
    root=Path(__file__).parent
    image=image.add_local_dir(root/'sliding_node_mem_v3','/root/research/sliding_node_mem_v3',ignore=['__pycache__'])
    image=image.add_local_file(root/'modal_sliding_v2_benchmark.py','/root/research/modal_sliding_v2_benchmark.py')
app=modal.App('tg-detect-v3-profile')


@app.function(image=image,gpu='L4',cpu=4,memory=16384,timeout=900,volumes={'/study':volume})
def measure():
    import itertools,json,torch
    from sliding_node_mem_upgrade.pipeline import prepared_batches
    from sliding_node_mem_v3.profile_latency import measure as profile
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    manifest=json.loads(Path('/study/prepared/manifest.json').read_text())
    attrs=json.loads(Path(manifest['node_attributes']).read_text())
    samples=[(x[7],x[3],x[4],x[5],x[6]) for x in itertools.islice(prepared_batches(manifest['files']['train'],attrs,batch=1024),72)]
    result=profile(samples,manifest['n_nodes'],'/tmp/latency')
    return result,Path('/tmp/latency/trace.json').read_bytes()


@app.local_entrypoint()
def main():
    import json
    result,trace=measure.remote()
    root=Path(__file__).resolve().parents[1]/'artifacts/sliding_v3/encoder_profile'
    root.mkdir(parents=True,exist_ok=True)
    (root/'result.json').write_text(json.dumps(result,indent=2))
    (root/'trace.json').write_bytes(trace)
    print(json.dumps(result,indent=2))
