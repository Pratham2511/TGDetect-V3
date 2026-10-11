"""Isolated five-epoch V3 versus ORTHRUS on the existing official CADETS split."""
from pathlib import Path
import modal
from modal_sliding_v2_benchmark import image as base_image,volume
image=base_image
if modal.is_local():
    root=Path(__file__).parent
    image=image.add_local_dir(root/'sliding_node_mem_v3','/root/research/sliding_node_mem_v3',ignore=['__pycache__'])
    image=image.add_local_file(root/'modal_sliding_v2_benchmark.py','/root/research/modal_sliding_v2_benchmark.py')
app=modal.App('tg-detect-v3-vs-orthrus-accuracy')
ROOT='/study/v3_vs_orthrus_20261010'


@app.function(image=image,gpu='L4',cpu=8,memory=32768,timeout=86400,volumes={'/study':volume},max_containers=2)
def train(arm):
    import json,time,torch,shutil
    volume.reload();out=Path(ROOT)/arm;out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    if arm=='V3':
        from sliding_node_mem_v3.prepare import prepare
        from sliding_node_mem_v3.run import run
        prepared=Path(ROOT)/'prepared_v3_local'
        if not (prepared/'manifest.json').exists():
            print('Preparing full V3 token/event bank on local scratch',flush=True)
            scratch=Path('/tmp/prepared_v3')
            prepare('/study/prepared/manifest.json',scratch)
            shutil.copytree(scratch,prepared);volume.commit()
        print('Starting V3 training',flush=True)
        result=run(prepared,out,epochs=5,batch=1024,device='cuda',seed=0,
                   frozen_eval=True,on_checkpoint=volume.commit)
    elif arm=='ORTHRUS':
        from sliding_node_mem_v3.compare_orthrus import run
        result=run('/study/prepared/manifest.json',out,epochs=5,batch=1024,seed=0,on_checkpoint=volume.commit)
    else:raise ValueError(arm)
    result.update(gpu=torch.cuda.get_device_name(),total_seconds=time.perf_counter()-started,
                  peak_cuda_bytes=torch.cuda.max_memory_allocated())
    (out/'result.json').write_text(json.dumps(result,indent=2));volume.commit();return result


@app.local_entrypoint()
def main(v3_only:bool=False):
    import json
    calls={arm:train.spawn(arm) for arm in (('V3',) if v3_only else ('V3','ORTHRUS'))}
    out=Path(__file__).resolve().parents[1]/'artifacts/v3_vs_orthrus';out.mkdir(parents=True,exist_ok=True)
    (out/('v3_retry_calls.json' if v3_only else 'calls.json')).write_text(json.dumps({k:v.object_id for k,v in calls.items()},indent=2))
    results={}
    for arm,call in calls.items():
        results[arm]=call.get()
        (out/(arm+'_accuracy.json')).write_text(json.dumps(results[arm],indent=2))
    if not v3_only:(out/'accuracy.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))
