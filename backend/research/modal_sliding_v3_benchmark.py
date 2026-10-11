"""Bounded L4 validation of tensor-ring TGNN on existing real prepared inputs."""
from pathlib import Path
import modal
from modal_sliding_v2_benchmark import image as base_image,volume

root=Path(__file__).parent
image=base_image
if modal.is_local():
    image=image.add_local_dir(root/'sliding_node_mem_v3','/root/research/sliding_node_mem_v3',ignore=['__pycache__'])
    image=image.add_local_file(root/'modal_sliding_v2_benchmark.py','/root/research/modal_sliding_v2_benchmark.py')
app=modal.App('tg-detect-sliding-v3-latency')


@app.function(image=image,gpu='L4',cpu=4,memory=16384,timeout=900,volumes={'/study':volume})
def measure():
    import itertools,json
    import torch
    from sliding_node_mem_upgrade.pipeline import prepared_batches
    from sliding_node_mem_v3.benchmark import benchmark
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    manifest=json.loads(Path('/study/prepared/manifest.json').read_text())
    attrs=json.loads(Path(manifest['node_attributes']).read_text())
    print('Declared nodes:',manifest['n_nodes'],'fully dense ring bytes avoided:',manifest['n_nodes']*5208,flush=True)
    results={}
    for size in (32,256,1024):
        samples=[(item[7],item[3],item[4],item[5],item[6]) for item in itertools.islice(
            prepared_batches(manifest['files']['train'],attrs,batch=size),72)]
        results[str(size)]=benchmark(samples,manifest['n_nodes'],f'/tmp/v3-{size}',device='cuda')
        print('Completed',size,{k:round(v['p50_ms'],3) for k,v in results[str(size)]['results'].items()},flush=True)
    from sliding_node_mem_v2.benchmark import synthetic_batch
    from sliding_node_mem_v3.benchmark import pack_samples
    from sliding_node_mem_v3.model import TGNNSlidingNodeMem
    from sliding_node_mem_v3.run import make_optimizer,train_unroll
    bank,batches,signatures,_=pack_samples([synthetic_batch(32,i) for i in range(4)],'/tmp/v3-training-smoke',False)
    bank.to_device('cuda')
    # This complete synthetic fixture has only u->(u+64) pairs. The cyclic
    # corruption is therefore a known non-edge, with the same destination type.
    for data in batches:
        data['negative_identity']=data['dst_identity'].roll(1)
        data['negative_valid']=data['negative_identity']!=data['dst_identity']
    model=TGNNSlidingNodeMem(256,bank,signatures=signatures).cuda()
    optimizer=make_optimizer(model)
    losses=[]
    for start in (0,2):
        loss,_=train_unroll(model,optimizer,batches[start:start+2]);losses.append(float(loss))
        assert model.pair_head.weight.grad.abs().sum()>0
        assert model.write_src[0].weight.grad.abs().sum()>0
    model.eval();snapshot=model.stream_state_dict()
    data={k:v.clone() for k,v in batches[0].items()};data['time']+=128
    with torch.no_grad():
        expected=model.forward_batch(data)[0]
        model.load_stream_state_dict(snapshot)
        torch.testing.assert_close(expected,model.forward_batch(data)[0],rtol=1e-4,atol=1e-5)
    results['training_smoke']=dict(losses=losses,status='CUDA unroll, pair/memory gradients and snapshot replay passed',scope='synthetic; no accuracy measurement')
    return results


@app.local_entrypoint()
def main():
    import json
    result=measure.remote()
    path=root.parent/'artifacts/sliding_v3/gpu_latency.json';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2))
    print('Saved',path)
