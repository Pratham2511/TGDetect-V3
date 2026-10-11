"""Authors' ORTHRUS architecture with V3's held-out selection/calibration protocol."""
import json
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from orthrus_comparison.official import OrthrusReference
from orthrus_comparison.run import files,sha256,measure
from orthrus_comparison.protocol import require_coverage
from sliding_node_mem_v2.scoring import EdgeWriter,read_edges,calibrated_scores


def run(manifest_path,output,epochs=5,batch=1024,seed=0,on_checkpoint=None):
    torch.set_num_threads(4);torch.manual_seed(seed);np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    manifest=json.loads(Path(manifest_path).read_text());require_coverage(manifest['days'])
    if manifest.get('dataset')!='CADETS_E3' or not manifest.get('official_preprocessing'):
        raise ValueError('Official prepared CADETS required')
    for p,h in manifest['file_sha256'].items():
        if sha256(p)!=h:raise ValueError('Data checksum mismatch: '+p)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    selection=sum(len(g.src) for g in files(manifest,'val'))//3
    model=OrthrusReference(manifest['n_nodes'],device='cuda')
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-5,weight_decay=1e-5)
    signature=dict(manifest=sha256(manifest_path),source=sha256(__file__),epochs=epochs,batch=batch,seed=seed)
    history=[];best=float('inf');first=0;best_epoch=0
    if (out/'resume.pt').exists():
        saved=torch.load(out/'resume.pt',map_location='cpu',weights_only=False)
        if saved['signature']!=signature:raise ValueError('Comparison resume mismatch')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        history=saved['history'];best=saved['best'];first=saved['epoch'];best_epoch=saved['best_epoch']
        torch.set_rng_state(saved['rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])

    def stream(split,training=False,limit=None,write=False):
        model.train(training);model.reset_state();count=0;total=torch.zeros((),device='cuda')
        full=SimpleNamespace(edge_type=torch.cat([g.msg[:,131:141] for g in files(manifest,split)]))
        writer=EdgeWriter(out/(split+'_edges.bin')) if write else None
        try:
            for graph in files(manifest,split):
                for lo in range(0,len(graph.src),batch):
                    if limit is not None and count>=limit:break
                    hi=min(lo+batch,len(graph.src),lo+limit-count if limit is not None else len(graph.src))
                    src,dst=graph.src[lo:hi],graph.dst[lo:hi]
                    times=graph.t[lo:hi].double()/1e9;msg=graph.msg[lo:hi]
                    with torch.set_grad_enabled(training):
                        loss=model(src.cuda(),dst.cuda(),times.cuda(),msg[:,:131].cuda(),msg[:,141:].cuda(),msg[:,131:141].cuda(),full,inference=True)
                        if training:
                            optimizer.zero_grad(set_to_none=True);loss.mean().backward();optimizer.step()
                    total+=loss.detach().sum();count+=len(src)
                    if writer:writer.append(src.numpy(),dst.numpy(),times.numpy(),msg[:,131:141].argmax(-1).numpy(),loss.detach().cpu().numpy())
                if limit is not None and count>=limit:break
        finally:
            if writer:writer.close()
        return float(total)/count

    for epoch in range(first,epochs):
        started=time.perf_counter();loss=stream('train',training=True);seconds=time.perf_counter()-started
        validation=stream('val',limit=selection)
        if validation<best:
            best=validation;best_epoch=epoch+1;torch.save(model.state_dict(),out/'best.pt')
        history.append(dict(epoch=epoch+1,loss=loss,validation_ce=validation,seconds=seconds))
        torch.save(dict(signature=signature,model=model.state_dict(),optimizer=optimizer.state_dict(),
                        history=history,best=best,epoch=epoch+1,best_epoch=best_epoch,rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all()),out/'resume.tmp')
        (out/'resume.tmp').replace(out/'resume.pt')
        (out/'progress.json').write_text(json.dumps(history,indent=2))
        if on_checkpoint:on_checkpoint()
        print('ORTHRUS',history[-1],flush=True)
    model.load_state_dict(torch.load(out/'best.pt',map_location='cpu',weights_only=False))
    started=time.perf_counter();val=stream('val',write=True);test=stream('test',write=True)
    threshold,scores,calibration=calibrated_scores(read_edges(out/'val_edges.bin')[selection:],read_edges(out/'test_edges.bin'),.001)
    truth=set(map(int,json.loads(Path(manifest['ground_truth_nodes']).read_text())))
    result=dict(status='complete',signature=signature,history=history,checkpoint_epoch=best_epoch,
        parameter_count=sum(p.numel() for p in model.parameters()),metrics=measure(scores,truth,{n for n,s in scores.items() if s>threshold}),
        validation_ce=val,test_ce=test,scoring_seconds=time.perf_counter()-started,
        unseen_ground_truth=sorted(truth-set(scores)),calibration=calibration,
        note='Unchanged ORTHRUS neural architecture; matched V3 benign validation checkpoint selection and calibrated node evidence, not released top-30 rule.')
    (out/'result.json').write_text(json.dumps(result,indent=2));return result
