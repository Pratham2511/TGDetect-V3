"""Chronological V2 training, resumable unrolls and uncapped calibrated alerts."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from orthrus_comparison.protocol import require_coverage, partition
from orthrus_comparison.run import sha256, measure
from sliding_node_mem_upgrade.pipeline import prepared_batches
from .model import TGNNSlidingNodeMem
from .training import train_unroll
from .scoring import EdgeWriter, read_edges, calibrated_scores
from .runtime import ordered_run

ARM = 'TGNN_Sliding_Node_Mem_V2'


@ordered_run
def run(manifest_path, output, epochs=5, batch=1024, unroll_batches=2,
        device='cpu', seed=0, profile='corrected', false_alert_rate=.001,
        window_seconds=900., checkpoint_seconds=300., on_checkpoint=None,
        cnn_backend='auto', model_options=None):
    if epochs < 1 or batch < 1 or unroll_batches < 1:
        raise ValueError('Positive epochs, batch and unroll length required')
    torch.set_num_threads(4); torch.manual_seed(seed); np.random.seed(seed)
    if str(device).startswith('cuda'):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    require_coverage(manifest['days'])
    if manifest.get('dataset') != 'CADETS_E3' or not manifest.get('official_preprocessing'):
        raise ValueError('Official CADETS E3 preprocessing required')
    # Require checksums for every input actually consumed, not just any supplied subset.
    consumed = [p for split in ('train','val','test') for p in manifest['files'][split]]
    consumed += [manifest['node_attributes'],manifest['ground_truth_nodes']]
    for path in consumed:
        expected = manifest['file_sha256'].get(path)
        if expected is None or sha256(path) != expected:
            raise ValueError('Missing or mismatched input checksum: '+path)
    attrs = json.loads(Path(manifest['node_attributes']).read_text())
    validation_count = sum(len(torch.load(path,map_location='cpu',weights_only=False).src)
                           for path in manifest['files']['val'])
    selection_count = validation_count//3
    if selection_count < 32 or validation_count-selection_count < 64:
        raise ValueError('Need >=96 validation events for chronological selection/calibration partitions')
    model = TGNNSlidingNodeMem(manifest['n_nodes'], profile=profile,
                               cnn_backend=cnn_backend, **(model_options or {})).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-5,weight_decay=1e-5)
    out = Path(output)/ARM; out.mkdir(parents=True,exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    sources = list(Path(__file__).parent.glob('*.py'))
    sources += list((root/'sliding_node_mem_upgrade').glob('*.py'))
    sources += list((root/'orthrus_comparison').glob('*.py'))
    sources += [root/p for p in ('node_memory_v3.py','node_memory_models.py','tgnn_components.py',
                                 'causal_tgn.py','pomsathit_baseline.py')]
    signature = dict(arm=ARM,manifest_sha256=sha256(manifest_path),epochs=epochs,batch=batch,
                     unroll_batches=unroll_batches,seed=seed,model=model.config,
                     false_alert_rate=false_alert_rate,window_seconds=window_seconds,
                     validation_partition='first third checkpoint selection, second third relation fit, final third threshold',
                     source_hashes={str(p.relative_to(root)):sha256(p) for p in sources})
    journal = out/'resume.pt'; history = []; start_epoch = 0; resume = None
    best_loss = float('inf'); best_epoch = 0
    if journal.exists():
        saved = torch.load(journal,map_location='cpu',weights_only=False)
        if saved['signature'] != signature:
            raise ValueError('Source/protocol/config mismatch: use a new output directory')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        history, start_epoch, resume = saved['history'],saved['epoch'],saved['stream']
        best_loss,best_epoch = saved['best_loss'],saved['best_epoch']
        torch.set_rng_state(saved['rng'])
        if str(device).startswith('cuda'):
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
    source_dir = out/'source'
    for path in sources:
        dest = source_dir/path.relative_to(root); dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(path.read_bytes())
    last_checkpoint = time.monotonic()

    def save(epoch,stream=None):
        nonlocal last_checkpoint
        saved = dict(signature=signature,model=model.state_dict(),optimizer=optimizer.state_dict(),
                     epoch=epoch,stream=stream,history=history,best_loss=best_loss,best_epoch=best_epoch,
                     rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else [])
        torch.save(saved,out/'resume.tmp'); (out/'resume.tmp').replace(journal)
        (out/'progress.json').write_text(json.dumps(history,indent=2))
        if on_checkpoint:
            on_checkpoint()
        last_checkpoint = time.monotonic()

    def batches(split):
        for item in prepared_batches(manifest['files'][split],attrs,batch=batch):
            _,_,_,src,dst,times,relations,events = item
            if any(partition(t) != ('validation' if split == 'val' else split) for t in times):
                raise ValueError('Event outside declared chronological split')
            yield events,src,dst,times,relations

    def evaluate(split,path=None,limit_events=None):
        model.eval(); model.reset_state()
        total = torch.zeros((),device=device); count = 0; scores = {}
        writer = EdgeWriter(path) if path else None
        started = time.perf_counter()
        try:
            with torch.no_grad():
                for data in batches(split):
                    if limit_events is not None:
                        remaining = limit_events-count
                        if remaining <= 0:
                            break
                        if remaining < len(data[0]):
                            data = tuple(values[:remaining] for values in data)
                    logits,_ = model.forward_events(*data)
                    targets = data[-1].to(device)
                    losses = F.cross_entropy(logits,targets,reduction='none')
                    total += losses.sum(); count += len(losses)
                    model.detach_state()
                    if writer:
                        values = losses.cpu().numpy()
                        writer.append(data[1],data[2],data[3],data[4].numpy(),values)
                        for u,v,value in zip(data[1],data[2],values):
                            scores[u] = max(scores.get(u,0.),float(value))
                            scores[v] = max(scores.get(v,0.),float(value))
            if not count:
                raise ValueError('Empty '+split+' split')
            value = float(total)/count
            if not np.isfinite(value):
                raise RuntimeError('Nonfinite evaluation loss')
            return value,scores,time.perf_counter()-started
        finally:
            if writer:
                writer.close()

    for epoch in range(start_epoch,epochs):
        model.train(); model.reset_state()
        skip = 0; count = 0; accumulated = torch.zeros((),device=device); elapsed = 0.
        if resume:
            model.load_stream_state_dict(resume['state'])
            skip,count,elapsed = resume['next_batch'],resume['count'],resume['seconds']
            accumulated += resume['total']
        started = time.perf_counter(); pending = []; next_batch = skip

        def update():
            nonlocal pending,count,accumulated
            loss,n = train_unroll(model,optimizer,pending)
            accumulated += loss*n; count += n; pending = []

        for index,data in enumerate(batches('train')):
            if index < skip:
                continue
            pending.append(data); next_batch = index+1
            if len(pending) == unroll_batches:
                update()
                if time.monotonic()-last_checkpoint >= checkpoint_seconds:
                    save(epoch,dict(state=model.stream_state_dict(),next_batch=next_batch,count=count,
                                    total=float(accumulated),seconds=elapsed+time.perf_counter()-started))
        if pending:
            update()
        if not count:
            raise ValueError('Empty train split')
        training_loss = float(accumulated)/count
        training_seconds = elapsed+time.perf_counter()-started
        validation_loss,_,validation_seconds = evaluate('val',limit_events=selection_count)
        if validation_loss < best_loss:
            best_loss,best_epoch = validation_loss,epoch+1
            torch.save(dict(signature=signature,epoch=best_epoch,model=model.state_dict()),out/'best.tmp')
            (out/'best.tmp').replace(out/'best.pt')
        history.append(dict(epoch=epoch+1,loss=training_loss,validation_loss=validation_loss,
                            seconds=training_seconds,validation_seconds=validation_seconds,
                            events=count,optimizer_steps=(next_batch+unroll_batches-1)//unroll_batches))
        resume = None; save(epoch+1)

    # Checkpoint selection uses benign validation loss, never attack labels.
    selected = torch.load(out/'best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(selected['model'])
    val_loss,val_scores,val_seconds = evaluate('val',out/'val_edges.bin')
    test_loss,test_scores,test_seconds = evaluate('test',out/'test_edges.bin')
    started = time.perf_counter()
    threshold,scores,calibration = calibrated_scores(read_edges(out/'val_edges.bin')[selection_count:],read_edges(out/'test_edges.bin'),
                                                     false_alert_rate,window_seconds)
    predicted = {node for node,score in scores.items() if score > threshold}
    truth = set(map(int,json.loads(Path(manifest['ground_truth_nodes']).read_text())))
    raw_threshold = max(val_scores.values())
    result = dict(signature=signature,status='complete',checkpoint_selection='minimum benign validation loss',
                  checkpoint_epoch=best_epoch,validation_loss=val_loss,test_loss=test_loss,
                  calibrated=measure(scores,truth,predicted),node_scores=scores,predicted_nodes=sorted(predicted),
                  calibrated_threshold=threshold if np.isfinite(threshold) else None,calibration=calibration,
                  raw_threshold_only=measure(test_scores,truth,{n for n,s in test_scores.items() if s>raw_threshold}),
                  parameter_count=sum(p.numel() for p in model.parameters()),history=history,
                  training_seconds=sum(h['seconds'] for h in history),
                  scoring_and_io_seconds=val_seconds+test_seconds,
                  calibration_and_metrics_seconds=time.perf_counter()-started,
                  unseen_ground_truth=sorted(truth-set(scores)),
                  note='No top-k cap. Disjoint chronological benign validation partitions for checkpoint selection, relation calibration and threshold. Improvements require held-out evaluation.')
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    if on_checkpoint:
        on_checkpoint()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',dest='manifest_path',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--epochs',type=int,default=5)
    parser.add_argument('--batch',type=int,default=1024)
    parser.add_argument('--unroll-batches',type=int,default=2)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--profile',choices=['compatible','corrected'],default='corrected')
    parser.add_argument('--cnn-backend',choices=['auto','conv','linear'],default='auto')
    parser.add_argument('--false-alert-rate',type=float,default=.001)
    parser.add_argument('--window-seconds',type=float,default=900.)
    run(**vars(parser.parse_args()))
