"""Train/pretrain tensor-ring TGNN and calibrate its combined anomaly evidence."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from orthrus_comparison.run import sha256,measure
from sliding_node_mem_v2.scoring import EdgeWriter,read_edges,calibrated_scores
from sliding_node_mem_v2.runtime import ordered_run
from .tokens import TokenBank
from .model import TGNNSlidingNodeMem
from .prepare import read_batches
from .pretrain import pretrain


def make_optimizer(model,lr=1e-5,embedding_lr=1e-4):
    token=model.event_encoder.tokens.weight
    return torch.optim.Adam([{'params':[token],'lr':embedding_lr},
        {'params':[p for p in model.parameters() if p is not token],'lr':lr}],weight_decay=1e-5)


def train_unroll(model,optimizer,batches,pair_weight=.1,compression_weight=.01):
    if not batches:raise ValueError('Empty unroll')
    optimizer.zero_grad(set_to_none=True);losses=[];count=0
    for data in batches:
        logits,aux=model.forward_batch(data);n=len(logits);count+=n
        loss=F.cross_entropy(logits,data['relation'].to(logits.device))+compression_weight*aux['compression_loss']
        if 'pair_loss' in aux:loss=loss+pair_weight*aux['pair_loss']
        losses.append(loss*n)
    loss=torch.stack(losses).sum()/count
    if not torch.isfinite(loss):raise RuntimeError('Nonfinite training objective')
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    optimizer.step();model.detach_state()
    return loss.detach(),count


@ordered_run
def run(prepared,output,epochs=5,batch=1024,unroll_batches=2,device='cpu',seed=0,
        initialization=None,pretrain_epochs=5,random_init=False,embedding_lr=1e-4,
        pair_weight=.1,pair_score_weight=.1,false_alert_rate=.001,resident_tokens=False,
        checkpoint_seconds=300.,on_checkpoint=None,model_options=None,frozen_eval=False):
    if min(epochs,batch,unroll_batches)<1 or min(embedding_lr,pair_weight,pair_score_weight)<0:
        raise ValueError('Invalid training configuration')
    torch.set_num_threads(4);torch.manual_seed(seed);np.random.seed(seed)
    if str(device).startswith('cuda'):
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=Path(prepared);manifest=json.loads((root/'manifest.json').read_text())
    for name,digest in manifest['checksums'].items():
        if sha256(root/name)!=digest:raise ValueError('Prepared checksum mismatch: '+name)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    profile='compatible' if manifest['compatible'] else 'corrected'
    init_seconds=0.
    if initialization is None and not random_init:
        initialization=out/'initial_embeddings.pt'
        if not Path(initialization).exists():
            print('Pretraining embeddings...',flush=True)
            started=time.perf_counter();pretrain(root,initialization,epochs=pretrain_epochs,device=device,seed=seed)
            init_seconds=time.perf_counter()-started
            print(f'Pretraining done in {init_seconds:.1f}s',flush=True)
    torch.manual_seed(seed)
    bank=TokenBank(root/'tokens')
    if resident_tokens:bank.to_device(device)
    model=TGNNSlidingNodeMem(manifest['n_nodes'],bank,num_relations=manifest['relations'],
         signatures=manifest['signatures'],profile=profile,**(model_options or {})).to(device)
    if initialization is not None:
        init=torch.load(initialization,map_location='cpu',weights_only=False)
        if init['source_manifest_sha256']!=sha256(root/'manifest.json') or init['token_config']!=bank.config:
            raise ValueError('Pretrained embeddings belong to different corpus/token configuration')
        with torch.no_grad():model.event_encoder.tokens.weight.copy_(init['tokens'])
    optimizer=make_optimizer(model,embedding_lr=embedding_lr)
    research=Path(__file__).resolve().parents[1]
    sources=[]
    for folder in ('sliding_node_mem_v3','sliding_node_mem_v2','sliding_node_mem_upgrade','orthrus_comparison'):
        sources+=list((research/folder).glob('*.py'))
    sources+=[research/p for p in ('node_memory_v3.py','node_memory_models.py','tgnn_components.py','temporal_cnn.py','causal_tgn.py','pomsathit_baseline.py')]
    signature=dict(prepared_sha256=sha256(root/'manifest.json'),model=model.config,epochs=epochs,batch=batch,
        unroll_batches=unroll_batches,seed=seed,embedding_lr=embedding_lr,pair_weight=pair_weight,
        pair_score_weight=pair_score_weight,false_alert_rate=false_alert_rate,frozen_eval=frozen_eval,
        initialization_sha256=None if initialization is None else sha256(initialization),
        sources={str(p.relative_to(research)):sha256(p) for p in sources})
    for path in sources:
        target=out/'source'/path.relative_to(research);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
    selection_count=manifest['events']['val']//3
    if selection_count<32:raise ValueError('At least 96 validation events required')
    best=float('inf');best_epoch=0;history=[];first=0;resume=None;last_save=time.monotonic()
    if (out/'resume.pt').exists():
        saved=torch.load(out/'resume.pt',map_location='cpu',weights_only=False)
        if saved['signature']!=signature:raise ValueError('Resume source/configuration mismatch')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        first,history,best,best_epoch,resume=(saved[k] for k in ('epoch','history','best','best_epoch','stream'))
        torch.set_rng_state(saved['rng'])
        if str(device).startswith('cuda'):torch.cuda.set_rng_state_all(saved['cuda_rng'])

    def save(epoch,stream=None):
        nonlocal last_save
        value=dict(signature=signature,epoch=epoch,stream=stream,model=model.state_dict(),optimizer=optimizer.state_dict(),
                   history=history,best=best,best_epoch=best_epoch,rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else [])
        torch.save(value,out/'resume.tmp');(out/'resume.tmp').replace(out/'resume.pt')
        last_save=time.monotonic()
        if on_checkpoint:on_checkpoint()

    def evaluate(split,limit=None,write=False):
        model.eval();model.reset_state();count=0;total=torch.zeros((),device=device)
        if frozen_eval and model.event_encoder._frozen_table is None:model.event_encoder.precompute()
        writer=EdgeWriter(out/(split+'_edges.bin')) if write else None
        try:
            with torch.no_grad():
                for data in read_batches(root/(split+'.bin'),batch):
                    if limit is not None:
                        if count>=limit:break
                        data={k:v[:limit-count] for k,v in data.items()}
                    logits,aux=model.forward_batch(data)
                    loss=F.cross_entropy(logits,data['relation'].to(device),reduction='none')
                    total+=loss.sum();count+=len(logits);model.detach_state()
                    if writer:
                        if 'pair_anomaly' in aux:loss=loss+pair_score_weight*aux['pair_anomaly']
                        writer.append(data['src'].numpy(),data['dst'].numpy(),data['time'].numpy(),data['relation'].numpy(),loss.cpu().numpy())
            if not count:raise ValueError('Empty '+split+' split')
            return float(total)/count
        finally:
            if writer:writer.close()

    for epoch in range(first,epochs):
        print(f'Epoch {epoch+1}/{epochs} starting (resume={resume is not None})...',flush=True)
        model.train();model.reset_state();count=0;total=torch.zeros((),device=device);elapsed=0.
        if resume:
            model.load_stream_state_dict(resume['state']);count=resume['count'];total+=resume['total'];elapsed=resume['seconds']
            print(f'  Resumed from batch count={count}',flush=True)
        started=time.perf_counter();pending=[];last_report=0
        for data in read_batches(root/'train.bin',batch,start=count):
            pending.append(data)
            if len(pending)<unroll_batches:continue
            loss,n=train_unroll(model,optimizer,pending,pair_weight);pending=[];total+=loss*n;count+=n
            if count-last_report>=100000:
                print(f'  Epoch {epoch+1}: {count} events, loss={float(total)/count:.4f}, {time.perf_counter()-started+elapsed:.0f}s',flush=True)
                last_report=count
            if time.monotonic()-last_save>=checkpoint_seconds:
                save(epoch,dict(state=model.stream_state_dict(),count=count,total=float(total),seconds=elapsed+time.perf_counter()-started))
                print(f'  Checkpoint saved at {count} events',flush=True)
        if pending:
            loss,n=train_unroll(model,optimizer,pending,pair_weight);total+=loss*n;count+=n
        if not count:raise ValueError('Empty training split')
        seconds=elapsed+time.perf_counter()-started
        print(f'Epoch {epoch+1} train done: {count} events, loss={float(total)/count:.4f}, {seconds:.1f}s. Evaluating...',flush=True)
        validation=evaluate('val',selection_count)
        if validation<best:
            best,best_epoch=validation,epoch+1
            torch.save(dict(signature=signature,model=model.state_dict(),epoch=best_epoch),out/'best.tmp')
            (out/'best.tmp').replace(out/'best.pt')
        history.append(dict(epoch=epoch+1,loss=float(total)/count,validation_ce=validation,seconds=seconds))
        resume=None;save(epoch+1)
        print(f'Epoch {epoch+1}/{epochs} complete: loss={history[-1]["loss"]:.4f} val_ce={validation:.4f} best_epoch={best_epoch} ({seconds:.1f}s)',flush=True)
    print(f'Training complete. Loading best checkpoint (epoch {best_epoch})...',flush=True)
    model.load_state_dict(torch.load(out/'best.pt',map_location='cpu',weights_only=False)['model'])
    print('Running final evaluation on val+test splits...',flush=True)
    started=time.perf_counter();val=evaluate('val',write=True);test=evaluate('test',write=True)
    threshold,scores,calibration=calibrated_scores(read_edges(out/'val_edges.bin')[selection_count:],read_edges(out/'test_edges.bin'),false_alert_rate)
    truth=set(json.loads((root/'ground_truth_nodes.json').read_text()))
    predictions={n for n,value in scores.items() if value>threshold}
    result=dict(status='complete',signature=signature,history=history,checkpoint_epoch=best_epoch,
        parameter_count=sum(p.numel() for p in model.parameters()),validation_ce=val,test_ce=test,
        pretraining_seconds_this_invocation=init_seconds,scoring_seconds=time.perf_counter()-started,
        calibration=calibration,threshold=threshold if np.isfinite(threshold) else None,
        metrics=measure(scores,truth,predictions),predicted_nodes=sorted(predictions),
        unseen_ground_truth=sorted(truth-set(scores)),negative_eligible=manifest['negative_eligible'],
        note='Combined relation CE + weighted pair anomaly is calibrated after fusion; minimum benign CE selects checkpoint. F1 improvements require held-out comparison.')
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--epochs',type=int,default=5)
    p.add_argument('--batch',type=int,default=1024);p.add_argument('--unroll-batches',type=int,default=2)
    p.add_argument('--initialization');p.add_argument('--pretrain-epochs',type=int,default=5)
    p.add_argument('--random-init',action='store_true');p.add_argument('--resident-tokens',action='store_true')
    p.add_argument('--embedding-lr',type=float,default=1e-4);p.add_argument('--pair-weight',type=float,default=.1)
    p.add_argument('--pair-score-weight',type=float,default=.1)
    p.add_argument('--frozen-eval',action='store_true',help='Precompute an FP32 identity lookup for frozen evaluation')
    p.add_argument('--model-options',type=json.loads,help='JSON model options, e.g. {"short_readout":"attention","static_short_read":true}')
    p.add_argument('--seed',type=int,default=0)
    print(json.dumps(run(**vars(p.parse_args())),indent=2))
