import copy
from collections import defaultdict,deque
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from sliding_node_mem_v3.tokens import TokenWriter,TokenBank,PackedEncoder
from sliding_node_mem_v3.model import TGNNSlidingNodeMem
from sliding_node_mem_v3.rings import CountTable
from sliding_node_mem_v2.model import TGNNSlidingNodeMem as V2
from sliding_node_mem_v2.encoding import EventEncoder


@pytest.fixture(autouse=True)
def setup():
    torch.set_num_threads(1);torch.manual_seed(42)


def fixture(tmp_path,n=96,corrected=False):
    writer=TokenWriter(tmp_path/'tokens',buckets=256,two_hash=corrected,ordered=corrected)
    rng=np.random.default_rng(42)
    src=rng.integers(0,5,n).tolist();dst=rng.integers(0,5,n).tolist()
    # Repeated nodes, self-edges, long hot-node runs and equal timestamps.
    src[10:35]=[0]*25;dst[20:30]=[0]*10
    times=(np.arange(n)//2).astype(float).tolist();relations=(np.arange(n)%3).tolist()
    events=[{'attributes':{'src':{'type':'process','path':f'/bin/p{u}'},'dst':{'type':'file','path':f'/tmp/f{v}'}}} for u,v in zip(src,dst)]
    columns=defaultdict(list)
    for i,event in enumerate(events):
        columns['query'].append(writer.add(dict(event,type=None)))
        columns['observed'].append(writer.add(dict(event,type=str(relations[i]))))
        columns['src_identity'].append(writer.add({'attributes':event['attributes']['src']}))
        columns['dst_identity'].append(writer.add({'attributes':event['attributes']['dst']}))
        columns['negative_identity'].append(writer.add({'attributes':{'type':'file','path':f'/other/{i}'}}))
    writer.close()
    columns.update(src=src,dst=dst,time=times,relation=relations,signature=relations,negative_valid=[True]*n)
    batch={k:torch.tensor(v,dtype=torch.float64 if k=='time' else torch.bool if k=='negative_valid' else torch.long) for k,v in columns.items()}
    return TokenBank(tmp_path/'tokens'),batch,(events,src,dst,times,relations)


def take(batch,lo,hi):return {k:v[lo:hi] for k,v in batch.items()}


def test_exact_negative_pair_index_preserves_samples(tmp_path):
    from sliding_node_mem_v3.prepare import Negatives
    attrs={str(i):{'type':str(i%2)} for i in range(20)}
    sql=Negatives(tmp_path/'sql.sqlite',attrs)
    ram=Negatives(tmp_path/'ram.sqlite',attrs,in_memory_pairs=True)
    src=[i//20 for i in range(300)];dst=[i%20 for i in range(300)]
    src.extend(range(15,20));dst.extend(range(15,20))
    for sampler in (sql,ram):sampler.observe(src,dst);sampler.finish()
    for i in range(1000):assert sql.sample(i%20,(i*7)%20)==ram.sample(i%20,(i*7)%20)
    for sampler in (sql,ram):sampler.db.close()


def test_frozen_encoder_invalidation_and_training(tmp_path):
    bank,batch,_=fixture(tmp_path,corrected=True)
    encoder=PackedEncoder(bank).eval()
    ids=batch['query']
    with torch.no_grad():expected=encoder(ids)
    assert encoder.precompute(batch_size=7)==bank.config['identities']*64*4
    with torch.no_grad():torch.testing.assert_close(encoder(ids),expected)
    # eval with gradients must bypass the detached table.
    encoder(ids).square().mean().backward()
    assert encoder.tokens.weight.grad.abs().sum()>0
    with torch.no_grad():encoder.output[0].weight.add_(.01)
    with torch.no_grad():encoder(ids)
    assert encoder._frozen_table is None
    encoder.precompute();encoder.train()
    assert encoder._frozen_table is None
    encoder.eval();encoder.precompute();encoder.load_state_dict(encoder.state_dict())
    assert encoder._frozen_table is None


def test_causal_attention_padding_and_future_independence():
    from sliding_node_mem_v3.temporal import CausalAttention
    model=CausalAttention()
    x=torch.randn(3,9,128,requires_grad=True)
    valid=torch.tensor([[False]*9,[False]*4+[True]*5,[True]*9])
    y=model(x,valid)
    assert torch.isfinite(y).all() and (y[~valid]==0).all()
    changed=x.detach().clone();changed[:,7:]+=100
    torch.testing.assert_close(y[:,:7],model(changed,valid)[:,:7])
    y.square().mean().backward()
    assert torch.isfinite(x.grad).all()


def test_temporal_graph_cpu_and_training_fallback():
    from sliding_node_mem_v3.graphs import TemporalGraphs
    from sliding_node_mem_v3.temporal import CausalAttention
    module=CausalAttention();graphs=TemporalGraphs()
    x=torch.randn(2,9,128,requires_grad=True);valid=torch.ones(2,9,dtype=torch.bool)
    torch.testing.assert_close(graphs(module,x,valid),module(x,valid))
    graphs(module,x,valid).square().mean().backward()
    assert x.grad is not None and not graphs.entries


@pytest.mark.parametrize('readout',['cnn','attention','none'])
def test_static_short_read_sequential_and_gradients(tmp_path,readout):
    bank,batch,_=fixture(tmp_path,corrected=True)
    a=TGNNSlidingNodeMem(8,bank,num_relations=3,signatures=3,short_readout=readout).eval()
    b=TGNNSlidingNodeMem(8,bank,num_relations=3,signatures=3,short_readout=readout,static_short_read=True).eval()
    b.load_state_dict(a.state_dict())
    losses=[0,0]
    for lo,hi in [(0,1),(1,20),(20,48),(48,96)]:
        x=a.forward_batch(take(batch,lo,hi))[0];y=b.forward_batch(take(batch,lo,hi))[0]
        torch.testing.assert_close(x,y,rtol=3e-4,atol=3e-5)
        losses[0]=losses[0]+x.square().mean();losses[1]=losses[1]+y.square().mean()
    for loss in losses:loss.backward()
    for (name,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad,q.grad,rtol=2e-3,atol=3e-5,msg=name)


@pytest.mark.parametrize('resident',[False,True])
def test_offline_encoder_matches_online_and_gradients(tmp_path,resident):
    bank,batch,args=fixture(tmp_path)
    if resident:bank.to_device('cpu')
    online=EventEncoder(buckets=256,ordered_values=False,cache_size=0)
    offline=PackedEncoder(bank);offline.load_state_dict(online.state_dict())
    a=online(args[0]);b=offline(batch['query'])
    torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
    a.square().mean().backward();b.square().mean().backward()
    for x,y in zip(online.parameters(),offline.parameters()):
        torch.testing.assert_close(x.grad,y.grad,rtol=3e-4,atol=3e-6)


def test_compatible_sequential_equivalence_against_current_v2(tmp_path):
    bank,batch,args=fixture(tmp_path)
    current=V2(8,num_relations=3,buckets=256,profile='compatible',cnn_backend='conv',compact_every=0).eval()
    ring=TGNNSlidingNodeMem(8,bank,num_relations=3,signatures=3,profile='compatible',exact_counts=True).eval()
    ring.load_state_dict(current.state_dict(),strict=False)
    expected=[]
    with torch.no_grad():
        for i in range(len(args[0])):
            logits,aux=current.forward_events(*[v[i:i+1] for v in args]);expected.append(logits)
            current.detach_state()
        actual=[]
        for lo,hi in ((0,17),(17,18),(18,71),(71,96)):
            logits,aux=ring.forward_batch(take(batch,lo,hi));actual.append(logits);ring.detach_state()
        torch.testing.assert_close(torch.cat(actual),torch.cat(expected),rtol=2e-4,atol=3e-5)


@pytest.mark.parametrize('corrected',[False,True])
def test_chunk_invariance_causality_and_snapshot(tmp_path,corrected):
    bank,batch,args=fixture(tmp_path,corrected=corrected)
    model=TGNNSlidingNodeMem(8,bank,num_relations=3,signatures=3,
                            profile='corrected' if corrected else 'compatible',count_width=32).eval()
    with torch.no_grad():
        full=model.forward_batch(batch)[0]
        model.reset_state();single=[]
        for i in range(96):single.append(model.forward_batch(take(batch,i,i+1))[0])
        torch.testing.assert_close(full,torch.cat(single),rtol=3e-4,atol=4e-5)
        model.reset_state();model.forward_batch(take(batch,0,48));saved=model.stream_state_dict()
        expected=model.forward_batch(take(batch,48,96))[0]
        model.load_stream_state_dict(saved)
        torch.testing.assert_close(expected,model.forward_batch(take(batch,48,96))[0])
        model.reset_state();changed={k:v.clone() for k,v in batch.items()}
        changed['relation'][-1]=2;changed['observed'][-1]=0
        torch.testing.assert_close(full,model.forward_batch(changed)[0],rtol=3e-4,atol=4e-5)


def test_count_min_matches_sequential_with_collisions():
    src=torch.tensor([0,0,1,0,2,1,1]);dst=torch.tensor([1,0,0,2,1,1,0]);sig=torch.tensor([0,0,0,1,0,1,0])
    a=CountTable(3,2,width=2);b=CountTable(3,2,width=2)
    actual=a(src,dst,sig)
    expected=torch.cat([b(src[i:i+1],dst[i:i+1],sig[i:i+1]) for i in range(len(src))])
    torch.testing.assert_close(actual,expected);torch.testing.assert_close(a.counts,b.counts)


def test_training_gradients_and_cross_batch_unroll(tmp_path):
    bank,batch,_=fixture(tmp_path,corrected=True)
    model=TGNNSlidingNodeMem(8,bank,num_relations=3,signatures=3)
    opt=torch.optim.Adam(model.parameters(),lr=1e-4)
    for lo in (0,48):
        opt.zero_grad();losses=[]
        for start in (lo,lo+24):
            data=take(batch,start,start+24);logits,aux=model.forward_batch(data)
            losses.append(F.cross_entropy(logits,data['relation'])+.01*aux['compression_loss']+.1*aux['pair_loss'])
        sum(losses).backward()
        for module in (model.event_encoder.tokens,model.gat_second,model.pair_head,model.admission,model.write_src):
            assert any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters())
        opt.step();model.detach_state()


def test_two_hash_does_not_add_embedding_parameters(tmp_path):
    from sliding_node_mem_v3.tokens import hashes
    bank,_,_=fixture(tmp_path,corrected=True)
    encoder=PackedEncoder(bank)
    assert encoder.tokens.weight.numel()==256*64
    seen={};collision=None
    for i in range(5000):
        a,b=hashes(str(i),256,True)
        if a in seen and seen[a][1]!=b:collision=(seen[a],(i,b));break
        seen[a]=(i,b)
    assert collision is not None
    with torch.no_grad():
        ids=torch.arange(min(12,bank.config['identities']))
        expected=encoder(ids);bank.to_device('cpu')
        torch.testing.assert_close(encoder(ids),expected)


def test_elapsed_affine_scan_matches_current_v2():
    from sliding_node_mem_v3.memory import endpoint_scan
    from sliding_node_mem_v2.memory import endpoint_scan as previous
    src=torch.tensor([0,0,1,0]);dst=torch.tensor([0,1,0,1]);times=torch.tensor([10.,20.,80.,100.],dtype=torch.float64)
    initial=torch.randn(2,4);weights=torch.rand(8,4)*.4;content=torch.randn(8,4)
    half=torch.tensor([10.,10.,50.,50.]);last=torch.tensor([0.,float('nan')],dtype=torch.float64)
    outputs=[]
    for fn in (previous,endpoint_scan):
        table=initial.clone().requires_grad_();w=weights.clone().requires_grad_();c=content.clone().requires_grad_()
        values=fn(table,src,dst,w,c,times=times,last_seen=last,half_lives=half)
        sum(v.square().sum() for v in values).backward()
        outputs.append(values+(table.grad,w.grad,c.grad))
    for a,b in zip(*outputs):torch.testing.assert_close(a,b,rtol=2e-5,atol=3e-6)


def test_compatible_gradient_and_update_parity(tmp_path):
    bank,batch,args=fixture(tmp_path,n=96)
    a=V2(8,num_relations=3,buckets=256,profile='compatible',cnn_backend='conv',compact_every=0).eval()
    b=TGNNSlidingNodeMem(8,bank,num_relations=3,signatures=3,profile='compatible',exact_counts=True).eval()
    b.load_state_dict(a.state_dict(),strict=False)
    optimizers=[torch.optim.Adam(m.parameters(),lr=1e-4) for m in (a,b)]
    for start in (0,32,64):
        for m,opt in zip((a,b),optimizers):
            opt.zero_grad()
            logits,aux=m.forward_events(*[v[start:start+32] for v in args]) if m is a else m.forward_batch(take(batch,start,start+32))
            (F.cross_entropy(logits,batch['relation'][start:start+32])+.01*aux['compression_loss']).backward()
        for (name,x),(other,y) in zip(a.named_parameters(),b.named_parameters()):
            assert name==other
            torch.testing.assert_close(x.grad,y.grad,rtol=5e-4,atol=8e-6)
        for m,opt in zip((a,b),optimizers):opt.step();m.detach_state()


def test_compatible_sticky_selection_reaches_quiet_endpoint(tmp_path):
    from sliding_node_mem_v3.benchmark import pack_samples
    pairs=[(0,2)]*8+[(0,1)]+[(0,2)]*12+[(3,4)]*10+[(1,3)]*12
    src,dst=map(list,zip(*pairs));times=list(map(float,range(len(pairs))))
    events=[{'attributes':{'src':{'type':'process','path':str(u)},'dst':{'type':'file','path':str(v)}}} for u,v in pairs]
    samples=[(events,src,dst,times,[0]*len(pairs))]
    bank,packed,signatures,_=pack_samples(samples,tmp_path/'quiet',True)
    a=V2(6,num_relations=2,profile='compatible',rare_count=0,novelty_threshold=3,compact_every=0).eval()
    b=TGNNSlidingNodeMem(6,bank,num_relations=2,signatures=signatures,profile='compatible',rare_count=0,novelty_threshold=3,exact_counts=True).eval()
    with torch.no_grad():a.importance_query.weight.zero_()
    b.load_state_dict(a.state_dict(),strict=False)
    with torch.no_grad():
        expected=[]
        for i in range(len(pairs)):
            expected.append(a.forward_events(*[v[i:i+1] for v in samples[0]])[0]);a.detach_state()
        actual=[]
        for lo,hi in ((0,9),(9,21),(21,31),(31,len(pairs))):
            actual.append(b.forward_batch(take(packed[0],lo,hi))[0]);b.detach_state()
        torch.testing.assert_close(torch.cat(actual),torch.cat(expected),rtol=3e-4,atol=3e-5)
        expected_ids=[i for i,_,_ in a.short[1]]
        rows=b.state.read(torch.tensor([1]));ring=b.state.chronological(rows,'short')[''][0,:,0]
        assert ring[ring>=0].tolist()==expected_ids


def test_corrected_balanced_quota_and_no_sticky_retention(tmp_path):
    bank,batch,_=fixture(tmp_path,corrected=True)
    m=TGNNSlidingNodeMem(8,bank,num_relations=3,signatures=3,rare_count=0,novelty_threshold=3).eval()
    history=defaultdict(lambda:deque(maxlen=8))
    from sliding_node_mem_v2.windows import choose_history
    expected=[]
    for i,(u,v) in enumerate(zip(batch['src'].tolist(),batch['dst'].tolist())):
        prior=choose_history(history[u],history[v],8)
        expected.append([-1]*(8-len(prior))+prior+[i])
        history[u].append(i)
        if u!=v:history[v].append(i)
    with torch.no_grad():_,aux=m.forward_batch(batch)
    assert aux['window_ids'].tolist()==expected
    assert not (m.state.short[...,0]>=0).any()
    assert not aux['selected'].any()
    assert (aux['admission']>0).all() # Differentiable soft write, separate from hard storage admission.


def test_sparse_ring_allocation_growth_and_cross_batch_credit(tmp_path):
    bank,batch,_=fixture(tmp_path,corrected=True)
    m=TGNNSlidingNodeMem(1000000,bank,num_relations=3,signatures=3,initial_capacity=1).eval()
    first=take(batch,0,1);first['src'][:]=999999;first['dst'][:]=42
    m.forward_batch(first)
    assert m.state.used==2 and m.state.capacity==2
    second={k:v.clone() for k,v in first.items()};second['time']+=1
    logits,_=m.forward_batch(second)
    F.cross_entropy(logits,second['relation']).backward()
    assert m.write_src[0].weight.grad.abs().sum()>0
    assert m.admission.weight.grad.abs().sum()>0
    m.detach_state()
    with torch.no_grad():
        data=take(batch,40,96);data['time']+=10
        m.forward_batch(data)
    assert m.state.capacity<=8 and m.state.used==7
    snapshot=m.stream_state_dict();assert len(snapshot['rings']['hist'])==7
    before=m.state.read(torch.tensor([42,999999]))['memory'].clone()
    m.load_stream_state_dict(snapshot)
    torch.testing.assert_close(m.state.read(torch.tensor([42,999999]))['memory'],before)


def test_preparation_pretraining_runner_and_resume(tmp_path):
    import json
    from test_sliding_v2 import make_manifest
    from orthrus_comparison.run import sha256
    from sliding_node_mem_v3.prepare import prepare,read_batches
    from sliding_node_mem_v3.pretrain import pretrain
    from sliding_node_mem_v3.run import run,make_optimizer
    source=tmp_path/'source';source.mkdir();manifest_path=make_manifest(source)
    manifest=json.loads(manifest_path.read_text())
    attrs={str(i):{'type':'process','path':f'/bin/task/{i}'} for i in range(64)}
    attrs['999']={'type':'file','path':'/secret/validation_only'}
    attrpath=source/'attrs.json';attrpath.write_text(json.dumps(attrs))
    manifest['file_sha256'][str(attrpath)]=sha256(attrpath);manifest_path.write_text(json.dumps(manifest))
    prepared=tmp_path/'prepared';meta=prepare(manifest_path,prepared,buckets=256)
    assert 'validation_only' not in (prepared/'benign_paths.jsonl').read_text()
    assert meta['negative_eligible']['train']>0 and meta['negative_eligible']['test']==0
    data=next(read_batches(prepared/'train.bin',64))
    assert (data['negative_identity'][data['negative_valid']]!=data['dst_identity'][data['negative_valid']]).all()
    initial=tmp_path/'initial.pt';result=pretrain(prepared,initial,epochs=1,projection_steps=8,batch=64)
    assert np.isfinite(result['projection_mse'])
    options=dict(prepared=prepared,epochs=2,batch=32,unroll_batches=2,initialization=initial,false_alert_rate=.1,
                 model_options={'count_width':64},seed=3)
    full=run(**options,output=tmp_path/'full')
    class Interrupt(Exception):pass
    def interrupt():raise Interrupt()
    with pytest.raises(Interrupt):run(**options,output=tmp_path/'resume',checkpoint_seconds=0,on_checkpoint=interrupt)
    resumed=run(**options,output=tmp_path/'resume')
    assert full['metrics']==resumed['metrics'] and full['checkpoint_epoch']==resumed['checkpoint_epoch']
    a=torch.load(tmp_path/'full/resume.pt',weights_only=False)
    b=torch.load(tmp_path/'resume/resume.pt',weights_only=False)
    for name in a['model']:torch.testing.assert_close(a['model'][name],b['model'][name])
    assert a['optimizer']['param_groups'][0]['lr']==1e-4
    assert a['optimizer']['param_groups'][1]['lr']==1e-5
