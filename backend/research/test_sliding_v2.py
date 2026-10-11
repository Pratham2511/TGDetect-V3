import copy
from collections import deque
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from sliding_node_mem_upgrade.model import TGNNSlidingNodeMem as Reference
from sliding_node_mem_v2.model import TGNNSlidingNodeMem
from sliding_node_mem_v2.encoding import EventEncoder
from sliding_node_mem_v2.layers import TemporalCNN
from sliding_node_mem_v2.windows import choose_history
from sliding_node_mem_v2.scoring import NodeEvidence, EdgeWriter, read_edges, calibrated_scores
from sliding_node_mem_v2.training import train_unroll


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(1)
    torch.manual_seed(42)


def sample(n=32, start=0):
    return ([{'attributes': {'src': {'type': 'process', 'path': f'/bin/p{i%3}'},
                              'dst': {'type': 'file', 'path': f'/tmp/f{i%5}'}}} for i in range(n)],
            [i%3 for i in range(n)], [3+i%2 for i in range(n)],
            [float(start+i) for i in range(n)], [i%3 for i in range(n)])


def model(**kwargs):
    return TGNNSlidingNodeMem(16, num_relations=3, buckets=256, **kwargs)


@pytest.mark.parametrize('batch', [1, 17, 65])
def test_dense_cnn_outputs_and_gradients(batch):
    a, b = TemporalCNN(backend='conv'), TemporalCNN(backend='linear')
    b.load_state_dict(a.state_dict())
    x = torch.randn(batch, 9, 128)
    valid = torch.arange(9)[None].expand(batch, -1) > 2
    outputs = []
    for m in (a,b):
        inputs = x.clone().requires_grad_()
        output = m(inputs*valid[...,None], valid)
        output.square().mean().backward()
        outputs.append((output, inputs.grad))
    for left,right in zip(outputs[0],outputs[1]):
        torch.testing.assert_close(left,right,atol=2e-6,rtol=2e-5)
    for left,right in zip(a.parameters(),b.parameters()):
        torch.testing.assert_close(left.grad,right.grad,atol=2e-6,rtol=2e-5)


def test_compatible_predictions_gradients_and_optimizer_updates():
    a = Reference(16, num_relations=3, buckets=256)
    b = model(profile='compatible', cnn_backend='conv', compact_every=1)
    b.load_state_dict(a.state_dict())
    optimizers = [torch.optim.Adam(m.parameters(), lr=1e-4) for m in (a,b)]
    for step in range(3):
        outputs = []
        for m, opt in zip((a,b), optimizers):
            torch.manual_seed(step)
            opt.zero_grad(set_to_none=True)
            logits, aux = m.forward_events(*sample(32, step*32))
            loss = F.cross_entropy(logits, torch.tensor(sample()[4])) + .01*aux['compression_loss']
            loss.backward()
            outputs.append(logits.detach())
            opt.step(); m.detach_state()
        torch.testing.assert_close(*outputs,rtol=2e-5,atol=2e-6)
        for left,right in zip(a.parameters(),b.parameters()):
            torch.testing.assert_close(left.grad,right.grad,rtol=3e-4,atol=3e-6)
            torch.testing.assert_close(left,right,rtol=2e-5,atol=2e-6)


@pytest.mark.parametrize('profile', ['compatible','corrected'])
def test_causality_and_batch_invariance(profile):
    m = model(profile=profile).eval()
    args = sample()
    with torch.no_grad():
        whole,_ = m.forward_events(*args)
        m.reset_state()
        steps = torch.cat([m.forward_events(*[v[i:i+1] for v in args])[0] for i in range(32)])
        torch.testing.assert_close(whole,steps,rtol=1e-4,atol=2e-5)
        m.reset_state(); changed = copy.deepcopy(args); changed[-1][-1] = 1
        other,_ = m.forward_events(*changed)
        torch.testing.assert_close(whole,other,rtol=1e-4,atol=2e-5)


def test_balanced_history_keeps_quiet_endpoint():
    chosen = choose_history(deque(range(1,9)), deque([0]),8)
    assert 0 in chosen and len(chosen) == 8 and chosen == sorted(chosen)


def test_balanced_vectorized_windows_match_sequential():
    from collections import defaultdict
    from sliding_node_mem_v2.windows import plan_windows
    rng=np.random.default_rng(7);history=defaultdict(deque);first=0
    for size in (1,11,8,39):
        s=rng.integers(0,9,size).tolist();d=rng.integers(0,9,size).tolist()
        planned=plan_windows(history,s,d,first,9)
        for j,(u,v) in enumerate(zip(s,d)):
            assert planned[j][planned[j]>=0].tolist()==choose_history(history[u],history[v],8)
            for node in {u,v}:
                history[node].append(first+j)
                if len(history[node])>8:history[node].popleft()
        first+=size


def test_order_sensitive_encoder_and_field_permutation():
    m = EventEncoder(buckets=256)
    values = m([{'attributes': {'cmd': 'a b b a'}}, {'attributes': {'cmd': 'b a a b'}}])
    assert not torch.allclose(values[0],values[1])
    values = m([{'attributes': {'a':1,'b':2}}, {'attributes': {'b':2,'a':1}}])
    torch.testing.assert_close(values[0],values[1])
    values.sum().backward()
    assert m.tokens.weight.grad is not None


def test_inference_cache_eviction_invalidation_and_gradients():
    m = EventEncoder(buckets=256, cache_size=2).eval()
    events = [{'attributes': {'path':f'/x/{i}'}} for i in range(5)]
    with torch.no_grad():
        expected = m.encode(events)
        first = m(events); second = m(events[::-1])
        torch.testing.assert_close(first,expected)
        torch.testing.assert_close(second,expected.flip(0))
        m.output[0].weight.add_(.1)
        torch.testing.assert_close(m(events),m.encode(events))
    m.train(); m(events).square().mean().backward()
    assert m.tokens.weight.grad is not None and not m._cache


def test_compaction_preserves_predictions_and_snapshot(tmp_path):
    m = model().eval()
    with torch.no_grad():
        for start in (0,64):
            data = ([{}]*64,[0]+[2]*63 if not start else [2]*64,
                    [1]+[3]*63 if not start else [3]*64,
                    list(map(float,range(start,start+64))),[0]*64)
            m.forward_events(*data); m.detach_state()
        a = model().eval(); a.load_state_dict(m.state_dict()); a.load_stream_state_dict(m.stream_state_dict())
        assert m.compact_state() > 0
        assert m.records[0].compressed.tensor.shape[0] == 1
        torch.testing.assert_close(m.forward_events(*sample(8,128))[0],a.forward_events(*sample(8,128))[0])


def test_partial_compaction_preserves_shared_rows_and_finishes_later():
    from sliding_node_mem_upgrade.model import TokenRef
    m = model().eval()
    expected = []
    for i in range(10):
        block = torch.randn(64,64)
        ref = TokenRef(block,7)
        expected.append(block[7].clone())
        m.short[0].append((i,float(i),ref))
        m.short[1].append((i,float(i),ref))
    m.compact_state(max_blocks=2)
    assert sum(ref.tensor.shape[0] == 1 for _,_,ref in m.short[0]) == 2
    m.compact_state()
    for i,((_,_,a),(_,_,b)) in enumerate(zip(m.short[0],m.short[1])):
        assert a.tensor is b.tensor and a.tensor.shape[0] == 1
        torch.testing.assert_close(a.tensor[a.row],expected[i])


def test_cross_batch_credit_survives_unroll():
    m = model().eval()
    m.forward_events([{}],[0],[1],[0.],[0])
    logits,_ = m.forward_events([{}],[0],[1],[1.],[1])
    F.cross_entropy(logits,torch.tensor([1])).backward()
    assert m.write_src[0].weight.grad.abs().sum() > 0
    assert m.admission.weight.grad.abs().sum() > 0
    # The pending state contains only touched rows, not a graph over every node.
    assert len(m._pending_memory[0]) == 2


@pytest.mark.parametrize('n',[1,17,128])
def test_planned_memory_scan_matches_reference_and_gradients(n):
    from node_memory_v3 import endpoint_scan as baseline
    from sliding_node_mem_v2.memory import endpoint_scan
    src=torch.arange(n)%11;dst=(torch.arange(n)*3)%11
    values=torch.randn(11,8);weights=torch.rand(2*n,8)*.3;content=torch.randn(2*n,8)
    outputs=[];grads=[]
    for fn in (baseline,endpoint_scan):
        x=values.clone().requires_grad_();w=weights.clone().requires_grad_();c=content.clone().requires_grad_()
        out=fn(x,src,dst,w,c)
        sum(t.square().sum() for t in out).backward()
        outputs.append(out);grads.append((x.grad,w.grad,c.grad))
    for a,b in zip(outputs[0]+grads[0],outputs[1]+grads[1]):
        torch.testing.assert_close(a,b,rtol=1e-5,atol=2e-6)


def test_elapsed_memory_decay_and_self_edges():
    from sliding_node_mem_v2.memory import endpoint_scan
    table=torch.ones(2,2);weights=torch.zeros(4,2);content=torch.ones(4,2)
    with torch.no_grad():
        ms,md,updated=endpoint_scan(table,[0,0],[0,1],weights,content,
            times=[10.,20.],last_seen=[0.,0.],half_lives=torch.tensor([10.,20.]))
    torch.testing.assert_close(ms[0],torch.tensor([.5,2**-.5]))
    torch.testing.assert_close(ms[0],md[0])
    torch.testing.assert_close(ms[1],torch.tensor([.25,.5]))
    torch.testing.assert_close(updated,torch.tensor([[.25,.5],[.25,.5]]))


def test_training_unroll_two_updates():
    m = model(); opt = torch.optim.Adam(m.parameters(),lr=1e-4)
    for j in range(2):
        loss,n = train_unroll(m,opt,[sample(16,j*32),sample(16,j*32+16)])
        assert torch.isfinite(loss) and n == 32 and not m._memory.requires_grad


def test_regular_traffic_does_not_erase_alert():
    score = NodeEvidence(window_seconds=10)
    score.update([0],[1],[0.],[12.]); before = score.scores()[0]
    score.update([0]*63,[1]*63,list(map(float,range(1,64))),[0.]*63)
    assert before == 12 and score.scores()[0] == 12


def test_retention_bounded_without_sticky_selection_and_floor():
    m = model(rare_count=0,novelty_threshold=3).eval()
    with torch.no_grad():
        m.importance_query.weight.zero_()
        _,aux = m.forward_events([{}]*64,[0]*64,[1]*64,list(map(float,range(64))),[0]*64)
    assert aux['admission'][-1] == 0
    assert sum(len(q) for q in m.short.values()) <= 8
    assert not m.records[63].selected


def test_columnar_scores_and_no_topk_alert_cap(tmp_path):
    def write(name,n,loss):
        with EdgeWriter(tmp_path/name) as writer:
            writer.append(np.arange(n),np.arange(n),np.arange(n),np.zeros(n),loss)
        return read_edges(tmp_path/name)
    val = write('val.bin',256,np.linspace(.01,1,256))
    test = write('test.bin',80,np.full(80,20.))
    threshold,scores,meta = calibrated_scores(val,test,false_alert_rate=.05)
    assert meta['supported'] and len([v for v in scores.values() if v>threshold]) == 80
    _,_,meta = calibrated_scores(val,test,false_alert_rate=1e-6)
    assert not meta['supported']


def test_reject_mismatched_stream_config():
    state = model().stream_state_dict()
    with pytest.raises(ValueError,match='config'):
        model(profile='compatible').load_stream_state_dict(state)


def make_manifest(tmp_path):
    import json
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from types import SimpleNamespace
    from orthrus_comparison.run import sha256
    paths = {}
    for split,day in [('train',3),('val',2),('test',6)]:
        n=192
        when=datetime(2018,4,day,tzinfo=ZoneInfo('America/New_York')).timestamp()
        msg=torch.zeros(n,272);msg[torch.arange(n),131+torch.arange(n)%3]=1
        graph=SimpleNamespace(src=torch.arange(n)%64,dst=(torch.arange(n)+1)%64,
                              t=(torch.arange(n,dtype=torch.float64)+when)*1e9,msg=msg)
        path=tmp_path/(split+'.pt');torch.save(graph,path);paths[split]=[str(path)]
    attrs=tmp_path/'attrs.json';attrs.write_text(json.dumps({str(i):{'type':'process'} for i in range(64)}))
    truth=tmp_path/'truth.json';truth.write_text('[1,2]')
    files=[p for rows in paths.values() for p in rows]+[str(attrs),str(truth)]
    manifest={'dataset':'CADETS_E3','official_preprocessing':True,'days':list(range(2,14)),
              'n_nodes':64,'files':paths,'node_attributes':str(attrs),'ground_truth_nodes':str(truth),
              'file_sha256':{p:sha256(p) for p in files}}
    path=tmp_path/'manifest.json';path.write_text(json.dumps(manifest));return path


def test_runner_checkpoint_selection_and_interrupted_resume(tmp_path):
    from sliding_node_mem_v2.run import run,ARM
    manifest=make_manifest(tmp_path)
    options=dict(manifest_path=manifest,epochs=2,batch=32,unroll_batches=2,
                 false_alert_rate=.1,model_options={'buckets':256},seed=12)
    completed=run(**options,output=tmp_path/'full')
    class Interrupt(Exception):pass
    def interrupt():raise Interrupt()
    with pytest.raises(Interrupt):
        run(**options,output=tmp_path/'resume',checkpoint_seconds=0,on_checkpoint=interrupt)
    resumed=run(**options,output=tmp_path/'resume')
    assert resumed['status']=='complete'
    assert completed['calibrated']==resumed['calibrated']
    assert completed['checkpoint_epoch']==resumed['checkpoint_epoch']
    assert resumed['calibration']['supported']
    for filename in ('resume.pt','best.pt'):
        a=torch.load(tmp_path/'full'/ARM/filename,weights_only=False)['model']
        b=torch.load(tmp_path/'resume'/ARM/filename,weights_only=False)['model']
        for key in a:
            torch.testing.assert_close(a[key],b[key])
