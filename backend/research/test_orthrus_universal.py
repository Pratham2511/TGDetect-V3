import copy
from types import SimpleNamespace
import pytest
import torch
from orthrus_comparison.universal import UniversalEventEncoder
from orthrus_comparison.models import TGNNNodeMemV4KV
from orthrus_comparison.official import OrthrusReference
from orthrus_comparison.protocol import identity, reconstruction_query, require_coverage


def test_missing_and_mixed_types():
    model = UniversalEventEncoder()
    events = [{}, {'type':'login','attributes':{'success':True,'user':None}},
              {'type':'network','attributes':{'ip':'2001:db8::1','port':443}},
              {'attributes':{'x':float('nan'),'n':float('inf'),'path':'C:\\Users\\abc', 'v':[1,'a',None]}},
              {'attributes':{'cloud':{'action':'CreateAccessKey'}}}]
    out = model(events)
    assert out.shape == (5,64) and torch.isfinite(out).all()
    out.square().mean().backward()
    assert model.tokens.weight.grad is not None


def test_permutation_and_batch_padding_invariance():
    torch.manual_seed(3)
    m = UniversalEventEncoder().eval()
    a = {'type':'login','attributes':{'user':'alice','ok':True}}
    b = {'type':'login','attributes':{'ok':True,'user':'alice'}}
    torch.testing.assert_close(m([a]),m([b]))
    torch.testing.assert_close(m([a])[0],m([a,{'attributes':{str(i):i for i in range(40)}}])[0],atol=1e-6,rtol=1e-5)


def test_no_current_relation_or_label_in_query():
    a = {'type':'subject','path':'/usr/bin/a','label':1,'action':'WRITE'}
    b = dict(a,label=0,action='READ')
    assert reconstruction_query(a,{}) == reconstruction_query(b,{})
    assert identity('a','hostlog','h','src',None) != identity('b','hostlog','h','src',None)
    with pytest.raises(ValueError):
        require_coverage([2,3,4,5,6])


def test_v4_kv_state_gradients_and_reload():
    torch.set_num_threads(1)
    torch.manual_seed(5)
    m = TGNNNodeMemV4KV(n_nodes=4)
    histories = [[{'attributes':{'src':{'path':'/tmp/a'}}}]+[None]*4,
                 [{'attributes':{'dst':{'remote_ip':'1.2.3.4'}}},{}]+[None]*3]
    ages=[[0]*5,[0,1,0,0,0]]
    logits, _ = m.forward_events(histories, ages,[0,1],[1,2],[1.,2.])
    loss = torch.nn.functional.cross_entropy(logits,torch.tensor([1,2]))
    loss.backward()
    assert torch.isfinite(loss) and m.event_encoder.tokens.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        m.forward_events(histories,ages,[0,1],[1,2],[0.,1.])
    clone = TGNNNodeMemV4KV(n_nodes=4)
    clone.load_state_dict(m.state_dict()); m.eval(); clone.eval(); m.reset_state()
    args = (histories,ages,[0,1],[1,2],[1.,2.])
    torch.testing.assert_close(m.forward_events(*args)[0],clone.forward_events(*args)[0])


def test_official_model_loss_and_no_current_relation_leak():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = OrthrusReference(4).eval()
    src,dst = torch.tensor([0,1]),torch.tensor([1,2])
    x = torch.randn(2,131)
    full = SimpleNamespace(edge_type=torch.eye(10)[:2])
    # Capture logits: changing target must not change predicted logits.
    captured=[]
    handle=model.decoder.lin_seq.register_forward_hook(lambda m,i,o: captured.append(o.detach().clone()))
    for target in (torch.eye(10)[:2],torch.eye(10)[2:4]):
        model.reset_state()
        value=model(src,dst,torch.tensor([1,2]),x,x,target,full,inference=True)
        assert torch.isfinite(value).all()
    handle.remove()
    torch.testing.assert_close(captured[0],captured[1])
    model.train(); model.reset_state()
    loss=model(src,dst,torch.tensor([1,2]),x,x,torch.eye(10)[:2],full)
    loss.backward()
    assert model.encoder.src_linear.weight.grad is not None


def test_matched_runner_synthetic(tmp_path,monkeypatch):
    """Exercise all arms and paper date gate; these scores are never research results."""
    import json
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from torch_geometric.data import TemporalData
    from orthrus_comparison.run import run, ARMS
    from orthrus_comparison.protocol import TRAIN_DAYS, VALIDATION_DAYS, TEST_DAYS
    from orthrus_comparison.models import TGNNV3Reconstruction
    original=TGNNV3Reconstruction.forward_events
    def checked(self,x,valid,ages,src,dst,times,**kw):
        assert valid[:,0].all() and (ages[:,0]==0).all()
        assert ((ages[:,1:]>0)|~valid[:,1:]).all()
        if x.shape[-1]==272:assert (x[:,0,131:141]==0).all()
        return original(self,x,valid,ages,src,dst,times,**kw)
    monkeypatch.setattr(TGNNV3Reconstruction,'forward_events',checked)
    torch.manual_seed(9)
    grouped={}
    for split,days in [('train',TRAIN_DAYS),('val',VALIDATION_DAYS),('test',TEST_DAYS)]:
        grouped[split]=[]
        for day in days:
            start=int(datetime(2018,4,day,12,tzinfo=ZoneInfo('America/New_York')).timestamp()*1e9)
            src=torch.arange(32);dst=src+32
            msg=torch.randn(32,272);msg[:,131:141]=torch.eye(10)[src%10]
            graph=TemporalData(src=src,dst=dst,t=torch.arange(32)*1000000000+start,msg=msg)
            path=tmp_path/f'{day}.pt';torch.save(graph,path);grouped[split].append(str(path))
    attrs=tmp_path/'attrs.json';attrs.write_text(json.dumps({str(n):{'type':'subject','path':f'/bin/{n}'} for n in range(64)}))
    truth=tmp_path/'truth.json';truth.write_text('[1, 2]')
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(dataset='CADETS_E3',
        official_preprocessing=True,days=list(range(2,14)),files=grouped,n_nodes=64,
        node_attributes=str(attrs),ground_truth_nodes=str(truth))))
    for arm in ARMS:
        result=run(manifest,tmp_path/'results',epochs=1,batch=32,arm=arm)
        assert result['status']=='complete'
        assert sum(result['threshold_only'][k] for k in ('tp','tn','fp','fn'))==64


def test_pinned_tracing_and_window_evaluation(tmp_path):
    """Attribution expands a seed causally and preserves the release's window rule."""
    import json
    import networkx as nx
    from orthrus_comparison.tracing import run_tracing
    g=nx.MultiDiGraph();g.add_edge('0','1',time=1);g.add_edge('1','2',time=2)
    first=tmp_path/'first.pt';torch.save(g,first)
    g2=nx.MultiDiGraph();g2.add_edge('0','3',time=3)
    second=tmp_path/'second.pt';torch.save(g2,second)
    truth=tmp_path/'truth.json';truth.write_text('[0,1,2]')
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(
        graph_files={'test':[str(first),str(second)]},ground_truth_nodes=str(truth))))
    result=tmp_path/'result.json';result.write_text(json.dumps(dict(status='complete',
        predicted_nodes=[1],node_scores={str(i):1. for i in range(4)})))
    measured=run_tracing(manifest,result,workers=1)
    assert measured['traced_nodes']==[0,1,2]
    assert measured['global_union']['tp']==3
    assert measured['released_last_window']['tp']==2
