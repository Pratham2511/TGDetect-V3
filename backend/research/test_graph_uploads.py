"""Upload, real V3 inference, dataset isolation and connected graph acceptance."""
import json
import time
import numpy as np
import pytest
from fastapi.testclient import TestClient
from dashboard.backend import app
from dashboard import graph_datasets as graphs
from dashboard.cadets import DATA, EVENT_DTYPE

client=TestClient(app)


def events_from_prepared(count=30):
    import sqlite3
    rows=np.memmap(DATA/'prepared/test.bin',dtype=EVENT_DTYPE,mode='r')[:count]
    events=[]
    with sqlite3.connect(DATA/'prepared/tokens/identities.sqlite') as db:
        for row in rows:
            attrs={}
            for role in ('src','dst'):
                value=json.loads(db.execute('SELECT identity FROM identities WHERE idx=?',(int(row[role+'_identity']),)).fetchone()[0])['attributes']
                attrs[role]={**value,'id':str(row[role]),'host':'cadets-test-host'}
            events.append(dict(time=float(row['time']),relation=int(row['relation']),**attrs))
    return events


def wait_ready(identifier):
    deadline=time.monotonic()+60
    while time.monotonic()<deadline:
        response=client.get('/api/graph-datasets/'+identifier)
        assert response.status_code==200,response.text
        data=response.json()
        if data['status']=='failed': pytest.fail(data['error'])
        if data['status']=='ready': return data
        time.sleep(.1)
    pytest.fail('Upload inference did not complete within 60 seconds')


def test_upload_real_inference_and_dynamic_connected_neighborhoods(tmp_path,monkeypatch):
    monkeypatch.setattr(graphs,'STORE',tmp_path)
    events=events_from_prepared(40)
    # Include a separate host/IP component to test that a focus excludes it.
    events.append(dict(time=events[-1]['time']+1,relation='CONNECT',src={'id':'extra-host','host':'disconnected-host','type':'subject','cmd_line':'curl'},dst={'id':'extra-ip','remote_ip':'192.0.2.200','type':'netflow'}))
    response=client.post('/api/graph-datasets?name=cadets-excerpt.json',content=json.dumps(events))
    assert response.status_code==202,response.text
    identifier=response.json()['id']
    ready=wait_ready(identifier)
    assert ready['events']==len(events) and ready['checkpoint_sha256']==graphs.CHECKPOINT_SHA
    options=client.get(f'/api/graph-datasets/{identifier}/entities?kind=host').json()
    assert {'ip','host','process'} <= set(options['available_kinds'])
    assert 'disconnected-host' in {n['label'] for n in options['items']}
    graph=client.get(f'/api/graph-datasets/{identifier}/graph?kind=ip&value=192.0.2.200').json()
    assert len(graph['nodes'])==2 and len(graph['events'])==1
    assert all(n['benchmark_member'] is None for n in graph['nodes'])
    assert all(n['alert']==(n['score']>graph['threshold']) for n in graph['nodes'])
    assert any(n['ip']=='192.0.2.200' for n in graph['nodes'])
    assert all(n['attributes'].get('host')!='cadets-test-host' for n in graph['nodes'])
    clicked=client.get(f'/api/graph-datasets/{identifier}/graph?node={graph["nodes"][0]["id"]}').json()
    assert {n['id'] for n in clicked['nodes']}=={n['id'] for n in graph['nodes']}
    assert identifier in {d['id'] for d in client.get('/api/graph-datasets').json()['items']}
    # Existing bundled data cannot leak into uploaded IDs or graphs.
    assert all(e['split']=='upload' for e in graph['events'])


def test_malicious_roots_keep_connected_benign_context():
    graph=client.get('/api/graph-datasets/cadets/graph?prediction=malicious').json()
    nodes={n['id']:n for n in graph['nodes']}
    assert graph['seeds'] and all(nodes[n]['alert'] for n in graph['seeds'])
    reachable=set(graph['seeds'])
    for _ in nodes:
        for e in graph['events']:
            if e['src'] in reachable or e['dst'] in reachable: reachable.update((e['src'],e['dst']))
    assert reachable==set(nodes)
    # All emitted nodes are context, not a destructive class-only filter.
    assert any(not n['alert'] for n in nodes.values())
    assert client.get('/api/graph-datasets/cadets/graph?kind=oops').status_code==422
    assert client.get('/api/graph-datasets/not-a-dataset/graph').status_code==404


def test_parsing_formats_errors_and_ignores_supplied_verdicts():
    csv=b'time,relation,src_ip,dst_ip,src_host\n1720000000,CONNECT,192.0.2.1,192.0.2.2,pc-one\n'
    nodes,events=graphs.parse_logs(csv,'network.csv')
    assert len(nodes)==2 and nodes[0]['attributes']['host']=='pc-one'
    assert events[0]['relation']==0
    rows=[dict(time='2026-10-11T10:00:00Z',relation='CONNECT',src='192.0.2.1',dst='192.0.2.2',alert=True,score=999),dict(time='2026-10-11T09:00:00Z',relation='READ',src='192.0.2.2',dst='192.0.2.1')]
    _,normalized=graphs.parse_logs('\n'.join(json.dumps(r) for r in rows).encode(),'logs.jsonl')
    assert normalized[0]['time']<normalized[1]['time']
    assert all('alert' not in e and 'score' not in e for e in normalized)
    for content in (b'[]',b'not json',b'[{"src":"a","dst":"b","time":1,"relation":"BOGUS"}]'):
        assert client.post('/api/graph-datasets?name=bad.json',content=content).status_code==422
    assert graphs.timestamp('2026-10-11T10:00:00+05:30')==graphs.timestamp('2026-10-11T04:30:00Z')


def test_upload_losses_match_original_gpu_prefix(tmp_path,monkeypatch):
    import sqlite3
    from dashboard.cadets import EDGE_DTYPE
    monkeypatch.setattr(graphs,'STORE',tmp_path)
    response=client.post('/api/graph-datasets?name=actual-cadets-prefix.json',content=json.dumps(events_from_prepared(1024)))
    assert response.status_code==202,response.text
    identifier=response.json()['id']; wait_ready(identifier)
    with sqlite3.connect(tmp_path/identifier/'graph.sqlite') as db:
        losses=np.array([r[0] for r in db.execute('SELECT loss FROM events ORDER BY idx')])
    saved=np.memmap(DATA/'test_edges.bin',dtype=EDGE_DTYPE,mode='r')[:1024]
    assert np.max(np.abs(losses-saved['loss']))<1e-4


def test_timestamp_offsets_and_event_attributes_do_not_leak_from_future():
    assert graphs.timestamp('2026-10-11T10:00:00-05:30')==graphs.timestamp('2026-10-11T15:30:00Z')
    records=[dict(time=1,relation='READ',src={'id':'p','cmd_line':'earlier'},dst='file'),dict(time=2,relation='READ',src={'id':'p','cmd_line':'later'},dst='file')]
    _,events=graphs.parse_logs(json.dumps(records).encode(),'events.json')
    assert events[0]['src_attributes']['cmd_line']=='earlier'
    assert events[1]['src_attributes']['cmd_line']=='later'
