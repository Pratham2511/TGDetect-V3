"""Real CADETS/checkpoint integration tests. No fabricated attack labels."""
import math
import numpy as np
from fastapi.testclient import TestClient
from dashboard.backend import app
from dashboard.cadets import DATA, EVENT_DTYPE, EDGE_DTYPE, RELATIONS, result

client = TestClient(app)


def test_summary_reproduces_artifact_and_full_dataset():
    response = client.get('/api/cadets/summary')
    assert response.status_code == 200, response.text
    summary = response.json()
    assert summary['total_events'] == 5537964
    assert summary['events'] == dict(train=3526541, val=105481, test=1905942)
    assert summary['metrics'] == result()['metrics']
    assert summary['alert_nodes'] == 5723
    assert summary['scored_nodes'] == 268153
    assert summary['metric_reproduction_max_error'] < 1e-6
    assert summary['metrics']['tp'] == 24 and summary['metrics']['fn'] == 44


def test_events_are_aligned_and_paginated_without_invented_train_scores():
    for split in ('train', 'val', 'test'):
        rows = np.memmap(DATA/'prepared'/(split+'.bin'), dtype=EVENT_DTYPE, mode='r')
        response = client.get('/api/cadets/events', params=dict(split=split, offset=17, limit=7))
        assert response.status_code == 200, response.text
        events = response.json()['items']
        assert len(events) == 7
        for i,event in enumerate(events,17):
            assert event['src'] == rows[i]['src'] and event['dst'] == rows[i]['dst']
            assert event['time'] == rows[i]['time']
            assert event['relation_name'] == RELATIONS[rows[i]['relation']]
            assert 'attack_label' not in event
            if split == 'train':
                assert event['loss'] is None and event['evidence'] is None
            else:
                saved = np.memmap(DATA/(split+'_edges.bin'), dtype=EDGE_DTYPE, mode='r')
                assert event['loss'] == float(saved[i]['loss'])
    node = int(rows[17]['src'])
    filtered = client.get('/api/cadets/events', params=dict(node=node,limit=50)).json()
    assert filtered['items'] and all(node in (e['src'],e['dst']) for e in filtered['items'])


def test_explanations_reproduce_node_scores_and_chain_uses_no_labels():
    nodes = client.get('/api/cadets/nodes?limit=3').json()['items']
    assert nodes and nodes[0]['score'] >= nodes[1]['score']
    for node in nodes:
        response = client.get(f'/api/cadets/nodes/{node["id"]}/explanation')
        assert response.status_code == 200, response.text
        explanation = response.json()
        assert math.isclose(explanation['reconstructed_score'],node['score'],abs_tol=1e-10)
        assert len(explanation['contributors']) <= 3
        assert all(node['id'] in (e['src'],e['dst']) for e in explanation['contributors'])
        chain_response = client.get(f'/api/cadets/nodes/{node["id"]}/chain?hops=2&limit=60')
        assert chain_response.status_code == 200, chain_response.text
        chain = chain_response.json()
        assert chain['labels_used'] is False
        assert len(chain['events']) <= 60
        assert chain['events'] == sorted(chain['events'],key=lambda e:(e['time'],e['idx']))
        ids = {n['id'] for n in chain['nodes']}
        assert all(e['src'] in ids and e['dst'] in ids for e in chain['events'])
        assert all(abs(e['time']-chain['anchor']) <= 900 for e in chain['events'])
        # Every reported entity must connect back to the seed in the returned
        # bounded graph, even when hub truncation occurs.
        reachable = {node['id']}
        for _ in range(len(ids)):
            for e in chain['events']:
                if e['src'] in reachable or e['dst'] in reachable:
                    reachable.update((e['src'],e['dst']))
        assert reachable == ids


def test_invalid_requests_and_missing_entities():
    for path in ('/events?split=invalid','/events?limit=1001','/events?node=-1','/nodes?offset=-1','/nodes/1/chain?hops=10'):
        assert client.get('/api/cadets'+path).status_code == 422
    assert client.get('/api/cadets/nodes/99999999/explanation').status_code == 404
    assert client.post('/api/cadets/replay',json={'batches':5}).status_code == 422
    assert client.post('/api/cadets/replay',json={'split':'bad'}).status_code == 422


def test_event_details_expose_actual_tokenized_endpoint_attributes():
    import json
    import sqlite3
    for split in ('train', 'val', 'test'):
        response = client.get(f'/api/cadets/events/{split}/0')
        assert response.status_code == 200, response.text
        detail = response.json()
        row = np.memmap(DATA/'prepared'/(split+'.bin'),dtype=EVENT_DTYPE,mode='r')[0]
        with sqlite3.connect(DATA/'prepared/tokens/identities.sqlite') as db:
            expected = json.loads(db.execute('SELECT identity FROM identities WHERE idx=?',(int(row['src_identity']),)).fetchone()[0])['attributes']
        assert detail['src_attributes'] == expected
    assert client.get('/api/cadets/events/test/99999999').status_code == 404
    assert client.get('/api/cadets/events/invalid/0').status_code == 422


def test_real_v3_replay_matches_saved_gpu_scores_and_resets():
    first = client.post('/api/cadets/replay',json={'split':'test','batches':1})
    assert first.status_code == 200, first.text
    first = first.json()
    assert first['events'] == 1024
    assert first['max_absolute_error_vs_saved_gpu_scores'] < 3e-4
    second = client.post('/api/cadets/replay',json={'split':'test','batches':1})
    assert second.status_code == 200, second.text
    np.testing.assert_allclose([e['combined_loss'] for e in first['scores']], [e['combined_loss'] for e in second.json()['scores']],atol=1e-7)


def test_constructed_chains_are_directed_chronological_and_label_free():
    for node in (674135, 239026, 373368):
        response = client.get(f'/api/cadets/nodes/{node}/construction?max_steps=5&limit=6')
        assert response.status_code == 200, response.text
        data = response.json()
        assert data['paths'] and data['labels_used'] is False
        for path in data['paths']:
            events = path['events']
            assert 1 <= len(events) <= 5
            assert events[path['anchor_step']]['idx'] == data['anchor']['idx']
            assert math.isclose(path['evidence_sum'],sum(e['evidence'] for e in events))
            for before,after in zip(events,events[1:]):
                assert before['dst'] == after['src']
                assert (before['time'],before['idx']) < (after['time'],after['idx'])
            assert len({e['id'] for e in events}) == len(events)
            assert all(abs(e['time']-data['anchor']['time']) <= 900 for e in events)
    # Direction controls preserve the anchor at the correct end of each path.
    for direction in ('forward','backward'):
        data = client.get(f'/api/cadets/nodes/674135/construction?direction={direction}').json()
        for path in data['paths']:
            assert path['anchor_step'] == (0 if direction=='forward' else len(path['events'])-1)
    assert client.get('/api/cadets/nodes/674135/construction?anchor_index=0').status_code == 422
    assert client.get('/api/cadets/nodes/674135/construction?max_steps=100').status_code == 422


def test_detection_graph_predictions_ips_and_observed_connections():
    from dashboard.cadets import database
    graph_response = client.get('/api/cadets/detection-graph')
    assert graph_response.status_code == 200, graph_response.text
    graph = graph_response.json()
    assert 4 <= len(graph['nodes']) <= 32
    assert graph['counts']['predicted_malicious'] > 0
    assert graph['counts']['predicted_benign'] > 0
    ids = {n['id'] for n in graph['nodes']}
    for n in graph['nodes']:
        assert n['alert'] == (n['score'] > graph['threshold'])
        assert n['prediction'] == ('predicted_malicious' if n['alert'] else 'predicted_benign')
        assert n['ip'] == n['attributes'].get('remote_ip')
    assert graph['events'] and len(graph['events']) <= 120
    with database() as db:
        for e in graph['events']:
            assert e['src'] in ids and e['dst'] in ids
            row = db.execute("SELECT src,dst,time FROM events WHERE split='test' AND idx=?", (e['idx'],)).fetchone()
            assert (e['src'],e['dst'],e['time']) == tuple(row)
    searched = client.get('/api/cadets/detection-graph?ip=128.55.12.167').json()
    assert searched['seeds']
    for n in searched['nodes']:
        if n['id'] in searched['seeds']:
            assert '128.55.12.167' in n['ip']
    benign = client.get('/api/cadets/detection-graph?node=0&limit=8').json()
    assert len(benign['nodes']) <= 8 and benign['seeds'] == [0]
    assert client.get('/api/cadets/detection-graph?limit=100').status_code == 422
    assert client.get('/api/cadets/detection-graph?node=99999999').status_code == 404
