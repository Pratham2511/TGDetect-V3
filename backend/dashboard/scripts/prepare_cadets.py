"""Verify downloaded V3 artifacts and atomically build the investigation index.

Run from repository root: python -m dashboard.scripts.prepare_cadets
No GPU or paid cloud execution is used.
"""
import heapq
import json
import math
import sqlite3
import time

import numpy as np
from dashboard.cadets import DATA, ROOT, EVENT_DTYPE, EDGE_DTYPE, CHECKPOINT_SHA, digest, research_imports


def build():
    research_imports()
    from sliding_node_mem_v2.scoring import calibrated_scores
    from sliding_node_mem_upgrade.scoring import RelationCalibrator
    from orthrus_comparison.run import measure
    DATA.mkdir(parents=True, exist_ok=True)
    result_path = DATA / 'result.json'
    if not result_path.exists():
        result_path.write_bytes((ROOT / 'artifacts/v3_vs_orthrus/V3_accuracy.json').read_bytes())
    result = json.loads(result_path.read_text())
    prepared = DATA / 'prepared'
    assert digest(DATA / 'best.pt') == CHECKPOINT_SHA, 'Checkpoint checksum mismatch'
    assert digest(DATA / 'node_attributes.json') == '07cb0af6853c113b3fc69e8fe874b56754f499936ddbea79786bf9b3db558641', 'Node attribute checksum mismatch'
    assert digest(prepared / 'manifest.json') == result['signature']['prepared_sha256'], 'Manifest checksum mismatch'
    manifest = json.loads((prepared / 'manifest.json').read_text())
    for name, expected in manifest['checksums'].items():
        assert digest(prepared / name) == expected, 'Prepared checksum mismatch: ' + name
    truth = set(json.loads((prepared / 'ground_truth_nodes.json').read_text()))
    validation = np.memmap(DATA/'val_edges.bin', dtype=EDGE_DTYPE, mode='r')
    test = np.memmap(DATA/'test_edges.bin', dtype=EDGE_DTYPE, mode='r')
    assert len(validation) == manifest['events']['val'] and len(test) == manifest['events']['test']
    remaining = validation[len(validation)//3:]
    calibrator = RelationCalibrator().fit(remaining['relation'][:len(remaining)//2], remaining['loss'][:len(remaining)//2])
    threshold, scores, calibration = calibrated_scores(remaining, test, result['signature']['false_alert_rate'])
    metrics = measure(scores, truth, {n for n,s in scores.items() if s > threshold})
    assert abs(threshold-result['threshold']) < 1e-10
    for key, value in metrics.items():
        # Empirical calibration uses NumPy transcendental operations. Tiny
        # platform differences can reorder tied scores in AP/AUC; all alert
        # decisions and confusion counts must still agree exactly.
        tolerance = 1e-6 if key in ('pr_ap', 'roc_auc') else 1e-10
        assert abs(value-result['metrics'][key]) < tolerance, 'Saved metrics differ: ' + key
    print('Verified checkpoint, prepared checksums and all test node metrics', flush=True)
    temp = DATA/'investigation.building.sqlite'
    if temp.exists():
        temp.unlink()
    db = sqlite3.connect(temp)
    try:
        db.executescript('''
            PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
            CREATE TABLE events (split TEXT NOT NULL,idx INTEGER NOT NULL,src INTEGER NOT NULL,dst INTEGER NOT NULL,time REAL NOT NULL,relation INTEGER NOT NULL,loss REAL,evidence REAL,PRIMARY KEY(split,idx)) WITHOUT ROWID;
            CREATE TABLE nodes (id INTEGER PRIMARY KEY,score REAL NOT NULL,alert INTEGER NOT NULL,benchmark_member INTEGER NOT NULL,attrs TEXT NOT NULL);
            CREATE TABLE metadata (key TEXT PRIMARY KEY,value TEXT NOT NULL);
        ''')
        for split in ('train', 'val', 'test'):
            rows = np.memmap(prepared/(split+'.bin'), dtype=EVENT_DTYPE, mode='r')
            edges = None if split == 'train' else np.memmap(DATA/(split+'_edges.bin'), dtype=EDGE_DTYPE, mode='r')
            assert len(rows) == manifest['events'][split]
            for lo in range(0, len(rows), 65536):
                block = rows[lo:lo+65536]
                if edges is not None:
                    saved = edges[lo:lo+len(block)]
                    for key in ('src', 'dst', 'time', 'relation'):
                        assert np.array_equal(block[key], saved[key]), 'Event/score alignment mismatch: ' + split
                    evidence = calibrator.transform(saved['relation'], saved['loss'])
                values = ((split,lo+i,int(row['src']),int(row['dst']),float(row['time']),int(row['relation']),
                           None if edges is None else float(saved['loss'][i]),None if edges is None else float(evidence[i])) for i,row in enumerate(block))
                db.executemany('INSERT INTO events VALUES (?,?,?,?,?,?,?,?)', values)
            db.commit()
            print(f'Indexed {split}: {len(rows):,} events', flush=True)
        attrs = json.loads((DATA/'node_attributes.json').read_text())
        db.executemany('INSERT INTO nodes VALUES (?,?,?,?,?)', ((n,s,int(s>threshold),int(n in truth),json.dumps(attrs.get(str(n),{}))) for n,s in scores.items()))
        del attrs
        meta = dict(events=manifest['events'], total_events=sum(manifest['events'].values()), scored_nodes=len(scores),
                    alert_nodes=sum(s>threshold for s in scores.values()), positive_scored_nodes=len(truth & scores.keys()),
                    calibration=calibration, index_version=1, recomputed_metrics=metrics,
                    metric_reproduction_max_error=max(abs(v-result['metrics'][k]) for k,v in metrics.items()),
                    manifest_sha256=digest(prepared/'manifest.json'),
                    score_files_sha256={split:digest(DATA/(split+'_edges.bin')) for split in ('val','test')})
        db.executemany('INSERT INTO metadata VALUES (?,?)', ((k,json.dumps(v)) for k,v in meta.items()))
        db.executescript('CREATE INDEX events_src ON events(split,src,time); CREATE INDEX events_dst ON events(split,dst,time); CREATE INDEX events_src_page ON events(split,src,idx,dst); CREATE INDEX events_dst_page ON events(split,dst,idx,src); CREATE INDEX nodes_risk ON nodes(alert,score DESC);')
        db.commit()
    finally:
        db.close()
    temp.replace(DATA/'investigation.sqlite')
    print('Index ready: '+str(DATA/'investigation.sqlite'), flush=True)


if __name__ == '__main__':
    build()
