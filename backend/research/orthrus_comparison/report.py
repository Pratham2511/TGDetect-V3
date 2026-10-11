"""Collect real study outputs without turning pending arms into zero scores."""
import argparse
import csv
import json
from pathlib import Path

from .run import ARMS


def report(root):
    root=Path(root); rows=[]; studies=[]
    for arm in ARMS:
        path=root/'results'/arm/'result.json'
        if not path.is_file():
            studies.append(dict(arm=arm,status='pending'));continue
        result=json.loads(path.read_text())
        record=dict(arm=arm,status=result['status'],signature=result['signature'],
                    parameter_count=result['parameter_count'],training_seconds=result['training_seconds'],
                    evaluation_seconds=result['evaluation_seconds'],
                    unseen_ground_truth=result['unseen_ground_truth'])
        for rule in ('threshold_only','released_top30_kmeans'):
            rows.append(dict(arm=arm,stage=rule,**result[rule]))
        trace=path.parent/'tracing'/'result.json'
        if trace.exists():
            record['tracing']=json.loads(trace.read_text())
            for rule in ('released_last_window','global_union'):
                rows.append(dict(arm=arm,stage='tracing_'+rule,**record['tracing'][rule]))
        else:record['tracing_status']='pending'
        studies.append(record)
    root.mkdir(parents=True,exist_ok=True)
    (root/'comparison.json').write_text(json.dumps(studies,indent=2))
    columns=['arm','stage','tp','fp','tn','fn','precision','recall','f1','mcc','pr_ap','roc_auc']
    with (root/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    lines=['# ORTHRUS / TGNN V3 / TGNN V4-KV: E3 CADETS', '',
           'All locally measured rows below use the shared manifest, benign relation reconstruction, and the final 20-epoch checkpoint. A missing row is pending. Synthetic checks and older sampled pilots are excluded.', '',
           '| Model | Stage | TP | FP | FN | Precision | Recall | F1 | MCC | PR-AP | ROC-AUC |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        values=[row['arm'],row['stage'],str(row['tp']),str(row['fp']),str(row['fn'])]
        values += ['—' if row[k] is None else f'{row[k]:.4f}' for k in ('precision','recall','f1','mcc','pr_ap','roc_auc')]
        lines.append('| '+' | '.join(values)+' |')
    if not rows:lines.append('| All three models | Pending full-data training | — | — | — | — | — | — | — | — | — |')
    lines += ['', '## Published paper reference (not measured in this run)', '',
              'Source: supplied `sec25cycle1-prepub-103-jiang-baoxiang.pdf`, Table 10. These figures establish a reference; exact reproduction is not assumed.', '',
              '| Paper system | TP | FP | TN | FN | Precision | MCC |',
              '|---|---:|---:|---:|---:|---:|---:|',
              '| ORTHRUS anomaly detection | 10 | 0 | 268085 | 58 | 1.00 | 0.38 |',
              '| ORTHRUS with tracing | 25 | 23 | 268062 | 43 | 0.52 | 0.44 |', '',
              'The released tracing evaluation assigns a repeated node its last-window prediction. Global-union tracing is also reported to expose that choice. The original upstream orchestration selected the epoch with best test MCC; this study selects the final epoch without test labels.', '',
              '## Run records', '']
    for study in studies:
        lines.append(f"- {study['arm']}: {study['status']}; tracing: {study.get('tracing',{}).get('status',study.get('tracing_status','pending'))}.")
        if 'parameter_count' in study:
            lines.append(f"  Parameters: {study['parameter_count']:,}; training seconds: {study['training_seconds']:.1f}; evaluation seconds: {study['evaluation_seconds']:.1f}; missing ground truth IDs: {study['unseen_ground_truth']}.")
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return studies


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True)
    report(parser.parse_args().root)
