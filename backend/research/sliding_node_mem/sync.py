"""Collect the new experiment into its own directory; never edit old ledgers."""
import argparse
import csv
import io
import json
from pathlib import Path
import time
import modal

ROOT=Path('artifacts/sliding_node_mem_e3')
ARMS=('ORTHRUS_Reference','TGNN_Sliding_Node_Mem')


def collect(checkpoints=False,epoch14=False):
    volume=modal.Volume.from_name('tg-detect-sliding-node-mem-e3-v1')
    ROOT.mkdir(parents=True,exist_ok=True)
    def fetch(remote,local):
        try: raw=b''.join(volume.read_file(remote))
        except FileNotFoundError:return None
        local.parent.mkdir(parents=True,exist_ok=True)
        temporary=local.with_suffix(local.suffix+'.tmp');temporary.write_bytes(raw);temporary.replace(local)
        return json.loads(raw) if local.suffix=='.json' else None
    for name in ('status.json','preflight.json','transfer.json','parameter_budget.json','performance.json'):
        fetch('/'+name,ROOT/name)
    rows=[]; progress={}
    for arm in ARMS:
        base=ROOT/'results'/arm
        history=fetch(f'/results/{arm}/progress.json',base/'progress.json')
        if history:progress[arm]=history[-1]
        for name in ('status.json','failure.json'):fetch(f'/results/{arm}/{name}',base/name)
        result=fetch(f'/results/{arm}/result.json',base/'result.json')
        trace=fetch(f'/results/{arm}/tracing/result.json',base/'tracing'/'result.json')
        if result:
            for stage in ('threshold_only','released_top30_kmeans'):
                rows.append(dict(arm=arm,stage=stage,**result[stage]))
            if trace:
                for stage in ('released_last_window','global_union'):
                    rows.append(dict(arm=arm,stage='tracing_'+stage,**trace[stage]))
            if checkpoints and not (base/'resume.pt').exists():
                fetch(f'/results/{arm}/resume.pt',base/'resume.pt')
    (ROOT/'comparison.json').write_text(json.dumps(rows,indent=2))
    if rows:
        fields=['arm','stage','tp','fp','tn','fn','precision','recall','f1','mcc','pr_ap','roc_auc']
        with (ROOT/'comparison.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    text='# Sliding node memory vs ORTHRUS: current experiment\n\n'
    text+='Both arms use full E3 CADETS, benign relation reconstruction, seed 0, 20 epochs, L4, batch 1024, Adam lr/weight decay 1e-5, and exactly 764,202 trainable detector parameters. Representation differs (shared KV vs released Word2Vec); fixed preprocessing and runtime state are excluded from the weight count. This is one seed; synthetic checks and prefix throughput are excluded from accuracy results.\n\n'
    text+='| Model | Saved epoch | Training loss |\n|---|---:|---:|\n'
    for arm in ARMS:
        p=progress.get(arm)
        text+=f"| {arm} | {p['epoch'] if p else 'pending'} | {p['loss'] if p else 'pending'} |\n"
    if rows:
        text+='\n| Model | Stage | TP | FP | FN | Precision | Recall | F1 | PR-AP |\n|---|---|---:|---:|---:|---:|---:|---:|---:|\n'
        for r in rows:
            text+=f"| {r['arm']} | {r['stage']} | {r['tp']} | {r['fp']} | {r['fn']} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['pr_ap']:.4f} |\n"
        text+='\nTracing changes binary alerts; its AP/AUC retain the detector ranking.\n'
    else:text+='\nEvaluation results are pending.\n'
    (ROOT/'COMPARISON.md').write_text(text)
    e14_status=fetch('/epoch14_status.json',ROOT/'epoch14_status.json') or {}
    e14rows=[];e14epochs={}
    for arm in ARMS:
        base=ROOT/'results_epoch14'/arm
        history=fetch(f'/results_epoch14/{arm}/progress.json',base/'progress.json')
        if history:e14epochs[arm]=history[-1]
        result=fetch(f'/results_epoch14/{arm}/result.json',base/'result.json')
        if result:
            for stage in ('threshold_only','released_top30_kmeans'):
                e14rows.append(dict(arm=arm,stage=stage,**result[stage]))
    (ROOT/'epoch14_comparison.json').write_text(json.dumps(e14rows,indent=2))
    if e14rows:
        fields=['arm','stage','tp','fp','tn','fn','precision','recall','f1','mcc','pr_ap','roc_auc']
        with (ROOT/'epoch14_comparison.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(e14rows)
    summary='# Matched 14-epoch CADETS E3 comparison\n\n'
    summary+='Both models were initialized fresh with seed 0, trained for 14 epochs on the same benign CADETS split, and evaluated with the same validation threshold and released top-30 KMeans rule. Each has 764,202 trainable parameters.\n\n'
    summary+='| Model | Saved epoch | Training loss |\n|---|---:|---:|\n'
    for arm in ARMS:
        p=e14epochs.get(arm);summary+=f"| {arm} | {p['epoch'] if p else 'pending'} | {p['loss'] if p else 'pending'} |\n"
    if e14rows:
        summary+='\n| Model | Evaluation | TP | FP | FN | Precision | Recall | F1 | PR-AUC |\n|---|---|---:|---:|---:|---:|---:|---:|---:|\n'
        for r in e14rows:summary+=f"| {r['arm']} | {r['stage']} | {r['tp']} | {r['fp']} | {r['fn']} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['pr_ap']:.4f} |\n"
    else:summary+='\nEvaluation results are pending.\n'
    (ROOT/'EPOCH14_COMPARISON.md').write_text(summary)
    status=json.loads((ROOT/'status.json').read_text()) if (ROOT/'status.json').exists() else {}
    current=e14_status if epoch14 else status
    print(json.dumps(dict(stage=current.get('stage','pending'),progress=e14epochs if epoch14 else progress,
                          metric_rows=len(e14rows) if epoch14 else len(rows))),flush=True)
    return current


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--follow',action='store_true');p.add_argument('--checkpoints',action='store_true');p.add_argument('--epoch14',action='store_true')
    a=p.parse_args()
    while True:
        status=collect(a.checkpoints,a.epoch14)
        if not a.follow or status.get('stage') in ('complete','failed'):break
        time.sleep(60)
