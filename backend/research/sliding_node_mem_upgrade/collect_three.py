"""Collect original five-epoch arm outputs and render comparable operating points."""
import argparse
import json
from pathlib import Path
import time
import modal

ARMS = ('ORTHRUS_Reference','TGNN_Sliding_Node_Mem_Upgrade','TGNN_Sliding_ORTHRUS_Encoder')
ROOT = '/comparison_upgrade_5ep_v1'


def collect(output):
    volume=modal.Volume.from_name('tg-detect-sliding-node-mem-e3-v1')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    results={};progress={}
    for arm in ARMS:
        local=output/(arm+'.json')
        if local.exists(): results[arm]=json.loads(local.read_text())
        else:
            try:
                result=json.loads(b''.join(volume.read_file(f'{ROOT}/{arm}/result.json')))
                results[arm]={k:v for k,v in result.items() if k not in ('node_scores','calibrated_node_scores')}
                local.write_text(json.dumps(results[arm],indent=2))
            except (modal.exception.NotFoundError, FileNotFoundError):pass
        try:progress[arm]=json.loads(b''.join(volume.read_file(f'{ROOT}/{arm}/progress.json')))
        except (modal.exception.NotFoundError, FileNotFoundError):pass
    (output/'progress.json').write_text(json.dumps(progress,indent=2))
    (output/'comparison.json').write_text(json.dumps(results,indent=2))
    lines=['# Three-model five-epoch comparison','',
           'Same CADETS E3 dataset/splits, batch 1024, seed 0 and five epochs from scratch. ORTHRUS retains its original loss; both TGDetect arms add the documented compression-consistency regularizer. The exact-ORTHRUS-encoder TGDetect arm has more parameters, as authorized.','']
    for rule in ('released_top30_kmeans','threshold_only','calibrated'):
        lines += [f'## {rule}','','| Model | Parameters | F1 | Precision | Recall | ROC-AUC | Average precision | TN | FP | FN | TP | Accuracy |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for arm in ARMS:
            result=results.get(arm,{})
            if rule not in result: continue
            m=result[rule];accuracy=(m['tn']+m['tp'])/sum(m[k] for k in ('tn','fp','fn','tp'))
            lines.append(f"| {arm} | {result.get('parameter_count')} | {m['f1']:.6f} | {m['precision']:.6f} | {m['recall']:.6f} | {m['roc_auc']:.6f} | {m['pr_ap']:.6f} | {m['tn']} | {m['fp']} | {m['fn']} | {m['tp']} | {accuracy:.6f} |")
        lines += ['']
    lines += ['Pending arms: '+', '.join(arm for arm in ARMS if arm not in results)]
    (output/'COMPARISON.md').write_text('\n'.join(lines)+'\n')
    return {'completed':list(results),'latest_epochs':{a:p[-1]['epoch'] if p else 0 for a,p in progress.items()}}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--follow',action='store_true')
    parser.add_argument('--output',default='artifacts/three_arm_five_epochs');args=parser.parse_args()
    while True:
        status=collect(args.output);print(json.dumps(status),flush=True)
        if not args.follow or len(status['completed'])==3:break
        time.sleep(30)
