"""Run independent three-seed temporal ablations on the official prepared split.

Each arm retains the runner's benign validation checkpoint selection and held-out
calibration. Results are never selected by test F1. Existing runs must match the
source/configuration signature to resume.
"""
import argparse
import json
from pathlib import Path
from .run import run


ARMS={
    'baseline':{},
    'short_attention':{'short_readout':'attention','static_short_read':True},
    'no_short_read':{'short_readout':'none','static_short_read':True},
    'causal_attention':{'temporal':'attention','short_readout':'attention','static_short_read':True},
}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--epochs',type=int,default=5)
    p.add_argument('--arms',nargs='+',choices=ARMS,default=list(ARMS))
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    p.add_argument('--resident-tokens',action='store_true')
    args=p.parse_args();results={}
    for arm in args.arms:
        for seed in args.seeds:
            result=run(args.prepared,Path(args.output)/arm/str(seed),epochs=args.epochs,
                       device=args.device,seed=seed,model_options=ARMS[arm],resident_tokens=args.resident_tokens)
            results[f'{arm}/{seed}']=dict(metrics=result['metrics'],parameters=result['parameter_count'])
            (Path(args.output)/'summary.json').write_text(json.dumps(results,indent=2))


if __name__=='__main__':main()
