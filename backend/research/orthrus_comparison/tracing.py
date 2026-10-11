"""Apply the pinned ORTHRUS dependency tracing algorithm to every study arm.

Ground truth is used only after tracing. Final-epoch node alerts seed the same
per-window component/degree tracing for all arms. Both the released evaluation
loop's last-window assignment and global union of traced nodes are reported.
"""
import ast
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import time

import networkx as nx
import numpy as np
import torch
from tqdm import tqdm

from .official import ROOT
from .run import measure, sha256


def upstream_tracer():
    path=ROOT/'src/triage/tracing_methods/depimpact_utils.py'
    tree=ast.parse(path.read_text())
    definitions=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
    namespace=dict(nx=nx,np=np,torch=torch,tqdm=tqdm,os=os,datetime=datetime)
    exec(compile(ast.Module(body=definitions,type_ignores=[]),str(path),'exec'),namespace)
    return namespace['DEPIMPACT']


def trace_window(task):
    index,path,seeds,scores=task
    graph=torch.load(path,map_location='cpu',weights_only=False)
    if any(not isinstance(n,str) for n in graph):
        raise ValueError('Pinned provenance graphs must use string node IDs')
    tracer=upstream_tracer(); nodes=set();details=[]
    for seed in seeds:
        started=time.perf_counter()
        traced=tracer(graph,str(seed),scores,'component','degree').run()
        nodes.update(map(int,traced))
        details.append(dict(seed=seed,nodes=sorted(map(int,traced)),seconds=time.perf_counter()-started))
    return index,dict(graph=path,nodes=sorted(nodes),seeds=details,
                       observed_nodes=sorted(map(int,graph.nodes())))


def run_tracing(manifest_path,result_path,workers=8,on_checkpoint=None):
    manifest=json.loads(Path(manifest_path).read_text())
    path=Path(result_path);result=json.loads(path.read_text())
    if result['status']!='complete': raise ValueError('Detector must finish before tracing')
    graphs=manifest.get('graph_files',{}).get('test')
    if not graphs: raise ValueError('Original upstream test graphs required for tracing')
    scores={int(n):float(v) for n,v in result['node_scores'].items()}
    truth=set(map(int,json.loads(Path(manifest['ground_truth_nodes']).read_text())))
    seeds=set(result['predicted_nodes']);tasks=[];observed={}
    for index,graph_path in enumerate(graphs):
        expected=manifest.get('file_sha256',{}).get(graph_path)
        if expected and sha256(graph_path)!=expected: raise ValueError('Original graph changed')
        graph=torch.load(graph_path,map_location='cpu',weights_only=False)
        present=set(map(int,graph.nodes()));observed[index]=present
        window_seeds=sorted(seeds & present)
        if window_seeds:tasks.append((index,graph_path,window_seeds,{str(n):scores[n] for n in present if n in scores}))
    out=path.parent/'tracing';out.mkdir(exist_ok=True)
    started=time.perf_counter();windows={}
    if workers==1:
        results=map(trace_window,tasks)
        pool=None
    else:
        pool=ProcessPoolExecutor(max_workers=workers)
        results=pool.map(trace_window,tasks)
    try:
        for index,info in results:
            windows[index]=info
            (out/f'window_{index:05d}.json').write_text(json.dumps(info,indent=2))
            if on_checkpoint:on_checkpoint()
    finally:
        if pool:pool.shutdown()
    union={n for info in windows.values() for n in info['nodes']}
    # Pinned tracing.get_new_stats overwrites a repeated node's y_hat in each
    # successive window; report that behavior without conflating it with union.
    last_assignment={}
    for index,present in observed.items():
        traced=set(windows.get(index,{}).get('nodes',[]))
        for n in present & set(scores):last_assignment[n]=n in traced
    released={n for n,v in last_assignment.items() if v}
    attacks={}
    if manifest.get('attack_nodes'):
        attacks=json.loads(Path(manifest['attack_nodes']).read_text())
    coverage={name:dict(ground_truth_nodes=len(nodes),
                        detector_hits=sorted(seeds & set(nodes)),
                        traced_hits=sorted(union & set(nodes))) for name,nodes in attacks.items()}
    summary=dict(status='complete',algorithm='pinned DEPIMPACT component/degree',
                 source_sha256=sha256(ROOT/'src/triage/tracing_methods/depimpact_utils.py'),
                 seconds=time.perf_counter()-started,windows_traced=len(windows),
                 total_seed_windows=len(tasks),traced_nodes=sorted(union),
                 released_last_window=measure(scores,truth,released),
                 global_union=measure(scores,truth,union),attack_group_coverage=coverage,
                 unseen_traced_nodes=sorted(union-set(scores)),
                 note='Final epoch selected without test labels; original tracing orchestration selected best test MCC epoch.')
    (out/'result.json').write_text(json.dumps(summary,indent=2))
    if on_checkpoint:on_checkpoint()
    return summary
