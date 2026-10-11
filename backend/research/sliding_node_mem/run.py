"""Isolated CADETS comparison; prior experiment source/results are untouched."""
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.cluster import KMeans
from orthrus_comparison.protocol import require_coverage, reconstruction_query, partition
from orthrus_comparison.run import sha256, measure, run as run_reference
from .model import TGNNSlidingNodeMem
from .budget import verify_budget, learned_count

ARM = 'TGNN_Sliding_Node_Mem'


def run(manifest_path, output, epochs=20, batch=1024, device='cpu', seed=0,
        on_checkpoint=None, checkpoint_seconds=300):
    torch.set_num_threads(4); torch.manual_seed(seed); np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = json.loads(Path(manifest_path).read_text())
    require_coverage(manifest['days'])
    if manifest.get('dataset') != 'CADETS_E3' or not manifest.get('official_preprocessing'):
        raise ValueError('Official full CADETS E3 preprocessing required')
    for p, expected in manifest['file_sha256'].items():
        if sha256(p) != expected: raise ValueError('Dataset checksum mismatch: ' + p)
    attrs = json.loads(Path(manifest['node_attributes']).read_text())
    budget = verify_budget()
    model = TGNNSlidingNodeMem(manifest['n_nodes']).to(device)
    if learned_count(model) != budget['trainable_parameters'][ARM]:
        raise ValueError('Instantiated detector differs from matched parameter budget')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
    root = Path(__file__).resolve().parents[1]
    sources = list(Path(__file__).parent.glob('*.py')) + [root/p for p in
        ('node_memory_v3.py', 'node_memory_models.py', 'tgnn_components.py', 'temporal_cnn.py',
         'causal_tgn.py', 'pomsathit_baseline.py', 'orthrus_comparison/universal.py',
         'orthrus_comparison/protocol.py', 'orthrus_comparison/run.py', 'orthrus_comparison/official.py')]
    sources += list((root/'orthrus_official'/'src').rglob('*.py'))
    signature = dict(arm=ARM, manifest_sha256=sha256(manifest_path), epochs=epochs,
                     batch=batch, seed=seed, model=model.config, parameter_budget=budget,
                     source_hashes={str(p.relative_to(root)): sha256(p) for p in sources})
    out = Path(output)/ARM; out.mkdir(parents=True, exist_ok=True)
    # Preserve original source and checkpoint when migrating this specific run.
    import shutil
    journal = out/'resume.pt'; history = []; epoch_start = 0; resume = None
    if journal.exists():
        # RNG byte tensors must remain on CPU; state_dict loading and Adam move
        # their own weights/state to the model device.
        saved = torch.load(journal, map_location='cpu', weights_only=False)
        if saved['signature'] != signature:
            old = saved['signature']
            legacy = json.loads((Path(__file__).parent/'allowed_performance_hashes.json').read_text())
            unchanged_protocol = {k:v for k,v in old.items() if k!='source_hashes'} == {k:v for k,v in signature.items() if k!='source_hashes'}
            if not unchanged_protocol or old['source_hashes'] not in legacy:
                raise ValueError('Resume source/protocol mismatch; not an approved performance migration')
            if (out/'source').exists() and not (out/'source_before_optimization').exists():
                (out/'source').rename(out/'source_before_optimization')
            if not (out/'resume_before_optimization.pt').exists(): shutil.copy2(journal,out/'resume_before_optimization.pt')
            (out/'performance_migration.json').write_text(json.dumps(dict(previous=old,current=signature,
                reason='Equivalent token cache, bulk short-window stacking, touched-node memory commit, packed CPU snapshots and CPU RNG restore'),indent=2))
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        history, epoch_start = saved['history'], saved['epoch']
        resume = saved.get('stream')
        torch.set_rng_state(saved['rng'].cpu())
        if device.startswith('cuda'): torch.cuda.set_rng_state_all([state.cpu() for state in saved['cuda_rng']])
    for source in sources:
        dest = out/'source'/source.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, dest)
    last_save = time.monotonic()

    def save(epoch, stream=None):
        nonlocal last_save
        state = dict(signature=signature, model=model.state_dict(), optimizer=optimizer.state_dict(),
                     epoch=epoch, history=history, stream=stream, rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else [])
        torch.save(state, out/'resume.tmp'); (out/'resume.tmp').replace(journal)
        (out/'progress.json').write_text(json.dumps(history, indent=2))
        if on_checkpoint: on_checkpoint()
        last_save = time.monotonic()

    def stream(split, training=False, epoch=0, previous=None):
        model.train(training); model.reset_state()
        start_graph = start_offset = 0; total = 0.; count = 0; elapsed = 0.
        if previous:
            model.load_stream_state_dict(previous['state'])
            start_graph, start_offset = previous['graph'], previous['offset']
            total, count, elapsed = previous['total'], previous['count'], previous['seconds']
        started = time.perf_counter(); scores = {}
        handle = (out/'test_edges.jsonl').open('w') if split == 'test' else None
        # Histories and token references are acyclic and released by reference
        # counting. Avoid cyclic-GC scans of the growing entity/event heap in
        # the per-batch path; restore normal GC at split/epoch boundaries.
        gc_was_enabled=gc.isenabled()
        gc.disable()
        try:
            for gi, path in enumerate(manifest['files'][split]):
                if gi < start_graph: continue
                g = torch.load(path, map_location='cpu', weights_only=False)
                for lo in range(start_offset if gi == start_graph else 0, len(g.src), batch):
                    hi = min(lo+batch, len(g.src))
                    s, d = g.src[lo:hi].tolist(), g.dst[lo:hi].tolist()
                    t = (g.t[lo:hi].double()/1e9).tolist()
                    if any(partition(v) != ('validation' if split == 'val' else split) for v in t):
                        raise ValueError('Event outside registered paper split')
                    targets = g.msg[lo:hi, 131:141].argmax(-1)
                    queries = [reconstruction_query(attrs.get(str(u), {}), attrs.get(str(v), {})) for u, v in zip(s, d)]
                    optimizer.zero_grad(set_to_none=True)
                    with torch.set_grad_enabled(training):
                        logits, aux = model.forward_events(queries, s, d, t, targets)
                        losses = F.cross_entropy(logits, targets.to(device), reduction='none')
                        loss = losses.mean()
                        if not torch.isfinite(loss): raise RuntimeError('Nonfinite relation loss')
                        if training: loss.backward(); optimizer.step()
                        model.detach_state()
                    total += float(loss.detach())*len(s); count += len(s)
                    if not training:
                        for u, v, when, value in zip(s, d, t, losses.detach().cpu().tolist()):
                            scores[u] = max(scores.get(u, 0.), value); scores[v] = max(scores.get(v, 0.), value)
                            if handle: handle.write(json.dumps(dict(srcnode=u, dstnode=v, t=when, loss=value))+'\n')
                    if training and time.monotonic()-last_save >= checkpoint_seconds:
                        seconds = elapsed + time.perf_counter()-started
                        status = dict(stage='training', epoch=epoch+1, events=count, loss=total/count,
                                      seconds=seconds, events_per_second=count/max(seconds, 1e-9),
                                      admission=float(aux['admission'].detach().mean()), short_slots=float(aux['short_count'].float().mean()))
                        (out/'status.json').write_text(json.dumps(status, indent=2)); print(status, flush=True)
                        save(epoch, dict(state=model.stream_state_dict(), graph=gi, offset=hi,
                                         total=total, count=count, seconds=seconds))
            if not count: raise ValueError('Empty split')
            return total/count, scores, elapsed+time.perf_counter()-started
        finally:
            if gc_was_enabled:gc.enable()
            if handle: handle.close()

    for epoch in range(epoch_start, epochs):
        loss, _, seconds = stream('train', True, epoch, resume)
        resume = None
        history.append(dict(epoch=epoch+1, loss=loss, seconds=seconds))
        save(epoch+1)
        print(ARM, epoch+1, loss, seconds, flush=True)
    started = time.perf_counter()
    (out/'status.json').write_text(json.dumps(dict(stage='evaluating', epoch=epochs)))
    if on_checkpoint: on_checkpoint()
    _, val, _ = stream('val'); _, test, _ = stream('test')
    truth = set(map(int, json.loads(Path(manifest['ground_truth_nodes']).read_text())))
    threshold = max(val.values()); above = {n for n, v in test.items() if v > threshold}
    selected = sorted(test, key=test.get)[-30:]
    if len(selected) < 30: raise ValueError('At least 30 test nodes required')
    cl = KMeans(n_clusters=2, random_state=0, n_init=10).fit(np.array([test[n] for n in selected]).reshape(-1, 1))
    predicted = {n for n, label in zip(selected, cl.labels_) if label == cl.cluster_centers_.argmax()}
    result = dict(signature=signature, status='complete', objective='benign relation cross entropy',
                  checkpoint_selection='final epoch', threshold=threshold,
                  threshold_only=measure(test, truth, above), released_top30_kmeans=measure(test, truth, predicted),
                  predicted_nodes=sorted(predicted), threshold_nodes=sorted(above), node_scores=test,
                  parameter_count=learned_count(model), parameter_budget=budget, checkpoint_epoch=epochs,
                  training_seconds=sum(h['seconds'] for h in history), evaluation_seconds=time.perf_counter()-started,
                  unseen_ground_truth=sorted(truth-set(test)),
                  note='No global rehearsal in this matched benign-objective comparison; stable-order tied timestamps.')
    (out/'result.json').write_text(json.dumps(result, indent=2))
    (out/'status.json').write_text(json.dumps(dict(stage='complete', epoch=epochs)))
    if on_checkpoint: on_checkpoint()
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True); p.add_argument('--output', required=True)
    p.add_argument('--arm', choices=[ARM, 'ORTHRUS_Reference'], default=ARM)
    p.add_argument('--epochs', type=int, default=20); p.add_argument('--batch', type=int, default=1024)
    p.add_argument('--device', default='cpu'); p.add_argument('--seed', type=int, default=0)
    a = vars(p.parse_args()); a['manifest_path'] = a.pop('manifest'); arm = a.pop('arm')
    (run if arm == ARM else run_reference)(**a)
