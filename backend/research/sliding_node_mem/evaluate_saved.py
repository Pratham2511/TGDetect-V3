"""Score a saved TGNN sliding checkpoint without running training."""
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans

from orthrus_comparison.protocol import require_coverage, reconstruction_query, partition
from orthrus_comparison.run import measure
from .model import TGNNSlidingNodeMem
from .budget import verify_budget, learned_count


def evaluate(manifest_path, checkpoint_path, output_path, device='cuda', batch=1024):
    torch.set_num_threads(4)
    manifest = json.loads(Path(manifest_path).read_text())
    require_coverage(manifest['days'])
    attrs = json.loads(Path(manifest['node_attributes']).read_text())
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = TGNNSlidingNodeMem(manifest['n_nodes']).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    if learned_count(model) != verify_budget()['trainable_parameters']['TGNN_Sliding_Node_Mem']:
        raise ValueError('Saved detector does not match the expected parameter budget')

    def score_split(split):
        model.reset_state()
        scores = {}
        total, count = 0.0, 0
        for path in manifest['files'][split]:
            graph = torch.load(path, map_location='cpu', weights_only=False)
            for lo in range(0, len(graph.src), batch):
                hi = min(lo + batch, len(graph.src))
                src = graph.src[lo:hi].tolist()
                dst = graph.dst[lo:hi].tolist()
                times = (graph.t[lo:hi].double() / 1e9).tolist()
                expected = 'validation' if split == 'val' else split
                if any(partition(t) != expected for t in times):
                    raise ValueError('Event outside registered paper split')
                targets = graph.msg[lo:hi, 131:141].argmax(-1)
                queries = [reconstruction_query(attrs.get(str(u), {}), attrs.get(str(v), {}))
                           for u, v in zip(src, dst)]
                with torch.no_grad():
                    logits, _ = model.forward_events(queries, src, dst, times, targets)
                    losses = torch.nn.functional.cross_entropy(
                        logits, targets.to(device), reduction='none')
                total += float(losses.sum().cpu())
                count += len(src)
                for u, v, loss in zip(src, dst, losses.cpu().tolist()):
                    scores[u] = max(scores.get(u, 0.0), loss)
                    scores[v] = max(scores.get(v, 0.0), loss)
                model.detach_state()
        if not count:
            raise ValueError('Empty evaluation split: ' + split)
        return total / count, scores

    started = time.perf_counter()
    _, val = score_split('val')
    _, test = score_split('test')
    truth = set(map(int, json.loads(Path(manifest['ground_truth_nodes']).read_text())))
    threshold = max(val.values())
    above = {n for n, value in test.items() if value > threshold}
    selected = sorted(test, key=test.get)[-30:]
    cluster = KMeans(n_clusters=2, random_state=0, n_init=10)
    labels = cluster.fit_predict(np.array([test[n] for n in selected]).reshape(-1, 1))
    high_cluster = int(cluster.cluster_centers_.argmax())
    predicted = {n for n, label in zip(selected, labels) if label == high_cluster}
    result = {
        'status': 'complete',
        'checkpoint_completed_epochs': len(checkpoint.get('history', [])),
        'checkpoint_epoch_field': checkpoint.get('epoch'),
        'checkpoint_stream_offset': (checkpoint.get('stream') or {}).get('offset'),
        'checkpoint_stream_graph': (checkpoint.get('stream') or {}).get('graph'),
        'checkpoint_note': 'Saved weights are from the latest checkpoint; if stream state is present, it includes partial next-epoch updates.',
        'parameter_count': learned_count(model),
        'threshold': threshold,
        'threshold_only': measure(test, truth, above),
        'released_top30_kmeans': measure(test, truth, predicted),
        'test_nodes': len(test),
        'evaluation_seconds': time.perf_counter() - started,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(result, indent=2))
    return result
