"""Real benign-prefix disposable training timing; never reports research accuracy."""
import gc
import json
import time
from pathlib import Path
import torch
from orthrus_comparison.protocol import reconstruction_query
from sliding_node_mem.model import TGNNSlidingNodeMem as Original
from .model import TGNNSlidingNodeMem as Upgrade
from .pipeline import QueryCache


def benchmark(manifest_path, checkpoint_path=None, device='cpu', measured_events=32768, warmup_events=8192, reverse=False):
    torch.set_num_threads(4)
    manifest = json.loads(Path(manifest_path).read_text())
    attrs = json.loads(Path(manifest['node_attributes']).read_text())
    s, d, t, r = [], [], [], []
    needed = warmup_events + measured_events
    for path in manifest['files']['train']:
        g = torch.load(path, map_location='cpu', weights_only=False)
        take = min(needed - len(s), len(g.src))
        s += g.src[:take].tolist(); d += g.dst[:take].tolist()
        t += (g.t[:take].double()/1e9).tolist(); r += g.msg[:take, 131:141].argmax(-1).tolist()
        if len(s) >= needed: break
    if len(s) < needed: raise ValueError('Insufficient real benign prefix')
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)['model'] if checkpoint_path else None
    report = {'device': torch.cuda.get_device_name() if device.startswith('cuda') else 'CPU',
              'warmup_events': warmup_events, 'measured_events': measured_events,
              'status': 'passed', 'scope': 'Disposable training on real benign prefix; no attack metrics or checkpoints',
              'runs': []}
    def sync():
        if device.startswith('cuda'): torch.cuda.synchronize()
    configurations = [('original', Original, 1024), ('upgrade', Upgrade, 512), ('upgrade', Upgrade, 1024), ('upgrade', Upgrade, 2048)]
    if state is None:
        torch.manual_seed(0)
        initial = Original(1)
        state = {k:v.detach().cpu().clone() for k,v in initial.state_dict().items()}
        del initial
    if reverse: configurations.reverse()
    for name, cls, size in configurations:
        torch.manual_seed(0)
        m = cls(manifest['n_nodes']).to(device)
        if state: m.load_state_dict(state)
        else:
            if name == 'original': state = {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
            else: m.load_state_dict(state)
        opt = torch.optim.Adam(m.parameters(), lr=1e-5, weight_decay=1e-5)
        cache = QueryCache(attrs)
        seconds, losses = [], []
        gc_was_enabled = gc.isenabled(); gc.disable()
        if device.startswith('cuda'): torch.cuda.reset_peak_memory_stats()
        try:
            for lo in range(0, len(s), size):
                hi = min(lo + size, len(s)); sync(); started = time.perf_counter()
                queries = [cache.query(u,v) for u,v in zip(s[lo:hi],d[lo:hi])] if name=='upgrade' else [
                    reconstruction_query(attrs.get(str(u),{}), attrs.get(str(v),{})) for u,v in zip(s[lo:hi],d[lo:hi])]
                opt.zero_grad(set_to_none=True)
                out, aux = m.forward_events(queries,s[lo:hi],d[lo:hi],t[lo:hi],r[lo:hi])
                loss = torch.nn.functional.cross_entropy(out, torch.tensor(r[lo:hi],device=device))
                if name=='upgrade': loss = loss + .01 * aux['compression_loss']
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite smoke loss')
                loss.backward(); opt.step(); m.detach_state(); sync()
                elapsed = time.perf_counter()-started
                if lo >= warmup_events: seconds.append(elapsed); losses.append(float(loss.detach()))
            report['runs'].append({'implementation': name, 'batch':size, 'parameters':sum(p.numel() for p in m.parameters()),
                'seconds':sum(seconds), 'events_per_second':measured_events/sum(seconds),
                'median_batch_seconds':float(torch.tensor(seconds).median()), 'first_loss':losses[0], 'last_loss':losses[-1],
                'peak_allocated_bytes':torch.cuda.max_memory_allocated() if device.startswith('cuda') else None,
                'runtime_memory_bytes':m._memory.numel()*m._memory.element_size(),
                'runtime_memory_rows':len(m._memory), 'observed_entities':len(m.history),
                'estimated_epoch_seconds':3526541/(measured_events/sum(seconds))})
        finally:
            if gc_was_enabled: gc.enable()
        del m,opt,out,loss,aux
        gc.collect()
        if device.startswith('cuda'): torch.cuda.empty_cache()
    baseline=next(row['events_per_second'] for row in report['runs'] if row['implementation']=='original')
    for row in report['runs']:row['speedup_vs_original_1024']=row['events_per_second']/baseline
    return report
