"""Compare warm streaming latency; synthetic timings are not detection results."""
import argparse
import gc
import hashlib
import cProfile
import io
import json
from pathlib import Path
import pstats
import time
import numpy as np
import torch
from sliding_node_mem_upgrade.model import TGNNSlidingNodeMem as Reference
from .model import TGNNSlidingNodeMem
from .runtime import stream_gc


def synthetic_batch(batch, index, reuse=True):
    base = index*batch
    events = []
    for i in range(batch):
        ident = i if reuse else base+i
        events.append({'attributes': {'src': {'type':'process','path':f'/bin/p{ident%17 if reuse else ident}'},
                                      'dst': {'type':'file','path':f'/tmp/f{ident%31 if reuse else ident}'}}})
    return events, [i%64 for i in range(batch)], [64+i%128 for i in range(batch)], \
        [float(base+i) for i in range(batch)], [i%10 for i in range(batch)]


def synchronize(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


def benchmark(device='cpu', batches=(1,32,256,1024), repeats=12, warmup=4,
              output='artifacts/sliding_v2/latency.json', reuse=True, input_batches=None,
              n_nodes=256):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    if repeats < 2 or warmup < 1:
        raise ValueError('At least two measured and one warmup batch required')
    result = dict(device=str(device), torch=torch.__version__,
                  workload='real prepared prefix' if input_batches is not None else 'synthetic repeated attributes' if reuse else 'synthetic unique attributes',
                  scope='forward latency with prepared CPU events; includes encoding, state update, and detach/maintenance',
                  gc_policy='cyclic GC disabled during each ordered stream, as in reference training runner',
                  source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
                  warmup=warmup, repeats=repeats, threads=1, results=[])
    for size in batches:
        torch.manual_seed(42)
        reference = Reference(n_nodes).to(device).eval()
        weights = {k:v.detach().cpu().clone() for k,v in reference.state_dict().items()}
        del reference
        samples = input_batches if input_batches is not None else [synthetic_batch(size,i,reuse) for i in range(warmup+repeats)]
        if len(samples) < warmup+repeats:
            raise ValueError('Insufficient real input batches for the requested measurement')
        measurements = {}
        # Rotate order with batch size to reduce systematic first-run bias.
        arms = ['reference','compatible','corrected']
        if size % 3:
            arms.reverse()
        for name in arms:
            gc.collect()
            torch.manual_seed(42)
            model = (Reference(n_nodes) if name == 'reference' else TGNNSlidingNodeMem(n_nodes,profile=name)).to(device).eval()
            if name != 'corrected':
                model.load_state_dict(weights)
            if str(device).startswith('cuda'):
                torch.cuda.reset_peak_memory_stats()
            durations = []
            with stream_gc(), torch.no_grad():
                for j,data in enumerate(samples[:warmup+repeats]):
                    synchronize(device); started = time.perf_counter()
                    model.forward_events(*data)
                    model.detach_state()
                    synchronize(device)
                    if j >= warmup:
                        durations.append(time.perf_counter()-started)
            count = sum(len(row[0]) for row in samples[warmup:warmup+repeats])
            values = np.asarray(durations)*1000
            measurements[name] = dict(p50_batch_ms=float(np.median(values)),
                                      p95_batch_ms=float(np.quantile(values,.95)),
                                      p99_batch_ms=float(np.quantile(values,.99)),
                                      events_per_second=count/sum(durations),
                                      amortized_ms_per_event=1000*sum(durations)/count,
                                      parameters=sum(p.numel() for p in model.parameters()),
                                      peak_cuda_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else None,
                                      cache_hits=getattr(model.event_encoder,'cache_hits',None),
                                      cache_misses=getattr(model.event_encoder,'cache_misses',None))
            del model
            if str(device).startswith('cuda'):
                torch.cuda.empty_cache()
        result['results'].append(dict(batch=size,arms=measurements,
            compatible_speedup=measurements['compatible']['events_per_second']/measurements['reference']['events_per_second'],
            corrected_speedup=measurements['corrected']['events_per_second']/measurements['reference']['events_per_second']))
    path = Path(output); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')
    return result


def profile_baseline(output='artifacts/sliding_v2/profile.txt', batch=1024):
    torch.set_num_threads(1)
    model = Reference(256).eval()
    with torch.no_grad():
        model.forward_events(*synthetic_batch(batch,0))
        profiler = cProfile.Profile(); profiler.enable()
        for j in range(1,5):
            model.forward_events(*synthetic_batch(batch,j))
        profiler.disable()
    buffer = io.StringIO()
    pstats.Stats(profiler,stream=buffer).sort_stats('cumulative').print_stats(35)
    path=Path(output); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(buffer.getvalue())


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--batches',type=int,nargs='+',default=[1,32,256,1024])
    parser.add_argument('--repeats',type=int,default=12)
    parser.add_argument('--warmup',type=int,default=4)
    parser.add_argument('--output',default='artifacts/sliding_v2/latency.json')
    parser.add_argument('--unique',action='store_true')
    args=vars(parser.parse_args()); args['reuse']=not args.pop('unique')
    print(json.dumps(benchmark(**args),indent=2))
