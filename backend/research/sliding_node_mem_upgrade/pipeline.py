"""Bounded CPU query preparation overlapped with ordered GPU batches."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import json
import torch
from orthrus_comparison.protocol import reconstruction_query


class QueryCache:
    def __init__(self, attrs, capacity=65536):
        self.attrs, self.capacity = attrs, capacity
        self.cache = OrderedDict()

    def query(self, u, v):
        key = (u, v)
        event = self.cache.pop(key, None)
        if event is None:
            event = reconstruction_query(self.attrs.get(str(u), {}), self.attrs.get(str(v), {}))
            event['_canonical_attributes'] = json.dumps(event['attributes'], sort_keys=True, separators=(',', ':'))
        self.cache[key] = event
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return event


def prepared_batches(paths, attrs, batch=1024, encoding_config=None, include_messages=False):
    """One producer owns the cache; two pending batches bound CPU memory.

    GPU state is touched only by the consumer in ingestion order. Windows in
    each batch execute together; overlapping entity memories never race.
    """
    cache = QueryCache(attrs)
    def produce():
        for gi, path in enumerate(paths):
            graph = torch.load(path, map_location='cpu', weights_only=False)
            for lo in range(0, len(graph.src), batch):
                hi = min(lo + batch, len(graph.src))
                src, dst = graph.src[lo:hi].tolist(), graph.dst[lo:hi].tolist()
                times = (graph.t[lo:hi].double() / 1e9).tolist()
                queries = [cache.query(u, v) for u, v in zip(src, dst)]
                targets = graph.msg[lo:hi, 131:141].argmax(-1)
                if encoding_config is not None:
                    from .encoding import canonical_identity, token_row
                    unique = {(None, q['_canonical_attributes']) for q in queries}
                    unique.update((str(r), q['_canonical_attributes']) for r,q in zip(targets.tolist(),queries))
                    for event_type, attributes in unique:
                        token_row(canonical_identity(event_type,attributes), *encoding_config)
                item = (gi, lo, hi, src, dst, times, targets, queries)
                yield item + (graph.msg[lo:hi],) if include_messages else item
    iterator = iter(produce())
    sentinel = object()
    def advance():
        return next(iterator, sentinel)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='event-preparation') as pool:
        future = pool.submit(advance)
        while True:
            item = future.result()
            if item is sentinel:
                break
            future = pool.submit(advance)
            yield item
