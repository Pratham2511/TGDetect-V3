"""Sequence-aware value encoding and bounded, weight-versioned inference cache."""
from collections import OrderedDict
from functools import lru_cache
import json
import numpy as np
import torch
from sliding_node_mem_upgrade.encoding import UpgradedUniversalEventEncoder, token_row, canonical_identity


@lru_cache(maxsize=32768)
def sequence_row(identity, buckets, max_fields, max_subtokens):
    """Unigrams plus position-tagged adjacent subtokens, sharing the token table.

    Unlike a weighted bag, this distinguishes equal positional-sum permutations.
    Hashing is still lossy; this is a bounded local-order encoder, not lossless text.
    """
    keys, words, kinds, numbers, truncated = token_row(identity,buckets,max_fields,max_subtokens)
    expanded = np.zeros((len(keys),2*max_subtokens),dtype=np.int64)
    expanded[:,:max_subtokens] = words
    if max_subtokens > 1:
        left,right = words[:,:-1],words[:,1:]
        positions = np.arange(1,max_subtokens,dtype=np.int64)[None]
        valid = (left != 0) & (right != 0)
        # The original tokenizer may retain the first and last half of a long
        # value. Suppress the possible gap edge in all full-length rows.
        full = (words != 0).sum(-1) == max_subtokens
        valid[full,(max_subtokens+1)//2-1] = False
        hashes = 1 + ((left*73856093) ^ (right*19349663) ^ (positions*83492791)) % (buckets-1)
        expanded[:,max_subtokens:max_subtokens+max_subtokens-1] = np.where(valid,hashes,0)
    return keys,expanded,kinds,numbers,truncated


class EventEncoder(UpgradedUniversalEventEncoder):
    def __init__(self, *args, ordered_values=True, cache_size=8192, **kwargs):
        super().__init__(*args, **kwargs)
        self.ordered_values = ordered_values
        self.cache_size = cache_size
        self.clear_cache()

    def clear_cache(self):
        self._cache = OrderedDict()
        self._cache_values = None
        self._cache_version = None
        self.cache_hits = self.cache_misses = 0

    def train(self, mode=True):
        if mode:
            self.clear_cache()
        return super().train(mode)

    def _apply(self, fn, recurse=True):
        self.clear_cache()
        return super()._apply(fn, recurse=recurse)

    def _load_from_state_dict(self, *args, **kwargs):
        self.clear_cache()
        return super()._load_from_state_dict(*args, **kwargs)

    def encode(self, events):
        identities = [canonical_identity(e.get('type'),e.get('_canonical_attributes') or
                      json.dumps(e.get('attributes'),sort_keys=True,separators=(',',':'))) for e in events]
        row_fn = sequence_row if self.ordered_values else token_row
        rows = [row_fn(key,self.buckets,self.max_fields,self.max_subtokens) for key in identities]
        width = max(len(row[0]) for row in rows)
        n = len(rows); length = self.max_subtokens*(2 if self.ordered_values else 1)
        keys=np.zeros((n,width),dtype=np.int64); kinds=np.zeros_like(keys)
        number=np.zeros((n,width,1),dtype=np.float32); mask=np.ones((n,width),dtype=bool)
        words=np.zeros((n,width,length),dtype=np.int64)
        for i,(key,word,kind,numeric,_) in enumerate(rows):
            size=len(key);keys[i,:size]=key;kinds[i,:size]=kind;number[i,:size]=numeric
            words[i,:size]=word;mask[i,:size]=False
        flat=words.reshape(-1,length); valid=flat!=0; counts=valid.sum(-1)
        offsets=np.concatenate(([0],counts.cumsum())).astype(np.int64)
        if self.ordered_values:
            weights=valid.astype(np.float32)/np.maximum(counts[:,None],1)
        else:
            positions=np.arange(length,dtype=np.float32)[None]
            weights=(1+positions/np.maximum(counts[:,None]-1,1))*valid
            weights/=np.maximum(weights.sum(-1,keepdims=True),1.)
        device=self.tokens.weight.device
        def tensor(a):
            return torch.as_tensor(a,device='cpu').to(device,non_blocking=True)
        values=torch.nn.functional.embedding_bag(tensor(flat[valid]),self.tokens.weight,tensor(offsets),
            mode='sum',include_last_offset=True,per_sample_weights=tensor(weights[valid].astype(np.float32)),padding_idx=0)
        values=values.reshape(n,width,self.dim)
        typed=self.kinds(tensor(kinds))+self.numeric(tensor(number))
        pairs=self.pair(torch.cat((self.tokens(tensor(keys)),values,typed),-1))
        pooled,_=self.attention(self.query.expand(n,-1,-1),pairs,pairs,
                                key_padding_mask=tensor(mask),need_weights=False)
        return self.output(pooled[:, 0])

    def forward(self, events):
        if not events:
            return self.tokens.weight.new_empty((0, self.dim))
        identities, unique, positions, index = [], [], {}, []
        for event in events:
            # Prepared attributes are immutable by contract, as in QueryCache.
            attributes = event.get('_canonical_attributes')
            if attributes is None:
                attributes = json.dumps(event.get('attributes'), sort_keys=True, separators=(',', ':'))
            identity = (event.get('type'), attributes)
            if identity not in positions:
                positions[identity] = len(unique)
                identities.append(identity)
                unique.append(event)
            index.append(positions[identity])
        device = self.tokens.weight.device
        gather = torch.tensor(index, device=device)
        if self.training or torch.is_grad_enabled() or self.cache_size == 0:
            return self.encode(unique)[gather]
        version = tuple(p._version for p in self.parameters())
        if version != self._cache_version:
            self.clear_cache()
            self._cache_version = version
        if self._cache_values is None:
            self._cache_values = self.tokens.weight.new_empty(self.cache_size, self.dim)
        # Gather hits BEFORE inserting misses: eviction may reuse their slots.
        hits, rows, missing = [], [], []
        for i, identity in enumerate(identities):
            row = self._cache.pop(identity, None)
            if row is None:
                missing.append(i)
            else:
                self._cache[identity] = row
                hits.append(i)
                rows.append(row)
        values = self.tokens.weight.new_empty(len(unique), self.dim)
        if hits:
            values[torch.tensor(hits, device=device)] = self._cache_values[torch.tensor(rows, device=device)]
        if missing:
            new = self.encode([unique[i] for i in missing])
            values[torch.tensor(missing, device=device)] = new
            # One bulk store even if a single call has more misses than capacity.
            destinations = {}
            for j, i in enumerate(missing):
                row = len(self._cache) if len(self._cache) < self.cache_size else self._cache.popitem(last=False)[1]
                self._cache[identities[i]] = row
                destinations[row] = j
            rows = list(destinations)
            self._cache_values[torch.tensor(rows, device=device)] = new[torch.tensor([destinations[r] for r in rows], device=device)]
        self.cache_hits += len(hits)
        self.cache_misses += len(missing)
        return values[gather]
