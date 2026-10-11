"""Cache only deterministic token IDs, never learned event embeddings."""
from functools import lru_cache
import json
import numpy as np
import torch
from orthrus_comparison.universal import UniversalEventEncoder, flatten, value_parts, bucket, KINDS


@lru_cache(maxsize=32768)
def token_row(identity, buckets, max_fields, max_subtokens):
    event=json.loads(identity); attrs=event.get('attributes') or {}
    if not isinstance(attrs,dict): attrs={'raw':attrs}
    pairs=list(flatten(attrs)); truncated=max(0,len(pairs)-max_fields)
    pairs=[('@event_type',event.get('type'))]+pairs[:max_fields]
    keys=np.zeros(len(pairs),dtype=np.int64)
    words=np.zeros((len(pairs),max_subtokens),dtype=np.int64)
    kinds=np.zeros(len(pairs),dtype=np.int64); numeric=np.zeros((len(pairs),1),dtype=np.float32)
    for j,(key,value) in enumerate(pairs):
        kind,parts,number=value_parts(value)
        keys[j]=bucket('key:'+key,buckets); kinds[j]=KINDS.index(kind); numeric[j,0]=number
        selected = parts if len(parts) <= max_subtokens else parts[:(max_subtokens+1)//2] + parts[-(max_subtokens//2):] if max_subtokens > 1 else parts[:1]
        ids=[bucket('value:'+p,buckets) for p in selected]
        words[j,:len(ids)]=ids
    return keys,words,kinds,numeric,truncated


class CachedUniversalEventEncoder(UniversalEventEncoder):
    def tokenize(self, events, device=None):
        try:
            rows=[token_row(json.dumps({'type':e.get('type'),'attributes':e.get('attributes')},
                                      sort_keys=True,separators=(',',':')),
                            self.buckets,self.max_fields,self.max_subtokens) for e in events]
        except (TypeError,ValueError):
            return super().tokenize(events,device)
        width=max((len(r[0]) for r in rows),default=1); count=len(rows)
        keys=np.zeros((count,width),dtype=np.int64)
        words=np.zeros((count,width,self.max_subtokens),dtype=np.int64)
        kinds=np.zeros((count,width),dtype=np.int64)
        numeric=np.zeros((count,width,1),dtype=np.float32)
        mask=np.ones((count,width),dtype=bool)
        for j,(k,w,t,n,_) in enumerate(rows):
            size=len(k); keys[j,:size]=k; words[j,:size]=w; kinds[j,:size]=t; numeric[j,:size]=n; mask[j,:size]=False
        target=device or self.tokens.weight.device
        return *(torch.from_numpy(a).to(target) for a in (keys,words,kinds,numeric,mask)), [r[-1] for r in rows]


@lru_cache(maxsize=32768)
def canonical_identity(event_type, attributes):
    return json.dumps({'type': event_type, 'attributes': json.loads(attributes)},
                      sort_keys=True, separators=(',', ':'))


class UpgradedUniversalEventEncoder(CachedUniversalEventEncoder):
    """Pool only real subtokens, preserving their order with bounded weights.

    Prepared queries carry canonical attributes, so observed relation variants
    reuse the same serialization. No learned embedding is cached across steps.
    """
    def forward(self, events):
        if not events:
            return self.tokens.weight.new_empty((0, self.dim))
        unique, indices, seen = [], [], {}
        for event in events:
            attributes = event.get('_canonical_attributes')
            if attributes is None:
                try:
                    attributes = json.dumps(event.get('attributes'), sort_keys=True, separators=(',', ':'))
                except (TypeError, ValueError):
                    return super().forward(events)
            identity = (event.get('type'), attributes)
            if identity not in seen:
                seen[identity] = len(unique)
                unique.append(canonical_identity(identity[0], attributes))
            indices.append(seen[identity])
        rows = [token_row(i, self.buckets, self.max_fields, self.max_subtokens) for i in unique]
        width = max(len(r[0]) for r in rows)
        keys = np.zeros((len(rows), width), dtype=np.int64)
        kinds = np.zeros_like(keys)
        numeric = np.zeros((len(rows), width, 1), dtype=np.float32)
        mask = np.ones((len(rows), width), dtype=bool)
        words = np.zeros((len(rows), width, self.max_subtokens), dtype=np.int64)
        for j, (key, word, kind, number, _) in enumerate(rows):
            size = len(key)
            keys[j, :size] = key; kinds[j, :size] = kind
            numeric[j, :size] = number; mask[j, :size] = False
            words[j, :size] = word
        flat_words = words.reshape(-1, self.max_subtokens)
        valid = flat_words != 0
        counts = valid.sum(-1)
        offsets = np.concatenate(([0], counts.cumsum())).astype(np.int64)
        positions = np.arange(self.max_subtokens, dtype=np.float32)[None]
        position_weights = (1. + positions / np.maximum(counts[:, None] - 1, 1)) * valid
        position_weights /= np.maximum(position_weights.sum(-1, keepdims=True), 1.)
        bags = flat_words[valid]
        weights = position_weights[valid].astype(np.float32)
        device = self.tokens.weight.device
        def tensor(a):
            return torch.as_tensor(a, device='cpu').to(device, non_blocking=True)
        value = torch.nn.functional.embedding_bag(
            tensor(np.asarray(bags, dtype=np.int64)), self.tokens.weight,
            tensor(np.asarray(offsets, dtype=np.int64)), mode='sum', include_last_offset=True,
            per_sample_weights=tensor(np.asarray(weights, dtype=np.float32)), padding_idx=0)
        value = value.reshape(len(rows), width, self.dim)
        typed = self.kinds(tensor(kinds)) + self.numeric(tensor(numeric))
        pairs = self.pair(torch.cat((self.tokens(tensor(keys)), value, typed), -1))
        pooled, _ = self.attention(self.query.expand(len(rows), -1, -1), pairs, pairs,
                                  key_padding_mask=tensor(mask), need_weights=False)
        return self.output(pooled[:, 0])[tensor(np.asarray(indices, dtype=np.int64))]
