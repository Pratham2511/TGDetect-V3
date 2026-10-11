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
        ids=[bucket('value:'+p,buckets) for p in parts[:max_subtokens]]
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
