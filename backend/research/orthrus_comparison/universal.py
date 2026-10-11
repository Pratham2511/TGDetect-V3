"""One shared, bounded-vocabulary encoder for unordered typed key/value sets.

No source-specific learned encoders. Numeric features supplement shared subtoken
embeddings; missing and nonfinite values get explicit types. No field positions.
"""
import hashlib
import json
import ipaddress
import math
import re
from functools import lru_cache
from collections.abc import Mapping

import torch
from torch import nn

KINDS = ('missing', 'bool', 'number', 'ip', 'path', 'text', 'list', 'object')


@lru_cache(maxsize=262144)
def bucket(text, size):
    return 1 + int.from_bytes(hashlib.blake2b(text.encode('utf8'), digest_size=8).digest(), 'big') % (size-1)


def value_parts(value):
    if value is None:
        return 'missing', ['<missing>'], 0.
    if isinstance(value, bool):
        return 'bool', [str(value).lower()], float(value)
    if isinstance(value, (int, float)):
        try:
            number = float(value)
        except OverflowError:
            number = math.inf
        if not math.isfinite(number):
            return 'missing', ['<nonfinite>'], 0.
        return 'number', ['<number>'], math.copysign(math.log1p(abs(number)), number)
    if isinstance(value, Mapping):
        # Leaf dictionaries are normally flattened before this point.
        return 'object', ['<object>'], 0.
    if isinstance(value, (list, tuple)):
        parts = ['<list>']
        for item in value[:32]:
            kind, words, number = value_parts(item)
            parts.extend([kind] + words)
            if kind=='number':parts.append(f'<signed_log:{number:.6g}>')
        return 'list', parts, math.log1p(len(value))
    text = str(value)
    try:
        addr = ipaddress.ip_address(text)
        bits = (8, 16, 24, 32) if addr.version == 4 else (32, 64, 96, 128)
        return 'ip', [str(ipaddress.ip_network(f'{addr}/{b}', strict=False)) for b in bits], 0.
    except ValueError:
        pass
    parts = re.findall(r'[\w]+|[^\w\s]', text[:4096], re.UNICODE) or ['<empty>']
    kind = 'path' if '/' in text or '\\' in text else 'text'
    return kind, parts, 0.


def flatten(attributes, prefix=(), depth=0):
    for key, value in sorted(attributes.items(), key=lambda p: str(p[0])):
        path = prefix + (str(key),)
        if isinstance(value, Mapping) and value and depth < 8:
            yield from flatten(value, path, depth+1)
        else:
            # Length-prefixed components prevent collision between nested and dotted keys.
            yield ''.join(f'{len(p)}:{p}' for p in path), value


class UniversalEventEncoder(nn.Module):
    def __init__(self, dim=64, buckets=16384, max_fields=64, max_subtokens=32):
        super().__init__()
        if dim<4 or dim%4 or buckets<2 or max_fields<0 or max_subtokens<1:
            raise ValueError('dim must be a positive multiple of 4; buckets>=2, fields>=0, subtokens>=1')
        self.dim, self.buckets = dim, buckets
        self.max_fields, self.max_subtokens = max_fields, max_subtokens
        self.tokens = nn.Embedding(buckets, dim, padding_idx=0)
        self.kinds = nn.Embedding(len(KINDS), dim)
        self.numeric = nn.Linear(1, dim, bias=False)
        self.pair = nn.Sequential(nn.Linear(3*dim, dim), nn.GELU(), nn.LayerNorm(dim))
        self.attention = nn.MultiheadAttention(dim, 4, dropout=0., batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, dim)*.02)
        self.output = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))

    def tokenize(self, events, device=None):
        target_device = device or self.tokens.weight.device
        device = torch.device('cpu')  # Build metadata on CPU; transfer once per batch.
        rows, truncations = [], []
        for event in events:
            attrs = event.get('attributes') or {}
            if not isinstance(attrs, Mapping):
                attrs = {'raw': attrs}
            pairs = list(flatten(attrs))
            truncated = max(0, len(pairs) - self.max_fields)
            # Always retain type token; an empty event cannot create all-masked attention.
            pairs = [('@event_type', event.get('type'))] + pairs[:self.max_fields]
            rows.append(pairs)
            truncations.append(truncated)
        n = max((len(row) for row in rows), default=1)
        keys,words,kinds,numeric,mask=[],[],[],[],[]
        for row in rows:
            rk,rw,rt,rn=[],[],[],[]
            for key, value in row:
                kind, parts, number = value_parts(value)
                ids = [bucket('value:'+p, self.buckets) for p in parts[:self.max_subtokens]]
                rk.append(bucket('key:'+key,self.buckets));rt.append(KINDS.index(kind))
                rw.append(ids+[0]*(self.max_subtokens-len(ids)));rn.append([number])
            padding=n-len(row)
            keys.append(rk+[0]*padding);kinds.append(rt+[0]*padding)
            words.append(rw+[[0]*self.max_subtokens]*padding)
            numeric.append(rn+[[0.]]*padding);mask.append([False]*len(row)+[True]*padding)
        arrays=(torch.tensor(keys,dtype=torch.long,device=device),
                torch.tensor(words,dtype=torch.long,device=device),
                torch.tensor(kinds,dtype=torch.long,device=device),
                torch.tensor(numeric,dtype=torch.float32,device=device),
                torch.tensor(mask,dtype=torch.bool,device=device))
        return *(t.to(target_device) for t in arrays), truncations

    def forward(self, events):
        if not events:
            return self.tokens.weight.new_empty((0,self.dim))
        # The same observed event occurs in several endpoint history windows.
        # Encode it once per batch, then gather; gradients from all uses sum.
        # No stochastic event-encoder layers are used, so this preserves output.
        unique=[];indices=[];seen={}
        for event in events:
            try:
                identity=json.dumps({'type':event.get('type'),'attributes':event.get('attributes')},
                                    sort_keys=True,separators=(',',':'))
            except (TypeError,ValueError):
                identity=None
            index=seen.get(identity) if identity is not None else None
            if index is None:
                index=len(unique);unique.append(event)
                if identity is not None:seen[identity]=index
            indices.append(index)
        key, word, kind, number, mask, _ = self.tokenize(unique)
        valid = word.ne(0).unsqueeze(-1)
        value = (self.tokens(word)*valid).sum(2)/valid.sum(2).clamp_min(1)
        typed = self.kinds(kind) + self.numeric(number)
        pairs = self.pair(torch.cat((self.tokens(key), value, typed), -1))
        pooled, _ = self.attention(self.query.expand(len(unique),-1,-1), pairs, pairs,
                                   key_padding_mask=mask, need_weights=False)
        out=self.output(pooled[:,0])
        return out[torch.tensor(indices,dtype=torch.long,device=out.device)]
