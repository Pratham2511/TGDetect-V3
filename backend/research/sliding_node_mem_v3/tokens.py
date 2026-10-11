"""Offline typed token rows and a vectorized memmap/resident token bank."""
from array import array
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import sqlite3
import numpy as np
import torch
from torch.nn import functional as F
from orthrus_comparison.universal import flatten,value_parts,bucket,KINDS
from sliding_node_mem_v2.encoding import EventEncoder


def hashes(text,buckets,two_hash):
    first=bucket(text,buckets)
    second=1+int.from_bytes(hashlib.blake2b(text.encode(),digest_size=8,person=b'tgnn-hash-two').digest(),'big')%(buckets-1)
    return [first,second] if two_hash else [first]


def field_dtype(length,hash_count):
    return np.dtype([('key','<i4',(hash_count,)),('word','<i4',(length,hash_count)),
                     ('kind','<i8'),('number','<f4')])


class TokenWriter:
    def __init__(self,path,buckets=5303,max_fields=64,max_subtokens=32,
                 two_hash=True,ordered=True):
        self.path=Path(path);self.path.mkdir(parents=True,exist_ok=False)
        self.config=dict(buckets=buckets,max_fields=max_fields,max_subtokens=max_subtokens,
                         two_hash=two_hash,ordered=ordered)
        self.dtype=field_dtype(max_subtokens*(2 if ordered else 1),2 if two_hash else 1)
        self.file=(self.path/'fields.bin').open('wb',buffering=1024*1024)
        self.db=sqlite3.connect(self.path/'identities.sqlite')
        self.db.execute('CREATE TABLE identities (identity TEXT PRIMARY KEY, idx INTEGER UNIQUE)')
        self.offsets=array('q',[0]);self.cache=OrderedDict();self.max_width=1

    def add(self,event):
        identity=json.dumps({'type':event.get('type'),'attributes':event.get('attributes')},sort_keys=True,separators=(',',':'))
        existing=self.cache.get(identity)
        if existing is not None:
            self.cache.move_to_end(identity);return existing
        row=self.db.execute('SELECT idx FROM identities WHERE identity=?',(identity,)).fetchone()
        if row is not None:
            index=row[0]
        else:
            index=len(self.offsets)-1
            self.db.execute('INSERT INTO identities VALUES (?,?)',(identity,index))
            attrs=event.get('attributes') or {}
            if not isinstance(attrs,dict):attrs={'raw':attrs}
            pairs=[('@event_type',event.get('type'))]+list(flatten(attrs))[:self.config['max_fields']]
            data=np.zeros(len(pairs),dtype=self.dtype)
            length=self.config['max_subtokens'];buckets=self.config['buckets'];two=self.config['two_hash']
            for j,(key,value) in enumerate(pairs):
                kind,parts,number=value_parts(value)
                selected=parts if len(parts)<=length else parts[:(length+1)//2]+parts[-(length//2):] if length>1 else parts[:1]
                tokens=['value:'+p for p in selected]
                if self.config['ordered']:
                    tokens += [f'pair:{i}:{json.dumps([selected[i-1],selected[i]],ensure_ascii=False)}'
                               for i in range(1,len(selected)) if not (len(parts)>length and i==(length+1)//2)]
                data['key'][j]=hashes('key:'+key,buckets,two)
                data['word'][j,:len(tokens)]=[hashes(p,buckets,two) for p in tokens]
                data['kind'][j]=KINDS.index(kind);data['number'][j]=number
            self.file.write(data.tobytes());self.offsets.append(self.offsets[-1]+len(data))
            self.max_width=max(self.max_width,len(data))
        self.cache[identity]=index
        if len(self.cache)>65536:self.cache.popitem(last=False)
        return index

    def close(self):
        self.file.close();self.db.commit();self.db.close()
        np.save(self.path/'offsets.npy',np.asarray(self.offsets,dtype=np.int64))
        config=dict(self.config,max_width=self.max_width,identities=len(self.offsets)-1,fields=self.offsets[-1],format=1)
        (self.path/'tokens.json').write_text(json.dumps(config,indent=2))


class TokenBank:
    def __init__(self,path):
        self.path=Path(path);self.config=json.loads((self.path/'tokens.json').read_text())
        c=self.config
        self.fields=np.memmap(self.path/'fields.bin',mode='r',dtype=field_dtype(c['max_subtokens']*(2 if c['ordered'] else 1),2 if c['two_hash'] else 1))
        self.offsets=np.load(self.path/'offsets.npy',mmap_mode='r')
        self.resident=None

    def to_device(self,device):
        # Optional: fitting the bank on GPU eliminates the remaining identity
        # transfer/memmap staging. Runtime JSON/tokenization is absent either way.
        self.resident={k:torch.from_numpy(np.array(self.fields[k])).to(device) for k in self.fields.dtype.names}
        self.resident['offsets']=torch.from_numpy(np.array(self.offsets)).to(device)
        return self

    def fetch(self,identities,device):
        width=self.config['max_width']
        if self.resident is not None:
            offsets=self.resident['offsets'];identities=identities.to(offsets.device)
            start=offsets[identities];sizes=offsets[identities+1]-start
            index=start[:,None]+torch.arange(width,device=offsets.device)
            mask=torch.arange(width,device=offsets.device)[None]>=sizes[:,None]
            index=index.clamp_max(len(self.fields)-1)
            data={k:self.resident[k][index].to(device) for k in self.fields.dtype.names}
            mask=mask.to(device)
        else:
            ids=identities.detach().cpu().numpy()
            start=self.offsets[ids];sizes=self.offsets[ids+1]-start
            index=np.minimum(start[:,None]+np.arange(width),len(self.fields)-1)
            mask=torch.from_numpy(np.arange(width)[None]>=sizes[:,None]).to(device)
            rows=self.fields[index]
            data={k:torch.from_numpy(np.array(rows[k])).to(device) for k in self.fields.dtype.names}
        for k in ('key','word'):
            data[k]=data[k].long().masked_fill(mask.reshape(mask.shape+(1,)*(data[k].ndim-2)),0)
        data['kind']=data['kind'].masked_fill(mask,0)
        data['number']=data['number'].masked_fill(mask,0)
        return data,mask


class PackedEncoder(EventEncoder):
    """Same learned typed-KV modules, with already-tokenized unique identities."""
    def __init__(self,bank,dim=64):
        super().__init__(dim=dim,buckets=bank.config['buckets'],cache_size=0,
                         max_fields=bank.config['max_fields'],max_subtokens=bank.config['max_subtokens'])
        self.bank=bank
        self._frozen_table=None
        self._frozen_signature=None

    def _weight_signature(self):
        return tuple((id(p),p._version,p.device,p.dtype) for p in self.parameters())

    def clear_frozen_table(self):
        self._frozen_table=None
        self._frozen_signature=None

    def _apply(self,fn,recurse=True):
        self.clear_frozen_table()
        return super()._apply(fn,recurse=recurse)

    def _load_from_state_dict(self,*args,**kwargs):
        self.clear_frozen_table()
        return super()._load_from_state_dict(*args,**kwargs)

    def train(self,mode=True):
        if mode:self.clear_frozen_table()
        return super().train(mode)

    @torch.no_grad()
    def precompute(self,batch_size=1024,dtype=None):
        """Explicit bounded-chunk eval preparation; table is not checkpointed.

        Default dtype preserves precision. Half precision is an opt-in ablation.
        Normal optimizer/load_state_dict updates invalidate the lookup.
        """
        if self.training:raise ValueError('Call eval() before precomputing identities')
        if batch_size<1:raise ValueError('Positive precompute batch size required')
        self.clear_frozen_table()
        device=self.tokens.weight.device
        table=torch.empty(self.bank.config['identities'],self.dim,device=device,
                          dtype=dtype or self.tokens.weight.dtype)
        for start in range(0,len(table),batch_size):
            ids=torch.arange(start,min(start+batch_size,len(table)),device=device)
            table[start:start+len(ids)]=self._encode(ids).to(table.dtype)
        self._frozen_table=table
        self._frozen_signature=self._weight_signature()
        return table.numel()*table.element_size()

    def forward(self,identities):
        if self._frozen_table is not None:
            if self._frozen_signature!=self._weight_signature():self.clear_frozen_table()
            elif not self.training and not torch.is_grad_enabled():
                return self._frozen_table[identities].to(self.tokens.weight.dtype)
        unique,inverse=torch.unique(identities,sorted=True,return_inverse=True)
        return self._encode(unique)[inverse]

    def _encode(self,unique):
        with torch.profiler.record_function('token_bank_fetch'):
            data,mask=self.bank.fetch(unique,self.tokens.weight.device)
        words=data['word'];valid=words[...,0]!=0
        counts=valid.sum(-1,keepdim=True)
        weights=valid.to(self.tokens.weight.dtype)
        if not self.bank.config['ordered']:
            positions=torch.arange(words.shape[-2],device=words.device)
            weights=weights*(1+positions/(counts-1).clamp_min(1))
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(1)
        # Summing two embeddings uses the SAME table; no extra token parameters.
        # Embedding-bag avoids materializing [U,F,L,H,64].
        flat=words.reshape(-1,words.shape[-2]*words.shape[-1])
        weight=weights[...,None].expand_as(words).reshape_as(flat)
        offsets=torch.arange(0,flat.numel()+1,flat.shape[1],device=words.device)
        value=F.embedding_bag(flat.flatten(),self.tokens.weight,offsets,
                              mode='sum',per_sample_weights=weight.flatten(),include_last_offset=True,padding_idx=0)
        value=value.reshape(len(unique),words.shape[1],self.dim)
        key=self.tokens(data['key']).sum(-2)
        typed=self.kinds(data['kind'])+self.numeric(data['number'][...,None])
        pairs=self.pair(torch.cat((key,value,typed),-1))
        pooled,_=self.attention(self.query.expand(len(unique),-1,-1),pairs,pairs,key_padding_mask=mask,need_weights=False)
        return self.output(pooled[:,0])
