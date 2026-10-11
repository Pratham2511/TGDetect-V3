"""Compile official prepared graphs once into token-ID/event memmaps."""
import argparse
from collections import OrderedDict
import json
from pathlib import Path
import sqlite3
import numpy as np
import torch
from orthrus_comparison.protocol import require_coverage,partition,reconstruction_query
from orthrus_comparison.run import sha256
from orthrus_comparison.universal import value_parts
from sliding_node_mem_upgrade.pipeline import prepared_batches
from .tokens import TokenWriter

EVENT_DTYPE=np.dtype([(k,'<i8') for k in ('src','dst','relation','query','observed','signature',
                                        'src_identity','dst_identity','negative_identity')]+[('time','<f8'),('negative_valid','?')])


def read_batches(path,batch=1024,start=0):
    if Path(path).stat().st_size%EVENT_DTYPE.itemsize:raise ValueError('Truncated event memmap')
    count=Path(path).stat().st_size//EVENT_DTYPE.itemsize
    if not count:return
    rows=np.memmap(path,mode='r',dtype=EVENT_DTYPE)
    for lo in range(start,len(rows),batch):
        yield {k:torch.from_numpy(np.array(rows[k][lo:lo+batch])) for k in EVENT_DTYPE.names}


class Negatives:
    """Same-type destination corruption, excluding all known TRAIN pairs.

    Uses training identities only. Future validation/test co-occurrences are
    deliberately unavailable. Unknown true pairs can still be false negatives.
    """
    def __init__(self,path,attrs,in_memory_pairs=False):
        self.db=sqlite3.connect(path)
        self.db.execute('CREATE TABLE pairs (src INTEGER,dst INTEGER,PRIMARY KEY(src,dst)) WITHOUT ROWID')
        self.attrs=attrs;self.destinations=set();self.train_nodes=set();self.cache=OrderedDict()
        self.rng=np.random.default_rng(0)
        self.in_memory_pairs=in_memory_pairs;self.known_pairs=None

    def observe(self,src,dst):
        self.db.executemany('INSERT OR IGNORE INTO pairs VALUES (?,?)',zip(src,dst))
        self.destinations.update(dst);self.train_nodes.update(src);self.train_nodes.update(dst)

    def finish(self):
        self.db.commit();self.groups={}
        for node in sorted(self.destinations):
            kind=str(self.attrs.get(str(node),{}).get('type'))
            self.groups.setdefault(kind,[]).append(node)
        if self.in_memory_pairs:
            # Exact packed Python integers avoid an SQLite transaction/lock per
            # candidate. Corpus preparation only; no GPU/runtime state change.
            self.pair_stride=max(self.destinations,default=-1)+1
            self.known_pairs={u*self.pair_stride+v for u,v in self.db.execute('SELECT src,dst FROM pairs')}

    def sample(self,u,v):
        kind=str(self.attrs.get(str(v),{}).get('type'))
        candidates=self.groups.get(kind,[])
        for _ in range(min(32,len(candidates))):
            candidate=candidates[int(self.rng.integers(len(candidates)))]
            if candidate==v:continue
            key=(u,candidate)
            found=(u*self.pair_stride+candidate in self.known_pairs) if self.known_pairs is not None else self.cache.get(key)
            if found is None:
                found=self.db.execute('SELECT 1 FROM pairs WHERE src=? AND dst=?',key).fetchone() is not None
                self.cache[key]=found
                if len(self.cache)>65536:self.cache.popitem(last=False)
            if not found:return candidate
        return None


def prepare(manifest_path,output,compatible=False,buckets=5303):
    manifest_path=Path(manifest_path);manifest=json.loads(manifest_path.read_text())
    require_coverage(manifest['days'])
    if manifest.get('dataset')!='CADETS_E3' or not manifest.get('official_preprocessing'):
        raise ValueError('Official CADETS E3 inputs required')
    consumed=[p for paths in manifest['files'].values() for p in paths]+[manifest['node_attributes'],manifest['ground_truth_nodes']]
    for path in consumed:
        if manifest['file_sha256'].get(path)!=sha256(path):raise ValueError('Input checksum mismatch: '+path)
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    attrs=json.loads(Path(manifest['node_attributes']).read_text())
    negative=Negatives(out/'training_pairs.sqlite',attrs,in_memory_pairs=True)
    for path in manifest['files']['train']:
        graph=torch.load(path,map_location='cpu',weights_only=False)
        negative.observe(graph.src.tolist(),graph.dst.tolist())
    negative.finish()
    print('Exact training-pair index:',len(negative.known_pairs),'pairs',flush=True)
    # Corpus is built only from nodes actually observed in benign training.
    with (out/'benign_paths.jsonl').open('w') as corpus:
        for node in sorted(negative.train_nodes):
            value=attrs.get(str(node),{}).get('path')
            if value is not None:
                corpus.write(json.dumps(value_parts(value)[1])+'\n')
    writer=TokenWriter(out/'tokens',buckets=buckets,two_hash=not compatible,ordered=not compatible)
    signatures={};counts={};negative_stats={}
    for split,paths in manifest['files'].items():
        count=eligible=0;last=-float('inf')
        with (out/(split+'.bin')).open('wb',buffering=1024*1024) as stream:
            for _,_,_,src,dst,times,relations,queries in prepared_batches(paths,attrs,batch=4096):
                if any(t<last or partition(t)!=('validation' if split=='val' else split) for t in times):
                    raise ValueError('Invalid chronological split')
                last=times[-1];rows=np.zeros(len(src),dtype=EVENT_DTYPE)
                rows['src']=src;rows['dst']=dst;rows['time']=times;rows['relation']=relations.numpy()
                for i,(u,v,event,relation) in enumerate(zip(src,dst,queries,relations.tolist())):
                    source,destination=event['attributes']['src'],event['attributes']['dst']
                    sig=(relation,str(source.get('type')),str(destination.get('type')))
                    if sig not in signatures:signatures[sig]=len(signatures)
                    rows['signature'][i]=signatures[sig]
                    rows['query'][i]=writer.add(dict(event,type=None))
                    rows['observed'][i]=writer.add(dict(event,type=str(relation)))
                    rows['src_identity'][i]=writer.add({'attributes':source})
                    rows['dst_identity'][i]=writer.add({'attributes':destination})
                    candidate=negative.sample(u,v) if split=='train' else None
                    negattrs=destination if candidate is None else reconstruction_query({},attrs.get(str(candidate),{}))['attributes']['dst']
                    negid=writer.add({'attributes':negattrs})
                    rows['negative_identity'][i]=negid
                    rows['negative_valid'][i]=candidate is not None and negid!=rows['dst_identity'][i]
                stream.write(rows.tobytes());previous=count;count+=len(rows);eligible+=int(rows['negative_valid'].sum())
                if count//100000!=previous//100000:print('Prepared',split,count,'events',flush=True)
        counts[split]=count;negative_stats[split]=eligible
    writer.close();negative.db.close()
    truth=json.loads(Path(manifest['ground_truth_nodes']).read_text())
    (out/'ground_truth_nodes.json').write_text(json.dumps(truth))
    files=[out/(s+'.bin') for s in counts]+[out/'benign_paths.jsonl',out/'ground_truth_nodes.json',
           out/'tokens/fields.bin',out/'tokens/offsets.npy',out/'tokens/tokens.json']
    result=dict(format=1,n_nodes=manifest['n_nodes'],relations=10,signatures=len(signatures),events=counts,
                compatible=compatible,negative_eligible=negative_stats,source_manifest_sha256=sha256(manifest_path),
                checksums={str(p.relative_to(out)):sha256(p) for p in files},
                corpus_scope='Unique node paths observed in benign training only',
                negative_scope='Same-type training destinations; reject all observed training pairs and identical endpoint attributes')
    (out/'manifest.json').write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',dest='manifest_path',required=True);p.add_argument('--output',required=True)
    p.add_argument('--compatible',action='store_true');p.add_argument('--buckets',type=int,default=5303)
    print(json.dumps(prepare(**vars(p.parse_args())),indent=2))
