"""Train Word2Vec's skip-gram/negative-sampling objective on benign path text.

    Fit the resulting word vectors into the model's shared hash table. This is
    corpus pretraining, not a claim of using ORTHRUS's original Word2Vec weights.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from orthrus_comparison.run import sha256
from .tokens import hashes


def pretrain(prepared,output,epochs=5,dim=64,window=3,negatives=5,max_vocab=100000,
             batch=4096,seed=0,device='cpu',projection_steps=200):
    if min(epochs,dim,window,negatives,max_vocab,batch,projection_steps)<1:raise ValueError('Positive pretraining settings required')
    root=Path(prepared);manifest=json.loads((root/'manifest.json').read_text())
    corpus=root/'benign_paths.jsonl'
    if sha256(corpus)!=manifest['checksums']['benign_paths.jsonl']:raise ValueError('Training corpus checksum mismatch')
    config=json.loads((root/'tokens/tokens.json').read_text())
    if sha256(root/'tokens/tokens.json')!=manifest['checksums']['tokens/tokens.json']:raise ValueError('Token configuration checksum mismatch')
    counts=Counter()
    with corpus.open() as stream:
        for line in stream:counts.update(json.loads(line))
    vocabulary=counts.most_common(max_vocab)
    if len(vocabulary)<2:raise ValueError('Need at least two training subtokens')
    lookup={word:i for i,(word,_) in enumerate(vocabulary)}
    rng=np.random.default_rng(seed);torch.manual_seed(seed)
    inputs=nn.Embedding(len(vocabulary),dim,device=device)
    outputs=nn.Embedding(len(vocabulary),dim,device=device)
    nn.init.uniform_(inputs.weight,-.5/dim,.5/dim);nn.init.zeros_(outputs.weight)
    distribution=torch.tensor([count**.75 for _,count in vocabulary],device=device)
    distribution/=distribution.sum()
    optimizer=torch.optim.Adam([inputs.weight,outputs.weight],lr=.01)
    history=[]
    for epoch in range(epochs):
        pending=[];total=0.;seen=0
        def update(pairs):
            ids=torch.tensor(pairs,dtype=torch.long,device=device)
            left=inputs(ids[:,0]);right=outputs(ids[:,1])
            sampled=torch.multinomial(distribution,len(ids)*negatives,replacement=True).reshape(len(ids),negatives)
            loss=(F.softplus(-(left*right).sum(-1))+F.softplus((left[:,None]*outputs(sampled)).sum(-1)).sum(-1)).mean()
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            return float(loss.detach())*len(ids)
        with corpus.open() as stream:
            for line in stream:
                words=[lookup[w] for w in json.loads(line) if w in lookup]
                for i,word in enumerate(words):
                    radius=int(rng.integers(1,window+1))
                    pending.extend((word,words[j]) for j in range(max(0,i-radius),min(len(words),i+radius+1)) if j!=i)
                    if len(pending)>=batch:
                        total+=update(pending);seen+=len(pending);pending=[]
        if pending:total+=update(pending);seen+=len(pending)
        if not seen:raise ValueError('Corpus contains no context pairs')
        history.append(total/seen)
    # Summation hashes share weights, so directly copying vectors into both
    # buckets would double them and mishandle collisions. Fit their sum instead.
    table=nn.Embedding(config['buckets'],dim,padding_idx=0,device=device)
    ids=torch.tensor([hashes('value:'+word,config['buckets'],config['two_hash']) for word,_ in vocabulary],device=device)
    target=inputs.weight.detach()
    touched=torch.unique(ids)
    with torch.no_grad():table.weight[touched]=0
    fitting=torch.optim.Adam([table.weight],lr=.02)
    for _ in range(projection_steps):
        chosen=torch.randint(len(vocabulary),(min(batch,len(vocabulary)),),device=device)
        loss=F.mse_loss(table(ids[chosen]).sum(1),target[chosen])
        fitting.zero_grad(set_to_none=True);loss.backward();fitting.step()
    with torch.no_grad():table.weight[0].zero_()
    result=dict(tokens=table.weight.detach().cpu(),token_config=config,source_manifest_sha256=sha256(root/'manifest.json'),
                corpus_sha256=sha256(corpus),objective='Word2Vec SGNS, followed by shared-hash sum fitting',
                epochs=epochs,seed=seed,vocabulary=len(vocabulary),history=history,projection_mse=float(loss.detach()))
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True);torch.save(result,path)
    return {k:v for k,v in result.items() if k!='tokens'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',required=True);p.add_argument('--output',required=True)
    p.add_argument('--epochs',type=int,default=5);p.add_argument('--device',default='cpu')
    print(json.dumps(pretrain(**vars(p.parse_args())),indent=2))
