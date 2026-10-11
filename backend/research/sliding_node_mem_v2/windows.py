"""Balanced endpoint history with chronological, batch-independent sampling."""
import numpy as np


def choose_history(source, destination, budget):
    """Reserve half per endpoint; deduplicate, then fill with newest leftovers."""
    quota = budget // 2
    selected = set(list(source)[-quota:]) | set(list(destination)[-quota:])
    union = set(source) | set(destination)
    for index in sorted(union-selected, reverse=True):
        if len(selected) >= budget:
            break
        selected.add(index)
    return sorted(selected)


def plan_windows(history, src, dst, first_id, window):
    size=len(src);budget=window-1
    if not size:
        return np.empty((0,budget),dtype=np.int64)
    src,dst=np.asarray(src,dtype=np.int64),np.asarray(dst,dtype=np.int64)
    prior_nodes,prior_ids=[],[]
    for node in np.unique(np.r_[src,dst]):
        ids=history.get(int(node),())
        prior_nodes.extend([node]*len(ids));prior_ids.extend(ids)
    prefix=len(prior_ids);current=first_id+np.arange(size)
    nodes=np.r_[np.asarray(prior_nodes,dtype=np.int64),src,dst]
    ids=np.r_[np.asarray(prior_ids,dtype=np.int64),current,current]
    order=np.lexsort((ids,nodes));sn,si=nodes[order],ids[order]
    keep=np.ones(len(order),dtype=bool)
    keep[1:]=(sn[1:]!=sn[:-1]) | (si[1:]!=si[:-1])
    ranks=np.empty(len(order),dtype=np.int64);ranks[order]=np.cumsum(keep)-1
    sn,si=sn[keep],si[keep];columns=[]
    for endpoints,positions in ((src,ranks[prefix:prefix+size]),(dst,ranks[prefix+size:])):
        offsets=positions[:,None]-np.arange(1,window)
        safe=np.maximum(offsets,0)
        valid=(offsets>=0)&(sn[safe]==endpoints[:,None])
        columns.append(np.where(valid,si[safe],-1))
    candidates=np.sort(np.concatenate(columns,axis=1),axis=1)
    unique=np.ones_like(candidates,dtype=bool);unique[:,1:]=candidates[:,1:]!=candidates[:,:-1]
    candidates=np.where(unique,candidates,-1)
    reserved=np.concatenate([col[:,:budget//2] for col in columns],axis=1)
    mandatory=(candidates[:,:,None]==reserved[:,None,:]).any(-1)&(candidates>=0)
    priority=candidates+mandatory*(first_id+size+1)
    chosen=np.argsort(priority,axis=1,kind='stable')[:,-budget:]
    return np.sort(np.take_along_axis(candidates,chosen,axis=1),axis=1)
