"""Vectorized chronological neighborhood planning for many overlapping windows."""
import numpy as np


def plan_windows(history, src, dst, first_id, window):
    size = len(src)
    if not size: return np.empty((0, window-1),dtype=np.int64)
    src, dst = np.asarray(src,dtype=np.int64), np.asarray(dst,dtype=np.int64)
    prior_nodes, prior_ids = [], []
    for node in np.unique(np.concatenate((src,dst))):
        ids = history.get(int(node), ())
        prior_nodes.extend([node]*len(ids)); prior_ids.extend(ids)
    prefix = len(prior_ids)
    current = first_id + np.arange(size)
    nodes = np.concatenate((np.asarray(prior_nodes,dtype=np.int64), src, dst))
    ids = np.concatenate((np.asarray(prior_ids,dtype=np.int64), current,current))
    order = np.lexsort((ids,nodes))
    sn, si = nodes[order], ids[order]
    keep = np.ones(len(order),dtype=bool)
    keep[1:] = (sn[1:] != sn[:-1]) | (si[1:] != si[:-1])
    ranks = np.empty(len(order),dtype=np.int64)
    ranks[order] = np.cumsum(keep)-1
    sn, si = sn[keep], si[keep]
    columns=[]
    for endpoints,positions in [(src,ranks[prefix:prefix+size]),(dst,ranks[prefix+size:])]:
        indices = positions[:,None] - np.arange(1,window)
        safe = np.maximum(indices,0)
        valid = (indices >= 0) & (sn[safe] == endpoints[:,None])
        columns.append(np.where(valid,si[safe],-1))
    candidates = np.sort(np.concatenate(columns,axis=1),axis=1)
    distinct = np.ones_like(candidates,dtype=bool)
    distinct[:,1:] = candidates[:,1:] != candidates[:,:-1]
    return np.sort(np.where(distinct,candidates,-1),axis=1)[:,-(window-1):]
