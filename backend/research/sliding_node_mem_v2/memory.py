"""Exact affine memory scan with CPU-planned indices and bounded scan depth.

Endpoint identities already live on the CPU in the streaming planner. Sorting
them there avoids CUDA nonzero/unique output-size synchronizations. Only the
largest per-entity run determines prefix depth, not the entire batch size.
"""
import numpy as np
import torch
import math


def endpoint_scan(table, src, dst, weights, content, times=None, last_seen=None, half_lives=None):
    src,dst = np.asarray(src,dtype=np.int64),np.asarray(dst,dtype=np.int64)
    batch = len(src)
    if not batch:
        return table[:0],table[:0],table
    nodes = np.concatenate((src,dst))
    chronological = np.concatenate((2*np.arange(batch),2*np.arange(batch)+1))
    order = np.lexsort((chronological,nodes)); ns = nodes[order]
    first = np.ones(len(ns),dtype=np.int64);first[1:] = ns[1:] != ns[:-1]
    group = np.cumsum(first)
    last_positions = np.r_[np.flatnonzero(first)[1:]-1,len(ns)-1]
    counts = np.diff(np.r_[-1,last_positions]); depth = int(counts.max())
    duplicates = np.r_[np.zeros(batch,dtype=np.int64),(src==dst).astype(np.int64)][order]
    arrays = [order,ns,first,group,duplicates,np.argsort(order),last_positions,(src==dst).astype(np.int64)]
    packed = torch.from_numpy(np.concatenate(arrays)).to(table.device,non_blocking=True)
    slices = list(packed.split([len(a) for a in arrays]))
    order,ns,first,group,duplicate,inverse,last,self_edge = slices
    a,b = 1-weights[order],weights[order]*content[order]
    decay = None
    if half_lives is not None:
        times = np.asarray(times,dtype=np.float64)
        sorted_times = np.concatenate((times,times))[arrays[0]]
        previous = np.r_[sorted_times[:1],sorted_times[:-1]]
        first_cpu = arrays[2].astype(bool)
        previous[first_cpu] = np.asarray(last_seen)[arrays[1][first_cpu]]
        elapsed = np.where(np.isnan(previous),0.,sorted_times-previous)
        if not np.isfinite(elapsed).all() or (elapsed < 0).any():
            raise ValueError('Finite chronological node-memory times required')
        elapsed = torch.from_numpy(elapsed.astype(np.float32)).to(table.device,non_blocking=True)
        decay = torch.exp(-math.log(2)*elapsed[:,None]/half_lives)
        a = a*decay
    a = torch.where(duplicate.bool()[:,None],torch.ones_like(a),a)
    b = torch.where(duplicate.bool()[:,None],torch.zeros_like(b),b)
    step = 1
    while step < depth:
        same = (group[step:] == group[:-step])[:,None]
        aa = torch.where(same,a[step:]*a[:-step],a[step:])
        bb = torch.where(same,a[step:]*b[:-step]+b[step:],b[step:])
        a,b = torch.cat((a[:step],aa)),torch.cat((b[:step],bb))
        step *= 2
    initial = table[ns]
    post = a*initial+b
    before = torch.where(first.bool()[:,None],initial,torch.cat((initial[:1],post[:-1])))
    if decay is not None:
        before = before*decay
    original = before[inverse]
    ms,md = original[:batch],original[batch:]
    md = torch.where(self_edge.bool()[:,None],ms,md)
    if torch.is_grad_enabled():
        updated = table.index_copy(0,ns[last],post[last])
    else:
        table.index_copy_(0,ns[last],post[last]);updated=table
    return ms,md,updated
