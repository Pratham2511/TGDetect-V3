"""GPU segmented affine scan with optional elapsed-time fast/slow decay."""
import math
import torch


def endpoint_scan(table, src, dst, weights, content, times=None, last_seen=None, half_lives=None):
    """Return strictly pre-event endpoint states and the final EMA table.

    weights/content are [2*B,D], ordered source events then destination events.
    Self-edges write once. Segmented affine composition is associative.
    """
    batch = len(src)
    if not batch:
        return table[src], table[dst], table
    ar = torch.arange(batch, device=src.device)
    nodes = torch.cat((src, dst))
    order = torch.argsort(nodes * (2 * batch + 1) + torch.cat((2 * ar, 2 * ar + 1)))
    ns = nodes[order]
    a = 1 - weights[order]
    b = weights[order] * content[order]
    first_node = torch.ones_like(ns, dtype=torch.bool)
    first_node[1:] = ns[1:] != ns[:-1]
    decay = None
    if half_lives is not None:
        clock = times.repeat(2)[order]
        previous = torch.cat((clock[:1],clock[:-1]))
        previous = torch.where(first_node,last_seen[ns],previous)
        elapsed = torch.where(previous.isnan(),torch.zeros_like(clock),clock-previous)
        decay = torch.exp(-math.log(2)*elapsed.to(table.dtype)[:,None]/half_lives)
        a = a*decay
    # Suppress the second write when both endpoints are the same entity.
    duplicate = torch.cat((torch.zeros_like(src, dtype=torch.bool), src == dst))[order]
    a = torch.where(duplicate[:, None], torch.ones_like(a), a)
    b = torch.where(duplicate[:, None], torch.zeros_like(b), b)
    first = torch.ones_like(ns, dtype=torch.bool)
    first[1:] = ns[1:] != ns[:-1]
    group = first.long().cumsum(0)
    # Fixed loop bound avoids a GPU -> CPU synchronization for maximum run size.
    step = 1
    while step < len(ns):
        same = group[step:] == group[:-step]
        tail_a = torch.where(same[:, None], a[step:] * a[:-step], a[step:])
        tail_b = torch.where(same[:, None], a[step:] * b[:-step] + b[step:], b[step:])
        a = torch.cat((a[:step], tail_a))
        b = torch.cat((b[:step], tail_b))
        step *= 2
    initial = table[ns]
    post = a * initial + b
    pre = torch.cat((initial[:1], post[:-1]))
    pre = torch.where(first[:, None], initial, pre)
    inverse = torch.argsort(order)
    if decay is not None:
        pre = pre*decay
    original = pre[inverse]
    ms, md = original[:batch], original[batch:]
    md = torch.where((src == dst)[:, None], ms, md)
    last = torch.ones_like(first)
    last[:-1] = first[1:]
    unique = ns[last]
    if torch.is_grad_enabled():
        updated = table.index_copy(0, unique, post[last])
    else:
        table.index_copy_(0, unique, post[last])
        updated = table
    return ms, md, updated
