"""Bounded chronological unrolls retain cross-batch memory-write gradients."""
import torch
from torch.nn import functional as F


def train_unroll(model, optimizer, batches, compression_weight=.01, clip_norm=1.):
    """One optimizer update AFTER all forward passes and a single backward.

    Batch tuples are (events, src, dst, times, relations). No intermediate
    detach, parameter update, or checkpoint may occur inside this unroll.
    """
    if not batches:
        raise ValueError('Empty training unroll')
    optimizer.zero_grad(set_to_none=True)
    totals, count = [], 0
    for events, src, dst, times, relations in batches:
        logits, aux = model.forward_events(events, src, dst, times, relations)
        targets = torch.as_tensor(relations, device=logits.device, dtype=torch.long)
        count += len(targets)
        totals.append(F.cross_entropy(logits, targets, reduction='sum')
                      + compression_weight*aux['compression_loss']*len(targets))
    loss = torch.stack(totals).sum()/count
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite unroll loss')
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm, error_if_nonfinite=True)
    optimizer.step()
    model.detach_state()
    return loss.detach(), count
