"""Bulk CPU metadata conversion and shared-block compressed-token gathers."""
import numpy as np
import torch
from sliding_node_mem_upgrade.model import gather_tokens as reference_gather


def tensor(values, device, dtype=None):
    mapping = {torch.float32:np.float32,torch.float64:np.float64,
               torch.int64:np.int64,torch.bool:np.bool_}
    array = np.asarray(values,dtype=mapping.get(dtype))
    return torch.from_numpy(array).to(device,non_blocking=True)


def gather_tokens(refs, zero):
    if not refs:
        return zero.new_empty((0,64))
    blocks = [zero.reshape(1,64)]; offsets = {}; next_offset = 1
    indices = np.empty(len(refs),dtype=np.int64)
    for i,ref in enumerate(refs):
        if ref is None:
            indices[i] = 0
            continue
        identity = id(ref.tensor)
        offset = offsets.get(identity)
        if offset is None:
            offset = next_offset; offsets[identity] = offset
            next_offset += ref.tensor.shape[0]; blocks.append(ref.tensor)
        indices[i] = offset+ref.row
    # Sparse old backing blocks must not turn this into a large bulk copy.
    if next_offset > 2*len(refs):
        return reference_gather(refs,zero)
    values = torch.cat(blocks)
    return values[torch.from_numpy(indices).to(zero.device,non_blocking=True)]
