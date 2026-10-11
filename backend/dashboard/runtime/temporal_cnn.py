"""Causal convolutions along event order, never along feature or batch axes."""
import torch
from torch import nn
from torch.nn import functional as F

from .tgnn_components import ComponentTGNN

CNN_ARMS = {
    "GNN_H": ("H",), "TGNN": ("T", "H"),
    "TGNN_Q": ("T", "H", "Q"), "TGNN_Q_A": ("T", "H", "Q", "A"),
    "CNN_H": ("C", "H"), "TGNN_C": ("T", "H", "C"),
    "TGNN_Q_C": ("T", "H", "Q", "C"),
    "TGNN_Q_A_C": ("T", "H", "Q", "A", "C"),
}


class CausalTemporalCNN(nn.Module):
    def __init__(self, channels=128, dropout=.2):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(channels, channels, 3, dilation=d) for d in (1,2)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence, valid):
        mask = valid[:, None, :]
        x = sequence.transpose(1,2) * mask
        for conv in self.convs:
            # For five-position sequences use the exact Conv1d contraction as GEMM.
            # Tap-major columns match the permuted [out, tap, in] kernel weights.
            dilation=conv.dilation[0]
            length=x.shape[-1]
            padded=F.pad(x,(2*dilation,0))
            windows=torch.cat([padded[:,:,k*dilation:k*dilation+length].transpose(1,2)
                               for k in range(3)],dim=-1)
            weights=conv.weight.permute(0,2,1).reshape(conv.out_channels,-1)
            convolved=F.linear(windows,weights,conv.bias).transpose(1,2)
            x = self.dropout(F.relu(convolved)) * mask
        return x.transpose(1,2)


class CNNOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.factors = frozenset(("C", "H"))
        self.projection = nn.Linear(10,128)
        self.cnn = CausalTemporalCNN()
        self.head = nn.Linear(128,2)

    def forward_events(self, x, valid, ages, src=None, dst=None, times=None):
        # Existing storage is [current, newest, ..., oldest]; reverse the time axis.
        sequence = F.relu(self.projection(x)).flip(1)
        h = self.cnn(sequence, valid.flip(1))[:, -1]
        return self.head(h), torch.zeros_like(h)


class CNNHybrid(ComponentTGNN):
    def __init__(self, factors):
        super().__init__(10, set(factors)-{"C"})
        self.cnn = CausalTemporalCNN()
        self.factors = frozenset(factors)

    def encode(self, x, valid, extra=None):
        batch, nodes, _ = x.shape
        adjacency = torch.eye(nodes, dtype=torch.bool, device=x.device).expand(batch,-1,-1).clone()
        adjacency[:,0,:] = valid
        for i, layer in enumerate(self.layers):
            x = layer(x, adjacency, extra if i == 0 else None)
        h = x[:,0]
        if self.history_gru is not None:
            state = torch.zeros_like(h)
            for j in range(nodes-1,0,-1):
                state = torch.where(valid[:,j,None], self.history_gru(x[:,j], state), state)
            h = h + state
        if self.query is not None:
            mask = valid[:,1:]
            scores = (self.query(h)[:,None] * self.key(x[:,1:])).sum(-1) / 128**.5
            weights = scores.masked_fill(~mask,-1e9).softmax(-1) * mask
            h = h + (weights[:,:,None]*self.value(x[:,1:])).sum(1)
        return h + self.cnn(x.flip(1), valid.flip(1))[:,-1]


def make_model(arm):
    factors = CNN_ARMS[arm]
    if arm == "CNN_H":
        return CNNOnly()
    if "C" in factors:
        return CNNHybrid(factors)
    return ComponentTGNN(10, factors)


def rehearsal_schedule(train_end=9830, epochs=200, batch_size=256, seed=0):
    """Exact original Algorithm R indices; all arms here have zero memory context."""
    import random
    import numpy as np
    rng = random.Random(seed+937)
    items, seen, schedule = [], 0, []
    for _ in range(epochs):
        for lo in range(0,train_end,batch_size):
            schedule.append(rng.sample(items, min(26,len(items))))
            for event in range(lo,min(lo+batch_size,train_end)):
                seen += 1
                if len(items) < train_end//10:
                    items.append(event)
                elif train_end//10:
                    j = rng.randrange(seen)
                    if j < train_end//10:
                        items[j] = event
    sizes = np.array([len(ids) for ids in schedule],dtype=np.int64)
    ids = np.zeros((len(schedule),26),dtype=np.int64)
    for i, samples in enumerate(schedule):
        ids[i,:len(samples)] = samples
    return ids, sizes
