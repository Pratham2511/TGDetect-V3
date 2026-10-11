"""Independent additions to the frozen TGNN; no changes to the reference files."""
from __future__ import annotations

import itertools
import random

import numpy as np
import torch
from torch import nn

from .causal_tgn import CausalTGNN


ARMS = {"GNN_H": ("H",), "GNN_H_A": ("H", "A")}
for bits in itertools.product([False, True], repeat=3):
    added = tuple(f for f, active in zip(("Q", "M", "A"), bits) if active)
    ARMS["TGNN" + ("_" + "_".join(added) if added else "")] = ("T", "H") + added
ARMS.update(TGNN_N=("T", "H", "N"), TGNN_W=("T", "H", "W"))


class ComponentTGNN(CausalTGNN):
    """Q: history GRU; M: causal EMA memory; A: extra attention; N: head norm.

    Q and M are independent: Q resets on each ego, M persists between events.
    W is exclusively a runner loss factor. H is exclusively replay policy.
    """

    def __init__(self, input_dim=10, factors=("T", "H")):
        factors = frozenset(factors)
        if factors - set("THQMANW"):
            raise ValueError("Unsupported component")
        super().__init__(input_dim, factors & set("THM"))
        self.factors = factors
        self.history_gru = nn.GRUCell(128, 128) if "Q" in factors else None
        self.query = nn.Linear(128, 128, bias=False) if "A" in factors else None
        self.key = nn.Linear(128, 128, bias=False) if "A" in factors else None
        self.value = nn.Linear(128, 128, bias=False) if "A" in factors else None
        self.norm = nn.LayerNorm(128) if "N" in factors else nn.Identity()
        self._plans = {}

    def encode(self, x, valid, extra=None):
        if self.history_gru is None and self.query is None:
            return super().encode(x, valid, extra)
        batch, nodes, _ = x.shape
        adjacency = torch.eye(nodes, dtype=torch.bool, device=x.device).expand(batch, -1, -1).clone()
        adjacency[:, 0, :] = valid
        for i, layer in enumerate(self.layers):
            x = layer(x, adjacency, extra if i == 0 else None)
        h = x[:, 0]
        if self.history_gru is not None:
            state = torch.zeros_like(h)
            # Ego history is newest first; reverse it and skip padded slots.
            for j in range(nodes - 1, 0, -1):
                proposed = self.history_gru(x[:, j], state)
                state = torch.where(valid[:, j, None], proposed, state)
            h = h + state
        if self.query is not None:
            mask = valid[:, 1:]
            scores = (self.query(h)[:, None] * self.key(x[:, 1:])).sum(-1) / (128 ** .5)
            # Empty history contributes exactly zero; finite fill avoids NaN gradients.
            weights = scores.masked_fill(~mask, -1e9).softmax(-1) * mask
            h = h + (weights[:, :, None] * self.value(x[:, 1:])).sum(1)
        return h

    def _memory_plan(self, src, dst, times, device, dtype):
        """Compile the exact scalar EMA recurrence into constant linear operators.

        Pending tied events are carried across batches. No labels or embeddings
        participate in planning. Coefficients preserve gradients inside a batch.
        """
        previous = tuple((s, d) for s, d, _ in self.pending)
        key = (tuple(src), tuple(dst), tuple(times), previous, self.last_time, str(device), dtype)
        if key in self._plans:
            return self._plans[key]
        if len(times) == 0 or np.any(np.diff(times) < 0):
            raise ValueError("Nonempty chronological batch required")
        if self.last_time is not None and times[0] < self.last_time:
            raise ValueError("Memory stream moved backwards")
        endpoints = previous + tuple(zip(map(int, src), map(int, dst)))
        nodes = sorted({node for pair in endpoints for node in pair})
        positions = {node: i for i, node in enumerate(nodes)}
        u, p, b = len(nodes), len(previous), len(src)
        state = np.eye(u, u + p + b, dtype=np.float32)
        context = []
        pending = list(range(p))
        last = self.last_time
        for i, (s, d, t) in enumerate(zip(src, dst, times)):
            if t != last:
                for event in pending:
                    for node in set(endpoints[event]):
                        j = positions[node]
                        state[j] *= .5
                        state[j, u + event] += .5
                pending = []
            context.append(.5 * (state[positions[int(s)]] + state[positions[int(d)]]))
            pending.append(p + i)
            last = t
        plan = (nodes, torch.as_tensor(np.stack(context), device=device, dtype=dtype),
                torch.as_tensor(state, device=device, dtype=dtype),
                tuple((endpoints[i][0], endpoints[i][1], i) for i in pending))
        self._plans[key] = plan
        return plan

    def forward_events(self, x, valid, ages, src, dst, times, intervention=None):
        if intervention not in {None, "reset_memory", "constant_time"}:
            raise ValueError("Unsupported intervention")
        extra = None
        if self.time_encoder is not None:
            time = torch.zeros_like(ages) if intervention == "constant_time" else ages
            extra = self.time_encoder(time.unsqueeze(-1)).cos() * valid.unsqueeze(-1)
        h = self.encode(x, valid, extra)
        if "M" not in self.factors or intervention == "reset_memory":
            context = torch.zeros_like(h)
            if intervention == "reset_memory":
                self.reset_state()
        else:
            nodes, read, update, pending = self._memory_plan(src, dst, times, h.device, h.dtype)
            zero = h.new_zeros(128)
            old = torch.stack([self.memory.get(node, zero) for node in nodes])
            values = torch.cat([torch.stack([v for _, _, v in self.pending]), h]) if self.pending else h
            basis = torch.cat([old, values])
            context = read @ basis
            updated = update @ basis
            self.memory.update(zip(nodes, updated.unbind()))
            self.pending = [(s, d, values[i]) for s, d, i in pending]
            self.last_time = float(times[-1])
        return self.head(self.norm(h + context)), context

    def replay(self, x, valid, ages, context):
        extra = None if self.time_encoder is None else self.time_encoder(ages.unsqueeze(-1)).cos() * valid.unsqueeze(-1)
        if "M" not in self.factors and torch.count_nonzero(context):
            raise ValueError("Memory supplied while disabled")
        return self.head(self.norm(self.encode(x, valid, extra) + context.detach()))


class EfficientReservoir:
    """Same Algorithm R/RNG as frozen Reservoir; clone only admitted snapshots."""
    def __init__(self, capacity, train_end, seed):
        self.capacity, self.train_end = capacity, train_end
        self.rng = random.Random(seed)
        self.seen = 0
        self.items = []

    def admit(self, event_id, context):
        if not 0 <= event_id < self.train_end:
            raise ValueError("Only training events may enter rehearsal")
        self.seen += 1
        if len(self.items) < self.capacity:
            self.items.append((int(event_id), context.detach().clone()))
        elif self.capacity:
            j = self.rng.randrange(self.seen)
            if j < self.capacity:
                self.items[j] = (int(event_id), context.detach().clone())

    def sample(self, size):
        return self.rng.sample(self.items, min(size, len(self.items)))
