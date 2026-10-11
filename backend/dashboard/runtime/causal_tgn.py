"""Small controlled temporal additions to the canonical adapted graph encoder."""
from __future__ import annotations

import random
import torch
from torch import nn

from .pomsathit_baseline import PomsathitAdapted


class CausalTGNN(PomsathitAdapted):
    def __init__(self, input_dim=10, factors=()):
        super().__init__(input_dim)
        self.factors = frozenset(factors)
        if self.factors - {'T', 'M', 'R', 'H'}:
            raise ValueError('Unimplemented factor requested')
        if 'R' in self.factors and 'M' not in self.factors:
            raise ValueError('R is a replacement memory update and requires M')
        self.time_encoder = nn.Linear(1, 128) if 'T' in self.factors else None
        self.cell = nn.GRUCell(128, 128) if 'R' in self.factors else None
        self.reset_state()

    def reset_state(self):
        self.memory = {}
        self.last_time = None
        self.pending = []

    def detach_state(self):
        self.memory = {k: v.detach() for k, v in self.memory.items()}
        self.pending = [(s, d, h.detach()) for s, d, h in self.pending]

    def _commit(self, intervention=None):
        if intervention not in {'frozen_memory', 'reset_memory', 'gru_noop'}:
            for s, d, h in self.pending:
                for node in {s, d}:
                    old = self.memory.get(node, torch.zeros_like(h))
                    new = self.cell(h, old) if self.cell is not None else .5 * old + .5 * h
                    self.memory[node] = new
        self.pending = []

    def forward_events(self, x, valid, ages, src, dst, times, intervention=None):
        """Each ego is causal. Score before update; equal-time events share prior state.

        Returns pre-event context for explicit stale-context training replay.
        Labels are deliberately absent from this interface.
        """
        extra = None
        if self.time_encoder is not None:
            age_input = torch.zeros_like(ages) if intervention == 'constant_time' else ages
            extra = self.time_encoder(age_input.unsqueeze(-1)).cos() * valid.unsqueeze(-1)
        h = self.encode(x, valid, extra)
        if 'M' not in self.factors:
            context = torch.zeros_like(h)
            return self.head(h), context
        contexts = []
        for i, (s, d, t) in enumerate(zip(src, dst, times)):
            s, d, t = int(s), int(d), float(t)
            if self.last_time is not None and t < self.last_time:
                raise ValueError('Memory stream moved backwards')
            if t != self.last_time:
                self._commit(intervention)
            if intervention == 'reset_memory':
                self.memory = {}
                self.pending = []
            zero = torch.zeros_like(h[i])
            contexts.append((self.memory.get(s, zero) + self.memory.get(d, zero)) * .5)
            self.pending.append((s, d, h[i]))
            self.last_time = t
        context = torch.stack(contexts)
        return self.head(h + context), context

    def replay(self, x, valid, ages, context):
        """Recompute graph, read a detached training snapshot, never mutate state."""
        extra = None if self.time_encoder is None else self.time_encoder(ages.unsqueeze(-1)).cos() * valid.unsqueeze(-1)
        if 'M' not in self.factors and torch.count_nonzero(context):
            raise ValueError('Memory supplied while M is disabled')
        return self.head(self.encode(x, valid, extra) + context.detach())


class Reservoir:
    """Uniform Algorithm R over training visits with dedicated RNG and hard bounds."""
    def __init__(self, capacity, train_end, seed):
        self.capacity, self.train_end = capacity, train_end
        self.rng = random.Random(seed)
        self.seen = 0
        self.items = []

    def admit(self, event_id, context):
        if not 0 <= event_id < self.train_end:
            raise ValueError('Only training events may enter rehearsal')
        self.seen += 1
        item = (int(event_id), context.detach().clone())
        if len(self.items) < self.capacity:
            self.items.append(item)
        elif self.capacity:
            j = self.rng.randrange(self.seen)
            if j < self.capacity:
                self.items[j] = item

    def sample(self, size):
        return self.rng.sample(self.items, min(size, len(self.items)))
