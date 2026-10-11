"""Causal, batch-invariant selective memory with separate endpoint reads.

Admission uses a fixed-rate reference EMA for novelty. Its coefficients depend
only on the current event, allowing exact parallel prefix reads without a
sequential learned-gate dependency. Reference and task memories share the same
endpoint-specific write space. Runtime state is excluded from checkpoints.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from node_memory_models import TGNNNodeMemSel, AdmissionController


def endpoint_scan(table, src, dst, weights, content):
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


class TGNNNodeMemV3(TGNNNodeMemSel):
    def __init__(self, input_dim=72, n_nodes=1, use_memory=True, gat_layers=2,
                 adaptive=True, memory_dim=64):
        super().__init__(input_dim, n_nodes, use_memory, gat_layers)
        # Remove unused V2 modules; report actual rather than padded parameters.
        del self.mem_read, self.write_mlp
        self.memory_dim = memory_dim
        self.adaptive = adaptive
        self.write_src = nn.Sequential(nn.Linear(128, memory_dim), nn.Tanh())
        self.write_dst = nn.Sequential(nn.Linear(128, memory_dim), nn.Tanh())
        self.read_src = nn.Linear(2 * memory_dim, 128, bias=False)
        self.read_dst = nn.Linear(2 * memory_dim, 128, bias=False)
        self.read_gate = nn.Linear(128, 2)
        nn.init.zeros_(self.read_gate.weight)
        nn.init.zeros_(self.read_gate.bias)
        self.candidate_threshold = nn.Parameter(torch.tensor(0.5))
        self.controller = AdmissionController(state_dim=7, r_min=.10, r_max=.30, r_init=.20)
        with torch.no_grad():
            self.controller.threshold.fill_(2.9)
        self.reset_state()

    def reset_state(self):
        self.Mem = self.Reference = None
        self.last_time = None
        self.memory, self.pending = {}, []

    def detach_state(self):
        if self.Mem is not None:
            self.Mem = self.Mem.detach()
            self.Reference = self.Reference.detach()

    def _context_forward(self, c, src, dst):
        """Factored out for causality, recurrence and gradient acceptance tests."""
        if not self.use_memory:
            return self.head(self.mem_norm(c)), {}
        if self.Mem is None:
            self.Mem = c.new_zeros(self.n_nodes, 2 * self.memory_dim)
            self.Reference = c.new_zeros(self.n_nodes, self.memory_dim)
        batch = len(c)
        ws, wd = self.write_src(c), self.write_dst(c)
        content = torch.cat((ws, wd))
        # Novelty reference is detached: only task writes learn from future loss.
        with torch.no_grad():
            rs, rd, self.Reference = endpoint_scan(
                self.Reference, src, dst, torch.full_like(content, .10), content.detach())
            ns = 1 - F.cosine_similarity(ws.detach(), rs, dim=1, eps=1e-6)
            nd = 1 - F.cosine_similarity(wd.detach(), rd, dim=1, eps=1e-6)
            novelty = .5 * (ns + nd)
            base = self.head(self.mem_norm(c)).softmax(-1)
            uncertainty = -(base * base.clamp_min(1e-9).log()).sum(-1) / math.log(2)
        importance = self.attn_score(c).sigmoid().squeeze(-1)
        cand = torch.maximum(((importance - self.candidate_threshold) / .1).sigmoid(),
                             (novelty >= self.nov_tau).to(c.dtype))
        feats = torch.stack((importance.detach(), novelty, uncertainty,
                             rs.norm(dim=1), rd.norm(dim=1),
                             (rs.norm(dim=1) > 0).float(), (rd.norm(dim=1) > 0).float()), dim=1)
        rate = self.controller.rate(feats)
        if self.adaptive:
            # Importance needs task credit even when novelty forces candidacy.
            score = self.controller.score(importance, novelty, uncertainty)
            temperature = self.controller.log_temp.exp().clamp(.01, 10)
            admit = cand * ((score - self.controller.threshold) / temperature).sigmoid()
        else:
            cand = torch.ones_like(cand)
            admit = torch.full_like(cand, .20)
        # Typical admission .2 gives effective weights .1 and .001.
        strength = c.new_tensor([.5] * self.memory_dim + [.005] * self.memory_dim)
        weights = torch.cat((admit, admit))[:, None] * strength
        expanded = torch.cat((content, content), dim=1)
        ms, md, self.Mem = endpoint_scan(self.Mem, src, dst, weights, expanded)
        gates = self.read_gate(c).tanh()
        z = c + gates[:, :1] * self.read_src(ms) + gates[:, 1:] * self.read_dst(md)
        logits = self.head(self.mem_norm(z))
        aux = {"admit": admit, "cand": cand, "novelty": novelty, "rate": rate,
               "read_gate": gates, "effective_fast_weight": admit * .5,
               "effective_slow_weight": admit * .005}
        return logits, aux

    def forward_events(self, x, valid, ages, src, dst, times, intervention=None):
        extra = None
        if self.time_encoder is not None:
            age = torch.zeros_like(ages) if intervention == "constant_time" else ages
            extra = self.time_encoder(age.unsqueeze(-1)).cos() * valid.unsqueeze(-1)
        c = self.encode(x, valid, extra)
        if not self.use_memory:
            return self.head(self.mem_norm(c)), {}
        src = torch.as_tensor(src, device=c.device, dtype=torch.long)
        dst = torch.as_tensor(dst, device=c.device, dtype=torch.long)
        result = self._context_forward(c, src, dst)
        self.last_time = float(times[-1]) if len(times) else self.last_time
        return result


def admission_penalties(aux, target=.20):
    """Candidate membership is counted once, matching actual write admission."""
    mass = aux['cand'].sum().clamp_min(1e-6)
    fraction = aux['admit'].sum() / mass
    rate = (aux['rate'] * aux['cand']).sum() / mass
    budget = (fraction - rate).square() + .25 * (rate - target).square()
    redundancy = (aux['admit'] * (1 - aux['novelty']).clamp_min(0)).sum() / mass
    return budget, redundancy


def make_mem_model(arm, input_dim=72, n_nodes=1, **kwargs):
    if arm not in ('TGNN_Node_Mem_V3', 'TGNN_Node_Mem_V3_Fixed', 'TGNN_Q_A_C'):
        raise ValueError(arm)
    return TGNNNodeMemV3(input_dim, n_nodes, use_memory=arm != 'TGNN_Q_A_C',
                         gat_layers=kwargs.get('gat_layers', 2),
                         adaptive=arm != 'TGNN_Node_Mem_V3_Fixed')
