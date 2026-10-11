"""TGNN_Sliding_Node_Mem: causal event-line-graph GAT + CNN + two memories.

The provenance graph consists of actual entity IDs and timestamped event edges.
One GAT operates on its causal line graph: slots connect only when their events
share an entity, with four directed endpoint-role features. The active window
has eight past incident events and the current event. No GRU is instantiated.

Input records must be prediction queries: for relation reconstruction the caller
must remove target aliases from attributes. The CADETS runner uses an allowlist.
Relations passed separately become observable only AFTER their own prediction.
Equal timestamps follow stable ingestion order. Runtime state is stream-local;
reset at split boundaries and detach after each optimizer step.
"""
from collections import defaultdict, deque, OrderedDict
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .encoding import UpgradedUniversalEventEncoder as UniversalEventEncoder
from node_memory_v3 import endpoint_scan
from .windows import plan_windows


@dataclass
class Record:
    index: int
    src: int
    dst: int
    time: float
    event: dict
    rare: bool = False
    selected: bool = False
    compressed: object = None


@dataclass
class TokenRef:
    # Keep a batched tensor and a row index, rather than a per-event autograd view.
    tensor: torch.Tensor
    row: int

    @property
    def requires_grad(self): return self.tensor.requires_grad


def gather_tokens(refs, zero):
    """One gather per backing batch, preserving order and gradients to that batch."""
    if not refs: return zero.new_empty((0,64))
    groups = {}
    padding = zero.reshape(1,64)
    for position, ref in enumerate(refs):
        tensor, row = (padding,0) if ref is None else (ref.tensor,ref.row)
        key=id(tensor)
        if key not in groups: groups[key]=(tensor,[],[])
        groups[key][1].append(row); groups[key][2].append(position)
    values=[]; order=[0]*len(refs);offset=0
    for tensor, rows, positions in groups.values():
        values.append(tensor.index_select(0,torch.tensor(rows,device=tensor.device)))
        for j, position in enumerate(positions):order[position]=offset+j
        offset+=len(rows)
    packed=torch.cat(values,0)
    return packed[torch.tensor(order,device=packed.device)]


def provenance_links(src, dst, valid):
    """[B,W,W,4] endpoint roles and causal, shared-entity adjacency."""
    roles = torch.stack((src[:, :, None] == src[:, None, :],
                         src[:, :, None] == dst[:, None, :],
                         dst[:, :, None] == src[:, None, :],
                         dst[:, :, None] == dst[:, None, :]), -1)
    width = src.shape[1]
    causal = torch.ones(width, width, dtype=torch.bool, device=src.device).tril()
    edges = roles.any(-1) & valid[:, :, None] & valid[:, None, :] & causal
    # Padded queries get a self-loop to avoid empty softmax; output is masked.
    edges = edges | torch.eye(width, dtype=torch.bool, device=src.device)[None]
    return edges, roles


class ProvenanceGAT(nn.Module):
    """Exactly one 128-dimensional, single-head additive GAT layer."""
    def __init__(self, dim=128):
        super().__init__()
        self.projection = nn.Linear(dim, dim)
        self.query = nn.Linear(dim, 1, bias=False)
        self.key = nn.Linear(dim, 1, bias=False)
        self.role = nn.Linear(4, 1, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, src, dst, valid):
        edges, roles = provenance_links(src, dst, valid)
        z = self.projection(x)
        scores = F.leaky_relu(self.query(z) + self.key(z).transpose(1, 2)
                              + self.role(roles.to(z.dtype)).squeeze(-1), .2)
        attention = scores.masked_fill(~edges, -torch.inf).softmax(-1)
        return self.norm(x + F.elu(attention @ z)) * valid[..., None]


class TemporalCNN(nn.Module):
    """Residual causal convolutions; receptive field 15 covers all nine slots."""
    def __init__(self, dim=128):
        super().__init__()
        self.convs = nn.ModuleList(nn.Conv1d(dim, dim, 3, dilation=d) for d in (1, 2, 4))
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in self.convs)

    def forward(self, x, valid):
        for conv, norm in zip(self.convs, self.norms):
            y = conv(F.pad(x.transpose(1, 2), (2 * conv.dilation[0], 0))).transpose(1, 2)
            x = norm(x + F.gelu(y)) * valid[..., None]
        return x


class TGNNSlidingNodeMem(nn.Module):
    def __init__(self, n_nodes, num_relations=10, window=9, short_slots=8,
                 event_dim=64, buckets=5303, novelty_threshold=.35, rare_count=2, head_dim=130):
        super().__init__()
        if n_nodes < 1 or num_relations < 2 or window not in (8, 9) or short_slots < 1 or head_dim < 1:
            raise ValueError('Positive node count, >=2 relations, window 8/9, positive short slots required')
        self.config = dict(upgrade="ordered-kv-local-rarity-sparse-memory-v1", n_nodes=n_nodes, num_relations=num_relations, window=window,
                           short_slots=short_slots, event_dim=event_dim, buckets=buckets,
                           novelty_threshold=novelty_threshold, rare_count=rare_count, head_dim=head_dim)
        self.n_nodes, self.num_relations = n_nodes, num_relations
        self.window, self.short_slots = window, short_slots
        self.novelty_threshold, self.rare_count = novelty_threshold, rare_count
        self.event_encoder = UniversalEventEncoder(dim=event_dim, buckets=buckets)
        self.project = nn.Linear(event_dim, 128)
        self.time = nn.Linear(1, 128)
        self.gat = ProvenanceGAT()
        self.cnn = TemporalCNN()
        self.importance_query = nn.Linear(128, 128, bias=False)
        self.importance_key = nn.Linear(128, 128, bias=False)
        self.compress = nn.Sequential(nn.Linear(256, 64), nn.Tanh())
        self.short_key = nn.Linear(64, 128, bias=False)
        self.short_value = nn.Linear(64, 128, bias=False)
        self.short_time = nn.Linear(1, 128, bias=False)
        self.short_role = nn.Embedding(3, 128)  # source, destination, both
        self.fusion = nn.Linear(256, 128)
        self.fusion_norm = nn.LayerNorm(128)
        self.write_src = nn.Sequential(nn.Linear(256, 64), nn.Tanh())
        self.write_dst = nn.Sequential(nn.Linear(256, 64), nn.Tanh())
        self.admission = nn.Linear(3, 1)
        self.read_src = nn.Linear(128, 128, bias=False)
        self.read_dst = nn.Linear(128, 128, bias=False)
        self.endpoint_gate = nn.Linear(128, 2)
        nn.init.zeros_(self.endpoint_gate.weight)
        nn.init.constant_(self.endpoint_gate.bias, -2.)
        self.head = nn.Sequential(nn.Linear(384, head_dim), nn.GELU(), nn.Linear(head_dim, num_relations))
        self.reset_state()

    def reset_state(self):
        self.history = defaultdict(deque)
        self.short = defaultdict(deque)
        self.records = {}
        self.counts = OrderedDict()
        self._memory = None
        self._entity_slots = {}
        self._entity_ids = []
        self._pending_memory = None
        self.local_counts = OrderedDict()
        self.next_index = 0
        self.last_time = None
        self._dirty_records = set()
        self._dirty_short = set()

    @property
    def Mem(self):
        self._flush_memory(detach=False)
        return self._memory

    @Mem.setter
    def Mem(self, value):
        self._memory = value
        self._pending_memory = None

    def _flush_memory(self, detach):
        if self._pending_memory is not None:
            nodes, updated = self._pending_memory
            if detach:
                with torch.no_grad(): self._memory.index_copy_(0, nodes, updated.detach())
            else:
                self._memory = self._memory.index_copy(0, nodes, updated)
            self._pending_memory = None
        if detach and self._memory is not None: self._memory = self._memory.detach()

    def detach_state(self):
        # Predictions use the differentiable touched-node scan. Commit only those
        # rows after backward, avoiding a full-table copy and gradient allocation.
        self._flush_memory(detach=True)
        detached = {}
        blocks = {}
        def detached_ref(value):
            key=id(value.tensor)
            if key not in blocks:blocks[key]=value.tensor.detach()
            return TokenRef(blocks[key],value.row)
        for i in self._dirty_records:
            r = self.records.get(i)
            if r is None: continue
            if r.compressed is not None:
                r.compressed = detached_ref(r.compressed)
                detached[i] = r.compressed
        for node in self._dirty_short:
            queue = self.short[node]
            def value_for(i, value):
                if i not in detached:
                    detached[i] = detached_ref(value) if value.requires_grad else value
                return detached[i]
            self.short[node] = deque((i, t, value_for(i, value)) for i, t, value in queue)
        self._dirty_records.clear()
        self._dirty_short.clear()

    def stream_state_dict(self):
        """Snapshot separately from learned weights; suitable for stream continuation."""
        import copy
        self.detach_state()
        # One bulk GPU-to-CPU copy for compressed tokens; deepcopy of individual
        # CUDA views would launch hundreds of thousands of tiny clone kernels.
        pool = {i:r.compressed for i,r in self.records.items() if r.compressed is not None}
        for q in self.short.values():
            for i, _, value in q: pool.setdefault(i, value)
        ids = list(pool); positions = {i:j for j,i in enumerate(ids)}
        values = gather_tokens([pool[i] for i in ids], self.project.weight.new_zeros(64)).detach().cpu()
        metadata = dict(format=4, history=dict(self.history), counts=self.counts,
                        next_index=self.next_index, last_time=self.last_time,
                        local_counts=self.local_counts, records=[(r.index,r.src,r.dst,r.time,r.event,r.rare,r.selected,
                                  positions.get(r.index)) for r in self.records.values()],
                        short={n:[(i,t,positions[i]) for i,t,_ in q] for n,q in self.short.items()})
        snapshot = copy.deepcopy(metadata)
        # Only observed entities can have nonzero memory after reset. The paper's
        # global ID space is much larger than this split's observed entity set.
        nodes=torch.tensor(self._entity_ids,dtype=torch.long,device=self.project.weight.device)
        snapshot.update(compressed_values=values, memory_nodes=nodes.cpu(),
                        memory_values=None if self._memory is None else self._memory[:len(nodes)].detach().cpu())
        return snapshot

    def load_stream_state_dict(self, state):
        import copy
        device = self.project.weight.device
        if state.get('format') in (2,3,4):
            values = state['compressed_values'].to(device).clone()
            self.history = defaultdict(deque, copy.deepcopy(state['history']))
            self.short = defaultdict(deque, {n:deque((i,t,TokenRef(values,p)) for i,t,p in q)
                                             for n,q in state['short'].items()})
            self.records = {row[0]:Record(*copy.deepcopy(row[:7]),
                            None if row[7] is None else TokenRef(values,row[7])) for row in state['records']}
            self.counts = copy.deepcopy(state['counts'])
            self.local_counts = copy.deepcopy(state.get('local_counts', OrderedDict()))
            self.next_index, self.last_time = state['next_index'], state['last_time']
            if state['format']==2:
                self.Mem = None if state['Mem'] is None else state['Mem'].to(device).clone()
                self._entity_ids = list(range(self.n_nodes))
                self._entity_slots = {node:node for node in self._entity_ids}
            else:
                self._entity_ids = state['memory_nodes'].tolist()
                self._entity_slots = {node:slot for slot,node in enumerate(self._entity_ids)}
                capacity = max(1, 1 << (max(1,len(self._entity_ids))-1).bit_length())
                self.Mem=None if state['memory_values'] is None else values.new_zeros(capacity,128)
                if self._memory is not None:
                    self._memory[:len(self._entity_ids)].copy_(state['memory_values'].to(device))
            self._dirty_records = set(); self._dirty_short = set()
            return
        for k, value in copy.deepcopy(state).items():
            if k not in ('history', 'short', 'records', 'counts', 'Mem', 'next_index', 'last_time'):
                raise ValueError('Unknown stream state field')
            setattr(self, k, value)
        if self.Mem is not None:
            self.Mem = self.Mem.to(device)
        for r in self.records.values():
            if r.compressed is not None: r.compressed = TokenRef(r.compressed.to(device).reshape(1,64),0)
        for n, q in self.short.items():
            self.short[n] = deque((i, t, TokenRef(v.to(device).reshape(1,64),0)) for i, t, v in q)
        self._dirty_records = set(self.records)
        self._dirty_short = set(self.short)

    def forward_events(self, events, src, dst, times, relations):
        """Predict and then observe a chronological batch, equivalent to single steps.

        Heavy KV/GAT/CNN work is batched; histories and selection advance per event.
        The affine endpoint scan preserves sequential memory reads and writes.
        Current event type is masked; attribute target aliases are caller-owned.
        """
        src, dst, times, relations = [v.detach().cpu().tolist() if torch.is_tensor(v) else list(v)
                                     for v in (src, dst, times, relations)]
        n = len(events)
        if not all(len(v) == n for v in (src, dst, times, relations)):
            raise ValueError('One endpoint pair, time and observed relation per event required')
        if any(not isinstance(u, int) or isinstance(u, bool) or u < 0 or u >= self.n_nodes for u in src + dst):
            raise ValueError('Map real, namespaced entity IDs to integer nodes; missing IDs need unique nodes')
        if any(not math.isfinite(t) for t in times) or any(a > b for a, b in zip(times, times[1:])):
            raise ValueError('Finite chronological timestamps required')
        if n and self.last_time is not None and times[0] < self.last_time:
            raise ValueError('Reset state before a new stream')
        if any(not isinstance(r, int) or r < 0 or r >= self.num_relations for r in relations):
            raise ValueError('Observed relation outside configured vocabulary')
        if not n:
            return self.project.weight.new_empty((0, self.num_relations)), {}
        device = self.project.weight.device
        # Construct real provenance neighborhoods from preceding event edges.
        windows, evictions, current = [], [], []
        planned = plan_windows(self.history,src,dst,self.next_index,self.window).tolist()
        for j, (event, u, v, t, relation) in enumerate(zip(events, src, dst, times, relations)):
            observed = dict(event, type=str(relation))
            attrs = event.get('attributes') or {}
            if not isinstance(attrs, dict): attrs = {}
            # Rarity uses structural categories, never unique paths/IDs or attack labels.
            signature = (relation, str(attrs.get('src', {}).get('type')) if isinstance(attrs.get('src'), dict) else '',
                         str(attrs.get('dst', {}).get('type')) if isinstance(attrs.get('dst'), dict) else '')
            count = self.counts.pop(signature, 0)
            self.counts[signature] = count + 1
            if len(self.counts) > 4096: self.counts.popitem(last=False)
            local_rare = False
            for node in dict.fromkeys((u, v)):
                local_key = (node, signature)
                local_count = self.local_counts.pop(local_key, 0)
                self.local_counts[local_key] = local_count + 1
                local_rare |= local_count < self.rare_count
            while len(self.local_counts) > 262144:
                self.local_counts.popitem(last=False)
            row = Record(self.next_index, u, v, float(t), observed, local_rare)
            prior = [index for index in planned[j] if index >= 0]
            windows.append([self.records[i] for i in prior] + [row])
            current.append(row)
            self.records[row.index] = row
            removed = []
            for node in dict.fromkeys((u, v)):
                queue = self.history[node]
                queue.append(row.index)
                if len(queue) >= self.window:
                    removed.append((node, self.records[queue.popleft()]))
            evictions.append(removed)
            self.next_index += 1
        # Re-encode retained raw events with current weights (no stale active tokens).
        observed_records = {r.index: r for row in windows for r in row}
        observed_records.update((r.index, r) for removed in evictions for _, r in removed)
        observed_list = list(observed_records.values())
        positions = {r.index: i for i, r in enumerate(observed_list)}
        query_events = [dict(e, type=None) for e in events]
        embedded = self.event_encoder(query_events + [r.event for r in observed_list])
        query, observed = self.project(embedded[:n]), self.project(embedded[n:])
        indices, sources, destinations, ages, masks = [], [], [], [], []
        for row in windows:
            pad = self.window - len(row)
            indices.append([0] * pad + [positions[r.index] for r in row])
            sources.append([-1] * pad + [r.src for r in row])
            destinations.append([-2] * pad + [r.dst for r in row])
            ages.append([0.] * pad + [math.log1p(row[-1].time-r.time) for r in row])
            masks.append([False] * pad + [True] * len(row))
        valid = torch.tensor(masks, device=device)
        raw = observed[torch.tensor(indices, device=device)]
        x = torch.cat((raw[:, :-1], query[:, None]), 1)
        x = x + self.time(torch.tensor(ages, device=device, dtype=x.dtype)[..., None]).cos()
        x = x * valid[..., None]
        h = self.cnn(self.gat(x, torch.tensor(sources, device=device),
                              torch.tensor(destinations, device=device), valid), valid)
        scores = (self.importance_query(h[:, -1])[:, None] * self.importance_key(h)).sum(-1) / math.sqrt(128)
        attention = scores.masked_fill(~valid, -torch.inf).softmax(-1)
        c = h[:, -1] + (attention[..., None] * h).sum(1)
        # Stable top-half selection, retaining ceil(0.5 * valid slots).
        ranking = scores.masked_fill(~valid, -torch.inf).argsort(dim=-1, descending=True, stable=True).argsort(dim=-1)
        top = (ranking < ((valid.sum(-1) + 1) // 2)[:, None]) & valid
        actual = observed[torch.tensor([positions[r.index] for r in current], device=device)]
        similarity = F.cosine_similarity(actual[:, None].detach(), raw[:, :-1].detach(), dim=-1)
        similarity = similarity.masked_fill(~valid[:, :-1], -1.)
        novelty = (1 - similarity.max(-1).values).clamp(0, 2)
        rare = torch.tensor([r.rare for r in current], device=device)
        exceptional = rare | (novelty >= self.novelty_threshold)
        admission = self.admission(torch.stack((attention[:, -1], novelty, rare.to(c.dtype)), -1)).sigmoid().squeeze(-1)
        # Small nonzero floor lets repetitive events train the long-memory writes.
        admission = .05 + .95 * admission * (top[:, -1] | exceptional).to(c.dtype)
        content = torch.cat((c, actual), -1)
        compressed = self.compress(content)
        selected_cpu, exceptional_cpu = top.detach().cpu().tolist(), exceptional.detach().cpu().tolist()
        # Read short memory BEFORE committing this event's selection/evictions.
        short_rows, short_ages, short_roles, short_valid = [], [], [], []
        zero = compressed.new_zeros(64)
        width = self.window
        for j, (row, now, removed) in enumerate(zip(windows, current, evictions)):
            entries = [(i, t, value, role) for role, node in enumerate((now.src, now.dst))
                       for i, t, value in self.short[node]]
            # One compressed event per slot, oldest to newest, sliding by ingestion ID.
            unique = {}
            for i, t, value, role in entries:
                unique[i] = (i, t, value, 2 if i in unique else role)
            entries = [unique[i] for i in sorted(unique)[-width:]]
            pad = width - len(entries)
            short_rows.append([None] * pad + [e[2] for e in entries])
            short_ages.append([0.] * pad + [math.log1p(now.time-e[1]) for e in entries])
            short_roles.append([0] * pad + [e[3] for e in entries])
            short_valid.append([False] * pad + [True] * len(entries))
            now.compressed = TokenRef(compressed,j)
            self._dirty_records.add(now.index)
            for r, keep in zip(row, selected_cpu[j][-len(row):]):
                r.selected = r.selected or keep
            now.selected = now.selected or exceptional_cpu[j]
            for node, old in removed:
                if old.selected:
                    self.short[node].append((old.index, old.time, old.compressed))
                    self._dirty_short.add(node)
                    if len(self.short[node]) > self.short_slots: self.short[node].popleft()
        # Many entities have not yet evicted an event. Skip their all-empty
        # compressed windows instead of running a second CNN on padded zeros.
        active_short = [j for j, row_mask in enumerate(short_valid) if any(row_mask)]
        short_context = c.new_zeros(n,128)
        short_count = torch.tensor([sum(row_mask) for row_mask in short_valid],device=device)
        if active_short:
            active = torch.tensor(active_short,device=device)
            sv = gather_tokens([value for j in active_short for value in short_rows[j]],zero).reshape(len(active_short),width,64)
            smask = torch.tensor([short_valid[j] for j in active_short],device=device)
            sx = self.short_value(sv) + self.short_time(torch.tensor(
                [short_ages[j] for j in active_short],device=device,dtype=c.dtype)[...,None]).cos()
            sx = sx + self.short_role(torch.tensor([short_roles[j] for j in active_short],device=device))
            sh = self.cnn(sx * smask[...,None],smask)
            sk = self.short_key(sv) + sh
            sa = (sk * c[active,None]).sum(-1) / math.sqrt(128)
            sa = sa.masked_fill(~smask,-1e9).softmax(-1) * smask
            short_context = short_context.index_copy(0,active,(sa[...,None]*sh).sum(1))
        fused = self.fusion_norm(c + self.fusion(torch.cat((c, short_context), -1)).sigmoid() * short_context)
        # Only touched nodes participate in the scan; scatter after the causal reads.
        # Compact rows are keyed by real entity IDs; grow only as entities appear.
        for node in src + dst:
            if node not in self._entity_slots:
                self._entity_slots[node] = len(self._entity_ids)
                self._entity_ids.append(node)
        s = torch.tensor([self._entity_slots[u] for u in src], device=device)
        d = torch.tensor([self._entity_slots[v] for v in dst], device=device)
        self._flush_memory(detach=False)
        required = len(self._entity_ids)
        if self._memory is None:
            capacity = min(self.n_nodes, max(64, 1 << (required - 1).bit_length()))
            self._memory = c.new_zeros(capacity, 128)
        elif required > len(self._memory):
            capacity = min(self.n_nodes, 1 << (required - 1).bit_length())
            self._memory = torch.cat((self._memory, c.new_zeros(capacity-len(self._memory),128)),0)
        nodes, inverse = torch.unique(torch.cat((s, d)), sorted=True, return_inverse=True)
        ws, wd = self.write_src(content), self.write_dst(content)
        writes = torch.cat((ws, wd))
        writes = torch.cat((writes, writes), -1)
        strength = c.new_tensor([.5] * 64 + [.005] * 64)
        weights = torch.cat((admission, admission))[:, None] * strength
        ms, md, updated = endpoint_scan(self._memory[nodes], inverse[:n], inverse[n:], weights, writes)
        if torch.is_grad_enabled(): self._pending_memory = (nodes, updated)
        else: self._memory.index_copy_(0, nodes, updated)
        gates = self.endpoint_gate(F.dropout(fused, p=.1, training=self.training)).sigmoid()
        logits = self.head(torch.cat((F.dropout(fused, p=.1, training=self.training), gates[:, :1] * self.read_src(ms), gates[:, 1:] * self.read_dst(md)), -1))
        for removed in evictions:
            for _, old in removed:
                if old.index not in self.history[old.src] and old.index not in self.history[old.dst]:
                    self.records.pop(old.index, None)
        self.last_time = float(times[-1])
        return logits, dict(admission=admission, novelty=novelty, rare=rare,
                            attention=attention, selected=top, short_count=short_count,
                            compression_loss=F.mse_loss(compressed, embedded[n:][torch.tensor(
                                [positions[r.index] for r in current], device=device)].detach()))
