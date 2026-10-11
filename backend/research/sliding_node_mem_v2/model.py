"""V2: exact execution optimizations plus separately selectable detection fixes.

profile='compatible' retains the upgrade architecture and stream semantics.
profile='corrected' needs retraining: balanced histories, one-time admission,
bounded rare/regular retention, contextual cosine compression and ordered pairs.
"""
import math
import numpy as np
from collections import defaultdict, deque, OrderedDict
import torch
from torch import nn
from torch.nn import functional as F
from sliding_node_mem_upgrade.model import TGNNSlidingNodeMem as Reference
from sliding_node_mem_upgrade.model import Record, TokenRef
from .tensors import tensor, gather_tokens
from sliding_node_mem_upgrade.windows import plan_windows as legacy_windows
from .memory import endpoint_scan
from .encoding import EventEncoder
from .layers import TemporalCNN
from .windows import plan_windows as balanced_windows


class TGNNSlidingNodeMem(Reference):
    def __init__(self, n_nodes, num_relations=10, window=9, short_slots=8,
                 event_dim=64, buckets=5303, novelty_threshold=.35, rare_count=2,
                 head_dim=130, profile="corrected", cnn_backend="auto",
                 cache_size=8192, compact_every=32, admission_floor=0., retention_half_life=3600.,
                 memory_half_lives=(3600.,86400.)):
        if profile not in ("compatible", "corrected"):
            raise ValueError("Unknown V2 profile")
        if event_dim != 64 or not 0 <= admission_floor <= 1 or compact_every < 0 or cache_size < 0 or retention_half_life <= 0:
            raise ValueError("event_dim=64, bounded admission floor and nonnegative cache/compaction required")
        self.corrected = profile == "corrected"
        self.compact_every = compact_every
        self.admission_floor = admission_floor
        self.retention_half_life = retention_half_life
        if memory_half_lives is not None and (len(memory_half_lives) != 2 or any(h <= 0 or not math.isfinite(h) for h in memory_half_lives)):
            raise ValueError('Two positive finite half-lives or None required')
        self.memory_half_lives = memory_half_lives
        super().__init__(n_nodes, num_relations, window, short_slots, event_dim,
                         buckets, novelty_threshold, rare_count, head_dim)
        self.event_encoder = EventEncoder(dim=event_dim, buckets=buckets,
                                         ordered_values=self.corrected, cache_size=cache_size)
        self.cnn = TemporalCNN(backend=cnn_backend)
        self.register_buffer('_write_strength',torch.tensor([.5]*64+[.005]*64),persistent=False)
        self.register_buffer('_half_lives',None if memory_half_lives is None else
                             torch.tensor([memory_half_lives[0]]*64+[memory_half_lives[1]]*64),persistent=False)
        self.config.update(upgrade="sliding-v2", profile=profile, cnn_backend=cnn_backend,
                           cache_size=cache_size, compact_every=compact_every,
                           admission_floor=admission_floor,retention_half_life=retention_half_life,
                           memory_half_lives=memory_half_lives)

    def reset_state(self):
        super().reset_state()
        self._retention = {}
        self._retention_expired = set()
        self._node_times = {}
        self._detach_count = 0
        self._retention_stats = [0,0]

    @property
    def retained_evictions(self):
        return self._retention_stats[0]

    @property
    def total_evictions(self):
        return self._retention_stats[1]

    def compact_state(self, max_blocks=None):
        """Repack only underoccupied detached blocks; preserve shared token rows."""
        pool = {i:r.compressed for i,r in self.records.items() if r.compressed is not None}
        for queue in self.short.values():
            for i, _, ref in queue:
                pool.setdefault(i, ref)
        blocks = {}
        for i, ref in pool.items():
            blocks.setdefault(id(ref.tensor), []).append((i, ref))
        replacements = {}
        released = 0
        compacted = 0
        for group in blocks.values():
            tensor = group[0][1].tensor
            if tensor.requires_grad or len(group)*2 >= tensor.shape[0]:
                continue
            compact = tensor.index_select(0, torch.tensor([r.row for _,r in group], device=tensor.device))
            released += (tensor.shape[0]-len(group))*tensor.shape[1]*tensor.element_size()
            replacements.update({i:TokenRef(compact,j) for j,(i,_) in enumerate(group)})
            compacted += 1
            if max_blocks is not None and compacted >= max_blocks:
                break
        if replacements:
            for i, record in self.records.items():
                if i in replacements:
                    record.compressed = replacements[i]
            for node, queue in self.short.items():
                self.short[node] = deque((i,t,replacements.get(i,ref)) for i,t,ref in queue)
        return released

    def detach_state(self):
        super().detach_state()
        self._detach_count += 1
        if self.compact_every and self._detach_count % self.compact_every == 0:
            # Bound GPU gathers/allocations in scheduled maintenance. A manual
            # compact_state() still repacks every eligible backing block.
            self.compact_state(max_blocks=8)

    def stream_state_dict(self):
        state = super().stream_state_dict()
        state['v2'] = dict(config=self.config, retention=dict(self._retention),node_times=dict(self._node_times),
                           retained_evictions=self.retained_evictions,
                           total_evictions=self.total_evictions, detach_count=self._detach_count)
        return state

    def load_stream_state_dict(self, state):
        extra = state.get('v2')
        if extra is None or extra['config'] != self.config:
            raise ValueError('V2 stream state/config mismatch; reset or use matching configuration')
        super().load_stream_state_dict({k:v for k,v in state.items() if k != 'v2'})
        self._retention = dict(extra['retention'])
        self._node_times = dict(extra['node_times'])
        self._retention_stats = [extra['retained_evictions'],extra['total_evictions']]
        self._detach_count = extra['detach_count']

    def _retain(self, node, old, now):
        """One admission decision per event; bounded regular/exception quotas."""
        self._retention_stats[1] += 1
        if not old.selected:
            return
        queue = self.short[node]
        queue.append((old.index, old.time, old.compressed))
        self._dirty_short.add(node)
        if self.corrected and self.short_slots > 1:
            exceptional = self._retention[old.index][1]
            quota = (self.short_slots+1)//2 if exceptional else self.short_slots//2
            candidates = [entry for entry in queue if self._retention[entry[0]][1] == exceptional]
            if len(candidates) > quota:
                victim = min(candidates, key=lambda e: (
                    self._retention[e[0]][0]*math.exp(-math.log(2)*(now-e[1])/self.retention_half_life), e[0]))
                queue.remove(victim)
                self._retention_expired.add(victim[0])
        elif len(queue) > self.short_slots:
            self._retention_expired.add(queue.popleft()[0])
        self._retention_stats[0] += any(i == old.index for i,_,_ in queue)

    def _memory_rows(self, nodes):
        """Overlay uncommitted differentiable rows without copying the full table."""
        rows = self._memory[nodes]
        if self._pending_memory is not None:
            previous, values = self._pending_memory
            locations = torch.searchsorted(previous, nodes)
            safe = locations.clamp_max(len(previous)-1)
            matches = previous[safe] == nodes
            rows = torch.where(matches[:,None], values[safe], rows)
        return rows

    def _stage_memory(self, nodes, updated):
        if not torch.is_grad_enabled():
            self._memory.index_copy_(0, nodes, updated)
            return
        if self._pending_memory is not None:
            previous, values = self._pending_memory
            locations = torch.searchsorted(nodes, previous).clamp_max(len(nodes)-1)
            keep = nodes[locations] != previous
            nodes = torch.cat((nodes, previous[keep]))
            updated = torch.cat((updated, values[keep]))
            order = nodes.argsort()
            nodes, updated = nodes[order], updated[order]
        self._pending_memory = (nodes, updated)

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
        planner = balanced_windows if self.corrected else legacy_windows
        planned = planner(self.history,src,dst,self.next_index,self.window).tolist()
        first_index = self.next_index
        for j, (event, u, v, t, relation) in enumerate(zip(events, src, dst, times, relations)):
            observed = dict(event, type=str(relation))
            attrs = event.get('attributes') or {}
            if not isinstance(attrs, dict): attrs = {}
            # Rarity uses structural categories, never unique paths/IDs or attack labels.
            signature = (relation, str(attrs.get('src', {}).get('type')) if isinstance(attrs.get('src'), dict) else '',
                         str(attrs.get('dst', {}).get('type')) if isinstance(attrs.get('dst'), dict) else '')
            if not self.corrected:
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
            row = Record(first_index+j, u, v, float(t), observed, local_rare)
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
        self.next_index += n
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
            ages.append([0.] * pad + [row[-1].time-r.time for r in row])
            masks.append([False] * pad + [True] * len(row))
        metadata = torch.as_tensor(np.asarray([indices,sources,destinations,masks],dtype=np.int64)).to(device,non_blocking=True)
        valid = metadata[3].bool()
        raw = observed[metadata[0]]
        x = torch.cat((raw[:, :-1], query[:, None]), 1)
        x = x + self.time(tensor(np.log1p(ages), device=device, dtype=x.dtype)[..., None]).cos()
        x = x * valid[..., None]
        h = self.cnn(self.gat(x, metadata[1], metadata[2], valid), valid)
        scores = (self.importance_query(h[:, -1])[:, None] * self.importance_key(h)).sum(-1) / math.sqrt(128)
        attention = scores.masked_fill(~valid, -torch.inf).softmax(-1)
        c = h[:, -1] + (attention[..., None] * h).sum(1)
        # Stable top-half selection, retaining ceil(0.5 * valid slots).
        ranking = scores.masked_fill(~valid, -torch.inf).argsort(dim=-1, descending=True, stable=True).argsort(dim=-1)
        top = (ranking < ((valid.sum(-1) + 1) // 2)[:, None]) & valid
        current_indices = tensor([positions[r.index] for r in current], device=device)
        actual = observed[current_indices]
        similarity = F.cosine_similarity(actual[:, None].detach(), raw[:, :-1].detach(), dim=-1)
        similarity = similarity.masked_fill(~valid[:, :-1], -1.)
        novelty = (1 - similarity.max(-1).values).clamp(0, 2)
        rare = tensor([r.rare for r in current], device=device)
        exceptional = rare | (novelty >= self.novelty_threshold)
        admission = self.admission(torch.stack((attention[:, -1], novelty, rare.to(c.dtype)), -1)).sigmoid().squeeze(-1)
        # Small nonzero floor lets repetitive events train the long-memory writes.
        floor = self.admission_floor if self.corrected else .05
        admission = floor + (1-floor) * admission * (top[:, -1] | exceptional).to(c.dtype)
        content = torch.cat((c, actual), -1)
        compressed = self.compress(content)
        if self.corrected:
            decisions = torch.stack(((top[:, -1] | exceptional).to(c.dtype),
                                     exceptional.to(c.dtype), attention[:, -1]), -1).detach().cpu().tolist()
        else:
            packed = torch.cat((top, exceptional[:,None]), -1).detach().cpu().tolist()
            selected_cpu = [row[:-1] for row in packed]
            exceptional_cpu = [row[-1] for row in packed]
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
            short_ages.append([0.] * pad + [now.time-e[1] for e in entries])
            short_roles.append([0] * pad + [e[3] for e in entries])
            short_valid.append([False] * pad + [True] * len(entries))
            now.compressed = TokenRef(compressed,j)
            self._dirty_records.add(now.index)
            if self.corrected:
                keep, exceptional_flag, priority = decisions[j]
                now.selected = bool(keep)
                self._retention[now.index] = (priority, bool(exceptional_flag), (now.src,now.dst))
                for node, old in removed:
                    self._retain(node, old, now.time)
            else:
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
        short_count = tensor([sum(row_mask) for row_mask in short_valid],device=device)
        if active_short:
            active = tensor(active_short,device=device)
            sv = gather_tokens([value for j in active_short for value in short_rows[j]],zero).reshape(len(active_short),width,64)
            short_meta = tensor([[short_valid[j] for j in active_short],
                                 [short_roles[j] for j in active_short]],device=device,dtype=torch.int64)
            smask = short_meta[0].bool()
            sx = self.short_value(sv) + self.short_time(tensor(
                np.log1p([short_ages[j] for j in active_short]),device=device,dtype=c.dtype)[...,None]).cos()
            sx = sx + self.short_role(short_meta[1])
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
        endpoint_slots = np.asarray([self._entity_slots[u] for u in src+dst],dtype=np.int64)
        required = len(self._entity_ids)
        if self._memory is None:
            capacity = min(self.n_nodes, max(64, 1 << (required - 1).bit_length()))
            self._memory = c.new_zeros(capacity, 128)
        elif required > len(self._memory):
            capacity = min(self.n_nodes, 1 << (required - 1).bit_length())
            self._memory = torch.cat((self._memory, c.new_zeros(capacity-len(self._memory),128)),0)
        cpu_nodes, inverse = np.unique(endpoint_slots,return_inverse=True)
        nodes = torch.from_numpy(cpu_nodes).to(device,non_blocking=True)
        write_content = torch.cat((fused, actual), -1) if self.corrected else content
        ws, wd = self.write_src(write_content), self.write_dst(write_content)
        writes = torch.cat((ws, wd))
        writes = torch.cat((writes, writes), -1)
        strength = self._write_strength
        weights = torch.cat((admission, admission))[:, None] * strength
        decay_args = {}
        if self.corrected and self.memory_half_lives is not None:
            last_seen = [self._node_times.get(self._entity_ids[i],float('nan')) for i in cpu_nodes]
            decay_args = dict(times=times,last_seen=last_seen,half_lives=self._half_lives)
        ms, md, updated = endpoint_scan(self._memory_rows(nodes), inverse[:n], inverse[n:], weights, writes, **decay_args)
        if self.corrected:
            for u,v,t in zip(src,dst,times):
                self._node_times[u] = self._node_times[v] = t
        self._stage_memory(nodes, updated)
        gates = self.endpoint_gate(F.dropout(fused, p=.1, training=self.training)).sigmoid()
        logits = self.head(torch.cat((F.dropout(fused, p=.1, training=self.training), gates[:, :1] * self.read_src(ms), gates[:, 1:] * self.read_dst(md)), -1))
        for removed in evictions:
            for _, old in removed:
                if old.index not in self.history[old.src] and old.index not in self.history[old.dst]:
                    self.records.pop(old.index, None)
                    if self.corrected:
                        self._retention_expired.add(old.index)
        self.last_time = float(times[-1])
        if self.corrected:
            # Both sides live in the same bounded latent space. The target mixes
            # observed attributes and local contextual features, with no label.
            target = F.normalize((c.reshape(n,2,64).mean(1) + actual.reshape(n,2,64).mean(1)).detach(), dim=-1)
            compression_loss = (1 - F.cosine_similarity(compressed, target, dim=-1)).mean()
            # Expired priorities must not grow with the number of stream events.
            # Only queues touched by this batch can drop an ID. Avoid an O(all
            # observed entities) scan in every forward pass.
            short_live = {i for node in set(src+dst) for i,_,_ in self.short[node]}
            for i in self._retention_expired:
                if i not in self.records and i not in short_live:
                    # An event may still be retained by its other, quiet endpoint.
                    endpoints = self._retention.get(i, (0,False,()))[2]
                    if not any(any(j == i for j,_,_ in self.short[u]) for u in endpoints):
                        self._retention.pop(i, None)
            self._retention_expired.clear()
        else:
            compression_loss = F.mse_loss(compressed, embedded[n:][current_indices].detach())
        return logits, dict(admission=admission, novelty=novelty, rare=rare,
                            attention=attention, selected=top, short_count=short_count,
                            compression_loss=compression_loss)
