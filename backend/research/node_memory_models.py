"""TGNN_Node_Mem: a learned adaptive memory-admission TGNN for streaming APT detection.

Three time scales, deliberately separated:
  * sliding window (seconds-minutes): the causal 5-slot ego graph -> local context c_t,
    produced by the SAME frozen encoder as TGNN_Q_A_C (GAT + time + history-GRU + attention
    + causal temporal CNN).  Reused verbatim so the comparison isolates node memory.
  * persistent per-entity node memory (long-term entity history): one 128-d state per graph
    node (process node = src, host node = dst).  Read before scoring; written AFTER scoring
    with a *learned, differentiable* gate -- the novelty here, replacing the fixed 0.5 EMA of
    the frozen M factor.
  * global replay (training knowledge): a bounded reservoir handled by the TRAINER, kept
    separate from node-memory admission (see modal_dedale_mem.ReplayBuffer).

Admission (spec faithful, v1 minimal core):
  Novelty_i   = 1 - cos(c_i, M_{v_i})                     (per entity, high when unseen)
  Importance  = sigma(w_a . c_i)                          (learned attention/importance proxy)
  Uncertainty = entropy(softmax(logits_i))                (optional, gamma)
  S_i         = alpha.Importance + beta.Novelty + gamma.Uncertainty
  r_t         = r_min + (r_max - r_min).sigmoid(f_phi(s_t))   (adaptive rate, ~0.25 at init)
  g_i         = r_t . sigmoid((S_i - theta) / T)          (soft differentiable write gate)
  M_v'        = (1 - g_i).M_v + g_i . MLP_write(c_i)       (per entity v in {src, dst})

Two parameter groups the trainer optimises separately:
  main_parameters()       (theta): encoder, memory read/write, fusion, head  <- L_task + lambda.L_replay
  controller_parameters() (phi):   f_phi, alpha/beta/gamma, theta_thr, T      <- L_future + budget + redundancy

Efficiency (v2): the per-entity memory is a dense table M[n_nodes,128] and every operation
is vectorised.  Reads/writes run in CHUNKS of `mem_chunk` events (TGN-style batch-granular
memory): all events in a chunk read the memory as of the chunk start and their writes are
fused last-wins, eliminating the per-event Python loop (~50x faster).  Causality is relaxed
from per-event to per-chunk; the gate still earns a task gradient because chunk-0 writes feed
chunk-1.. reads within the same (un-detached) training batch.  The encoder is also trimmed to
`gat_layers` GAT layers (default 2) for the new arch; the baselines keep their own depth.
Set adaptive=False for the fixed-rate ablation control (g_i := r_fixed, no controller).
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from tgnn_components import ComponentTGNN
from temporal_cnn import CausalTemporalCNN, CNNHybrid
from pomsathit_baseline import CausalGATLayer

R_MIN, R_MAX, R_INIT = 0.05, 0.50, 0.25


class AdmissionController(nn.Module):
    """phi: maps a per-event state s_t -> an adaptive write rate r_t in [r_min, r_max],
    and combines (importance, novelty, uncertainty) into a candidate score S_t, finally
    producing the soft gate g_t = r_t * sigmoid((S_t - theta)/T). All phi-only parameters."""

    def __init__(self, state_dim=7, hidden=32, r_min=R_MIN, r_max=R_MAX, r_init=R_INIT):
        super().__init__()
        self.r_min, self.r_max = float(r_min), float(r_max)
        self.net = nn.Sequential(nn.Linear(state_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
        # score weights alpha/beta/gamma kept positive via softplus(raw); init ~[1,1,0.3]
        self.score_raw = nn.Parameter(torch.tensor([0.5413, 0.5413, -1.0], dtype=torch.float32))
        self.threshold = nn.Parameter(torch.tensor(0.5))      # theta
        self.log_temp = nn.Parameter(torch.tensor(0.0))       # log T  (T = exp(log_temp))
        nn.init.zeros_(self.net[-1].weight)
        # bias so sigmoid(bias) maps to r_init on the [r_min, r_max] range
        frac = min(max((r_init - r_min) / (r_max - r_min), 1e-4), 1 - 1e-4)
        nn.init.constant_(self.net[-1].bias, math.log(frac / (1 - frac)))

    def score(self, importance, novelty, uncertainty):
        a, b, g = F.softplus(self.score_raw)
        return a * importance + b * novelty + g * uncertainty

    def rate(self, state):
        return self.r_min + (self.r_max - self.r_min) * torch.sigmoid(self.net(state).squeeze(-1))

    def gate(self, state, importance, novelty, uncertainty):
        r_t = self.rate(state)
        S = self.score(importance, novelty, uncertainty)
        T = self.log_temp.exp().clamp(1e-2, 10.0)
        g = r_t * torch.sigmoid((S - self.threshold) / T)
        return g, r_t, S
class TGNNNodeMem(ComponentTGNN):
    """TGNN_Q_A_C-style encoder (GAT+T+Q+A+C, trimmed to `gat_layers`) + persistent per-entity
    node memory with a learned, differentiable write gate.  adaptive=False -> fixed-rate control.
    Fully vectorised: dense memory table + chunked batch-granular reads/writes (no Python loop)."""

    encode = CNNHybrid.encode  # same local-context encoder family as TGNN_Q_A_C

    def __init__(self, input_dim=72, n_nodes=1, adaptive=True, fixed_rate=R_INIT,
                 mem_chunk=128, gat_layers=2):
        super().__init__(input_dim, ("T", "H", "Q", "A"))
        # user-requested efficiency: trim the inherited 3-layer GAT stack to `gat_layers` (default 2)
        if gat_layers != len(self.layers):
            self.layers = nn.ModuleList(
                [CausalGATLayer(input_dim)] + [CausalGATLayer(128) for _ in range(gat_layers - 1)])
        self.cnn = CausalTemporalCNN()
        self.factors = frozenset(("T", "H", "Q", "A", "C"))
        self.adaptive = bool(adaptive)
        self.fixed_rate = float(fixed_rate)
        self.n_nodes = int(n_nodes)
        self.mem_chunk = int(mem_chunk)
        self.mem_read = nn.Linear(128, 128)
        self.mem_norm = nn.LayerNorm(128)
        self.write_mlp = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 128))
        self.importance = nn.Linear(128, 1)
        self.controller = AdmissionController(state_dim=7)
        # memory read starts at zero -> z == c (no memory influence) until it is learned in
        nn.init.zeros_(self.mem_read.weight); nn.init.zeros_(self.mem_read.bias)
        self.reset_state()

    def reset_state(self):
        self.Mem = None          # dense [n_nodes,128], lazily built on the model's device
        self.Seen = None         # bool [n_nodes]; False -> entity never written (novelty := 1)
        self.last_time = None
        self.memory = {}; self.pending = []   # kept only for inherited-API compatibility

    def detach_state(self):
        if self.Mem is not None:
            self.Mem = self.Mem.detach()

    def _ensure_memory(self, ref):
        if self.Mem is None or self.Mem.device != ref.device:
            self.Mem = ref.new_zeros(self.n_nodes, 128)
            self.Seen = torch.zeros(self.n_nodes, dtype=torch.bool, device=ref.device)

    def main_parameters(self):
        """theta: encoder + memory read/write/fusion + head (trained on task + replay)."""
        return [p for n, p in self.named_parameters()
                if not (n.startswith("controller.") or n.startswith("importance."))]

    def controller_parameters(self):
        """phi: admission controller + importance feature (trained on future + budget + redundancy)."""
        return list(self.controller.parameters()) + list(self.importance.parameters())

    def replay_logits(self, x, valid, ages):
        """Memory-free window classification for GLOBAL replay (anti-forgetting), kept strictly
        separate from the per-entity node-memory stream so out-of-order replay never corrupts it."""
        extra = None
        if self.time_encoder is not None:
            extra = self.time_encoder(ages.unsqueeze(-1)).cos() * valid.unsqueeze(-1)
        return self.head(self.mem_norm(self.encode(x, valid, extra)))

    def _write(self, sidx, didx, g, wc, ms, md):
        """Fuse this chunk's gated writes into the dense memory, last-write-wins per node.
        Differentiable w.r.t. g (phi) and wc (theta); deterministic via a stable sort."""
        flat = torch.stack([sidx, didx], dim=1).reshape(-1)                 # [2c] event-interleaved
        g2 = torch.stack([g, g], dim=1).reshape(-1)                         # [2c]
        wc2 = torch.stack([wc, wc], dim=1).reshape(-1, 128)                 # [2c,128]
        old = torch.stack([ms, md], dim=1).reshape(-1, 128)                 # [2c,128] (chunk-start)
        target = (1 - g2)[:, None] * old + g2[:, None] * wc2               # [2c,128] new values
        order = torch.argsort(flat, stable=True)                           # group equal node ids
        sorted_idx = flat[order]
        keep = torch.ones_like(sorted_idx, dtype=torch.bool)
        keep[:-1] = sorted_idx[1:] != sorted_idx[:-1]                      # last of each group wins
        nodes = sorted_idx[keep]
        self.Mem = self.Mem.index_copy(0, nodes, target[order[keep]])      # unique idx -> deterministic
        self.Seen = self.Seen.index_fill(0, flat, True)

    def forward_events(self, x, valid, ages, src, dst, times, intervention=None):
        """Score each event against its CURRENT per-entity memory, then stage a gated write.
        Events are processed in chunks of `mem_chunk`: a chunk reads the memory as of its start
        and fuses its writes last-wins (batch-granular causality).  Returns (logits[B,2], aux)
        with per-event gate / novelty / rate for the controller's budget & redundancy losses."""
        extra = None
        if self.time_encoder is not None:
            age = torch.zeros_like(ages) if intervention == "constant_time" else ages
            extra = self.time_encoder(age.unsqueeze(-1)).cos() * valid.unsqueeze(-1)
        c = self.encode(x, valid, extra)                       # [B,128] batched (heavy part)
        self._ensure_memory(c)
        imp = torch.sigmoid(self.importance(c.detach())).squeeze(-1)   # [B] controller feature (phi)
        wc = self.write_mlp(c)                                 # [B,128] write content (theta)
        B = c.shape[0]
        s_all = torch.as_tensor(src, device=c.device, dtype=torch.long)
        d_all = torch.as_tensor(dst, device=c.device, dtype=torch.long)
        logits, gates, novelties, rates = [], [], [], []
        for lo in range(0, B, self.mem_chunk):
            hi = min(lo + self.mem_chunk, B)
            sl = slice(lo, hi)
            sidx, didx = s_all[sl], d_all[sl]
            cc = c[sl]
            ms, md = self.Mem[sidx], self.Mem[didx]            # [c,128] read (theta via mem_read)
            seen_s, seen_d = self.Seen[sidx], self.Seen[didx]
            z = cc + self.mem_read(0.5 * (ms + md))
            logit = self.head(self.mem_norm(z))                # [c,2]
            # controller signals are DETACHED encoder/memory outputs: phi reacts to, but never
            # reshapes, the representation.  theta decides WHAT to write (wc); phi HOW MUCH (g).
            ccd, msd, mdd = cc.detach(), ms.detach(), md.detach()
            one = torch.ones(hi - lo, device=c.device)
            nov_s = torch.where(seen_s, 1 - F.cosine_similarity(ccd, msd, dim=1, eps=1e-6), one)
            nov_d = torch.where(seen_d, 1 - F.cosine_similarity(ccd, mdd, dim=1, eps=1e-6), one)
            nov = 0.5 * (nov_s + nov_d)                        # [c]
            if self.adaptive:
                p = logit.detach().softmax(-1)
                unc = -(p * (p + 1e-9).log()).sum(-1) / math.log(2)       # [c]
                state = torch.stack([imp[sl], nov, unc, msd.norm(dim=1), mdd.norm(dim=1),
                                     seen_s.float(), seen_d.float()], dim=1)   # [c,7]
                g, r_t, _ = self.controller.gate(state, imp[sl], nov, unc)    # [c]
            else:
                g = torch.full((hi - lo,), self.fixed_rate, device=c.device); r_t = g
            self._write(sidx, didx, g, wc[sl], ms, md)
            logits.append(logit); gates.append(g); novelties.append(nov); rates.append(r_t)
        self.last_time = float(times[-1]) if len(times) else self.last_time
        aux = {"gate": torch.cat(gates), "novelty": torch.cat(novelties), "rate": torch.cat(rates)}
        return torch.cat(logits), aux


class TGNNNodeMemSel(ComponentTGNN):
    """V2 -- Local SELECTIVE memory admission + persistent node memory + global rehearsal.

    This is the user's actual research idea, implemented as THREE separated stages instead
    of the single soft gate of V1 (TGNN_Node_Mem):
      Stage 1  CANDIDATE SELECTION: events whose learned attention/importance a_i is in the
               top `sel_frac` of the batch, OR whose novelty >= `nov_tau`, enter the pool.
               Soft (sigmoid round the batch quantile) so the attention head earns a gradient;
               novel events are always admitted (set-union, not averaged).
      Stage 2  ADAPTIVE RETENTION FRACTION (phi): admit_i = sigma((S_i - theta)/T) restricted
               to candidates; a controller rate r_t in [R2_MIN, R2_MAX] sets the *target
               fraction of candidates* to keep (budget), decoupled from write strength.
      Stage 3  WRITE STRENGTH: EMA blend weight = admit_i * write_strength; the per-node memory
               transition is applied in CHRONOLOGICAL order (segment scan), so E1->E2->E3 chain
               (M1 from M0, M2 from M1, ...) instead of V1's "last-write-wins off the same M0".

    Reads use batch-start memory (TGN convention, keeps GAT/CNN batched); only the lightweight
    per-node EMA transition is chronological.  The controller (phi) is trained on a future-utility
    signal via TBPTT over K batches (its writes feed later-batch reads inside one un-detached
    window) plus budget + redundancy; everything else is theta.  use_memory=False yields the
    param-identical, 2-GAT TGNN_Q_A_C baseline (same module set, memory path simply bypassed) so
    the two TGNNs match parameter-for-parameter at inference and differ only in the mechanism."""

    encode = CNNHybrid.encode
    replay_logits = TGNNNodeMem.replay_logits
    reset_state = TGNNNodeMem.reset_state
    detach_state = TGNNNodeMem.detach_state
    _ensure_memory = TGNNNodeMem._ensure_memory

    def __init__(self, input_dim=72, n_nodes=1, use_memory=True, gat_layers=2,
                 sel_frac=0.6, nov_tau=0.5, write_strength=0.5,
                 r_min=0.10, r_max=0.30, r_init=0.20):
        super().__init__(input_dim, ("T", "H", "Q", "A"))
        if gat_layers != len(self.layers):
            self.layers = nn.ModuleList(
                [CausalGATLayer(input_dim)] + [CausalGATLayer(128) for _ in range(gat_layers - 1)])
        self.cnn = CausalTemporalCNN()
        self.factors = frozenset(("T", "H", "Q", "A", "C"))
        self.use_memory = bool(use_memory)
        self.n_nodes = int(n_nodes)
        self.sel_frac = float(sel_frac)
        self.nov_tau = float(nov_tau)
        self.write_strength = float(write_strength)
        self.mem_read = nn.Linear(128, 128)
        self.mem_norm = nn.LayerNorm(128)
        self.write_mlp = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 128))
        self.attn_score = nn.Linear(128, 1)   # Stage-1 selection score (importance, theta)
        self.controller = AdmissionController(state_dim=7, r_min=r_min, r_max=r_max, r_init=r_init)
        # memory read starts at zero -> z == c, i.e. identical to the no-memory baseline until learned
        nn.init.zeros_(self.mem_read.weight); nn.init.zeros_(self.mem_read.bias)
        self.reset_state()

    def main_parameters(self):
        """theta: encoder + memory read/write + selection head + fusion + head."""
        return [p for n, p in self.named_parameters() if not n.startswith("controller.")]

    def controller_parameters(self):
        """phi: retention controller only (future-utility + budget + redundancy)."""
        return list(self.controller.parameters())

    def _chrono_write(self, s_all, d_all, w, x):
        """Persist a batch of gated writes in per-node CHRONOLOGICAL order (fixes V1's last-wins
        collapse).  Each event writes BOTH its endpoints (src then dst); the per-node EMA
        M <- (1-w)M + w.x is a first-order linear recurrence, so each node's own event run is
        folded with a SEGMENTED associative scan (Hillis-Steele: ceil(log2 L) fully vectorised
        passes) instead of a Python loop over scan positions.  Identical result to the sequential
        fold, but no per-position full-slice index_copy -- that loop ran once per event touching
        the busiest node and copied the whole [G,128] slice each step, the hub-heavy bottleneck.
        Fully differentiable in (w: phi, x: theta); deterministic (unique sort keys; only
        where/mul/add, no division)."""
        B = s_all.shape[0]
        if B == 0:
            return
        dev = w.device
        ar = torch.arange(B, device=dev)
        nodes = torch.cat([s_all, d_all])
        ordk = torch.cat([2 * ar, 2 * ar + 1])            # src before dst within an event
        w2 = torch.cat([w, w]); x2 = torch.cat([x, x])
        sort_key = nodes.long() * (2 * B + 1) + ordk      # unique -> deterministic chronological order
        perm = torch.argsort(sort_key)
        ns, ws, xs = nodes[perm], w2[perm], x2[perm]
        L = ns.shape[0]
        new_grp = torch.ones_like(ns, dtype=torch.bool)
        new_grp[1:] = ns[1:] != ns[:-1]
        grp = torch.cumsum(new_grp.long(), 0) - 1         # 0..G-1 group id per write
        idx = torch.arange(L, device=dev)
        starts = torch.nonzero(new_grp, as_tuple=False).squeeze(1)
        sizes = torch.diff(torch.cat([starts, idx.new_tensor([L])]))
        max_len = int(sizes.max().item())                 # longest single-node run this batch
        # element transform (a,b): M -> a*M + b.  compose(earlier,later): a=ea*la, b=la*eb+lb.
        a = 1.0 - ws                                      # [L]
        b = ws.unsqueeze(1) * xs                          # [L,128]
        d = 1
        while d < max_len:                                # <= ceil(log2 max_len) vectorised passes
            prev_a = torch.zeros_like(a); prev_a[d:] = a[:-d]
            prev_b = torch.zeros_like(b); prev_b[d:] = b[:-d]
            prev_g = grp.new_full((L,), -1); prev_g[d:] = grp[:-d]
            same = (idx >= d) & (prev_g == grp)           # fold only within one node's own run
            new_a = torch.where(same, prev_a * a, a)
            new_b = torch.where(same.unsqueeze(1), a.unsqueeze(1) * prev_b + b, b)
            a, b = new_a, new_b
            d *= 2
        is_last = torch.ones_like(ns, dtype=torch.bool)
        is_last[:-1] = new_grp[1:]                         # final write of each node run
        uniq = ns[is_last]                                 # [G] distinct nodes (group order)
        M_new = a[is_last].unsqueeze(1) * self.Mem[uniq] + b[is_last]   # a_tot*M0 + b_tot
        self.Mem = self.Mem.index_copy(0, uniq, M_new)     # one scatter of the folded slice
        self.Seen = self.Seen.index_fill(0, nodes, True)

    def forward_events(self, x, valid, ages, src, dst, times, intervention=None):
        extra = None
        if self.time_encoder is not None:
            age = torch.zeros_like(ages) if intervention == "constant_time" else ages
            extra = self.time_encoder(age.unsqueeze(-1)).cos() * valid.unsqueeze(-1)
        c = self.encode(x, valid, extra)                       # [B,128] batched encoder
        if not self.use_memory:                                # param-identical TGNN_Q_A_C baseline
            return self.head(self.mem_norm(c)), {}
        self._ensure_memory(c)
        B = c.shape[0]
        s_all = torch.as_tensor(src, device=c.device, dtype=torch.long)
        d_all = torch.as_tensor(dst, device=c.device, dtype=torch.long)
        ms, md = self.Mem[s_all], self.Mem[d_all]              # batch-start read (TGN convention)
        seen_s, seen_d = self.Seen[s_all], self.Seen[d_all]
        z = c + self.mem_read(0.5 * (ms + md))
        logit = self.head(self.mem_norm(z))                    # [B,2]
        # ---- controller signals are DETACHED: phi reacts to, never reshapes, the representation
        cd = c.detach()
        one = torch.ones(B, device=c.device)
        nov_s = torch.where(seen_s, 1 - F.cosine_similarity(cd, ms.detach(), dim=1, eps=1e-6), one)
        nov_d = torch.where(seen_d, 1 - F.cosine_similarity(cd, md.detach(), dim=1, eps=1e-6), one)
        nov = 0.5 * (nov_s + nov_d)                            # [B]
        p = logit.detach().softmax(-1)
        unc = -(p * (p + 1e-9).log()).sum(-1) / math.log(2)    # [B]
        # ---- Stage 1: candidate pool = top-sel_frac attention  UNION  novel events
        a = torch.sigmoid(self.attn_score(c)).squeeze(-1)      # [B] learned importance (theta)
        q = torch.quantile(a.detach(), 1.0 - self.sel_frac)    # keep the top sel_frac by score
        cand = torch.maximum(torch.sigmoid((a - q) / 0.05),
                             (nov >= self.nov_tau).float())    # soft, novel always included
        # ---- Stage 2: retention (phi).  admit in [0,1] on candidates; r_t = target kept-fraction
        imp = a.detach()
        S = self.controller.score(imp, nov, unc)               # phi weights * [imp,nov,unc]
        T = self.controller.log_temp.exp().clamp(1e-2, 10.0)
        admit = cand * torch.sigmoid((S - self.controller.threshold) / T)   # [B]
        feats = torch.stack([imp, nov, unc, ms.detach().norm(dim=1), md.detach().norm(dim=1),
                             seen_s.float(), seen_d.float()], dim=1)        # [B,7]
        agg = (feats * cand[:, None]).sum(0) / cand.sum().clamp(min=1.0)    # mean over candidates
        r_t = self.controller.rate(agg.unsqueeze(0)).squeeze(0)            # scalar in [R2_MIN,R2_MAX]
        # ---- Stage 3: write strength (theta content, phi admission) in chronological order
        self._chrono_write(s_all, d_all, admit * self.write_strength, self.write_mlp(c))
        self.last_time = float(times[-1]) if len(times) else self.last_time
        aux = {"admit": admit, "cand": cand, "novelty": nov, "rate": r_t}
        return logit, aux


def make_mem_model(arm, input_dim=72, n_nodes=1, mem_chunk=128, gat_layers=2):
    if arm == "TGNN_Node_Mem":
        return TGNNNodeMem(input_dim, n_nodes, adaptive=True, mem_chunk=mem_chunk, gat_layers=gat_layers)
    if arm == "TGNN_Node_Mem_Fixed":
        return TGNNNodeMem(input_dim, n_nodes, adaptive=False, mem_chunk=mem_chunk, gat_layers=gat_layers)
    if arm == "TGNN_Node_Mem_Sel":
        return TGNNNodeMemSel(input_dim, n_nodes, use_memory=True, gat_layers=gat_layers)
    if arm == "TGNN_Q_A_C":   # param-identical 2-GAT baseline: same modules, memory path bypassed
        return TGNNNodeMemSel(input_dim, n_nodes, use_memory=False, gat_layers=gat_layers)
    raise ValueError(arm)
