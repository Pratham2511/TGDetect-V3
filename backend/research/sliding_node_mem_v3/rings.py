"""Device-side chronological ring planning, including repeated batch endpoints."""
from dataclasses import dataclass
import torch
from torch import nn


@dataclass
class RingPlan:
    before: torch.Tensor
    final: torch.Tensor
    writes: torch.Tensor


def plan_ring(inverse, seed_valid, write_valid, batch):
    """Indices into [chronological seed rows, src writes, dst writes].

    All reads precede their event's writes. Invalid writes do not consume slots.
    This segmented plan handles arbitrarily many appearances of a node in a
    batch; a gather of just the initial ring does not.
    """
    device = inverse.device
    nodes, width = seed_valid.shape
    prefix = nodes * width
    seed_nodes = torch.arange(nodes, device=device).repeat_interleave(width)
    owner = torch.cat((seed_nodes, inverse))
    clock = torch.cat((torch.arange(width,device=device).repeat(nodes),
                       torch.arange(batch,device=device).repeat(2)+width))
    order = (owner*(batch+width+1)+clock).argsort(stable=True)
    valid = torch.cat((seed_valid.flatten(),write_valid))
    cumulative = valid[order].long().cumsum(0)
    # One variable-size operation per ring, not a host operation per event.
    accepted = order[valid[order]]
    accepted = torch.cat((accepted,order.new_full((1,),-1)))
    ranks = torch.empty_like(order); ranks[order] = cumulative
    prior = ranks[prefix:]-write_valid.long()
    # A self-edge destination shares the source's strictly pre-event read.
    self_edge = inverse[:batch] == inverse[batch:]
    prior[batch:] = torch.where(self_edge,prior[:batch],prior[batch:])
    offsets = torch.arange(width,0,-1,device=device)

    def lookup(end, node):
        positions = end[:,None]-offsets
        safe = positions.clamp(0,len(accepted)-1)
        refs = accepted[safe]
        good = (positions>=0)&(refs>=0)&(owner[refs.clamp_min(0)]==node[:,None])
        return refs.masked_fill(~good,-1)

    before = lookup(prior,inverse)
    end = torch.zeros(nodes,dtype=torch.long,device=device)
    end.scatter_reduce_(0,owner,ranks,reduce='amax',include_self=True)
    final = lookup(end,torch.arange(nodes,device=device))
    writes = torch.zeros(nodes,dtype=torch.long,device=device)
    writes.scatter_add_(0,inverse,write_valid.long())
    return RingPlan(before,final,writes)


def gather(values, indices, fill=0):
    result = values[indices.clamp_min(0)]
    shape = indices.shape+(1,)*(values.ndim-1)
    return torch.where((indices>=0).reshape(shape),result,torch.full_like(result,fill))


def merge_windows(src, dst, width, balanced=False):
    """Chronological deduplicated IDs, optionally reserving four per endpoint."""
    candidates = torch.cat((src,dst),-1).sort(-1).values
    distinct = torch.ones_like(candidates,dtype=torch.bool)
    distinct[:,1:] = candidates[:,1:] != candidates[:,:-1]
    candidates = candidates.masked_fill(~distinct,-1)
    if balanced:
        quota = width//2
        reserved = torch.cat((src[:,-quota:],dst[:,-quota:]),-1)
        mandatory = (candidates[:,:,None]==reserved[:,None,:]).any(-1)&(candidates>=0)
        # Lexicographic priority avoids scaling/overflow with global event IDs.
        order = candidates.argsort(dim=-1,stable=True)
        order = order.gather(1,mandatory.gather(1,order).long().argsort(dim=-1,stable=True))
        return candidates.gather(1,order[:,-width:]).sort(-1).values
    return candidates.sort(-1).values[:,-width:]


class RingState(nn.Module):
    """Fixed storage plus sparse differentiable rows during a training unroll."""
    def __init__(self, nodes, width=8, initial_capacity=16384):
        super().__init__()
        self.nodes,self.width = nodes,width
        self.initial_capacity=min(nodes,initial_capacity)
        self.capacity=self.used=0
        self.register_buffer('slot_map',None,persistent=False)
        self.spec = {
            'hist':((width,5),torch.long,-1), # event, source, destination, identity, first-selected-at
            'hist_t':((width,),torch.float64,0),
            'hist_c':((width,64),torch.float32,0),
            'hist_ptr':((),torch.long,0),
            'short':((width,2),torch.long,-1), # event and admission event (order)
            'short_t':((width,),torch.float64,0),
            'short_c':((width,64),torch.float32,0),
            'short_ptr':((),torch.long,0),
            'memory':((128,),torch.float32,0),
            'last_time':((),torch.float64,float('nan')),
        }
        for name in self.spec:
            self.register_buffer(name,None,persistent=False)
        self.pending = None

    def allocate(self, device, nodes):
        if self.slot_map is None:
            self.slot_map=torch.full((self.nodes,),-1,dtype=torch.long,device=device)
        new=nodes[self.slot_map[nodes]<0]
        required=self.used+len(new)
        if required>self.capacity:
            capacity=min(self.nodes,max(self.initial_capacity,1<<(required-1).bit_length()))
            for name,(shape,dtype,fill) in self.spec.items():
                old=getattr(self,name)
                extra=torch.full((capacity-self.capacity,)+shape,fill,dtype=dtype,device=device)
                setattr(self,name,extra if old is None else torch.cat((old,extra)))
            self.capacity=capacity
        self.slot_map[new]=torch.arange(self.used,required,device=device)
        self.used=required

    def reset(self):
        self.pending = None
        self.used=0
        if self.slot_map is not None:self.slot_map.fill_(-1)
        for name in self.spec:
            value = getattr(self,name)
            if value is not None:
                value.fill_(self.spec[name][2])

    def read(self, nodes):
        slots=self.slot_map[nodes]
        rows = {k:getattr(self,k)[slots] for k in self.spec}
        if self.pending is not None:
            previous,values = self.pending
            at = torch.searchsorted(previous,nodes).clamp_max(len(previous)-1)
            found = previous[at]==nodes
            for k in rows:
                shape = (len(nodes),)+(1,)*(rows[k].ndim-1)
                rows[k] = torch.where(found.reshape(shape),values[k][at],rows[k])
        return rows

    def stage(self, nodes, rows):
        if not torch.is_grad_enabled():
            slots=self.slot_map[nodes]
            for k in rows:
                getattr(self,k).index_copy_(0,slots,rows[k])
            return
        if self.pending is not None:
            previous,values = self.pending
            at = torch.searchsorted(nodes,previous).clamp_max(len(nodes)-1)
            keep = nodes[at]!=previous
            all_nodes = torch.cat((nodes,previous[keep]))
            order = all_nodes.argsort()
            rows = {k:torch.cat((rows[k],values[k][keep]))[order] for k in rows}
            nodes = all_nodes[order]
        self.pending = nodes,rows

    def detach(self):
        if self.pending is not None:
            nodes,rows = self.pending
            slots=self.slot_map[nodes]
            with torch.no_grad():
                for k in rows:
                    getattr(self,k).index_copy_(0,slots,rows[k].detach())
            self.pending = None

    def chronological(self, rows, prefix):
        ptr = rows[prefix+'_ptr']
        order = (ptr[:,None]+torch.arange(self.width,device=ptr.device))%self.width
        return {suffix:rows[prefix+suffix].gather(1,order.reshape(order.shape+(1,)*(rows[prefix+suffix].ndim-2)).expand_as(rows[prefix+suffix]))
                for suffix in ('','_t','_c')}

    def write_ring(self, rows, prefix, payload, plan):
        ptr = (rows[prefix+'_ptr']+plan.writes)%self.width
        # final payload is oldest-to-newest; ptr points at the next overwrite.
        order = (torch.arange(self.width,device=ptr.device)[None]-ptr[:,None])%self.width
        for suffix,values in payload.items():
            values = gather(values,plan.final,fill=-1 if suffix=='' else 0)
            index = order.reshape(order.shape+(1,)*(values.ndim-2)).expand_as(values)
            rows[prefix+suffix] = values.gather(1,index)
        rows[prefix+'_ptr'] = ptr


class CountTable(nn.Module):
    """Fixed GPU count-min table with chronological within-batch updates.

    exact=True uses a collision-free dense table for small parity fixtures.
    It has no LRU eviction and is not a replacement oracle after LRU saturation.
    """
    def __init__(self,nodes,signatures,width=262144,depth=4,exact=False):
        super().__init__()
        if min(nodes,signatures,width,depth)<1:raise ValueError('Positive count table dimensions required')
        self.exact,self.signatures = exact,signatures
        self.width = nodes*signatures if exact else width
        self.depth = 1 if exact else depth
        self.register_buffer('counts',torch.zeros(self.depth,self.width,dtype=torch.long),persistent=False)

    def reset(self):
        self.counts.zero_()

    def forward(self,src,dst,signature):
        batch = len(src);device=src.device
        # Interleaving defines source-then-destination collision semantics.
        nodes=torch.stack((src,dst),1).flatten()
        sig=signature[:,None].expand(-1,2).flatten()
        valid=torch.stack((torch.ones_like(src,dtype=torch.bool),src!=dst),1).flatten()
        key=nodes*self.signatures+sig
        row=torch.arange(self.depth,device=device)[:,None]
        hashed=key[None] if self.exact else ((key[None]*(73856093+row*19349663))^(key[None]>>13)^(83492791*(row+1)))%self.width
        addresses=(hashed+row*self.width).flatten()
        valid=valid.repeat(self.depth)
        order=addresses.argsort(stable=True);bucket=addresses[order]
        increments=valid[order].long();prefix=increments.cumsum(0)
        first=torch.ones_like(valid);first[1:]=bucket[1:]!=bucket[:-1]
        base=torch.where(first,prefix-increments,torch.zeros_like(prefix)).cummax(0).values
        prior_sorted=prefix-base-increments
        prior=torch.empty_like(prior_sorted);prior[order]=prior_sorted
        estimates=self.counts.flatten()[addresses]+prior
        self.counts.flatten().scatter_add_(0,addresses,valid.long())
        result=estimates.reshape(self.depth,-1).amin(0).reshape(batch,2)
        result[:,1]=torch.where(src==dst,result[:,0],result[:,1])
        return result
