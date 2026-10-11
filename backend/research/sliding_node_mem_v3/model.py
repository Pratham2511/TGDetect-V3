"""Tensor-ring TGNN. No Python loop over events in the forward path."""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from sliding_node_mem_upgrade.model import TGNNSlidingNodeMem as Reference
from sliding_node_mem_v2.layers import TemporalCNN
from .memory import endpoint_scan
from .rings import RingState,CountTable,plan_ring,gather,merge_windows
from .tokens import PackedEncoder
from .temporal import CausalAttention
from .graphs import TemporalGraphs


class TGNNSlidingNodeMem(Reference):
    def __init__(self,n_nodes,bank,num_relations=10,signatures=1,profile='corrected',
                 count_width=262144,count_depth=4,exact_counts=False,gat_layers=None,
                 tied_gat=False,pair_head=None,temperature=.15,initial_capacity=16384,
                 memory_half_lives=(3600.,86400.),short_readout='cnn',static_short_read=False,temporal='cnn',temporal_graphs=False,**kwargs):
        if profile not in ('compatible','corrected') or temperature<=0:
            raise ValueError('Valid profile and positive retention temperature required')
        if kwargs.get('window',9)!=9 or kwargs.get('short_slots',8)!=8:
            raise ValueError('V3 implements eight-slot endpoint rings and nine-slot windows')
        kwargs.pop('buckets',None)
        super().__init__(n_nodes,num_relations=num_relations,buckets=bank.config['buckets'],**kwargs)
        self.corrected=profile=='corrected'
        if short_readout not in ('cnn','attention','none'):
            raise ValueError('short_readout must be cnn, attention or none')
        self.short_readout=short_readout
        self.static_short_read=static_short_read
        if temporal_graphs and not static_short_read:raise ValueError('Temporal graphs require static_short_read')
        self.temporal_graphs=TemporalGraphs() if temporal_graphs else None
        if not self.corrected and (bank.config['two_hash'] or bank.config['ordered']):
            raise ValueError('Parity profile requires legacy single-hash token rows')
        self.gat_layers=gat_layers if gat_layers is not None else (2 if self.corrected else 1)
        if self.gat_layers not in (1,2):raise ValueError('One or two GAT layers required')
        self.tied_gat=tied_gat
        self.gat_second=copy.deepcopy(self.gat) if self.gat_layers==2 and not tied_gat else None
        self.event_encoder=PackedEncoder(bank)
        if temporal not in ('cnn','attention'):raise ValueError('temporal must be cnn or attention')
        self.cnn=TemporalCNN(backend='auto') if temporal=='cnn' else CausalAttention()
        if initial_capacity<1:raise ValueError('Positive initial ring capacity required')
        self.state=RingState(n_nodes,initial_capacity=initial_capacity)
        self.counter=CountTable(n_nodes,signatures,count_width,count_depth,exact_counts)
        self.pair_head=nn.Bilinear(64,64,1) if (self.corrected if pair_head is None else pair_head) else None
        self.temperature=temperature
        if memory_half_lives is not None and (len(memory_half_lives)!=2 or any(not math.isfinite(h) or h<=0 for h in memory_half_lives)):
            raise ValueError('Two positive finite half-lives or None required')
        self.register_buffer('half_lives',None if memory_half_lives is None or not self.corrected else
                             torch.tensor([memory_half_lives[0]]*64+[memory_half_lives[1]]*64),persistent=False)
        self.register_buffer('write_strength',torch.tensor([.5]*64+[.005]*64),persistent=False)
        self.config.update(version='tensor-rings-v3',profile=profile,signatures=signatures,
            count_width=count_width,count_depth=count_depth,exact_counts=exact_counts,
            gat_layers=self.gat_layers,tied_gat=tied_gat,pair_head=self.pair_head is not None,
            temperature=temperature,initial_capacity=initial_capacity,memory_half_lives=memory_half_lives,token_config=bank.config,
            short_readout=short_readout,static_short_read=static_short_read,temporal=temporal,temporal_graphs=temporal_graphs)
        self.reset_state()

    def reset_state(self):
        self.next_index=0;self.last_time=None
        if hasattr(self,'state'):self.state.reset();self.counter.reset()

    def detach_state(self):
        self.state.detach()

    def forward_events(self,*args,**kwargs):
        raise TypeError('V3 consumes offline token IDs: use forward_batch(batch)')

    def _temporal(self,x,valid):
        return self.cnn(x,valid) if self.temporal_graphs is None else self.temporal_graphs(self.cnn,x,valid)

    def stream_state_dict(self):
        self.detach_state()
        return dict(config=self.config,next_index=self.next_index,last_time=None if self.last_time is None else self.last_time.cpu(),
                    used=self.state.used,slot_map=None if self.state.slot_map is None else self.state.slot_map.cpu().clone(),
                    rings={k:None if getattr(self.state,k) is None else getattr(self.state,k)[:self.state.used].detach().cpu().clone() for k in self.state.spec},
                    counts=self.counter.counts.detach().cpu().clone())

    def load_stream_state_dict(self,saved):
        if saved['config']!=self.config:raise ValueError('V3 stream configuration mismatch')
        self.reset_state();device=self.project.weight.device
        self.state.slot_map=None if saved['slot_map'] is None else saved['slot_map'].to(device).clone()
        self.state.used=self.state.capacity=saved['used']
        for k,v in saved['rings'].items():
            setattr(self.state,k,None if v is None else v.to(device).clone())
        self.counter.counts.copy_(saved['counts'].to(device))
        self.next_index=saved['next_index']
        self.last_time=None if saved['last_time'] is None else saved['last_time'].to(device)

    def forward_batch(self,batch):
        device=self.project.weight.device
        data={k:torch.as_tensor(v,device=device) for k,v in batch.items()}
        src,dst,times,relations=(data[k] for k in ('src','dst','time','relation'))
        n=len(src)
        if not n:raise ValueError('Empty batch')
        if any(len(v)!=n for v in data.values()):raise ValueError('Mismatched batch columns')
        valid_input=(src>=0).all()&(dst>=0).all()&(src<self.n_nodes).all()&(dst<self.n_nodes).all()
        valid_input &= torch.isfinite(times).all()&(times[1:]>=times[:-1]).all()
        valid_input &= (relations>=0).all()&(relations<self.num_relations).all()
        valid_input &= (data['signature']>=0).all()&(data['signature']<self.counter.signatures).all()
        if self.last_time is not None:valid_input &= times[0]>=self.last_time
        torch._assert_async(valid_input,'Invalid endpoints, relations or chronological timestamps')
        ids=torch.arange(n,device=device)+self.next_index
        endpoints=torch.cat((src,dst));nodes,inverse=torch.unique(endpoints,sorted=True,return_inverse=True)
        self.state.allocate(device,nodes);rows=self.state.read(nodes)
        old=self.state.chronological(rows,'hist')
        infinite=2**60
        current=torch.stack((ids,src,dst,data['observed'],torch.full_like(ids,infinite)),-1)
        hist_meta=torch.cat((old[''].reshape(-1,5),current,current))
        hist_time=torch.cat((old['_t'].flatten(),times,times))
        write_valid=torch.cat((torch.ones_like(src,dtype=torch.bool),src!=dst))
        hp=plan_ring(inverse,old[''][...,0]>=0,write_valid,n)
        incident=gather(hist_meta[:,0],hp.before,fill=-1)
        past=merge_windows(incident[:n],incident[n:],8,self.corrected)
        window_ids=torch.cat((past,ids[:,None]),1)
        # Unique event bank; a sentinel guarantees safe padded lookups.
        raw_bank=torch.cat((current.new_full((1,5),-1),old[''].reshape(-1,5),current))
        bank_ids,bank_inverse=torch.unique(raw_bank[:,0],sorted=True,return_inverse=True)
        first=torch.full_like(bank_ids,len(raw_bank))
        first.scatter_reduce_(0,bank_inverse,torch.arange(len(raw_bank),device=device),reduce='amin',include_self=True)
        meta=raw_bank[first]
        bank_times=torch.cat((times.new_zeros(1),old['_t'].flatten(),times))[first]
        current_at=torch.searchsorted(bank_ids,ids)
        wi=torch.searchsorted(bank_ids,window_ids)
        want=[data['query'],meta[:,3].clamp_min(0)]
        if self.pair_head is not None:want += [data['src_identity'],data['dst_identity'],data['negative_identity']]
        encoded=self.event_encoder(torch.cat(want))
        query=self.project(encoded[:n]);observed_emb=encoded[n:n+len(meta)]
        observed=self.project(observed_emb)
        actual=observed[current_at]
        raw=observed[wi];valid=window_ids>=0
        x=torch.cat((raw[:,:-1],query[:,None]),1)
        ages=(times[:,None]-bank_times[wi]).masked_fill(~valid,0)
        x=(x+self.time(ages.log1p().to(x.dtype)[...,None]).cos())*valid[...,None]
        h=self.gat(x,meta[wi,1],meta[wi,2],valid)
        if self.gat_layers==2:
            h=(self.gat if self.tied_gat else self.gat_second)(h,meta[wi,1],meta[wi,2],valid)
        h=self._temporal(h,valid)
        scores=(self.importance_query(h[:,-1])[:,None]*self.importance_key(h)).sum(-1)/math.sqrt(128)
        attention=scores.masked_fill(~valid,-torch.inf).softmax(-1)
        c=h[:,-1]+(attention[...,None]*h).sum(1)
        similarity=F.cosine_similarity(actual[:,None].detach(),raw[:,:-1].detach(),dim=-1).masked_fill(~valid[:,:-1],-1)
        novelty=(1-similarity.amax(-1)).clamp(0,2)
        rare=(self.counter(src,dst,data['signature'])<self.rare_count).any(-1)
        exceptional=rare|(novelty>=self.novelty_threshold)
        gate=self.admission(torch.stack((attention[:,-1],novelty,rare.to(c.dtype)),-1)).sigmoid().squeeze(-1)
        # first_selected prevents a later query from retroactively admitting an
        # earlier eviction. Compatibility reproduces the old sticky rule exactly.
        selected_at=meta[:,4].clone();selected_at=selected_at.masked_fill(bank_ids<0,infinite)
        if self.corrected:
            selected_at=selected_at.scatter_reduce(0,current_at,torch.where(exceptional,ids,infinite),reduce='amin',include_self=True)
            soft_keep=torch.where(rare,torch.ones_like(novelty),torch.sigmoid((novelty-self.novelty_threshold)/self.temperature))
            admission=gate*soft_keep
            top=exceptional[:,None]
        else:
            ranking=scores.masked_fill(~valid,-torch.inf).argsort(dim=-1,descending=True,stable=True).argsort(-1)
            top=(ranking<((valid.sum(-1)+1)//2)[:,None])&valid
            when=ids[:,None].expand_as(wi).masked_fill(~top,infinite)
            selected_at=selected_at.scatter_reduce(0,wi.flatten(),when.flatten(),reduce='amin',include_self=True)
            selected_at=selected_at.scatter_reduce(0,current_at,torch.where(exceptional,ids,infinite),reduce='amin',include_self=True)
            admission=.05+.95*gate*(top[:,-1]|exceptional).to(c.dtype)
        content=torch.cat((c,actual),-1);compressed=self.compress(content)
        bank_c=torch.cat((compressed.new_zeros(1,64),old['_c'].reshape(-1,64),compressed))[first]
        # Evictions are committed AFTER the current short-memory read.
        evicted=gather(hist_meta,hp.before[:,0],fill=-1)
        ev_at=torch.searchsorted(bank_ids,evicted[:,0].contiguous())
        admitted=(evicted[:,0]>=0)&(selected_at[ev_at]<=ids.repeat(2))&write_valid
        old_short=self.state.chronological(rows,'short')
        sp=plan_ring(inverse,old_short[''][...,0]>=0,admitted,n)
        short_meta=torch.cat((old_short[''].reshape(-1,2),torch.stack((evicted[:,0],ids.repeat(2)),-1)))
        short_c=torch.cat((old_short['_c'].reshape(-1,64),bank_c[ev_at]))
        short_t=torch.cat((old_short['_t'].flatten(),bank_times[ev_at]))
        short_incident=gather(short_meta[:,0],sp.before,fill=-1)
        chosen=merge_windows(short_incident[:n],short_incident[n:],9)
        candidates=torch.cat((short_incident[:n],short_incident[n:]),1)
        candidate_refs=torch.cat((sp.before[:n],sp.before[n:]),1)
        match=chosen[:,:,None]==candidates[:,None,:]
        positions=match.long().argmax(-1)
        refs=candidate_refs.gather(1,positions)
        smask=chosen>=0
        refs=refs.masked_fill(~smask,-1)
        roles=torch.where(match[:,:,:8].any(-1),0,1)
        roles=torch.where(match[:,:,:8].any(-1)&match[:,:,8:].any(-1),2,roles)
        active=torch.arange(n,device=device) if self.static_short_read else smask.any(-1).nonzero().flatten()
        short_context=c.new_zeros(n,128)
        if len(active) and self.short_readout!='none':
            sv=gather(short_c,refs[active]);st=gather(short_t,refs[active]);sm=smask[active]
            age=(times[active,None]-st).masked_fill(~sm,0)
            sx=self.short_value(sv)+self.short_time(age.log1p().to(c.dtype)[...,None]).cos()+self.short_role(roles[active])
            sh=self._temporal(sx*sm[...,None],sm) if self.short_readout=='cnn' else sx*sm[...,None]
            sk=self.short_key(sv)+sh
            sa=((sk*c[active,None]).sum(-1)/math.sqrt(128)).masked_fill(~sm,-1e9).softmax(-1)*sm
            pooled=(sa[...,None]*sh).sum(1)
            short_context=pooled if self.static_short_read else short_context.index_copy(0,active,pooled)
        fused=self.fusion_norm(c+self.fusion(torch.cat((c,short_context),-1)).sigmoid()*short_context)
        write_content=torch.cat((fused,actual),-1) if self.corrected else content
        writes=torch.cat((self.write_src(write_content),self.write_dst(write_content)))
        weights=admission.repeat(2)[:,None]*self.write_strength
        ms,md,updated=endpoint_scan(rows['memory'],inverse[:n],inverse[n:],weights,writes.repeat(1,2),
                                  times=times,last_seen=rows['last_time'],half_lives=self.half_lives)
        gates=self.endpoint_gate(F.dropout(fused,p=.1,training=self.training)).sigmoid()
        logits=self.head(torch.cat((F.dropout(fused,p=.1,training=self.training),gates[:,:1]*self.read_src(ms),gates[:,1:]*self.read_dst(md)),-1))
        # Commit final ring rows in one scatter per tensor, with unique node IDs.
        hist_at=torch.searchsorted(bank_ids,hist_meta[:,0].contiguous())
        hist_meta=torch.cat((hist_meta[:,:4],selected_at[hist_at,None]),1)
        self.state.write_ring(rows,'hist',{'':hist_meta,'_t':hist_time,'_c':bank_c[hist_at]},hp)
        self.state.write_ring(rows,'short',{'':short_meta,'_t':short_t,'_c':short_c},sp)
        rows['memory']=updated
        rows['last_time']=torch.full_like(rows['last_time'],-torch.inf).scatter_reduce(0,inverse,times.repeat(2),reduce='amax',include_self=True)
        self.state.stage(nodes,rows)
        if not self.corrected:
            # Legacy Record.selected is shared by BOTH endpoints, including a
            # quiet endpoint absent from this batch. Propagate changed flags to
            # those ring copies before losing the busy endpoint's active copy.
            changed=(bank_ids>=0)&(selected_at<meta[:,4])
            affected=torch.unique(meta[changed,1:3].flatten(),sorted=True)
            at=torch.searchsorted(nodes,affected).clamp_max(len(nodes)-1)
            extra=affected[nodes[at]!=affected]
            if len(extra):
                other=self.state.read(extra)
                old_ids=other['hist'][...,0]
                pos=torch.searchsorted(bank_ids,old_ids.contiguous()).clamp_max(len(bank_ids)-1)
                matched=(bank_ids[pos]==old_ids)&(old_ids>=0)
                flags=torch.where(matched,torch.minimum(other['hist'][...,4],selected_at[pos]),other['hist'][...,4])
                other['hist']=torch.cat((other['hist'][...,:4],flags[...,None]),-1)
                self.state.stage(extra,other)
        self.next_index+=n;self.last_time=times[-1].detach()
        target=F.normalize((c.reshape(n,2,64).mean(1)+actual.reshape(n,2,64).mean(1)).detach(),dim=-1)
        compression_loss=(1-F.cosine_similarity(compressed,target,dim=-1)).mean() if self.corrected else F.mse_loss(compressed,observed_emb[current_at].detach())
        aux=dict(compression_loss=compression_loss,admission=admission,novelty=novelty,rare=rare,
                 attention=attention,selected=top,short_count=smask.sum(-1),window_ids=window_ids)
        if self.pair_head is not None:
            left,right,negative=encoded[n+len(meta):].split(n)
            pos=self.pair_head(left,right).squeeze(-1);neg=self.pair_head(left,negative).squeeze(-1)
            mask=data['negative_valid'].to(pos.dtype)
            # Unknown/ambiguous negatives contribute neither positive nor negative
            # training examples, avoiding an all-positive collapse on tiny graphs.
            aux['pair_loss']=((F.softplus(-pos)+F.softplus(neg))*mask).sum()/(2*mask.sum().clamp_min(1))
            aux['pair_anomaly']=F.softplus(-pos)
        return logits,aux
