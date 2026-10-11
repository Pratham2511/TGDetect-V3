"""TGNN V3 relation-reconstruction adapter and proposed V4 with shared KV encoder."""
import torch
from torch import nn
from node_memory_v3 import TGNNNodeMemV3
from .universal import UniversalEventEncoder


class TGNNV3Reconstruction(TGNNNodeMemV3):
    """V3 backbone with relation logits instead of binary attack logits.

    The native memory admission/entropy rule is retained. Current relation must
    be masked by the caller; history may contain observed past relations.
    """
    def __init__(self, input_dim, n_nodes, num_relations=10):
        super().__init__(input_dim=input_dim, n_nodes=n_nodes)
        self.head = nn.Linear(128, num_relations)


class TGNNNodeMemV4KV(nn.Module):
    """One learned KV encoder shared across sources, followed by V3 graph memory.

    A distinct V4 research branch, not the earlier time-decay V4 checkpoint format.
    Tokenization is independent of labels. History must be strictly past.
    Missing optional fields produce finite embeddings, not guaranteed accuracy.
    """
    def __init__(self, n_nodes, num_relations=10, event_dim=64, **encoder_options):
        super().__init__()
        self.event_encoder = UniversalEventEncoder(dim=event_dim, **encoder_options)
        self.temporal = TGNNV3Reconstruction(event_dim, n_nodes, num_relations)

    def reset_state(self):
        self.temporal.reset_state()

    def detach_state(self):
        self.temporal.detach_state()

    def forward_events(self, histories, ages, src, dst, times):
        """Current query at slot 0, followed by newest-to-oldest past events."""
        device = self.event_encoder.tokens.weight.device
        lengths = [len(h) for h in histories]
        if not lengths or min(lengths) < 1 or len(set(lengths)) != 1:
            raise ValueError('Supply equally padded histories including a current query')
        width = lengths[0]
        events = [e if e is not None else {} for h in histories for e in h]
        x = self.event_encoder(events).reshape(len(histories), width, -1)
        valid = torch.tensor([[e is not None for e in h] for h in histories], device=device)
        ages = torch.as_tensor(ages, dtype=x.dtype, device=device)
        t = torch.as_tensor(times, dtype=torch.float64, device=device)
        if ages.shape != valid.shape or not torch.isfinite(ages).all() or (ages<0).any():
            raise ValueError('Finite nonnegative ages with one value per history slot required')
        if not valid[:,0].all() or (ages[:,0]!=0).any() or ((ages[:,1:]<=0)&valid[:,1:]).any():
            raise ValueError('Slot 0 must be current; valid history slots require strictly positive ages')
        if t.ndim!=1 or len(t)!=len(histories) or len(src)!=len(t) or len(dst)!=len(t):
            raise ValueError('One timestamp and endpoint pair required per event')
        if not torch.isfinite(t).all() or (t[1:] < t[:-1]).any():
            raise ValueError('Finite chronological timestamps required for graph memory')
        if self.temporal.last_time is not None and t[0] < self.temporal.last_time:
            raise ValueError('Reset state before starting a new chronological stream')
        return self.temporal.forward_events(x, valid, ages, src, dst, t)
