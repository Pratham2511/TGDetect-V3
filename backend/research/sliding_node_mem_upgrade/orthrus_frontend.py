"""Exact authors' ORTHRUS encoder/graph feed with TGDetect temporal processing."""
import torch
from torch import nn
from orthrus_comparison.official import OrthrusReference
from .model import TGNNSlidingNodeMem


class GraphEventBridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(128, 64)
        self.relations = nn.Embedding(10, 64)

    def forward(self, events):
        if not events:
            return self.projection.weight.new_empty((0,64))
        features = torch.stack([event['_graph_features'] for event in events])
        values = self.projection(features)
        relation = torch.tensor([int(event['type']) if event.get('type') is not None else 0
                                 for event in events], device=values.device)
        observed = torch.tensor([event.get('type') is not None for event in events],
                                device=values.device, dtype=values.dtype)
        return values + self.relations(relation) * observed[:,None]


class OrthrusSlidingNodeMem(TGNNSlidingNodeMem):
    def __init__(self, n_nodes, device='cpu'):
        super().__init__(n_nodes)
        reference = OrthrusReference(n_nodes, device=device)
        self.frontend = reference.encoder
        self.event_encoder = GraphEventBridge()
        self.config['frontend'] = 'exact_authors_orthrus_encoder_and_neighbor_loader'
        self.config['history_features'] = 'graph outputs at observation time, detached across optimizer steps'
        self._graph_dirty = []
        self.to(device)

    def reset_state(self):
        super().reset_state()
        self._graph_dirty = []
        if hasattr(self, 'frontend'):
            self.frontend.reset_state()
            reindexer = self.frontend.graph_reindexer
            reindexer.x_src_cache = reindexer.x_dst_cache = None

    def encode_graph_batch(self, events, src, dst, times, messages, full_data):
        device = self.project.weight.device
        src = torch.as_tensor(src, device=device); dst = torch.as_tensor(dst, device=device)
        hs, hd = self.frontend(torch.stack((src,dst)), torch.as_tensor(times,device=device), None,
                              (messages[:,:131].to(device), messages[:,141:].to(device)),
                              full_data, inference=True)
        features = torch.cat((hs,hd),-1)
        return [dict(event, _graph_features=features[i]) for i,event in enumerate(events)]

    def forward_events(self, *args, **kwargs):
        first = self.next_index
        result = super().forward_events(*args, **kwargs)
        self._graph_dirty.extend(range(first,self.next_index))
        return result

    def detach_state(self):
        super().detach_state()
        for index in self._graph_dirty:
            record = self.records.get(index)
            if record is not None:
                record.event['_graph_features'] = record.event['_graph_features'].detach()
        self._graph_dirty.clear()

    def stream_state_dict(self):
        raise RuntimeError('ORTHRUS frontend arm checkpoints at epoch boundaries; no partial graph-state snapshots')
