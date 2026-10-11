"""Load unmodified model class definitions from the pinned authors' release.

AST extraction omits broad database/plotting imports, not model statements.
The end-to-end reference remains research/orthrus_official/src/orthrus.py.
"""
import ast
from pathlib import Path
from typing import Tuple
import torch
from torch import nn, Tensor
from torch_geometric.nn import TransformerConv
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1] / 'orthrus_official'


def classes():
    namespace = dict(torch=torch, nn=nn, Tensor=Tensor, Tuple=Tuple, TransformerConv=TransformerConv)
    definitions = {
        'src/encoders.py': {'GraphTransformer','OrthrusEncoder'},
        'src/decoders.py': {'EdgeTypeDecoder'},
        'src/temporal.py': {'LastNeighborLoader'},
        'src/data_utils.py': {'GraphReindexer'},
    }
    for path, names in definitions.items():
        tree = ast.parse((ROOT/path).read_text())
        selected = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
        if {n.name for n in selected} != names:
            raise ValueError('Upstream class definitions changed')
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(ROOT/path), 'exec'), namespace)
    return namespace


class OrthrusReference(nn.Module):
    def __init__(self, n_nodes, input_dim=131, relations=10, device='cpu'):
        super().__init__()
        c = classes()
        self.reindexer = c['GraphReindexer'](n_nodes, device)
        graph = c['GraphTransformer'](100,128,64,relations,.25,nn.ReLU(),8)
        self.encoder = c['OrthrusEncoder'](graph,c['LastNeighborLoader'](n_nodes,20,device),
            input_dim,100,True,self.reindexer,['edge_type'],device,n_nodes,relations)
        self.decoder = c['EdgeTypeDecoder'](64,relations,self.loss,0.,2,nn.ReLU())
        self.to(device)

    @staticmethod
    def loss(logits, target, inference=False):
        return F.cross_entropy(logits,target,reduction='none' if inference else 'mean')

    def reset_state(self):
        self.encoder.reset_state()
        self.reindexer.x_src_cache = self.reindexer.x_dst_cache = None

    def forward(self, src, dst, times, x_src, x_dst, relation_onehot, full_data, inference=False):
        h_src, h_dst = self.encoder(torch.stack((src,dst)), times, None,
                                   (x_src,x_dst),full_data,inference=inference)
        return self.decoder(h_src,h_dst,relation_onehot,inference=inference)
