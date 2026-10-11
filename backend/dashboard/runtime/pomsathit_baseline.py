"""Canonical first-session skeleton, deliberately not an invented reproduction.

See BASELINE_PAPER_SPEC.md. Unknown architecture entries are explicit nulls.
No training or cloud work is triggered by importing this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


class UnresolvedPaperSpecification(RuntimeError):
    """Raised rather than silently substituting an adapted model."""


@dataclass(frozen=True)
class PaperSpecification:
    gnn_layers: int = 3
    hidden_dim: int = 128
    activation: str = 'ReLU'
    dropout: float = 0.2
    aggregation_family: str = 'attention'
    prediction_target: str = 'node'
    output_activation: str = 'softmax'
    optimizer: str = 'Adam'
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    epochs_per_scenario: int = 200
    seed_count: int = 5
    rehearsal_fraction: float = 0.1
    buffer_update: str = 'reservoir'
    generative_replay_in_proposed_model: bool = False


UNRESOLVED = (
    'original_dataset_and_split', 'input_dimension_and_feature_encoder',
    'node_label_mapping_and_class_count', 'graph_window_and_edge_semantics',
    'attention_equation_heads_and_self_loops', 'dropout_locations',
    'loss_and_class_weighting', 'rehearsal_sample_unit_and_capacity_schedule',
    'replay_sampling_and_lambda', 'initialization_and_seed_values',
    'validation_and_threshold_protocol',
)


def architecture_manifest():
    return {
        'schema_version': 1,
        'model_id': 'POMSATHIT_PAPER_SPEC',
        'status': 'BLOCKED_UNRESOLVED_SPECIFICATION',
        'implementation_stage': 'skeleton_only',
        'paper_doi': '10.1109/ACCESS.2025.3639270',
        'paper_explicit': asdict(PaperSpecification()),
        'parameter_count': None, 'trainable_parameter_count': None,
        'input_dimensions': ['N', None], 'output_dimensions': ['N', None],
        'tensor_shapes': [
            {'stage': 'input', 'shape': ['N', None]},
            *[{'stage': f'gnn_{i}', 'shape': ['N', 128]} for i in range(1, 4)],
            {'stage': 'node_head', 'shape': ['N', None]},
        ],
        'trainable_modules': None,
        'aggregation_equation': 'h_next[v] = sigma(W * AGGREGATE({h[v]} union {h[u]: u in N(v)}))',
        'attention_equation': None, 'activation_locations': 'generic Eq.3; exact operator unresolved',
        'dropout_locations': None,
        'optimizer_settings': {'name': 'Adam', 'lr': 0.001, 'weight_decay': 0.00001, 'betas': None, 'eps': None},
        'rehearsal_settings': {'fraction': 0.1, 'update': 'reservoir', 'unit': None, 'lambda': None, 'sampling': None},
        'random_seed_policy': {'independent_runs': 5, 'values': None},
        'unresolved': list(UNRESOLVED),
        'certified_executable_architecture': False,
    }


class PomsathitPaperSpec:
    """Reserved model API. Construction is blocked until the spec is resolved."""

    def __init__(self):
        raise UnresolvedPaperSpecification('POMSATHIT_PAPER_SPEC unresolved: ' + ', '.join(UNRESOLVED))


import torch
from torch import nn
from torch.nn import functional as F


class CausalGATLayer(nn.Module):
    """Single-head additive attention on independent, masked ego graphs."""

    def __init__(self, in_dim, hidden=128, dropout=.2):
        super().__init__()
        self.projection = nn.Linear(in_dim, hidden)
        self.a_src = nn.Parameter(torch.empty(hidden))
        self.a_dst = nn.Parameter(torch.empty(hidden))
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        nn.init.xavier_uniform_(self.a_src.view(1, -1))
        nn.init.xavier_uniform_(self.a_dst.view(1, -1))

    def forward(self, x, adjacency, extra=None):
        z = self.projection(x)
        if extra is not None:
            z = z + extra
        scores = F.leaky_relu((z @ self.a_dst).unsqueeze(2) + (z @ self.a_src).unsqueeze(1), .2)
        weights = scores.masked_fill(~adjacency, float('-inf')).softmax(-1)
        return self.dropout(F.relu(weights @ z))


class PomsathitAdapted(nn.Module):
    """Explicit event-node adaptation; see ADAPTED_PROTOCOL.md, never paper-exact."""

    def __init__(self, input_dim=None):
        super().__init__()
        if input_dim is None:
            raise UnresolvedPaperSpecification('An explicit adapted input dimension is required')
        self.input_dim = input_dim
        self.layers = nn.ModuleList([CausalGATLayer(input_dim), CausalGATLayer(128), CausalGATLayer(128)])
        self.head = nn.Linear(128, 2)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def encode(self, x, valid, extra=None):
        batch, nodes, _ = x.shape
        adjacency = torch.eye(nodes, dtype=torch.bool, device=x.device).expand(batch, -1, -1).clone()
        adjacency[:, 0, :] = valid
        # Padding self-loops avoid empty-softmax NaNs, but no valid node reads it.
        for layer_id, layer in enumerate(self.layers):
            x = layer(x, adjacency, extra if layer_id == 0 else None)
        return x[:, 0]

    def forward(self, x, valid):
        return self.head(self.encode(x, valid))

    def adapted_manifest(self):
        return {
            'model_id': 'POMSATHIT_ADAPTED', 'protocol': 'ADAPTED_PROTOCOL.md',
            'paper_exact': False, 'input_dim': self.input_dim, 'output_dim': 2,
            'parameter_count': sum(p.numel() for p in self.parameters()),
            'trainable_modules': {n: {'shape': list(p.shape), 'parameters': p.numel()}
                                  for n, p in self.named_parameters()},
            'tensor_shapes': [['B', 5, self.input_dim], *[['B', 5, 128] for _ in range(3)], ['B', 2]],
            'aggregation': 'softmax_j(LeakyReLU(a_dst.W.h_i+a_src.W.h_j,.2)) weighted W.h_j',
            'activation': 'ReLU after aggregation, each of 3 layers',
            'dropout': '.2 after ReLU, each layer; no attention dropout',
            'optimizer': {'name': 'Adam', 'lr': .001, 'weight_decay': .00001, 'betas': [.9,.999], 'eps': 1e-8},
            'seeds': [0,1,2,3,4], 'pilot_epochs': 3, 'paper_epochs': 200,
            'batch_size': 256, 'loss': 'unweighted cross entropy',
            'rehearsal': 'optional H, .1 capacity Algorithm R; see adapted protocol',
        }


if __name__ == '__main__':
    path = Path(__file__).with_name('BASELINE_ARCHITECTURE.json')
    path.write_text(json.dumps(architecture_manifest(), indent=2) + '\n')
    print(f'Wrote {path.name}: blocked skeleton, no invented parameter count.')
