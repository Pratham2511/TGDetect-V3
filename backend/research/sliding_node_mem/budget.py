"""Require equal detector parameters before either matched training arm starts."""
from orthrus_comparison.official import OrthrusReference
from .model import TGNNSlidingNodeMem

TARGET = 764202


def learned_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def verify_budget():
    # Node histories, memory and neighbor caches are runtime state, not weights.
    # Counts are independent of entity count. Avoid allocating full dataset state.
    reference = OrthrusReference(1)
    proposed = TGNNSlidingNodeMem(1)
    counts = {'ORTHRUS_Reference': learned_count(reference),
              'TGNN_Sliding_Node_Mem': learned_count(proposed)}
    if len(set(counts.values())) != 1 or counts['ORTHRUS_Reference'] != TARGET:
        raise ValueError(f'Trainable parameter budget mismatch: {counts}')
    return dict(status='matched', trainable_parameters=counts,
                sliding_token_buckets=proposed.config['buckets'],
                sliding_decoder_width=proposed.config['head_dim'],
                sliding_token_table_parameters=proposed.event_encoder.tokens.weight.numel(),
                note='Counts include all trainable detector tensors; no unused padding parameters. '
                     'Fixed Word2Vec preprocessing and per-entity runtime state are excluded. '
                     'Equal weight counts do not imply equal compute, state size or capacity.')
