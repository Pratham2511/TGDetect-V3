# What the new comparison can establish

The implemented reference uses the unchanged ORTHRUS graph transformer and relation decoder from the authors' pinned release. V3 keeps its history/graph/memory backbone and replaces binary attack classification with benign relation reconstruction. V4-KV adds one shared event encoder in front of that backbone. All three consume the same prepared CADETS graphs, dates, endpoint attributes, and relation targets.

| Property | ORTHRUS reference | V3 reconstruction | V4-KV |
|---|---|---|---|
| Event attributes | Released Word2Vec vectors | Same Word2Vec vectors | Typed unordered key-value set |
| Current relation | Hidden from predictor | Hidden from current query | Hidden from current query |
| History | Last 20 entity neighbors | Five event slots and entity memory | Five event slots and entity memory |
| Persistent learned entity summary | No recurrent node memory | Fast/slow memory and novelty reference | Same memory mechanism |
| Encoder fitting | Transductive Word2Vec | Same transductive Word2Vec | Shared encoder fitted on benign training |
| Registered parameters | 764,202 | 350,516 | 1,406,516 |
| Optional fields | Fixed released feature contract | Fixed vector width | Omitted or explicitly missing fields |
| Alerts | Relation reconstruction loss | Relation reconstruction loss | Relation reconstruction loss |
| Attribution | Released dependency tracing | Same tracing of V3 alerts | Same tracing of V4 alerts |

## Why V3 can help relative to a local GRU

A GRU that is reset for each five-event window can summarize only that window. V3 also retains a summary for each entity between windows. Separate source and destination reads retain role information, and fast/slow write strengths provide different history horizons. Its causal scan reads the pre-event state before incorporating the current event. A learned admission scorer can reduce the influence of repeated low-value activity.

These mechanisms explain plausible advantages; they do not prove the cause of an observed improvement. Earlier DEDALE results came from a different supervised protocol with warm starts. The new CADETS experiment tests benign reconstruction and must be assessed on its own results. Memory can also preserve misleading activity, dilute rare events, or lose useful context when identities change. Gradients are detached between training batches, so long retained history does not imply equally long gradient credit.

## What V4 changes

Each event becomes a set of typed pairs. The same bounded token table handles keys and value subtokens, a shared projection handles numerical magnitude, and attention combines available pairs into a 64-dimensional event token. There is no learned encoder selected by Windows, network, cloud, or another source. Field positions are absent, preserving dictionary-order invariance. Empty events retain a reserved type token, which prevents an all-masked attention operation.

This removes the need to expand a fixed vector whenever a source introduces a new field. It also lets one training path observe different log sources. It does not automatically correlate entities across sources: the identity policy scopes identifiers by source and host, and unknown IDs remain event-local. Actual cross-source identities need explicit, auditable correlation rules.

Missing optional data produces finite representations, but missing evidence can still reduce detection accuracy. The event encoder is trainable, so an unfamiliar source or key can be poorly represented until benign examples are available. Hashing bounds the vocabulary and creates collisions. Field and subtoken limits can drop information. Value subtokens are pooled as a bag in this first implementation, so their ordering is not retained; commands and paths that differ only in token order can collide semantically. This is a limitation to test with a shared position-sensitive value encoder in a later ablation.

The matched experiment intentionally uses the original V3 admission rule. Its entropy normalization was designed for two classes, and its auxiliary rate network receives no gradient under cross-entropy alone. A later controlled ablation should normalize entropy by log(number of relations), reintroduce budget regularization, and vary the history length. Those changes should receive their own experiment identity.

## Research claims and next experiments

[Deep Sets](https://arxiv.org/abs/1703.06114) and [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html) establish unordered-set representations and attention pooling. The research hypothesis here is the combination of a shared typed event representation and causal entity memory for heterogeneous security telemetry. This work does not establish novelty of that combination.

A close recent source is [MeCoLog (August 2026)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0356907): it represents parameters as key/value/type tuples, combines them with template embeddings, and trains a sequence model for cross-system adaptation. Its type-specific value branches and template encoder differ from this implementation's shared bounded token table and entity memory. These architectural differences are an inference from the paper and our code; they do not establish novelty or superiority.

[LogBERT](https://arxiv.org/abs/2103.04475) models normal log sequences through self-supervised tasks, while [LogLLM](https://arxiv.org/abs/2411.08561) combines pretrained language models for log sequence classification. V4-KV borrows structured-token organization, but its small encoder is trained from scratch and does not acquire pretrained language knowledge through tokenization alone. The CADETS head predicts ten provenance relations; a mixed-source deployment requires an appropriate shared target vocabulary or a separately evaluated masked-attribute objective.

CADETS provides provenance telemetry. It cannot establish accuracy across cloud, identity-provider, firewall, and host logs merely because their JSON events pass through the same interface. The next substantive study should use a real mixed-source capture, measure missing-field sensitivity, evaluate source-held-out transfer, and compare the shared encoder with modality-specific alternatives under the same parameter and training budgets. Numeric categorical fields, key aliases, command ordering, identity correlation, and calibrated false-alert rates deserve separate ablations.

The collected result table reports both anomaly detection and traced-node attribution. Published paper values are references, and all new measured rows identify their actual data, code, checkpoint selection, and feature-fitting scope.
