/**
 * TGDetect domain types — mirror the backend Python schemas exactly.
 *
 * Source of truth:
 *   - `graph_builder/schema.py`     (TGEvent, NodeType, RelationType, parquet schemas)
 *   - `graph_builder/attack_tracker.py` (AttackChain, AttackChainSummary)
 *   - `graph_builder/builder.py`     (GraphStats)
 *   - `graph_builder/labeler.py`     (LabelMode, LabelerStats)
 *   - `graph_builder/temporal.py`    (Snapshot, SnapshotMeta)
 *   - `models/tgnn.py`              (TGNNModelConfig)
 *   - `scripts/train_tgnn.py`       (TrainingRun, EpochMetrics)
 *   - `scripts/evaluate_tgnn.py`    (EvaluationMetrics, Prediction)
 *
 * These types MUST stay 1:1 with the backend so that swapping
 * the client connects directly to the FastAPI backend with zero mock fabrication.
 */

// ─────────────────────────────────────────────────────────────────────────────
// Enums (backend `str, Enum` classes — values are UPPER_CASE strings)
// ─────────────────────────────────────────────────────────────────────────────

export type NodeType =
  | 'USER'
  | 'HOST'
  | 'PROCESS'
  | 'FILE'
  | 'IP'
  | 'DOMAIN'
  | 'SOCKET'
  | 'UNKNOWN';

export type RelationType =
  | 'OBSERVED_ON'
  | 'LOGON'
  | 'EXECUTES'
  | 'READS'
  | 'WRITES'
  | 'DELETES'
  | 'CONNECTS_TO'
  | 'AUTHENTICATES_TO'
  | 'NETWORK_FLOW'
  | 'EXPLOIT'
  | 'LATERAL_MOVE'
  | 'EXFILTRATE'
  | 'DISCOVER'
  | 'IMPACT'
  | 'GENERIC';

/** 0 = benign, 1 = malicious. Backend clamps any other value to 0. */
export type EventLabel = 0 | 1;

export type LabelMode = 'parser' | 'force' | 'heuristic';

export type ChainStrategy = 'chain_id' | 'causal_parent' | 'entity_time';

/** Backend `parsers.PARSERS` registry keys. Only these two exist. */
export type DatasetKind = 'dedale' | 'ctu13' | 'synthetic' | 'mordor' | 'synthetic_demo';

/** CLI `--split-mode` for `train_tgnn.py`. */
export type SplitMode = 'time' | 'block';

/** CLI `--select-metric` for `train_tgnn.py`. */
export type SelectMetric = 'auc_pr' | 'f1' | 'auc_roc' | 'best_val_f1';

/** CLI `--node-feature-mode` for `build_snapshots.py`. */
export type NodeFeatureMode = 'type_degree' | 'type_only';

/** CLI `--edge-feature-mode` for `build_snapshots.py`. */
export type EdgeFeatureMode = 'relation_time' | 'relation_only';

/** CLI `--split` for `evaluate_tgnn.py`. */
export type EvalSplit = 'train' | 'val' | 'test';

// ─────────────────────────────────────────────────────────────────────────────
// TGEvent — the unified backend event (graph_builder/schema.py:63)
// ─────────────────────────────────────────────────────────────────────────────

export interface TGEvent {
  /** Unique per event. Mordor: `{source_tag}_{RecordNumber}_{idx}`. */
  event_id: string;
  /** Epoch seconds (float). Post `coerce_ts()` normalization. */
  ts: number;
  /** Canonical `type:value` form (e.g. `user:alice`, `ip:10.0.0.5`). */
  src_id: string;
  src_type: NodeType;
  dst_id: string;
  dst_type: NodeType;
  relation: RelationType | string; // backend permits dataset-specific upper-cased strings
  label: EventLabel;
  /** Free-form MITRE tactic strings. Synthetic uses underscored forms. */
  tactics: string[];
  /** Stage label (e.g. `"execution"`). `-1` / `"-1"` normalized to null. */
  apt_stage: string | null;
  /** Dataset/source identifier (`"synthetic"`, `"mordor"`, `"smoke"`, override). */
  source_tag: string;
  /** Ground-truth chain id. May be `null`. Heuristic synthesises `heur_chain_NNNNNN`. */
  chain_id: string | null;
  /** `event_id` of the causally-preceding event. Used by `CausalParentTracker`. */
  causal_parent: string | null;
  /** Free-form per-dataset metadata. Stored as JSON string in parquet. */
  attrs: Record<string, unknown>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Node / Edge (graph_builder/schema.py — parquet thin projections)
// ─────────────────────────────────────────────────────────────────────────────

/** `nodes.parquet` row — 7 columns. Note the column name is `malicious_events`. */
export interface GraphNode {
  node_id: string;
  node_type: NodeType;
  first_seen_ts: number;
  last_seen_ts: number;
  out_degree: number;
  in_degree: number;
  malicious_events: number;
}

/** `edges.parquet` row — 9 columns. */
export interface GraphEdge {
  event_id: string;
  src_id: string;
  dst_id: string;
  relation: RelationType | string;
  ts: number;
  label: EventLabel;
  chain_id: string | null;
  causal_parent: string | null;
  source_tag: string;
}

// ─────────────────────────────────────────────────────────────────────────────
// Attack chain (graph_builder/attack_tracker.py)
// ─────────────────────────────────────────────────────────────────────────────

/** `chains_summary.parquet` row — 12 columns. */
export interface AttackChainSummary {
  chain_id: string;
  strategy: ChainStrategy;
  num_events: number;
  num_nodes: number;
  start_ts: number;
  end_ts: number;
  duration_s: number;
  /** Run-length-compressed tactic list (consecutive dupes collapsed). */
  tactic_sequence: string[];
  stage_sequence: string[];
  relation_sequence: (RelationType | string)[];
  nodes: string[];
  event_ids: string[];
}

/** Per-chain subgraph JSON written to `graphs/chains/<chain_id>.json`. */
export interface ChainSubgraphNode {
  id: string;
  node_type: NodeType;
}

export interface ChainSubgraphEdge {
  event_id: string;
  src_id: string;
  dst_id: string;
  relation: RelationType | string;
  ts: number;
  label: EventLabel;
  chain_id: string | null;
  causal_parent: string | null;
  source_tag: string;
  tactics: string[];
  apt_stage: string | null;
}

export interface ChainSubgraph {
  chain_id: string;
  strategy: ChainStrategy;
  nodes: ChainSubgraphNode[];
  edges: ChainSubgraphEdge[];
}

// ─────────────────────────────────────────────────────────────────────────────
// Graph statistics (scripts/build_graph.py:145 → graph_stats.json)
// ─────────────────────────────────────────────────────────────────────────────

export interface NormalizationStats {
  seen: number;
  accepted: number;
  rejected: number;
  reject_reasons: Record<string, number>;
}

export interface LabelingStats {
  mode: LabelMode;
  total_events: number;
  seed_indicator_hits: number;
  propagated_events: number;
  malicious_events: number;
  benign_events: number;
  malicious_ratio: number;
  top_reasons: Record<string, number>;
}

export interface GraphSummaryStats {
  total_events: number;
  total_nodes: number;
  total_edges: number;
  benign_events: number;
  malicious_events: number;
  node_types: Record<string, number>;
  relation_types: Record<string, number>;
  source_tags: Record<string, number>;
  tactics: Record<string, number>;
  earliest_timestamp: number;
  latest_timestamp: number;
  timestamp_span_s: number;
  out_of_order_events: number;
  networkx_nodes?: number;
  networkx_edges?: number;
}

export interface AttackSummaryStats {
  malicious_events_tracked: number;
  malicious_events_dropped: number;
  total_chains: number;
  chains_by_strategy: Record<ChainStrategy, number>;
  events_per_chain_histogram: Record<string, number>;
  ungrouped_malicious_events: number;
  dangling_causal_parents: number;
}

export interface GraphOutputs {
  events: string;
  edges: string;
  nodes: string;
  chains: string;
  subgraphs_written: number;
}

export interface GraphStats {
  dataset: string;
  input: string;
  elapsed_s: number;
  normalization: NormalizationStats;
  labeling: LabelingStats;
  graph: GraphSummaryStats;
  attacks: AttackSummaryStats;
  outputs: GraphOutputs;
}

/** Alternate shape produced by `scripts/merge_graphs.py`. */
export interface MergedGraphStats {
  merged_from: string[];
  num_events: number;
  num_edges: number;
  num_nodes: number;
  malicious_events: number;
  benign_events: number;
  temporal_span_s: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Dataset / Processing (scripts/build_graph.py CLI args)
// ─────────────────────────────────────────────────────────────────────────────

/** Frontend domain representation of an ingestible raw dataset. */
export interface Dataset {
  /** Stable identifier for processing job. */
  id: string;
  name: string;
  kind: DatasetKind;
  /** Path or URL to the raw input (file or directory). */
  source: string;
  /** Optional `--metadata-dir` for Mordor scenario YAMLs. */
  metadata_dir?: string | null;
  /** Bytes (raw size of source). */
  size_bytes: number;
  /** Pre-build estimated event count (post-parse). May be `null` until scanned. */
  estimated_events?: number | null;
  created_at: number | null;
  /** Most recent processing job id, if any. */
  last_job_id?: string | null;
  tags?: string[];
  provenance?: string;
  num_raw_events?: number;
  time_span_s?: number | null;
  is_sample?: boolean;
  description?: string;
}

export interface OverviewTelemetry {
  dataset: string;
  dataset_id: string;
  dataset_kind: string;
  provenance: string;
  total_events: number;
  total_nodes: number;
  total_edges: number;
  benign_events: number;
  malicious_events: number;
  malicious_ratio: number;
  chain_count: number;
  node_types: Record<string, number>;
  relation_types: Record<string, number>;
  source_tags: Record<string, number>;
  tactics: Record<string, number>;
  time_span_s: number | null;
  input: string;
  pipeline_status: string;
}

/**
 * Processing job state — frontend mirror of the backend `build_graph.py`
 * pipeline stages. Future backend integration may report live stage via
 * SSE/WebSocket; for now these are user-driven transitions.
 */
export type ProcessingState =
  | 'idle'
  | 'queued'
  | 'parsing'
  | 'normalizing'
  | 'labeling'
  | 'building_graph'
  | 'exporting'
  | 'reconstructing_chains'
  | 'completed'
  | 'failed';

export interface ProcessingConfig {
  kind: DatasetKind;
  source_tag: string;
  label_mode: LabelMode;
  /** `--label` for force mode; 0 or 1. */
  force_label: EventLabel | null;
  /** `--label-window` seconds (heuristic mode default 300). */
  label_window_s: number;
  /** `--no-label-propagation` flag. */
  no_label_propagation: boolean;
  /** `--strategies` (comma-separated in CLI; here as array). */
  strategies: ChainStrategy[];
  /** `--window` seconds (entity_time default 86400). */
  chain_window_s: number;
  /** `--max-hops` (entity_time default 2). */
  chain_max_hops: number;
  /** `--max-subgraphs` (default 1000). */
  max_subgraphs: number;
  /** `--chunk-size` parquet row group size (default 100000). */
  chunk_size: number;
  /** `--limit` for testing/dev. */
  limit: number | null;
  /** `--networkx` flag (compute in-memory NetworkX graph too). */
  use_networkx: boolean;
}

export interface ProcessingJob {
  id: string;
  dataset_id: string;
  dataset_name: string;
  config: ProcessingConfig;
  state: ProcessingState;
  /** 0..1 progress within the current stage. */
  progress: number;
  /** Human-readable current activity. */
  current_step: string;
  /** Output directory (where parquet artifacts land). */
  output_dir: string;
  /** Subgraph output dir (`--graphs-out`). */
  graphs_dir: string | null;
  started_at: number | null;
  ended_at: number | null;
  /** Elapsed seconds (live-ticking while running). */
  elapsed_s: number | null;
  /** Final stats path (`graph_stats.json`) once `completed`. */
  stats_path: string | null;
  /** Error message if `state === 'failed'`. */
  error: string | null;
}

// ─────────────────────────────────────────────────────────────────────────────
// Artifacts (parquet files produced by build_graph.py)
// ─────────────────────────────────────────────────────────────────────────────

export type ArtifactKind =
  | 'events'
  | 'edges'
  | 'nodes'
  | 'chains_summary'
  | 'graph_stats'
  | 'subgraph';

export interface ArtifactSchemaField {
  name: string;
  type: 'string' | 'int8' | 'int64' | 'float64' | 'list<string>' | 'list<object>' | 'json' | 'object' | string;
  nullable: boolean;
  description: string;
}

export interface ArtifactMeta {
  kind: ArtifactKind;
  /** File path on the backend filesystem (e.g. `data/processed/x/events.parquet`). */
  path: string;
  /** Format: `parquet` / `json` / `json_dir`. */
  format: 'parquet' | 'json' | 'json_dir';
  size_bytes: number | null;
  row_count: number | null;
  schema: ArtifactSchemaField[];
  /** Preview rows (parsed; `attrs` is parsed JSON for events). */
  preview: Record<string, any>[];
}

// ─────────────────────────────────────────────────────────────────────────────
// TGNN model (models/tgnn.py)
// ─────────────────────────────────────────────────────────────────────────────

export interface TGNNModelConfig {
  in_channels: number;
  edge_dim: number;
  hidden_channels: number; // default 64
  out_channels: number; // default 64 (GRU hidden)
  num_gnn_layers: number; // default 2 (SAGEConv)
  num_rnn_layers: number; // default 1 (GRU)
  dropout: number; // default 0.3
  node_types: number | null;
  num_relations: number | null;
  learning_rate?: number;
  weight_decay?: number;
  gnn_type?: string;
  rnn_type?: string;
  has_node_classifier?: boolean;
  has_snapshot_classifier?: boolean;
  has_edge_classifier?: boolean;
  target?: 'node' | 'edge';
}

export interface TGNNModelSummary {
  architecture: string;
  model_name?: string;
  architecture_type?: string;
  target?: 'node' | 'edge';
  gnn_operator: string;
  temporal_aggregator: string;
  output_heads: string[];
  loss: string;
  has_attention: boolean;
  has_transformer: boolean;
  has_llm: boolean;
  total_parameters?: number;
  trainable_parameters?: number;
  non_trainable_parameters?: number;
  bn_running_stats?: number;
  total_state_dict_elements?: number;
  checkpoint_path?: string;
  config: TGNNModelConfig;
  layers?: Array<{
    name: string;
    type: string;
    input_dim?: number;
    output_dim?: number;
    num_features?: number;
    input_size?: number;
    hidden_size?: number;
    num_layers?: number;
    in_features?: number;
    out_features?: number;
  }>;
}

export interface ModelMeta {
  id: string;
  name: string;
  description: string;
  target: 'node' | 'edge';
  dataset_id: string;
  dataset_name: string;
  checkpoint: string;
  checkpoint_path: string;
  trainable_parameters: number;
  total_parameters: number;
  in_channels: number;
  edge_dim: number;
  output_heads: string[];
  has_edge_classifier: boolean;
  evaluation_run_id?: string;
  evaluation_status: string;
  metrics?: {
    roc_auc?: number;
    pr_auc?: number;
    f1?: number;
    precision?: number;
    recall?: number;
    accuracy?: number;
    recall_1pct_fpr?: number;
    samples?: number;
    positives?: number;
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Snapshots (graph_builder/temporal.py + scripts/build_snapshots.py)
// ─────────────────────────────────────────────────────────────────────────────

export interface SnapshotMeta {
  dataset: string;
  window_size_s: number;
  stride_s: number;
  num_snapshots: number;
  node_feature_mode: NodeFeatureMode;
  edge_feature_mode: EdgeFeatureMode;
  num_node_types: number;
  num_relations: number;
  node_feature_dim: number;
  edge_feature_dim: number;
  earliest_ts: number;
  latest_ts: number;
}

export interface SnapshotInfo {
  index: number;
  sequence?: number;
  window_start_ts: number;
  window_end_ts: number;
  num_nodes: number;
  num_edges: number;
  num_malicious_nodes: number;
  num_malicious_edges: number;
  snapshot_label: EventLabel;
  num_benign_edges?: number;
  num_unlabeled_edges?: number;
  label?: EventLabel;
  path?: string;
}

// ─────────────────────────────────────────────────────────────────────────────
// Training & evaluation
// ─────────────────────────────────────────────────────────────────────────────

export interface TrainingConfig {
  snapshots_dir?: string | null;
  out_dir?: string | null;
  epochs?: number | null;
  batch_size?: number | null;
  lr?: number | null;
  weight_decay?: number | null;
  grad_clip?: number | null;
  hidden_channels?: number | null;
  out_channels?: number | null;
  gnn_layers?: number | null;
  rnn_layers?: number | null;
  dropout?: number | null;
  window_size?: number | null; // # snapshots per training sequence (default 10)
  seq_stride?: number | null;
  val_ratio?: number | null;
  test_ratio?: number | null;
  split_mode?: SplitMode | string | null;
  block_size?: number | null;
  select_metric?: SelectMetric | string | null;
  no_threshold_tuning?: boolean | null;
  pos_weight?: number | null; // auto-computed if null
  seed?: number | null;
}

export interface EpochMetrics {
  epoch: number;
  train_loss: number;
  loss: number;
  auc_roc: number | null;
  auc_pr: number | null;
  threshold: number | null;
  f1: number;
  accuracy: number;
  precision: number;
  recall: number;
}

export interface TrainingRun {
  id: string;
  name: string;
  dataset_id: string;
  config: TrainingConfig;
  started_at: number | string | null;
  ended_at: number | string | null;
  elapsed_s: number | null;
  current_epoch: number | null;
  total_epochs: number | null;
  best_epoch: number | null;
  best_metric: SelectMetric | string | null;
  best_score: number | null;
  best_val_f1: number | null;
  threshold: number | null;
  history: EpochMetrics[];
  checkpoint_path: string | null;
  final_model_path: string | null;
  state: 'running' | 'completed' | 'failed' | 'stopped' | string;
  error: string | null;
}

export interface PredictionRow {
  sequence: number;
  node_id: string;
  probability: number;
  prediction: EventLabel;
  ground_truth: EventLabel;
  snapshot_label: EventLabel;
}

export interface ConfusionMatrix {
  /** [[TN, FP], [FN, TP]] — matches backend `metrics_<split>.json`. */
  tn: number;
  fp: number;
  fn: number;
  tp: number;
}

export interface EvaluationMetrics {
  threshold: number;
  num_samples: number;
  num_positive: number;
  num_negative: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  /** `null` if only one class present in the split. */
  auc_roc: number | null;
  /** `null` if only one class present in the split. */
  auc_pr: number | null;
  confusion_matrix: ConfusionMatrix | null;
  recall_at_1pct_fpr?: number | null;
  threshold_best?: number | null;
  threshold_at_1pct_fpr?: number | null;
  saved_threshold?: number | null;
  f1_at_saved_threshold?: number | null;
  accuracy_at_saved_threshold?: number | null;
}

export interface EvaluationRun {
  id: string;
  training_run_id: string;
  checkpoint_path: string;
  split: EvalSplit;
  target?: 'node' | 'edge';
  metrics: EvaluationMetrics;
  predictions_path: string | null;
  started_at: number | null;
  ended_at: number | null;
  checkpoint?: string;
  evaluated_at?: number | null;
}

// ─────────────────────────────────────────────────────────────────────────────
// Service-level request types (filters)
// ─────────────────────────────────────────────────────────────────────────────

export interface EventFilter {
  /** Substring match on `event_id`. */
  event_id_query?: string;
  /** Subset of label values. Empty = all. */
  labels?: EventLabel[];
  /** Subset of node types — matches if `src_type` OR `dst_type` is in set. */
  node_types?: NodeType[];
  /** Subset of relation types. */
  relations?: (RelationType | string)[];
  /** Subset of `source_tag`. */
  source_tags?: string[];
  /** Subset of `tactics` — event matches if ANY tactic overlaps. */
  tactics?: string[];
  /** Subset of `chain_id` — exact match. */
  chain_ids?: string[];
  /** `ts >= start_ts`. */
  start_ts?: number;
  /** `ts <= end_ts`. */
  end_ts?: number;
  /** Page size. */
  limit?: number;
  /** Page offset. */
  offset?: number;
}

export interface ChainFilter {
  strategies?: ChainStrategy[];
  /** `chain_id` substring match. */
  chain_id_query?: string;
  min_events?: number;
  max_events?: number;
  min_duration_s?: number;
  max_duration_s?: number;
  limit?: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// UI status — loading / error / empty states
// ─────────────────────────────────────────────────────────────────────────────

export type LoadState = 'idle' | 'loading' | 'success' | 'empty' | 'partial' | 'failed' | 'unavailable';

export interface AsyncResult<T> {
  state: LoadState;
  data: T | null;
  error: string | null;
}


// ─────────────────────────────────────────────────────────────────────────────
// Dataset Upload & Validation
// ─────────────────────────────────────────────────────────────────────────────

export interface ValidationDiagnosticError {
  line: number;
  column?: string | null;
  message: string;
}

export interface ValidationDiagnosticWarning {
  line?: number | null;
  message: string;
}

export interface ValidationReport {
  status: 'valid' | 'valid_with_warnings' | 'invalid';
  format: string;
  format_label?: string;
  detected_format: string;
  total_rows_inspected: number;
  valid_rows: number;
  invalid_rows: number;
  error_count: number;
  warning_count: number;
  errors: ValidationDiagnosticError[];
  warnings: ValidationDiagnosticWarning[];
  columns_detected: string[];
  missing_required_columns: string[];
  column_mapping?: Record<string, string>;
  features_extracted?: string[];
  sample_events: any[];
  saved_path?: string;
  filename?: string;
  file_info?: {
    filename: string;
    size_bytes: number;
    readable: boolean;
    archive_type?: string | null;
    members?: string[];
  };
}
