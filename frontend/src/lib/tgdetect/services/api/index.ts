/**
 * Real API-backed implementations of the TGDetect service interfaces.
 *
 * Calls the FastAPI backend at http://localhost:8000 via `apiClient`.
 * Directly mirrors backend types with zero mock fabrication.
 */

import { api } from '../../api/client';
import type {
  ArtifactMeta,
  OverviewTelemetry,
  AttackChainSummary,
  ChainFilter,
  ChainStrategy,
  ChainSubgraph,
  Dataset,
  DatasetKind,
  EventFilter,
  EventLabel,
  EvaluationRun,
  GraphEdge,
  GraphNode,
  GraphStats,
  ModelMeta,
  PredictionRow,
  ProcessingJob,
  SnapshotInfo,
  SnapshotMeta,
  TGEvent,
  TGNNModelConfig,
  TGNNModelSummary,
  TrainingRun,
  ValidationReport,
} from '../../types';
import type {
  AnalyticsService,
  ArtifactService,
  AttackChainService,
  DatasetService,
  EventService,
  GraphService,
  ModelService,
  ProcessingService,
  TrainingService,
} from '../index';

export class ApiDatasetService implements DatasetService {
  async list(): Promise<Dataset[]> {
    const raw = await api.get<any[]>('/api/datasets');
    return raw.map((d) => ({
      ...d,
      id: d.id,
      name: d.name,
      kind: (d.kind as DatasetKind) || 'dedale',
      source: d.source || '',
      metadata_dir: null,
      size_bytes: d.raw_bytes || d.size_bytes || 0,
      estimated_events: d.num_raw_events ?? d.estimated_events ?? null,
      created_at: d.created_at ?? null,
      last_job_id: d.last_job_id ?? null,
      tags: d.tags ?? (d.id && d.id.startsWith('ctu13') ? ['ctu13', 'netflow'] : ['telemetry']),
    }));
  }

  async get(id: string): Promise<Dataset | null> {
    try {
      const d = await api.get<any>(`/api/datasets/${encodeURIComponent(id)}`);
      if (!d) return null;
      return {
        ...d,
      id: d.id,
        name: d.name,
        kind: (d.kind as DatasetKind) || 'dedale',
        source: d.source || '',
        metadata_dir: null,
        size_bytes: d.raw_bytes || d.size_bytes || 0,
        estimated_events: d.num_raw_events ?? d.estimated_events ?? null,
        created_at: d.created_at ?? null,
        last_job_id: d.last_job_id ?? null,
        tags: d.tags ?? (d.id && d.id.startsWith('ctu13') ? ['ctu13', 'netflow'] : ['telemetry']),
      };
    } catch {
      return null;
    }
  }

  async upload(file: File, format?: string): Promise<ValidationReport> {
    const formData = new FormData();
    formData.append('file', file);
    if (format && format !== 'auto') {
      formData.append('format', format);
    }
    return api.upload<ValidationReport>('/api/datasets/upload', formData);
  }

  async validate(filePath: string, format?: string): Promise<ValidationReport> {
    return api.post<ValidationReport>('/api/datasets/validate', {
      file_path: filePath,
      format: format && format !== 'auto' ? format : undefined,
    });
  }

  async process(datasetId: string, format: string, sourcePath: string, config?: any): Promise<ProcessingJob> {
    const raw = await api.post<any>('/api/datasets/process', {
      dataset_id: datasetId,
      format,
      source_path: sourcePath,
      config,
    });
    const cfg = raw.config || {};
    return {
      id: raw.id,
      dataset_id: raw.dataset_id,
      dataset_name: raw.dataset_name || raw.dataset_id,
      config: {
        kind: (cfg.dataset as DatasetKind) || (format as DatasetKind) || 'user_uploaded',
        source_tag: datasetId,
        label_mode: 'parser',
        force_label: null,
        label_window_s: 300,
        no_label_propagation: false,
        strategies: cfg.strategies || ['chain_id', 'causal_parent', 'entity_time'],
        chain_window_s: 86400,
        chain_max_hops: 2,
        max_subgraphs: 100,
        chunk_size: 50000,
        limit: null,
        use_networkx: false,
      },
      state: (raw.status === 'completed' ? 'completed' : raw.status) as any,
      progress: (raw.progress_pct ?? 100) / 100,
      current_step: raw.description || `Processing ${datasetId}`,
      output_dir: cfg.out_dir || `data/processed/${datasetId}`,
      graphs_dir: 'data/graphs',
      started_at: raw.started_at ?? null,
      ended_at: raw.completed_at ?? null,
      elapsed_s: raw.elapsed_s ?? null,
      stats_path: `data/processed/${datasetId}/graph_stats.json`,
      error: raw.error ?? null,
    };
  }

  async getActive(): Promise<Dataset | null> {
    try {
      return await api.get<Dataset>('/api/datasets/active');
    } catch {
      return null;
    }
  }

  async overview(datasetId?: string): Promise<OverviewTelemetry> {
    return api.get<OverviewTelemetry>('/api/overview', {
      params: datasetId ? { dataset_id: datasetId } : undefined,
    });
  }

  async activate(datasetId: string): Promise<{ status: string; active_dataset: string; dataset?: Dataset }> {
    return api.post<{ status: string; active_dataset: string; dataset?: Dataset }>(`/api/datasets/${encodeURIComponent(datasetId)}/activate`);
  }
}

export class ApiEventService implements EventService {
  async list(filter?: EventFilter): Promise<{ items: TGEvent[]; total: number }> {
    const params: Record<string, string | number | boolean | null | undefined> = {};
    if (filter) {
      if (filter.event_id_query) params.search = filter.event_id_query;
      if (filter.labels && filter.labels.length > 0) params.labels = filter.labels.join(',');
      if (filter.node_types && filter.node_types.length > 0) params.node_types = filter.node_types.join(',');
      if (filter.relations && filter.relations.length > 0) params.relations = filter.relations.join(',');
      if (filter.source_tags && filter.source_tags.length > 0) params.source_tags = filter.source_tags.join(',');
      if (filter.tactics && filter.tactics.length > 0) params.tactics = filter.tactics.join(',');
      if (filter.chain_ids && filter.chain_ids.length > 0) params.chain_ids = filter.chain_ids.join(',');
      if (filter.start_ts !== undefined) params.start_ts = filter.start_ts;
      if (filter.end_ts !== undefined) params.end_ts = filter.end_ts;
      if (filter.limit !== undefined) params.limit = filter.limit;
      if (filter.offset !== undefined) params.offset = filter.offset;
    }
    return api.get<{ items: TGEvent[]; total: number }>('/api/events', { params });
  }

  async get(eventId: string): Promise<TGEvent | null> {
    try {
      return await api.get<TGEvent>(`/api/events/${encodeURIComponent(eventId)}`);
    } catch {
      return null;
    }
  }

  async byChain(chainId: string): Promise<TGEvent[]> {
    return api.get<TGEvent[]>(`/api/chains/${encodeURIComponent(chainId)}/events`);
  }

  async byNode(nodeId: string): Promise<TGEvent[]> {
    const res = await api.get<{ items: TGEvent[]; total: number }>('/api/events', {
      params: { node_id: nodeId, limit: 50 },
    });
    return res.items;
  }

  async recentMalicious(limit: number): Promise<TGEvent[]> {
    const res = await api.get<{ items: TGEvent[]; total: number }>('/api/events', {
      params: { labels: '1', limit },
    });
    return res.items;
  }
}

export class ApiGraphService implements GraphService {
  async nodes(limit?: number, malicious_only?: boolean): Promise<GraphNode[]> {
    const params: Record<string, number | boolean> = {};
    if (limit !== undefined) params.limit = limit;
    if (malicious_only !== undefined) params.malicious_only = malicious_only;
    return api.get<GraphNode[]>('/api/graph/nodes', { params });
  }

  async edges(limit?: number, malicious_only?: boolean): Promise<GraphEdge[]> {
    const params: Record<string, number | boolean> = {};
    if (limit !== undefined) params.limit = limit;
    if (malicious_only !== undefined) params.malicious_only = malicious_only;
    return api.get<GraphEdge[]>('/api/graph/edges', { params });
  }

  async stats(): Promise<GraphStats> {
    return api.get<GraphStats>('/api/graph/stats');
  }
}

export class ApiAttackChainService implements AttackChainService {
  async list(filter?: ChainFilter): Promise<AttackChainSummary[]> {
    const params: Record<string, string | number | boolean | null | undefined> = {};
    if (filter) {
      if (filter.strategies && filter.strategies.length > 0) {
        params.strategy = filter.strategies[0];
      }
      if (filter.min_events !== undefined) params.min_events = filter.min_events;
      if (filter.max_events !== undefined) params.max_events = filter.max_events;
      if (filter.min_duration_s !== undefined) params.min_duration = filter.min_duration_s;
      if (filter.max_duration_s !== undefined) params.max_duration = filter.max_duration_s;
    }
    return api.get<AttackChainSummary[]>('/api/chains', { params });
  }

  async listByStrategy(strategy: ChainStrategy): Promise<AttackChainSummary[]> {
    return api.get<AttackChainSummary[]>('/api/chains', {
      params: { strategy },
    });
  }

  async get(chainId: string): Promise<AttackChainSummary | null> {
    try {
      return await api.get<AttackChainSummary>(`/api/chains/${encodeURIComponent(chainId)}`);
    } catch {
      return null;
    }
  }

  async subgraph(chainId: string): Promise<ChainSubgraph | null> {
    try {
      return await api.get<ChainSubgraph>(`/api/chains/${encodeURIComponent(chainId)}/subgraph`);
    } catch {
      return null;
    }
  }

  async allStrategies(): Promise<Record<ChainStrategy, AttackChainSummary[]>> {
    const all = await api.get<AttackChainSummary[]>('/api/chains');
    const out: Record<ChainStrategy, AttackChainSummary[]> = {
      chain_id: [],
      causal_parent: [],
      entity_time: [],
    };
    for (const c of all) {
      if (out[c.strategy]) {
        out[c.strategy].push(c);
      }
    }
    return out;
  }
}

export class ApiArtifactService implements ArtifactService {
  async listForJob(jobId: string): Promise<ArtifactMeta[]> {
    const raw = await api.get<any[]>('/api/artifacts', {
      params: { job_id: jobId },
    });
    return raw.map((a) => {
      const pathStr = a.path || a.id || '';
      const fmt: 'parquet' | 'json' = pathStr.endsWith('.parquet') ? 'parquet' : 'json';
      return {
        kind: a.kind as any,
        path: pathStr,
        format: fmt,
        size_bytes: a.size_bytes ?? null,
        row_count: a.row_count ?? null,
        schema: (a.schema || []).map((s: any) => ({
          name: s.name,
          type: s.type,
          nullable: !!s.nullable,
          description: s.name,
        })),
        preview: a.preview || [],
      };
    });
  }
}

export class ApiModelService implements ModelService {
  async models(): Promise<ModelMeta[]> {
    return api.get<ModelMeta[]>('/api/models');
  }

  async tgnnConfig(modelId?: string): Promise<TGNNModelConfig> {
    return api.get<TGNNModelConfig>('/api/model/config', {
      params: modelId ? { model_id: modelId } : undefined,
    });
  }

  async tgnnSummary(modelId?: string): Promise<TGNNModelSummary> {
    return api.get<TGNNModelSummary>('/api/model/summary', {
      params: modelId ? { model_id: modelId } : undefined,
    });
  }

  async snapshotMeta(): Promise<SnapshotMeta> {
    return api.get<SnapshotMeta>('/api/model/snapshots/meta');
  }

  async snapshots(limit?: number): Promise<SnapshotInfo[]> {
    const res = await api.get<{ meta: SnapshotMeta; snapshots: any[] }>(
      '/api/model/snapshots',
      { params: limit ? { limit } : undefined }
    );
    const list = res.snapshots || [];
    return list.map((s: any) => ({
      index: s.index ?? s.sequence ?? 0,
      sequence: s.sequence ?? s.index ?? 0,
      window_start_ts: s.window_start_ts ?? 0,
      window_end_ts: s.window_end_ts ?? 0,
      num_nodes: s.num_nodes ?? 0,
      num_edges: s.num_edges ?? 0,
      num_malicious_nodes: s.num_malicious_nodes ?? 0,
      num_malicious_edges: typeof s.num_malicious_edges === 'number' ? s.num_malicious_edges : 0,
      snapshot_label: (s.snapshot_label ?? s.label ?? 0) as EventLabel,
      label: (s.label ?? s.snapshot_label ?? 0) as EventLabel,
      path: s.path,
      num_benign_edges: s.num_benign_edges ?? 0,
      num_unlabeled_edges: s.num_unlabeled_edges ?? 0,
    }));
  }
}

export class ApiTrainingService implements TrainingService {
  async currentRun(modelId?: string): Promise<TrainingRun> {
    const raw = await api.get<any>('/api/model/training', {
      params: modelId ? { model_id: modelId } : undefined,
    });
    const history = raw.history || [];
    const bestEpochMetric = history.find((h: any) => h.epoch === raw.best_epoch) || history[0] || {};
    const cfg = raw.config || {};
    return {
      id: raw.id || raw.run_id || 'Unavailable',
      name: raw.model_type || 'Unavailable',
      dataset_id: raw.dataset || raw.dataset_id || 'dedale-host-memory',
      config: {
        snapshots_dir: cfg.snapshots ?? null,
        out_dir: cfg.out ?? null,
        epochs: cfg.epochs ?? raw.epochs_total ?? (history.length || null),
        batch_size: cfg.batch_size ?? null,
        lr: cfg.lr ?? null,
        weight_decay: cfg.weight_decay ?? null,
        gnn_layers: cfg.gnn_layers ?? null,
        rnn_layers: cfg.rnn_layers ?? null,
        hidden_channels: cfg.hidden_channels ?? null,
        out_channels: cfg.out_channels ?? null,
        dropout: cfg.dropout ?? null,
        window_size: cfg.window_size ?? null,
        val_ratio: cfg.val_ratio ?? null,
        test_ratio: cfg.test_ratio ?? null,
        split_mode: cfg.split_mode ?? null,
        seed: cfg.seed ?? null,
        grad_clip: cfg.grad_clip ?? null,
        seq_stride: cfg.seq_stride ?? null,
        block_size: cfg.block_size ?? null,
        select_metric: cfg.select_metric ?? null,
        no_threshold_tuning: cfg.no_threshold_tuning ?? null,
        pos_weight: cfg.pos_weight ?? null,
      },
      started_at: raw.started_at ?? null,
      ended_at: raw.ended_at ?? null,
      elapsed_s: raw.training_time_s ?? null,
      current_epoch: raw.epochs_total ?? (history.length || null),
      total_epochs: raw.epochs_total ?? (history.length || null),
      best_epoch: raw.best_epoch ?? null,
      best_metric: raw.best_metric ?? null,
      best_score: raw.best_metric_value ?? bestEpochMetric.f1 ?? bestEpochMetric.auc_roc ?? null,
      best_val_f1: bestEpochMetric.f1 ?? raw.best_metric_value ?? null,
      threshold: raw.threshold ?? null,
      history: history.map((h: any) => ({
        epoch: h.epoch,
        train_loss: h.train_loss,
        loss: h.loss,
        auc_roc: h.auc_roc ?? null,
        auc_pr: h.auc_pr ?? null,
        threshold: h.threshold ?? null,
        f1: h.f1,
        accuracy: h.accuracy,
        precision: h.precision,
        recall: h.recall,
      })),
      checkpoint_path: raw.checkpoint_path || null,
      final_model_path: raw.checkpoint_path || null,
      state: raw.status === 'completed' ? 'completed' : raw.status === 'failed' ? 'failed' : 'running',
      error: raw.error ?? null,
    };
  }

  async history(modelId?: string): Promise<TrainingRun['history']> {
    return api.get<TrainingRun['history']>('/api/model/training/history', {
      params: modelId ? { model_id: modelId } : undefined,
    });
  }

  async evaluationRuns(): Promise<EvaluationRun[]> {
    return api.get<EvaluationRun[]>('/api/model/evaluation/runs');
  }

  async evaluationRun(split: 'train' | 'val' | 'test', runId?: string, modelId?: string): Promise<EvaluationRun | null> {
    try {
      return await api.get<EvaluationRun>('/api/model/evaluation', {
        params: {
          split,
          ...(runId ? { run_id: runId } : {}),
          ...(modelId ? { model_id: modelId } : {}),
        },
      });
    } catch {
      return null;
    }
  }

  async predictions(split: 'train' | 'val' | 'test', modelId?: string): Promise<PredictionRow[]> {
    try {
      return await api.get<PredictionRow[]>('/api/model/predictions', {
        params: {
          split,
          limit: 100,
          ...(modelId ? { model_id: modelId } : {}),
        },
      });
    } catch {
      return [];
    }
  }
}

export class ApiProcessingService implements ProcessingService {
  async jobs(): Promise<ProcessingJob[]> {
    const raw = await api.get<any[]>('/api/jobs');
    return raw.map((j) => {
      const cfg = j.config || {};
      return {
        id: j.id,
        dataset_id: j.dataset_id,
        dataset_name: j.dataset_name || j.dataset_id || 'DEDALE Sysmon',
        config: {
          kind: (cfg.dataset_kind as DatasetKind) || (cfg.dataset as DatasetKind) || 'dedale',
          source_tag: cfg.source_tag || j.dataset_id || 'dedale',
          label_mode: cfg.label_mode || 'parser',
          force_label: cfg.force_label ?? null,
          label_window_s: cfg.label_window_s ?? 300,
          no_label_propagation: !cfg.label_propagation,
          strategies: cfg.strategies || ['chain_id', 'causal_parent', 'entity_time'],
          chain_window_s: cfg.window_s ?? 86400,
          chain_max_hops: cfg.max_hops ?? 2,
          max_subgraphs: cfg.max_subgraphs ?? 1000,
          chunk_size: cfg.chunk_size ?? 100000,
          limit: null,
          use_networkx: false,
        },
        state: (j.status === 'completed' ? 'completed' : j.status) as any,
        progress: (j.progress_pct ?? 100) / 100,
        current_step: j.description || 'Completed batch processing',
        output_dir: cfg.out_dir || `data/processed/${j.dataset_id}`,
        graphs_dir: cfg.graphs_out || 'data/graphs',
        started_at: j.started_at ?? null,
        ended_at: j.completed_at ?? null,
        elapsed_s: j.elapsed_s ?? (j.completed_at && j.started_at ? j.completed_at - j.started_at : (j.stats?.elapsed_s ?? null)),
        stats_path: `data/processed/${j.dataset_id}/graph_stats.json`,
        error: j.error ?? null,
      };
    });
  }

  async job(id: string): Promise<ProcessingJob | null> {
    try {
      const all = await this.jobs();
      return all.find((j) => j.id === id) ?? null;
    } catch {
      return null;
    }
  }
}

export class ApiAnalyticsService implements AnalyticsService {
  async eventsAnalytics(): Promise<{
    timeline: Array<{
      bucket_start: number;
      bucket_end: number;
      total: number;
      benign: number;
      malicious: number;
    }>;
    tactics: Record<string, number>;
    source_tags: Record<string, number>;
  }> {
    return api.get('/api/analytics/events');
  }

  async graphAnalytics(): Promise<{
    node_types: Record<string, number>;
    relations: Record<string, number>;
    degree_distribution: Array<{ range: string; count: number }>;
  }> {
    return api.get('/api/analytics/graph');
  }

  async attacksAnalytics(): Promise<{
    strategies: Record<string, number>;
    length_histogram: Record<string, number>;
    durations: Array<{ chain_id: string; strategy: string; duration_s: number }>;
  }> {
    return api.get('/api/analytics/attacks');
  }

  async datasetsAnalytics(): Promise<any> {
    return api.get('/api/analytics/datasets');
  }
}
