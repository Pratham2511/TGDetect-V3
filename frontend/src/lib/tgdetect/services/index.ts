/**
 * TGDetect authoritative service layer.
 *
 * All frontend services communicate directly with the FastAPI backend.
 * Zero mock mode. Zero fabricated data. Pure production API integration.
 */

import {
  ApiAnalyticsService,
  ApiArtifactService,
  ApiAttackChainService,
  ApiDatasetService,
  ApiEventService,
  ApiGraphService,
  ApiModelService,
  ApiProcessingService,
  ApiTrainingService,
} from './api';
import type {
  ArtifactMeta,
  AttackChainSummary,
  ChainFilter,
  ChainStrategy,
  ChainSubgraph,
  Dataset,
  EventFilter,
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
  OverviewTelemetry,
} from '../types';

// ─────────────────────────────────────────────────────────────────────────────
// Service interfaces
// ─────────────────────────────────────────────────────────────────────────────

export interface DatasetService {
  list(): Promise<Dataset[]>;
  get(id: string): Promise<Dataset | null>;
  getActive(): Promise<Dataset | null>;
  overview(datasetId?: string): Promise<OverviewTelemetry>;
  upload(file: File, format?: string): Promise<ValidationReport>;
  validate(filePath: string, format?: string): Promise<ValidationReport>;
  process(datasetId: string, format: string, sourcePath: string, config?: any): Promise<ProcessingJob>;
  activate(datasetId: string): Promise<{ status: string; active_dataset: string; dataset?: Dataset }>;
}

export interface EventService {
  list(filter?: EventFilter): Promise<{ items: TGEvent[]; total: number }>;
  get(eventId: string): Promise<TGEvent | null>;
  byChain(chainId: string): Promise<TGEvent[]>;
  byNode(nodeId: string): Promise<TGEvent[]>;
  recentMalicious(limit: number): Promise<TGEvent[]>;
}

export interface GraphService {
  nodes(limit?: number, malicious_only?: boolean): Promise<GraphNode[]>;
  edges(limit?: number, malicious_only?: boolean): Promise<GraphEdge[]>;
  stats(): Promise<GraphStats>;
}

export interface AttackChainService {
  list(filter?: ChainFilter): Promise<AttackChainSummary[]>;
  listByStrategy(strategy: ChainStrategy): Promise<AttackChainSummary[]>;
  get(chainId: string): Promise<AttackChainSummary | null>;
  subgraph(chainId: string): Promise<ChainSubgraph | null>;
  allStrategies(): Promise<Record<ChainStrategy, AttackChainSummary[]>>;
}

export interface ArtifactService {
  listForJob(jobId: string): Promise<ArtifactMeta[]>;
}

export interface ModelService {
  models(): Promise<ModelMeta[]>;
  tgnnConfig(modelId?: string): Promise<TGNNModelConfig>;
  tgnnSummary(modelId?: string): Promise<TGNNModelSummary>;
  snapshotMeta(): Promise<SnapshotMeta>;
  snapshots(limit?: number): Promise<SnapshotInfo[]>;
}

export interface TrainingService {
  currentRun(modelId?: string): Promise<TrainingRun>;
  history(modelId?: string): Promise<TrainingRun['history']>;
  evaluationRuns(): Promise<EvaluationRun[]>;
  evaluationRun(split: 'train' | 'val' | 'test', runId?: string, modelId?: string): Promise<EvaluationRun | null>;
  predictions(split: 'train' | 'val' | 'test', modelId?: string): Promise<PredictionRow[]>;
}

export interface ProcessingService {
  jobs(): Promise<ProcessingJob[]>;
  job(id: string): Promise<ProcessingJob | null>;
}

export interface AnalyticsService {
  eventsAnalytics(): Promise<{
    timeline: Array<{
      bucket_start: number;
      bucket_end: number;
      total: number;
      benign: number;
      malicious: number;
    }>;
    tactics: Record<string, number>;
    source_tags: Record<string, number>;
  }>;
  graphAnalytics(): Promise<{
    node_types: Record<string, number>;
    relations: Record<string, number>;
    degree_distribution: Array<{ range: string; count: number }>;
  }>;
  attacksAnalytics(): Promise<{
    strategies: Record<string, number>;
    length_histogram: Record<string, number>;
    durations: Array<{ chain_id: string; strategy: string; duration_s: number }>;
  }>;
  datasetsAnalytics(): Promise<any>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Authoritative Production Services (Connected directly to backend)
// ─────────────────────────────────────────────────────────────────────────────

export const datasetService: DatasetService = new ApiDatasetService();
export const eventService: EventService = new ApiEventService();
export const graphService: GraphService = new ApiGraphService();
export const chainService: AttackChainService = new ApiAttackChainService();
export const artifactService: ArtifactService = new ApiArtifactService();
export const modelService: ModelService = new ApiModelService();
export const trainingService: TrainingService = new ApiTrainingService();
export const processingService: ProcessingService = new ApiProcessingService();
export const analyticsService: AnalyticsService = new ApiAnalyticsService();
