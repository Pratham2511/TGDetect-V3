'use client';

import { useEffect, useState } from 'react';
import {
  analyticsService,
  artifactService,
  chainService,
  datasetService,
  eventService,
  graphService,
  modelService,
  processingService,
  trainingService,
} from './index';
import { useModel } from '@/lib/model-context';
import { useDataset } from '@/lib/dataset-context';
import type {
  ArtifactMeta,
  AttackChainSummary,
  ChainStrategy,
  ChainSubgraph,
  Dataset,
  EventFilter,
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
  EvaluationRun,
  OverviewTelemetry,
} from '../types';

// ─────────────────────────────────────────────────────────────────────────────
// Lightweight hook — wraps a promise-returning service call in state
// (loading / success / error). UI is decoupled from the service impl.
// ─────────────────────────────────────────────────────────────────────────────

export type AsyncState = 'idle' | 'loading' | 'success' | 'empty' | 'failed';

export interface AsyncValue<T> {
  state: AsyncState;
  data: T | null;
  error: string | null;
  reload: () => void;
}

function useAsync<T>(fn: () => Promise<T>, deps: unknown[]): AsyncValue<T> {
  const { activeModelId } = useModel();
  const [state, setState] = useState<AsyncState>('loading');
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [reloadCount, setReloadCount] = useState(0);

  useEffect(() => {
    let cancelled = false;
    // Reset to loading state when deps change. This setState-in-effect is
    // intentional and idiomatic for async data fetching.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setState('loading');
    setData(null);
    setError(null);
    fn()
      .then((d) => {
        if (cancelled) return;
        setData(d);
        setState('success');
        if (Array.isArray(d) && d.length === 0) setState('empty');
        else if (d == null || (typeof d === 'object' && !Array.isArray(d) && Object.keys(d as object).length === 0)) {
          // Treat empty object as empty for some services.
        }
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err?.message ?? String(err));
        setState('failed');
      });
    return () => { cancelled = true; };
  }, [...deps, reloadCount, activeModelId]);

  return { state, data, error, reload: () => setReloadCount((n) => n + 1) };
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-service hooks — UI imports these
// ─────────────────────────────────────────────────────────────────────────────

export function useDatasets() {
  const { datasetVersion } = useDataset();
  return useAsync<Dataset[]>(() => datasetService.list(), [datasetVersion]);
}

export function useEvents(filter?: EventFilter) {
  const { datasetVersion } = useDataset();
  return useAsync(
    () => eventService.list(filter),
    [datasetVersion, filter?.event_id_query, filter?.labels?.join(','), filter?.node_types?.join(','), filter?.relations?.join(','), filter?.source_tags?.join(','), filter?.tactics?.join(','), filter?.chain_ids?.join(','), filter?.start_ts, filter?.end_ts, filter?.limit, filter?.offset],
  );
}

export function useEvent(eventId: string | null) {
  const { datasetVersion } = useDataset();
  return useAsync(() => (eventId ? eventService.get(eventId) : Promise.resolve(null)), [eventId, datasetVersion]);
}

export function useChainEvents(chainId: string | null) {
  const { datasetVersion } = useDataset();
  return useAsync(() => (chainId ? eventService.byChain(chainId) : Promise.resolve([])), [chainId, datasetVersion]);
}

export function useNodeEvents(nodeId: string | null) {
  const { datasetVersion } = useDataset();
  return useAsync(() => (nodeId ? eventService.byNode(nodeId) : Promise.resolve([])), [nodeId, datasetVersion]);
}

export function useRecentMalicious(limit: number) {
  const { datasetVersion } = useDataset();
  return useAsync(() => eventService.recentMalicious(limit), [limit, datasetVersion]);
}

export function useGraphNodes(limit: number = 1000, maliciousOnly: boolean = false) {
  const { datasetVersion } = useDataset();
  return useAsync<GraphNode[]>(() => graphService.nodes(limit, maliciousOnly), [limit, maliciousOnly, datasetVersion]);
}

export function useGraphEdges(limit: number = 2000, maliciousOnly: boolean = false) {
  const { datasetVersion } = useDataset();
  return useAsync<GraphEdge[]>(() => graphService.edges(limit, maliciousOnly), [limit, maliciousOnly, datasetVersion]);
}

export function useGraphStats() {
  const { datasetVersion } = useDataset();
  return useAsync<GraphStats>(() => graphService.stats(), [datasetVersion]);
}

export function useChains() {
  const { datasetVersion } = useDataset();
  return useAsync<AttackChainSummary[]>(() => chainService.list(), [datasetVersion]);
}

export function useChain(chainId: string | null) {
  const { datasetVersion } = useDataset();
  return useAsync(() => (chainId ? chainService.get(chainId) : Promise.resolve(null)), [chainId, datasetVersion]);
}

export function useChainSubgraph(chainId: string | null) {
  const { datasetVersion } = useDataset();
  return useAsync<ChainSubgraph | null>(() => (chainId ? chainService.subgraph(chainId) : Promise.resolve(null)), [chainId, datasetVersion]);
}

export function useChainsByStrategy(strategy: ChainStrategy | null) {
  const { datasetVersion } = useDataset();
  return useAsync(() => (strategy ? chainService.listByStrategy(strategy) : Promise.resolve([])), [strategy, datasetVersion]);
}

export function useJobs() {
  return useAsync<ProcessingJob[]>(() => processingService.jobs(), []);
}

export function useJob(id: string | null) {
  return useAsync(() => (id ? processingService.job(id) : Promise.resolve(null)), [id]);
}

export function useArtifacts(jobId?: string | null) {
  const { datasetVersion, activeDatasetId } = useDataset();
  return useAsync<ArtifactMeta[]>(
    () => artifactService.listForJob(jobId || `job-${activeDatasetId}`),
    [jobId, activeDatasetId, datasetVersion]
  );
}

export function useModels() {
  return useAsync<ModelMeta[]>(() => modelService.models(), []);
}

export function useTGNNConfig(modelId?: string) {
  return useAsync<TGNNModelConfig>(() => modelService.tgnnConfig(modelId), [modelId]);
}

export function useTGNNSummary(modelId?: string) {
  return useAsync<TGNNModelSummary>(() => modelService.tgnnSummary(modelId), [modelId]);
}

export function useSnapshotMeta() {
  return useAsync<SnapshotMeta>(() => modelService.snapshotMeta(), []);
}

export function useSnapshots(limit?: number) {
  return useAsync<SnapshotInfo[]>(() => modelService.snapshots(limit), [limit]);
}

export function useTrainingRun(modelId?: string) {
  return useAsync<TrainingRun>(() => trainingService.currentRun(modelId), [modelId]);
}

export function useEvaluationRuns() {
  return useAsync<EvaluationRun[]>(() => trainingService.evaluationRuns(), []);
}

export function useEvaluationRun(split: 'train' | 'val' | 'test', runId?: string, modelId?: string) {
  return useAsync<EvaluationRun | null>(
    () => trainingService.evaluationRun(split, runId, modelId),
    [split, runId, modelId]
  );
}

export function usePredictions(split: 'train' | 'val' | 'test', modelId?: string) {
  return useAsync<PredictionRow[]>(() => trainingService.predictions(split, modelId), [split, modelId]);
}

export function useEventsAnalytics() {
  const { datasetVersion } = useDataset();
  return useAsync(() => analyticsService.eventsAnalytics(), [datasetVersion]);
}

export function useGraphAnalytics() {
  const { datasetVersion } = useDataset();
  return useAsync(() => analyticsService.graphAnalytics(), [datasetVersion]);
}

export function useAttacksAnalytics() {
  const { datasetVersion } = useDataset();
  return useAsync(() => analyticsService.attacksAnalytics(), [datasetVersion]);
}

export function useDatasetsAnalytics() {
  const { datasetVersion } = useDataset();
  return useAsync(() => analyticsService.datasetsAnalytics(), [datasetVersion]);
}

export function useOverview(datasetId?: string) {
  const { datasetVersion, activeDatasetId } = useDataset();
  const targetId = datasetId || activeDatasetId;
  return useAsync<OverviewTelemetry>(
    () => datasetService.overview(targetId),
    [targetId, datasetVersion]
  );
}
