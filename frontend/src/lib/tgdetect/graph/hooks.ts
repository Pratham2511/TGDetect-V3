'use client';

/**
 * React bindings for the attack-reconstruction engine.
 *
 * These mirror the `useAsync` pattern in services/hooks.ts (loading / success /
 * empty / failed, re-run on dataset or model change, cancellation on unmount)
 * but call the pure engine in ./causality instead of a single service method,
 * so a summary can fan out across several /api/events fetches safely.
 */

import { useEffect, useState } from 'react';

import { useDataset } from '@/lib/dataset-context';
import { useModel } from '@/lib/model-context';

import { eventService, graphService } from '../services';
import type { NodeType } from '../types';
import { aggregateAnomaly, buildSuspects, extractAttackSummary } from './causality';
import type { AttackSummary, ExtractOptions, Suspect } from './types';

export type AsyncState = 'idle' | 'loading' | 'success' | 'empty' | 'failed';

export interface AsyncValue<T> {
  state: AsyncState;
  data: T | null;
  error: string | null;
  reload: () => void;
}

/** Local async runner: re-runs when deps, the active model, or a manual reload change. */
function useAsync<T>(fn: () => Promise<T>, deps: unknown[], skip = false): AsyncValue<T> {
  const { activeModelId } = useModel();
  const [state, setState] = useState<AsyncState>(skip ? 'idle' : 'loading');
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [reloadCount, setReloadCount] = useState(0);

  useEffect(() => {
    if (skip) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setState('idle');
      setData(null);
      setError(null);
      return;
    }
    let cancelled = false;
    setState('loading');
    setData(null);
    setError(null);
    fn()
      .then((d) => {
        if (cancelled) return;
        setData(d);
        setState(Array.isArray(d) && d.length === 0 ? 'empty' : 'success');
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : String(err));
        setState('failed');
      });
    return () => {
      cancelled = true;
    };
  }, [...deps, reloadCount, activeModelId, skip]);

  return { state, data, error, reload: () => setReloadCount((n) => n + 1) };
}
// ─────────────────────────────────────────────────────────────────────────────
// Public hooks
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Ranked triage list (§4.4). Fetches flagged nodes + a sample of malicious
 * events, aggregates per-node model anomaly, and returns the top suspects.
 */
export function useSuspects(limit = 40, sample = 400): AsyncValue<Suspect[]> {
  const { datasetVersion } = useDataset();
  return useAsync<Suspect[]>(
    async () => {
      const [nodes, events] = await Promise.all([
        graphService.nodes(Math.max(limit, 200), true),
        eventService.recentMalicious(sample),
      ]);
      const anomaly = aggregateAnomaly(events);
      return buildSuspects(nodes, anomaly).slice(0, limit);
    },
    [limit, sample, datasetVersion],
  );
}

/**
 * Attack-summary subgraph (§4.5) around a focal node. Null focal → idle.
 * Re-runs whenever the focal, any extract option, the dataset, or the model
 * changes, with the previous run cancelled.
 */
export function useAttackSummary(
  focalId: string | null,
  focalType: NodeType,
  opts: ExtractOptions,
): AsyncValue<AttackSummary | null> {
  const { datasetVersion } = useDataset();
  return useAsync<AttackSummary | null>(
    () => (focalId ? extractAttackSummary(focalId, focalType, opts) : Promise.resolve(null)),
    [focalId, focalType, opts.windowSec, opts.maxNodes, opts.maxHosts, datasetVersion],
    focalId == null,
  );
}