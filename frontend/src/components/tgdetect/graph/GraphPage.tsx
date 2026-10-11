'use client';

/**
 * Attack-investigation workspace (ORTHRUS §4.4–4.5).
 *
 * Three panes wire the reconstruction engine to the analyst:
 *   • left   — TriageRail: ranked suspicious nodes (§4.4); picking one sets the focal.
 *   • centre — AttackGraphCanvas: the backward→focal→forward attack-summary subgraph
 *              (§4.5) with trace / window / size controls and a plain-English report.
 *   • right  — NodeExplanationPanel: why the selected node is flagged + its evidence.
 *
 * No 1000-node dump and no force simulation: the summary is a bounded client-side
 * extraction and the canvas paints a precomputed layered layout.
 */

import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { ChevronDown, Layers, Route, Target } from 'lucide-react';

import { useDataset } from '@/lib/dataset-context';
import { formatDurationLong, formatInt, shortNodeId } from '@/lib/tgdetect/formatters';
import { useAttackSummary, useSuspects } from '@/lib/tgdetect/graph/hooks';
import type { ExtractOptions, SummaryNode, TraceMode } from '@/lib/tgdetect/graph/types';
import type { NodeType } from '@/lib/tgdetect/types';
import { cn } from '@/lib/utils';

import { EmptyState, ErrorState, LoadingState, MonoId, SectionTitle } from '../shared/pills';
import { AttackGraphCanvas } from './AttackGraphCanvas';
import { NodeExplanationPanel } from './NodeExplanationPanel';
import { TriageRail } from './TriageRail';

/** Discrete half-window stops (seconds) the window slider steps through. */
const WINDOW_STOPS = [60, 300, 900, 1800, 3600, 10_800, 21_600, 86_400];
const DEFAULT_WINDOW_IDX = 2; // ±15 min

const TRACE_MODES: { value: TraceMode; label: string }[] = [
  { value: 'backward', label: 'Backward' },
  { value: 'both', label: 'Both' },
  { value: 'forward', label: 'Forward' },
];
/** Compact labelled stat chip for the reconstruction report. */
function Chip({ label, value, tone }: { label: string; value: string; tone?: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-md border border-[hsl(var(--border))] bg-[hsl(var(--background))] px-2 py-1">
      <span className="text-[10px] uppercase tracking-wide text-[hsl(var(--muted-foreground))]">{label}</span>
      <span className={cn('mono text-[11px] font-semibold', tone ?? 'text-[hsl(var(--foreground))]')}>{value}</span>
    </span>
  );
}

/** A labelled range slider with a live readout. */
function RangeControl({
  label,
  value,
  min,
  max,
  step,
  onChange,
  readout,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  readout: string;
}) {
  return (
    <label className="flex min-w-[136px] flex-1 flex-col gap-1">
      <span className="flex items-center justify-between gap-2 text-[10px] uppercase tracking-wide text-[hsl(var(--muted-foreground))]">
        {label}
        <span className="mono text-[11px] font-semibold text-[hsl(var(--foreground))]">{readout}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="h-1.5 w-full cursor-pointer accent-[hsl(var(--primary))]"
      />
    </label>
  );
}
export function GraphPage({ onNavigate }: { onNavigate?: (page: string, ctx?: Record<string, unknown>) => void }) {
  const { activeDataset, activeDatasetId, datasetVersion } = useDataset();

  const [focal, setFocal] = useState<{ id: string; type: NodeType } | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [traceMode, setTraceMode] = useState<TraceMode>('both');
  const [windowIdx, setWindowIdx] = useState(DEFAULT_WINDOW_IDX);
  const [maxNodes, setMaxNodes] = useState(60);
  const [maxHosts, setMaxHosts] = useState(6);
  const [reportOpen, setReportOpen] = useState(true);

  const opts: ExtractOptions = useMemo(
    () => ({ windowSec: WINDOW_STOPS[windowIdx], maxNodes, maxHosts }),
    [windowIdx, maxNodes, maxHosts],
  );

  const suspects = useSuspects(40, 400);
  const summary = useAttackSummary(focal?.id ?? null, focal?.type ?? 'PROCESS', opts);
  const summaryData = summary.data;

  // Re-seed from the new dataset's top suspect when the dataset changes.
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setFocal(null);
    setSelectedNodeId(null);
  }, [datasetVersion]);

  // Auto-select the highest-priority suspect as the initial focal.
  useEffect(() => {
    if (focal) return;
    const top = suspects.data?.[0];
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (top) setFocal({ id: top.id, type: top.type });
  }, [suspects.data, focal]);

  // Default the node selection to the focal whenever the summary (re)loads.
  useEffect(() => {
    if (!summaryData) return;
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setSelectedNodeId((cur) =>
      cur && summaryData.nodes.some((n) => n.id === cur) ? cur : summaryData.focalId,
    );
  }, [summaryData]);

  const selectedNode: SummaryNode | null = useMemo(() => {
    if (!summaryData || !selectedNodeId) return null;
    return summaryData.nodes.find((n) => n.id === selectedNodeId) ?? null;
  }, [summaryData, selectedNodeId]);

  const pickSuspect = (id: string, type: NodeType): void => {
    setFocal({ id, type });
    setSelectedNodeId(id);
  };
  const openEvents = (nodeId: string): void => onNavigate?.('events', { nodeId });
  // Non-canvas centre states (loading / empty / error).
  const centerState: ReactNode = (() => {
    if (!focal) {
      if (suspects.state === 'failed') return <ErrorState message={suspects.error ?? 'Failed to load suspects'} />;
      if (suspects.state === 'empty' || (suspects.state === 'success' && (suspects.data?.length ?? 0) === 0)) {
        return (
          <EmptyState
            icon={Target}
            title="No suspicious nodes"
            description="This dataset has no malicious-labelled or model-flagged nodes to reconstruct."
          />
        );
      }
      return <LoadingState label="Finding suspicious nodes…" />;
    }
    if (summary.state === 'failed') return <ErrorState message={summary.error ?? 'Failed to reconstruct attack'} />;
    return <LoadingState label="Reconstructing attack…" />;
  })();

  const rightPane: ReactNode = summaryData ? (
    <NodeExplanationPanel
      node={selectedNode}
      summary={summaryData}
      onOpenEvents={openEvents}
      className="h-[600px] xl:col-span-3 xl:h-full"
    />
  ) : (
    <div className="tg-panel flex h-[600px] flex-col items-center justify-center p-6 text-center xl:col-span-3 xl:h-full">
      <Target className="mb-2 size-7 text-[hsl(var(--muted-foreground))] opacity-50" />
      <p className="text-sm font-semibold text-[hsl(var(--foreground))]">No reconstruction yet</p>
      <p className="mt-1 text-xs text-[hsl(var(--muted-foreground))]">Pick a suspect to see why its nodes are flagged.</p>
    </div>
  );

  return (
    <div className="flex flex-col gap-3">
      {/* header */}
      <div className="tg-panel flex flex-wrap items-center gap-x-4 gap-y-2 px-3.5 py-2.5">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <Target className="size-4 text-[hsl(var(--danger))]" />
            <SectionTitle className="inline">Attack Reconstruction</SectionTitle>
            <span className="mono rounded border border-[hsl(var(--border))] bg-[hsl(var(--background))] px-2 py-0.5 text-[10px] font-semibold text-[hsl(var(--primary))]">
              {activeDataset?.name ?? activeDatasetId ?? 'ACTIVE DATASET'}
            </span>
          </div>
          <p className="mt-0.5 text-[11px] text-[hsl(var(--muted-foreground))]">
            Backward → focal → forward tracing (ORTHRUS §4.5). Pick a suspect to reconstruct a bounded attack summary.
          </p>
        </div>
        {focal && (
          <div className="flex items-center gap-2 text-[11px] text-[hsl(var(--muted-foreground))]">
            <span>Focal</span>
            <MonoId truncateAt={28} className="text-[12px]">{shortNodeId(focal.id)}</MonoId>
          </div>
        )}
      </div>
      {/* workspace */}
      <div className="grid grid-cols-1 gap-3 xl:h-[82vh] xl:grid-cols-12">
        <TriageRail
          suspects={suspects.data ?? []}
          state={suspects.state}
          error={suspects.error}
          selectedId={focal?.id ?? null}
          onSelect={pickSuspect}
          className="h-[380px] xl:col-span-3 xl:h-full"
        />

        {/* centre: controls + canvas + report */}
        <div className="flex h-[560px] min-h-0 flex-col gap-3 xl:col-span-6 xl:h-full">
          <div className="tg-panel flex flex-wrap items-center gap-x-5 gap-y-3 px-3 py-2.5">
            <div className="flex items-center gap-0.5 rounded-md border border-[hsl(var(--border))] bg-[hsl(var(--background))] p-0.5">
              {TRACE_MODES.map((m) => (
                <button
                  key={m.value}
                  type="button"
                  onClick={() => setTraceMode(m.value)}
                  className={cn(
                    'inline-flex items-center gap-1 rounded px-2 py-1 text-[11px] font-medium transition-colors',
                    traceMode === m.value
                      ? 'bg-[hsl(var(--primary)/0.14)] text-[hsl(var(--primary))]'
                      : 'text-[hsl(var(--muted-foreground))] hover:text-[hsl(var(--foreground))]',
                  )}
                >
                  <Route className="size-3" /> {m.label}
                </button>
              ))}
            </div>
            <RangeControl
              label="Trace window"
              value={windowIdx}
              min={0}
              max={WINDOW_STOPS.length - 1}
              step={1}
              onChange={setWindowIdx}
              readout={`±${formatDurationLong(WINDOW_STOPS[windowIdx])}`}
            />
            <RangeControl label="Max nodes" value={maxNodes} min={20} max={150} step={10} onChange={setMaxNodes} readout={String(maxNodes)} />
            <RangeControl label="Pivot hosts" value={maxHosts} min={2} max={16} step={1} onChange={setMaxHosts} readout={String(maxHosts)} />
          </div>

          <div className="relative min-h-0 flex-1">
            {focal && summaryData ? (
              <AttackGraphCanvas
                summary={summaryData}
                selectedNodeId={selectedNodeId}
                onSelectNode={setSelectedNodeId}
                traceMode={traceMode}
              />
            ) : (
              <div className="tg-panel flex h-full items-center justify-center">{centerState}</div>
            )}
          </div>
          {summaryData && (
            <div className="tg-panel shrink-0">
              <button
                type="button"
                onClick={() => setReportOpen((v) => !v)}
                className="tg-hud-header flex w-full items-center justify-between"
              >
                <span className="inline-flex items-center gap-1.5">
                  <Layers className="size-3.5 text-[hsl(var(--primary))]" />
                  <SectionTitle className="inline">Reconstruction report</SectionTitle>
                </span>
                <ChevronDown
                  className={cn(
                    'size-4 text-[hsl(var(--muted-foreground))] transition-transform',
                    reportOpen ? '' : '-rotate-90',
                  )}
                />
              </button>
              {reportOpen && (
                <div className="max-h-[26vh] space-y-3 overflow-y-auto p-3">
                  <div className="flex flex-wrap gap-1.5">
                    <Chip label="Nodes" value={formatInt(summaryData.stats.renderedNodes)} />
                    <Chip label="Events" value={formatInt(summaryData.stats.renderedEvents)} />
                    <Chip label="Hosts" value={formatInt(summaryData.stats.hostsTouched)} />
                    <Chip label="Co-proc" value={formatInt(summaryData.stats.colocatedProcesses)} />
                    <Chip label="Malicious" value={formatInt(summaryData.stats.maliciousEvents)} tone="text-[hsl(var(--danger))]" />
                    <Chip label="Flagged" value={formatInt(summaryData.stats.flaggedEvents)} tone="text-[hsl(var(--danger))]" />
                    <Chip label="Anom max" value={`${Math.round(summaryData.stats.anomalyMax * 100)}%`} />
                  </div>
                  <div className="space-y-2">
                    {summaryData.narrative.map((para) => (
                      <p key={para} className="text-[11px] leading-relaxed text-[hsl(var(--muted-foreground))]">
                        {para}
                      </p>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {rightPane}
      </div>
    </div>
  );
}
