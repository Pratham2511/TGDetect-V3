'use client';

import { useMemo } from 'react';
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  Boxes,
  CheckCircle2,
  Clock,
  Cpu,
  Crosshair,
  Database,
  ExternalLink,
  Eye,
  FileText,
  GitBranch,
  Layers,
  Network,
  Radio,
  RefreshCw,
  Search,
  ShieldAlert,
  ShieldCheck,
  Tag,
  Target,
  TrendingUp,
  Zap,
} from 'lucide-react';
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import {
  useGraphStats,
  useRecentMalicious,
  useTGNNSummary,
  useEvaluationRun,
  useEventsAnalytics,
  useOverview,
} from '@/lib/tgdetect/services/hooks';
import { useModel } from '@/lib/model-context';
import { useDataset } from '@/lib/dataset-context';
import {
  formatBytes,
  formatDurationLong,
  formatEpoch,
  formatEpochTime,
  formatInt,
  formatPercent,
} from '@/lib/tgdetect/formatters';
import { CHART_AXIS_STYLE, CHART_COLORS, CHART_GRID_STYLE, CHART_TOOLTIP_STYLE } from '@/lib/tgdetect/chart-constants';
import {
  EmptyState,
  ErrorState,
  LabelPill,
  LoadingState,
  MonoId,
  RelationPill,
  SectionTitle,
} from '../shared/pills';
import type { Dataset, GraphStats, TGEvent } from '@/lib/tgdetect/types';

export function OverviewPage({ onNavigate }: { onNavigate?: (page: string, ctx?: Record<string, unknown>) => void }) {
  const { activeModel, activeModelId, apiHealth } = useModel();
  const { activeDataset, activeDatasetId } = useDataset();
  const overviewRes = useOverview();
  const statsRes = useGraphStats();
  const recentMalRes = useRecentMalicious(8);
  const evalRes = useEvaluationRun('test', undefined, activeModelId);
  const summaryRes = useTGNNSummary(activeModelId);

  // Derive active telemetry values (from overview endpoint or stats fallback)
  const o = overviewRes.data;
  const isBenchmark = activeDatasetId === 'dedale-host-memory';
  const totalEvents = o?.total_events ?? statsRes.data?.graph?.total_events ?? 0;
  const maliciousEvents = o?.malicious_events ?? statsRes.data?.graph?.malicious_events ?? 0;
  const benignEvents = o?.benign_events ?? statsRes.data?.graph?.benign_events ?? Math.max(0, totalEvents - maliciousEvents);
  const totalNodes = o?.total_nodes ?? statsRes.data?.graph?.total_nodes ?? 0;
  const totalEdges = o?.total_edges ?? statsRes.data?.graph?.total_edges ?? totalEvents;
  const chainCount = o?.chain_count ?? statsRes.data?.attacks?.total_chains ?? 0;
  const timeSpan = o?.time_span_s ?? statsRes.data?.graph?.timestamp_span_s ?? null;
  const datasetName = o?.dataset ?? activeDataset?.name ?? activeDatasetId;
  const provenance = o?.provenance ?? activeDataset?.provenance ?? 'Active telemetry partition';

  return (
    <div className="space-y-4">
      {/* SECTION A — ACTIVE TELEMETRY (Primary Operational Command Surface) */}
      <ActiveTelemetryCommandDeck
        activeDatasetId={activeDatasetId}
        datasetName={datasetName}
        provenance={provenance}
        isBenchmark={isBenchmark}
        totalEvents={totalEvents}
        maliciousEvents={maliciousEvents}
        benignEvents={benignEvents}
        totalNodes={totalNodes}
        totalEdges={totalEdges}
        chainCount={chainCount}
        timeSpan={timeSpan}
        apiHealth={apiHealth}
        activeDataset={activeDataset}
        onNavigate={onNavigate}
      />

      {/* Primary Analytical Split: SECTION B (Temporal Activity) & SECTION C (Topology & Threat Matrix) */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* SECTION B: Temporal Activity (2/3 width) */}
        <TemporalActivityPanel
          stats={statsRes.data}
          totalEvents={totalEvents}
          className="lg:col-span-2"
          onNavigate={onNavigate}
        />

        {/* SECTION C: Graph Topology & Threat Matrix (1/3 width) */}
        <GraphTopologyPanel
          stats={statsRes.data}
          totalNodes={totalNodes}
          totalEdges={totalEdges}
          chainCount={chainCount}
          onNavigate={onNavigate}
        />
      </div>

      {/* Investigative Forensic Stream: Recent Malicious Telemetry */}
      <RecentMaliciousWorkbench
        events={recentMalRes.data ?? []}
        loading={recentMalRes.state === 'loading'}
        datasetName={datasetName}
        onViewAll={() => onNavigate?.('events', { labelFilter: 1 })}
        onEventClick={(eventId) => onNavigate?.('events', { eventId })}
        onNavigate={onNavigate}
      />

      {/* SECTION D — PRODUCTION MODEL BENCHMARK (Strictly Separated from Live Telemetry) */}
      <ModelBenchmarkSection
        activeModel={activeModel}
        evalData={evalRes.data}
        summaryData={summaryRes.data}
        onNavigate={onNavigate}
      />
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// SECTION A: Active Telemetry Command Deck
// ─────────────────────────────────────────────────────────────────────────────

interface ActiveTelemetryProps {
  activeDatasetId: string;
  datasetName: string;
  provenance: string;
  isBenchmark: boolean;
  totalEvents: number;
  maliciousEvents: number;
  benignEvents: number;
  totalNodes: number;
  totalEdges: number;
  chainCount: number;
  timeSpan: number | null;
  apiHealth: string;
  activeDataset: Dataset | null;
  onNavigate?: (page: string, ctx?: Record<string, unknown>) => void;
}

function ActiveTelemetryCommandDeck({
  activeDatasetId,
  datasetName,
  provenance,
  isBenchmark,
  totalEvents,
  maliciousEvents,
  benignEvents,
  totalNodes,
  totalEdges,
  chainCount,
  timeSpan,
  apiHealth,
  activeDataset,
  onNavigate,
}: ActiveTelemetryProps) {
  const malRatio = totalEvents > 0 ? (maliciousEvents / totalEvents) * 100 : 0;

  return (
    <div className="tg-panel hud-bracket p-5 space-y-4 border border-[hsl(var(--border))] bg-[hsl(var(--card))] shadow-sm">
      {/* Top Header Rail */}
      <div className="flex items-center justify-between gap-4 flex-wrap border-b border-[hsl(var(--border))] pb-3.5">
        <div className="flex items-center gap-3">
          <div className="size-9 rounded bg-[hsl(var(--primary)/0.12)] border border-[hsl(var(--primary)/0.3)] flex items-center justify-center text-[hsl(var(--primary))] shrink-0">
            <Radio className="size-4.5 animate-pulse" />
          </div>
          <div>
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-xs uppercase tracking-widest font-mono text-[hsl(var(--primary))] font-bold">
                ACTIVE TELEMETRY
              </span>
              <span className="text-[10px] text-[hsl(var(--muted-foreground))]">/</span>
              <h2 className="text-sm font-bold text-[hsl(var(--foreground))] font-mono">
                {datasetName}
              </h2>
              {isBenchmark ? (
                <span className="text-[9px] font-mono px-2 py-0.5 rounded border border-amber-500/30 bg-amber-500/10 text-amber-500 font-bold uppercase">
                  ◈ REFERENCE BENCHMARK
                </span>
              ) : (
                <span className="text-[9px] font-mono px-2 py-0.5 rounded border border-emerald-500/40 bg-emerald-500/15 text-emerald-500 font-bold uppercase flex items-center gap-1">
                  <span className="size-1.5 rounded-full bg-emerald-500 animate-pulse" />
                  ● LIVE INGESTED TELEMETRY
                </span>
              )}
            </div>
            <div className="text-[11px] font-mono text-[hsl(var(--muted-foreground))] mt-0.5 flex items-center gap-3">
              <span>Origin: <strong className="text-[hsl(var(--foreground))]">{provenance}</strong></span>
              <span>•</span>
              <span>API Gateway: <strong className={apiHealth === 'connected' ? 'text-emerald-500' : 'text-rose-500'}>{apiHealth.toUpperCase()}</strong></span>
            </div>
          </div>
        </div>

        {onNavigate && (
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => onNavigate('datasets')}
              className="px-3 py-1.5 text-xs font-mono border border-[hsl(var(--border))] rounded hover:bg-[hsl(var(--card-hover))] transition-colors flex items-center gap-1.5"
            >
              <Database className="size-3 text-[hsl(var(--muted-foreground))]" />
              <span>Switch Dataset</span>
            </button>
            <button
              type="button"
              onClick={() => onNavigate('datasets', { upload: true })}
              className="px-3 py-1.5 text-xs font-mono font-semibold bg-[hsl(var(--primary))] text-[hsl(var(--primary-foreground))] rounded hover:opacity-90 transition-opacity flex items-center gap-1.5 shadow-xs"
            >
              <Boxes className="size-3" />
              <span>Ingest Telemetry</span>
            </button>
          </div>
        )}
      </div>

      {/* High-Density Inline Metrics Strip (Level 3 Hierarchy) */}
      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-3">
        <InlineTelemetryMetric
          label="EVENT VOLUME"
          value={formatInt(totalEvents)}
          sub="recorded telemetry events"
          color="cyan"
        />
        <InlineTelemetryMetric
          label="MALICIOUS EVENTS"
          value={formatInt(maliciousEvents)}
          sub={`${malRatio.toFixed(2)}% threat ratio`}
          color={maliciousEvents > 0 ? 'crimson' : 'green'}
        />
        <InlineTelemetryMetric
          label="BENIGN EVENTS"
          value={formatInt(benignEvents)}
          sub={`${(100 - malRatio).toFixed(1)}% normal`}
          color="green"
        />
        <InlineTelemetryMetric
          label="UNIQUE NODES"
          value={formatInt(totalNodes)}
          sub="entity entities"
          color="cyan"
        />
        <InlineTelemetryMetric
          label="GRAPH EDGES"
          value={formatInt(totalEdges)}
          sub="process-to-host observations"
          color="cyan"
        />
        <InlineTelemetryMetric
          label="ATTACK CHAINS"
          value={formatInt(chainCount)}
          sub="causal paths"
          color={chainCount > 0 ? 'amber' : 'gray'}
        />
        <InlineTelemetryMetric
          label="TEMPORAL SPAN"
          value={timeSpan ? formatDurationLong(timeSpan) : '—'}
          sub="capture duration"
          color="teal"
        />
      </div>

      {/* Data Provenance Inspector Bar */}
      <div className="p-2.5 rounded bg-[hsl(var(--background))] border border-[hsl(var(--border))] flex items-center justify-between gap-4 text-xs font-mono flex-wrap">
        <div className="flex items-center gap-4 flex-wrap">
          <span className="text-[10px] uppercase tracking-wider text-[hsl(var(--muted-foreground))] font-bold flex items-center gap-1">
            <FileText className="size-3 text-[hsl(var(--primary))]" />
            DATA PROVENANCE:
          </span>
          <span className="text-[hsl(var(--muted-foreground))]">
            Dataset ID: <strong className="text-[hsl(var(--foreground))]">{activeDatasetId}</strong>
          </span>
          <span className="text-[hsl(var(--muted-foreground))]">
            Format: <strong className="text-[hsl(var(--primary))]">DEDALE Sysmon</strong>
          </span>
          <span className="text-[hsl(var(--muted-foreground))]">
            Artifacts: <strong className="text-emerald-500">Recorded NPZ + JSON exports</strong>
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[10px] uppercase font-bold text-emerald-500 flex items-center gap-1">
            <span className="size-1.5 rounded-full bg-emerald-500" />
            TELEMETRY ACTIVE
          </span>
        </div>
      </div>
    </div>
  );
}

function InlineTelemetryMetric({
  label,
  value,
  sub,
  color = 'cyan',
}: {
  label: string;
  value: string;
  sub: string;
  color?: 'cyan' | 'crimson' | 'green' | 'amber' | 'teal' | 'gray';
}) {
  const colorMap = {
    cyan: 'text-[hsl(var(--primary))]',
    crimson: 'text-rose-500',
    green: 'text-emerald-500',
    amber: 'text-amber-500',
    teal: 'text-[hsl(var(--teal))]',
    gray: 'text-[hsl(var(--muted-foreground))]',
  };

  return (
    <div className="p-3 rounded bg-[hsl(var(--background))] border border-[hsl(var(--border))] flex flex-col justify-between space-y-1">
      <span className="text-[9px] uppercase tracking-widest text-[hsl(var(--muted-foreground))] font-mono font-bold">
        {label}
      </span>
      <span className={`text-lg font-mono font-bold leading-none ${colorMap[color]}`}>
        {value}
      </span>
      <span className="text-[10px] font-mono text-[hsl(var(--muted-foreground))] truncate">
        {sub}
      </span>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// SECTION B: Temporal Activity Panel (Truthful timeline or domain-specific empty state)
// ─────────────────────────────────────────────────────────────────────────────

function TemporalActivityPanel({
  stats,
  totalEvents,
  className,
  onNavigate,
}: {
  stats: GraphStats | null;
  totalEvents: number;
  className?: string;
  onNavigate?: (page: string) => void;
}) {
  const { data: analytics, state } = useEventsAnalytics();

  const data = useMemo(() => {
    const rawTimeline = analytics?.timeline;
    if (rawTimeline && rawTimeline.length > 0) {
      return rawTimeline.map((b) => ({
        label: formatEpochTime(b.bucket_start).slice(0, 5),
        benign: b.benign,
        malicious: b.malicious,
        unlabeled: (b as any).unlabeled ?? 0,
      }));
    }
    return [];
  }, [analytics]);

  return (
    <div className={`tg-card p-4 space-y-3 flex flex-col justify-between ${className ?? ''}`}>
      <div className="flex items-center justify-between border-b border-[hsl(var(--border))] pb-2">
        <SectionTitle
          right={
            <div className="flex gap-3 text-[10px] font-mono">
              <span className="flex items-center gap-1.5">
                <span className="size-2 rounded-full bg-emerald-500" />
                Labeled benign
              </span>
              <span className="flex items-center gap-1.5">
                <span className="size-2 rounded-full bg-rose-500" />
                Labeled malicious
              </span>
            </div>
          }
        >
          Temporal Activity & Ingestion Density
        </SectionTitle>
        <span className="text-[10px] font-mono text-[hsl(var(--muted-foreground))]">
          ACTIVE DATASET TIMELINE
        </span>
      </div>

      {state === 'loading' ? (
        <div className="h-[220px] flex items-center justify-center text-xs font-mono text-[hsl(var(--muted-foreground))]">
          <LoadingState label="Reading temporal event stream from active dataset..." />
        </div>
      ) : data.length === 0 || totalEvents === 0 ? (
        <div className="h-[220px] flex flex-col items-center justify-center text-center p-6 space-y-2 bg-[hsl(var(--background))] rounded border border-[hsl(var(--border))]">
          <Clock className="size-6 text-[hsl(var(--muted-foreground))] opacity-40" />
          <div className="text-xs font-mono font-bold text-[hsl(var(--foreground))] uppercase tracking-wider">
            TEMPORAL SIGNAL ABSENT
          </div>
          <p className="text-[11px] font-mono text-[hsl(var(--muted-foreground))] max-w-sm">
            No timestamped event relationships were produced from the active dataset partition.
          </p>
          {onNavigate && (
            <button
              type="button"
              onClick={() => onNavigate('datasets')}
              className="mt-2 px-2.5 py-1 text-[11px] font-mono rounded border border-[hsl(var(--border))] hover:bg-[hsl(var(--card-hover))]"
            >
              Select Another Dataset →
            </button>
          )}
        </div>
      ) : (
        <>
          <div className="text-[10px] text-[hsl(var(--muted-foreground))] font-mono">
            {stats ? `${formatEpoch(stats.graph.earliest_timestamp)} → ${formatEpoch(stats.graph.latest_timestamp)}` : 'Active horizon'} · {data.length} temporal buckets
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -16 }}>
              <defs>
                <linearGradient id="overviewBenignGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#10b981" stopOpacity={0.7} />
                  <stop offset="95%" stopColor="#10b981" stopOpacity={0.05} />
                </linearGradient>
                <linearGradient id="overviewMalGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#f43f5e" stopOpacity={0.8} />
                  <stop offset="95%" stopColor="#f43f5e" stopOpacity={0.05} />
                </linearGradient>
              </defs>
              <CartesianGrid {...CHART_GRID_STYLE} />
              <XAxis dataKey="label" {...CHART_AXIS_STYLE} interval="preserveStartEnd" />
              <YAxis {...CHART_AXIS_STYLE} width={36} />
              <Tooltip
                {...CHART_TOOLTIP_STYLE}
                formatter={(value: any, name: any) => [
                  typeof value === 'number' && !Number.isNaN(value) ? formatInt(value) : '0',
                  name === 'benign' ? 'Labeled benign' : name === 'unlabeled' ? 'Unknown ground truth' : 'Labeled malicious',
                ]}
              />
              <Area type="monotone" dataKey="unlabeled" stackId="1" stroke="#94a3b8" fill="#94a3b8" isAnimationActive={false} />
              <Area type="monotone" dataKey="benign" stackId="1" stroke="#10b981" strokeWidth={1.5} fill="url(#overviewBenignGrad)" isAnimationActive={false} />
              <Area type="monotone" dataKey="malicious" stackId="1" stroke="#f43f5e" strokeWidth={1.5} fill="url(#overviewMalGrad)" isAnimationActive={false} />
            </AreaChart>
          </ResponsiveContainer>
        </>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// SECTION C: Graph Topology Panel
// ─────────────────────────────────────────────────────────────────────────────

function GraphTopologyPanel({
  stats,
  totalNodes,
  totalEdges,
  chainCount,
  onNavigate,
}: {
  stats: GraphStats | null;
  totalNodes: number;
  totalEdges: number;
  chainCount: number;
  onNavigate?: (page: string) => void;
}) {
  const nodeTypes = Object.entries(stats?.graph?.node_types || {});
  const relTypes = Object.entries(stats?.graph?.relation_types || {});
  const attacks = stats?.attacks;

  if (!stats || totalNodes === 0) {
    return (
      <div className="tg-card p-4 flex flex-col items-center justify-center text-center space-y-2 bg-[hsl(var(--background))] border border-[hsl(var(--border))]">
        <Network className="size-6 text-[hsl(var(--muted-foreground))] opacity-40" />
        <div className="text-xs font-mono font-bold text-[hsl(var(--foreground))] uppercase tracking-wider">
          NO GRAPH TOPOLOGY AVAILABLE
        </div>
        <p className="text-[11px] font-mono text-[hsl(var(--muted-foreground))]">
          This dataset contains no extracted graph nodes or edge relationships.
        </p>
      </div>
    );
  }

  return (
    <div className="tg-card p-4 space-y-4 flex flex-col justify-between">
      <div>
        <div className="flex items-center justify-between border-b border-[hsl(var(--border))] pb-2 mb-3">
          <SectionTitle>Graph Topology & Threat Matrix</SectionTitle>
          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[hsl(var(--primary)/0.1)] text-[hsl(var(--primary))] border border-[hsl(var(--primary)/0.2)]">
            LIVE TELEMETRY
          </span>
        </div>

        {/* Node Entity Distribution */}
        <div className="space-y-2 mb-4">
          <div className="text-[10px] font-mono uppercase tracking-wider text-[hsl(var(--muted-foreground))] flex items-center justify-between">
            <span>Entity Types</span>
            <span className="font-bold text-[hsl(var(--foreground))]">{formatInt(totalNodes)} Total</span>
          </div>
          <div className="space-y-1.5">
            {nodeTypes.map(([type, count]) => {
              const pct = Math.round((count / Math.max(1, totalNodes)) * 100);
              return (
                <div key={type} className="p-2 rounded bg-[hsl(var(--background))] border border-[hsl(var(--border))] space-y-1">
                  <div className="flex items-center justify-between text-xs font-mono">
                    <span className="font-semibold text-[hsl(var(--foreground))]">{type}</span>
                    <span className="text-[hsl(var(--muted-foreground))]">{formatInt(count)} ({pct}%)</span>
                  </div>
                  <div className="h-1.5 w-full rounded-full bg-[hsl(var(--muted)/0.3)] overflow-hidden">
                    <div className="h-full rounded-full bg-cyan-500" style={{ width: `${Math.max(5, pct)}%` }} />
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Edge Relations */}
        <div className="space-y-2 mb-4">
          <div className="text-[10px] font-mono uppercase tracking-wider text-[hsl(var(--muted-foreground))] flex items-center justify-between">
            <span>Temporal Relations</span>
            <span className="font-bold text-[hsl(var(--foreground))]">{formatInt(totalEdges)} Edges</span>
          </div>
          <div className="space-y-1.5">
            {relTypes.map(([rel, count]) => {
              const pct = Math.round((count / Math.max(1, totalEdges)) * 100);
              return (
                <div key={rel} className="p-2 rounded bg-[hsl(var(--background))] border border-[hsl(var(--border))] space-y-1">
                  <div className="flex items-center justify-between text-xs font-mono">
                    <span className="font-semibold text-[hsl(var(--primary))]">{rel}</span>
                    <span className="text-[hsl(var(--muted-foreground))]">{formatInt(count)} ({pct}%)</span>
                  </div>
                  <div className="h-1.5 w-full rounded-full bg-[hsl(var(--muted)/0.3)] overflow-hidden">
                    <div className="h-full rounded-full bg-emerald-500" style={{ width: `${Math.max(5, pct)}%` }} />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Attack Attribution Footer */}
      <div className="p-3 rounded border border-amber-500/30 bg-amber-500/5 space-y-2">
        <div className="flex items-center justify-between text-xs">
          <span className="font-bold text-amber-500 flex items-center gap-1.5 font-mono">
            <GitBranch className="size-3.5" />
            Attack Reconstruction
          </span>
          {onNavigate && (
            <button
              type="button"
              onClick={() => onNavigate('chains')}
              className="text-[10px] text-amber-500 hover:underline font-mono"
            >
              Explore Chains →
            </button>
          )}
        </div>
        <div className="grid grid-cols-2 gap-2 text-[11px] font-mono">
          <div>
            <span className="text-[hsl(var(--muted-foreground))]\">Total Chains:</span>{' '}
            <strong className="text-[hsl(var(--foreground))]">{formatInt(chainCount)}</strong>
          </div>
          <div>
            <span className="text-[hsl(var(--muted-foreground))]\">Tracked:</span>{' '}
            <strong className="text-rose-500">{formatInt(attacks?.malicious_events_tracked ?? 0)}</strong>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Recent Malicious Workbench
// ─────────────────────────────────────────────────────────────────────────────

function RecentMaliciousWorkbench({
  events,
  loading,
  datasetName,
  onViewAll,
  onEventClick,
  onNavigate,
}: {
  events: TGEvent[];
  loading: boolean;
  datasetName: string;
  onViewAll?: () => void;
  onEventClick?: (eventId: string) => void;
  onNavigate?: (page: string) => void;
}) {
  return (
    <div className="tg-card p-4 space-y-3">
      <div className="flex items-center justify-between border-b border-[hsl(var(--border))] pb-2">
        <div className="flex items-center gap-2">
          <SectionTitle>Recent Malicious Telemetry Stream</SectionTitle>
          <span className="text-[9px] font-mono px-1.5 py-0.2 rounded border border-rose-500/30 bg-rose-500/10 text-rose-500 font-bold">
            THREAT RADAR
          </span>
        </div>
        {onViewAll && events.length > 0 && (
          <button onClick={onViewAll} className="text-xs font-mono text-[hsl(var(--primary))] hover:underline">
            view all malicious events →
          </button>
        )}
      </div>

      {loading ? (
        <LoadingState label="Inspecting active dataset for threat events…" />
      ) : events.length === 0 ? (
        <div className="p-6 text-center space-y-2 bg-[hsl(var(--background))] rounded border border-[hsl(var(--border))]">
          <ShieldCheck className="size-6 text-emerald-500 mx-auto opacity-70" />
          <div className="text-xs font-mono font-bold text-[hsl(var(--foreground))]">
            ZERO MALICIOUS ANOMALIES DETECTED
          </div>
          <p className="text-[11px] font-mono text-[hsl(var(--muted-foreground))] max-w-sm mx-auto">
            No malicious events detected in telemetry partition '{datasetName}'.
          </p>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-xs font-mono">
            <thead>
              <tr className="text-left text-[10px] uppercase tracking-wider text-[hsl(var(--muted-foreground))] border-b border-[hsl(var(--border))]">
                <th className="py-2 pr-3">Event ID</th>
                <th className="py-2 pr-3">Timestamp</th>
                <th className="py-2 pr-3">Source Entity</th>
                <th className="py-2 pr-3">→</th>
                <th className="py-2 pr-3">Target Entity</th>
                <th className="py-2 pr-3">Relation</th>
                <th className="py-2 pr-3">Tactics</th>
                <th className="py-2 pr-3">Causal Chain</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[hsl(var(--border)/0.5)]">
              {events.map((e) => (
                <tr
                  key={e.event_id}
                  className="hover:bg-[hsl(var(--card-hover))] cursor-pointer transition-colors"
                  onClick={() => onEventClick?.(e.event_id)}
                >
                  <td className="py-2 pr-3 font-semibold text-[hsl(var(--primary))]"><MonoId>{e.event_id}</MonoId></td>
                  <td className="py-2 pr-3 text-[11px] text-[hsl(var(--muted-foreground))]">{formatEpochTime(e.ts)}</td>
                  <td className="py-2 pr-3 text-[11px] font-semibold text-[hsl(var(--foreground))]">{e.src_id}</td>
                  <td className="py-2 pr-3 text-[hsl(var(--muted-foreground))]">→</td>
                  <td className="py-2 pr-3 text-[11px] text-[hsl(var(--foreground))]">{e.dst_id}</td>
                  <td className="py-2 pr-3"><RelationPill relation={e.relation} /></td>
                  <td className="py-2 pr-3 text-[11px]">
                    {e.tactics && e.tactics.length > 0 ? (
                      <span className="text-rose-500 font-bold">{e.tactics.join(', ')}</span>
                    ) : (
                      <span className="text-[hsl(var(--muted-foreground))]">botnet</span>
                    )}
                  </td>
                  <td className="py-2 pr-3">
                    {e.chain_id ? (
                      <span className="px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-500 border border-amber-500/30 text-[10px]">
                        {e.chain_id.slice(0, 14)}
                      </span>
                    ) : (
                      <span className="text-[hsl(var(--muted-foreground))]">—</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// SECTION D: Production Model Benchmark (Strictly Separated from Live Telemetry)
// ─────────────────────────────────────────────────────────────────────────────

function ModelBenchmarkSection({
  activeModel,
  evalData,
  summaryData,
  onNavigate,
}: {
  activeModel: any;
  evalData: any;
  summaryData: any;
  onNavigate?: (page: string, ctx?: Record<string, unknown>) => void;
}) {
  const m = evalData?.metrics ?? activeModel?.metrics ?? {};
  const rocAuc = m.auc_roc ?? m.roc_auc ?? null;
  const prAuc = m.auc_pr ?? m.pr_auc ?? null;
  const f1 = m.f1 ?? null;
  const recall1Pct = m.recall_at_1pct_fpr ?? m.recall_1pct_fpr ?? null;
  const precision = m.precision ?? null;
  const recall = m.recall ?? null;

  if (!activeModel || rocAuc == null || prAuc == null || f1 == null || precision == null || recall == null) return <LoadingState label="Loading checkpoint benchmark…" />;
  return (
    <div className="tg-panel p-5 border border-indigo-500/30 bg-indigo-500/5 space-y-4">
      {/* Benchmark Header with Explicit Origin Guardrail */}
      <div className="flex items-center justify-between gap-4 flex-wrap border-b border-indigo-500/20 pb-3">
        <div className="flex items-center gap-3">
          <div className="size-9 rounded bg-indigo-500/20 border border-indigo-500/40 flex items-center justify-center text-indigo-400 shrink-0">
            <Cpu className="size-5" />
          </div>
          <div>
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-xs uppercase tracking-widest font-mono text-indigo-400 font-bold">
                PRODUCTION MODEL BENCHMARK
              </span>
              <span className="text-[10px] text-[hsl(var(--muted-foreground))]">/</span>
              <span className="text-sm font-bold font-mono text-[hsl(var(--foreground))]">
                {activeModel?.name ?? 'Loading model…'} ({activeModel?.id ?? '—'})
              </span>
              <span className="text-[9px] font-mono px-2 py-0.5 rounded border border-indigo-500/40 bg-indigo-500/10 text-indigo-300 font-bold uppercase">
                {activeModel?.target?.toUpperCase() ?? 'EDGE'} CLASSIFIER
              </span>
            </div>
            <div className="text-[11px] font-mono text-[hsl(var(--muted-foreground))] mt-0.5">
              Spatiotemporal Architecture: <strong className="text-[hsl(var(--foreground))]">{summaryData?.gnn_operator ?? activeModel?.description}</strong> · Trainable Params: <strong className="text-cyan-400">{formatInt(activeModel?.trainable_parameters ?? 0)}</strong>
            </div>
          </div>
        </div>

        {onNavigate && (
          <button
            type="button"
            onClick={() => onNavigate('model', { sub: 'evaluation' })}
            className="px-3 py-1.5 text-xs font-mono rounded border border-indigo-500/30 bg-indigo-500/10 text-indigo-300 hover:bg-indigo-500/20 transition-colors flex items-center gap-1.5"
          >
            <Crosshair className="size-3" />
            <span>Inspect Model Architecture →</span>
          </button>
        )}
      </div>

      {/* Explicit Guardrail Notice */}
      <div className="p-2.5 rounded bg-[hsl(var(--background))] border border-indigo-500/20 text-xs font-mono text-[hsl(var(--muted-foreground))] flex items-center gap-2">
        <span className="text-indigo-400 text-sm">◈</span>
        <span>
          <strong>HELD-OUT BENCHMARK EVALUATION:</strong> The metrics below come from the selected checkpoint on the chronological DEDALE test split ({formatInt(m.num_samples ?? m.samples ?? 0)} recorded events). They are <em>not</em> derived from your live uploaded dataset telemetry.
        </span>
      </div>

      {/* Benchmark KPI Grid */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        <BenchmarkKpiCard
          label="ROC-AUC"
          value={rocAuc.toFixed(4)}
          hint="Threshold-free ranking"
          badge="Generalization"
        />
        <BenchmarkKpiCard
          label="PR-AP"
          value={prAuc.toFixed(4)}
          hint="Average precision"
          badge="Precision-Recall"
        />
        <BenchmarkKpiCard
          label="F1"
          value={f1.toFixed(4)}
          hint="Validation-selected threshold"
          badge={`τ = ${m.threshold?.toFixed(4) ?? "validation"}`}
        />
        <BenchmarkKpiCard
          label="RECALL @ 1% FPR"
          value={recall1Pct == null ? "Unavailable" : `${(recall1Pct * 100).toFixed(2)}%`}
          hint="Requires a measured operating point"
          badge="SOC Benchmark"
          highlight
        />
        <BenchmarkKpiCard
          label="PRECISION"
          value={precision.toFixed(4)}
          hint="Positive predictive rate"
          badge="validation threshold"
        />
        <BenchmarkKpiCard
          label="RECALL"
          value={recall.toFixed(4)}
          hint="True positive rate"
          badge="validation threshold"
        />
      </div>
    </div>
  );
}

function BenchmarkKpiCard({
  label,
  value,
  hint,
  badge,
  highlight = false,
}: {
  label: string;
  value: string;
  hint: string;
  badge: string;
  highlight?: boolean;
}) {
  return (
    <div className={`p-3 rounded bg-[hsl(var(--background))] border flex flex-col justify-between space-y-1.5 ${
      highlight ? 'border-indigo-500/50 ring-1 ring-indigo-500/20' : 'border-[hsl(var(--border))]'
    }`}>
      <div className="flex items-center justify-between">
        <span className="text-[9px] uppercase tracking-wider text-[hsl(var(--muted-foreground))] font-mono font-bold">
          {label}
        </span>
        <span className="text-[8px] font-mono px-1 py-0.2 rounded bg-indigo-500/15 text-indigo-300 font-semibold">
          {badge}
        </span>
      </div>
      <div className="text-lg font-mono font-bold text-[hsl(var(--foreground))]">
        {value}
      </div>
      <div className="text-[10px] font-mono text-[hsl(var(--muted-foreground))] truncate">
        {hint}
      </div>
    </div>
  );
}
