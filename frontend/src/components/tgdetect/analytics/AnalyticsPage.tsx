'use client';

import { useMemo, useState } from 'react';
import {
  BarChart3,
  Boxes,
  Database,
  GitBranch,
  LineChart as LineChartIcon,
  Network,
  Tag,
  TrendingUp,
} from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { useDataset } from '@/lib/dataset-context';
import {
  useChainsByStrategy,
  useGraphStats,
  useSnapshots,
} from '@/lib/tgdetect/services/hooks';
import {
  bucketize,
  formatDurationLong,
  formatFloat,
  formatInt,
  formatPercent,
} from '@/lib/tgdetect/formatters';
import {
  NODE_TYPE_META,
  RELATION_TYPE_META,
} from '@/lib/tgdetect/constants';
import { CHART_AXIS_STYLE, CHART_COLORS, CHART_GRID_STYLE, CHART_TOOLTIP_STYLE, tokenToChartHex } from '@/lib/tgdetect/chart-constants';
import { EmptyState, LoadingState, SectionTitle } from '../shared/pills';
import type { ChainStrategy } from '@/lib/tgdetect/types';

type AnalyticsTab = 'events' | 'graph' | 'attacks' | 'datasets';

export function AnalyticsPage() {
  const { activeDataset, activeDatasetId } = useDataset();
  const [tab, setTab] = useState<AnalyticsTab>(() => {
    if (typeof window !== 'undefined') {
      const params = new URLSearchParams(window.location.search);
      const sub = params.get('sub') || window.location.hash.replace('#', '');
      if (sub === 'events' || sub === 'graph' || sub === 'attacks' || sub === 'datasets') {
        return sub as AnalyticsTab;
      }
    }
    return 'events';
  });
  return (
    <div className="space-y-3">
      {/* Active Dataset Provenance Header */}
      <div className="tg-card p-3 flex items-center justify-between gap-3 flex-wrap">
        <div className="flex items-center gap-2">
          <span className={`size-2 rounded-full ${activeDataset?.provenance === 'benchmark' ? 'bg-amber-400' : 'bg-emerald-400 animate-pulse'}`} />
          <SectionTitle className="inline">Security Analytics Engine</SectionTitle>
          <span className="text-[10px] font-mono px-2 py-0.5 rounded border border-[hsl(var(--border))] bg-[hsl(var(--background))] text-[hsl(var(--primary))] font-semibold">
            {activeDataset?.name ?? activeDatasetId ?? 'ACTIVE DATASET'}
          </span>
        </div>
        <div className="text-[10px] font-mono text-[hsl(var(--muted-foreground))]">
          PROVENANCE: {activeDataset?.provenance === 'benchmark' ? 'CANONICAL EVALUATION BENCHMARK' : 'LIVE UPLOADED TELEMETRY'} · PARTITION: {activeDatasetId ?? 'default'}
        </div>
      </div>

      <div className="bg-[hsl(var(--card))] p-1.5 rounded-lg border border-[hsl(var(--border))] flex flex-wrap gap-1 shadow-xs">
        {([
          { id: 'events', label: 'Event Analytics', icon: LineChartIcon },
          { id: 'graph', label: 'Graph Analytics', icon: Network },
          { id: 'attacks', label: 'Attack Analytics', icon: GitBranch },
          { id: 'datasets', label: 'Dataset Analytics', icon: Database },
        ] as { id: AnalyticsTab; label: string; icon: typeof BarChart3 }[]).map((t) => {
          const Icon = t.icon;
          const isActive = tab === t.id;
          return (
            <button
              key={t.id}
              type="button"
              onClick={() => setTab(t.id)}
              className={`px-3 py-1.5 text-xs rounded transition-all flex items-center gap-1.5 ${
                isActive
                  ? 'bg-[hsl(var(--primary)/0.15)] text-[hsl(var(--primary))] border border-[hsl(var(--primary)/0.3)] font-semibold shadow-xs'
                  : 'text-[hsl(var(--muted-foreground))] hover:text-[hsl(var(--foreground))] hover:bg-[hsl(var(--card-hover))]'
              }`}
            >
              <Icon className="size-3.5" />
              {t.label}
            </button>
          );
        })}
      </div>

      {tab === 'events' && <EventAnalytics />}
      {tab === 'graph' && <GraphAnalytics />}
      {tab === 'attacks' && <AttackAnalytics />}
      {tab === 'datasets' && <DatasetAnalytics />}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Event analytics
// ─────────────────────────────────────────────────────────────────────────────

function EventAnalytics() {
  const statsRes = useGraphStats();
  const snapsRes = useSnapshots(150);

  // ALL hooks before any early return — call unconditionally
  const eventsOverTime = useMemo(() => {
    return (snapsRes.data ?? []).map((s) => {
      const edges = s.num_edges ?? 0;
      const malEdges = s.num_malicious_edges ?? 0;
      const benign = s.num_benign_edges ?? 0;
      return {
        idx: s.index ?? (s as any).sequence ?? 0,
        benign,
        malicious: malEdges,
        unlabeled: s.num_unlabeled_edges ?? 0,
      };
    });
  }, [snapsRes.data]);

  const tacticData = useMemo(() => {
    const stats = statsRes.data;
    if (!stats) return [];
    return Object.entries(stats.graph.tactics)
      .map(([k, v]) => ({ name: k.replace(/_/g, ' '), value: v }))
      .sort((a, b) => b.value - a.value);
  }, [statsRes.data]);

  const sourceData = useMemo(() => {
    const stats = statsRes.data;
    if (!stats) return [];
    return Object.entries(stats.graph.source_tags).map(([k, v]) => ({ name: k, value: v }));
  }, [statsRes.data]);

  if (statsRes.state === 'loading' || snapsRes.state === 'loading') return <LoadingState label="Loading…" />;
  if (!statsRes.data) return <EmptyState title="No stats" />;
  const stats = statsRes.data;

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <MetricCard label="Total events" value={formatInt(stats.graph.total_events)} icon={LineChartIcon} color="info" />
        <MetricCard label="Benign" value={formatInt(stats.graph.benign_events)} icon={Tag} color="success" />
        <MetricCard label="Malicious" value={formatInt(stats.graph.malicious_events)} icon={BarChart3} color="danger" />
        <MetricCard label="Malicious ratio" value={formatPercent(stats.labeling.malicious_ratio, 2)} icon={TrendingUp} color="amber" />
      </div>

      <div className="tg-card p-4">
        <SectionTitle right={<div className="flex gap-3 text-[10px]">
          <span className="flex items-center gap-1"><span className="size-2 rounded-full bg-[hsl(var(--success))]" />benign</span>
          <span className="flex items-center gap-1"><span className="size-2 rounded-full bg-[hsl(var(--danger))]" />malicious</span>
        </div>}>
          Events per time bucket
        </SectionTitle>
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={eventsOverTime} margin={{ top: 8, right: 8, bottom: 0, left: -16 }}>
            <CartesianGrid {...CHART_GRID_STYLE} />
            <XAxis dataKey="idx" {...CHART_AXIS_STYLE} />
            <YAxis {...CHART_AXIS_STYLE} width={36} />
            <Tooltip
              {...CHART_TOOLTIP_STYLE}
              formatter={(value: any, name: any) => [
                typeof value === 'number' && !Number.isNaN(value) ? formatInt(value) : '0',
                name,
              ]}
            />
            <Bar isAnimationActive={false} dataKey="unlabeled" stackId="1" fill="#94a3b8" />
            <Bar isAnimationActive={false} dataKey="benign" stackId="1" fill={CHART_COLORS.green} />
            <Bar isAnimationActive={false} dataKey="malicious" stackId="1" fill={CHART_COLORS.red} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        <div className="tg-card p-4">
          <SectionTitle>Mitre tactic distribution</SectionTitle>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={tacticData} layout="vertical" margin={{ top: 0, right: 8, bottom: 0, left: 0 }}>
              <CartesianGrid {...CHART_GRID_STYLE} horizontal={false} />
              <XAxis type="number" {...CHART_AXIS_STYLE} />
              <YAxis type="category" dataKey="name" {...CHART_AXIS_STYLE} width={120} />
              <Tooltip {...CHART_TOOLTIP_STYLE} />
              <Bar isAnimationActive={false} dataKey="value" radius={[0, 2, 2, 0]}>
                {tacticData.map((_, i) => (
                  <Cell key={i} fill={CHART_COLORS.violet} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="tg-card p-4">
          <SectionTitle>Source tag distribution</SectionTitle>
          <ResponsiveContainer width="100%" height={260}>
            <PieChart>
              <Pie isAnimationActive={false} data={sourceData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label>
                {sourceData.map((_, i) => (
                  <Cell key={i} fill={[CHART_COLORS.cyan, CHART_COLORS.violet, CHART_COLORS.teal][i % 3]} />
                ))}
              </Pie>
              <Tooltip {...CHART_TOOLTIP_STYLE} />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Graph analytics
// ─────────────────────────────────────────────────────────────────────────────

function GraphAnalytics() {
  const statsRes = useGraphStats();
  const nodesRes = useSnapshots(80);

  const nodeTypeData = useMemo(() => {
    const stats = statsRes.data;
    if (!stats) return [];
    return Object.entries(stats.graph.node_types)
      .map(([k, v]) => ({
        name: k,
        value: v,
        color: tokenToChartHex(NODE_TYPE_META[k as keyof typeof NODE_TYPE_META]?.color ?? 'gray'),
      }));
  }, [statsRes.data]);

  const relationData = useMemo(() => {
    const stats = statsRes.data;
    if (!stats) return [];
    return Object.entries(stats.graph.relation_types)
      .map(([k, v]) => ({
        name: k,
        label: RELATION_TYPE_META[k as keyof typeof RELATION_TYPE_META]?.label ?? k.replace(/_/g, ' '),
        value: v,
        isAttack: RELATION_TYPE_META[k as keyof typeof RELATION_TYPE_META]?.is_attack ?? false,
      }))
      .sort((a, b) => b.value - a.value);
  }, [statsRes.data]);

  const degreeDist = useMemo(() => {
    const sn = nodesRes.data ?? [];
    return sn.map((s) => ({
      idx: s.index ?? (s as any).sequence ?? 0,
      nodes: s.num_nodes ?? 0,
      edges: s.num_edges ?? 0,
    }));
  }, [nodesRes.data]);

  if (statsRes.state === 'loading') return <LoadingState label="Loading…" />;
  if (!statsRes.data) return <EmptyState title="No stats" />;
  const stats = statsRes.data;
  const avgDegree = stats.graph.total_nodes ? (stats.graph.total_edges / stats.graph.total_nodes).toFixed(2) : '0.00';

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <MetricCard label="Total nodes" value={formatInt(stats.graph.total_nodes)} icon={Boxes} color="purple" />
        <MetricCard label="Total edges" value={formatInt(stats.graph.total_edges)} icon={Network} color="teal" />
        <MetricCard label="Avg degree" value={avgDegree} icon={TrendingUp} color="info" />
        <MetricCard label="Out-of-order events" value={formatInt(stats.graph.out_of_order_events)} icon={BarChart3} color="amber" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
        <div className="tg-card p-4">
          <SectionTitle>Node type distribution</SectionTitle>
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie isAnimationActive={false} data={nodeTypeData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={70} innerRadius={40}>
                {nodeTypeData.map((entry, i) => (
                  <Cell key={i} fill={entry.color} stroke="hsl(var(--card))" strokeWidth={2} />
                ))}
              </Pie>
              <Tooltip {...CHART_TOOLTIP_STYLE} />
            </PieChart>
          </ResponsiveContainer>
        </div>
        <div className="tg-card p-4 lg:col-span-2">
          <SectionTitle right={<div className="flex gap-3 text-[10px]">
            <span className="flex items-center gap-1"><span className="size-2 rounded-full bg-[hsl(var(--info))]" />standard</span>
            <span className="flex items-center gap-1"><span className="size-2 rounded-full bg-[hsl(var(--danger))]" />attack relation</span>
          </div>}>
            Relation distribution
          </SectionTitle>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={relationData} layout="vertical" margin={{ top: 0, right: 8, bottom: 0, left: 0 }}>
              <CartesianGrid {...CHART_GRID_STYLE} horizontal={false} />
              <XAxis type="number" {...CHART_AXIS_STYLE} />
              <YAxis type="category" dataKey="label" {...CHART_AXIS_STYLE} width={90} />
              <Tooltip {...CHART_TOOLTIP_STYLE} />
              <Bar isAnimationActive={false} dataKey="value" radius={[0, 2, 2, 0]}>
                {relationData.map((entry, i) => (
                  <Cell key={i} fill={entry.isAttack ? CHART_COLORS.red : CHART_COLORS.cyan} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="tg-card p-4">
        <SectionTitle right={<div className="text-[10px] text-[hsl(var(--muted-foreground))]">temporal span: {formatDurationLong(stats.graph.timestamp_span_s)}</div>}>
          Graph size over 512-event visualization blocks
        </SectionTitle>
        <ResponsiveContainer width="100%" height={240}>
          <LineChart data={degreeDist} margin={{ top: 8, right: 8, bottom: 0, left: -16 }}>
            <CartesianGrid {...CHART_GRID_STYLE} />
            <XAxis dataKey="idx" {...CHART_AXIS_STYLE} />
            <YAxis {...CHART_AXIS_STYLE} width={36} />
            <Tooltip
              {...CHART_TOOLTIP_STYLE}
              formatter={(value: any, name: any) => [
                typeof value === 'number' && !Number.isNaN(value) ? formatInt(value) : '0',
                name,
              ]}
            />
            <Line isAnimationActive={false} type="monotone" dataKey="nodes" stroke={CHART_COLORS.violet} strokeWidth={1.5} dot={false} />
            <Line isAnimationActive={false} type="monotone" dataKey="edges" stroke={CHART_COLORS.cyan} strokeWidth={1.5} dot={false} />
          </LineChart>
        </ResponsiveContainer>
        <div className="flex gap-3 text-[10px] mt-1">
          <span className="flex items-center gap-1"><span className="size-2 rounded-full" style={{ background: CHART_COLORS.violet }} />nodes per block</span>
          <span className="flex items-center gap-1"><span className="size-2 rounded-full" style={{ background: CHART_COLORS.cyan }} />events per block</span>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Attack analytics
// ─────────────────────────────────────────────────────────────────────────────

function AttackAnalytics() {
  const statsRes = useGraphStats();
  const chainIdRes = useChainsByStrategy('chain_id');
  const causalRes = useChainsByStrategy('causal_parent');
  const entityRes = useChainsByStrategy('entity_time');

  const chainLengthBuckets = useMemo(() => {
    const allChains = [
      ...(chainIdRes.data ?? []),
      ...(causalRes.data ?? []),
      ...(entityRes.data ?? []),
    ];
    const lengths = allChains.map((c) => c.num_events);
    return bucketize(lengths, 8);
  }, [chainIdRes.data, causalRes.data, entityRes.data]);

  const chainDurationBuckets = useMemo(() => {
    const allChains = [
      ...(chainIdRes.data ?? []),
      ...(causalRes.data ?? []),
      ...(entityRes.data ?? []),
    ];
    const durations = allChains.map((c) => c.duration_s);
    return bucketize(durations, 8);
  }, [chainIdRes.data, causalRes.data, entityRes.data]);

  const strategyComparison = useMemo(() => {
    return (['chain_id', 'causal_parent', 'entity_time'] as ChainStrategy[]).map((s) => {
      const list =
        s === 'chain_id' ? chainIdRes.data :
        s === 'causal_parent' ? causalRes.data :
        entityRes.data;
      const arr = list ?? [];
      const avgEvents = arr.length > 0 ? arr.reduce((sum, c) => sum + (c.num_events ?? 0), 0) / arr.length : 0;
      const avgDuration = arr.length > 0 ? arr.reduce((sum, c) => sum + (c.duration_s ?? 0), 0) / arr.length : 0;
      const avgNodes = arr.length > 0 ? arr.reduce((sum, c) => sum + (c.num_nodes ?? 0), 0) / arr.length : 0;
      return { strategy: s, chains: arr.length, avgEvents, avgDuration, avgNodes };
    });
  }, [chainIdRes.data, causalRes.data, entityRes.data]);

  const tacticInChains = useMemo(() => {
    const counts = new Map<string, number>();
    for (const c of [...(chainIdRes.data ?? [])]) {
      for (const t of c.tactic_sequence ?? []) counts.set(t, (counts.get(t) ?? 0) + 1);
    }
    return Array.from(counts.entries())
      .map(([name, value]) => ({ name: name.replace(/_/g, ' '), value }))
      .sort((a, b) => b.value - a.value);
  }, [chainIdRes.data]);

  if (statsRes.state === 'loading') return <LoadingState label="Loading…" />;
  if (!statsRes.data) return <EmptyState title="No stats" />;
  const stats = statsRes.data;

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <MetricCard label="Total chains" value={formatInt(stats.attacks.total_chains)} icon={GitBranch} color="teal" />
        <MetricCard label="Malicious events tracked" value={formatInt(stats.attacks.malicious_events_tracked)} icon={BarChart3} color="danger" />
        <MetricCard label="Ungrouped malicious" value={formatInt(stats.attacks.ungrouped_malicious_events)} icon={Tag} color="amber" />
        <MetricCard label="Dangling causal parents" value={formatInt(stats.attacks.dangling_causal_parents)} icon={TrendingUp} color="info" />
      </div>

      <div className="tg-card p-4">
        <SectionTitle>Strategy comparison</SectionTitle>
        <div className="overflow-x-auto mt-2">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-left text-[10px] uppercase tracking-wide text-[hsl(var(--muted-foreground))] border-b border-[hsl(var(--border))]">
                <th className="py-1.5 pr-3">strategy</th>
                <th className="py-1.5 pr-3">chains</th>
                <th className="py-1.5 pr-3">avg events</th>
                <th className="py-1.5 pr-3">avg nodes</th>
                <th className="py-1.5 pr-3">avg duration</th>
              </tr>
            </thead>
            <tbody>
              {strategyComparison.map((s) => (
                <tr key={s.strategy} className="border-b border-[hsl(var(--border)/0.5)]">
                  <td className="py-1.5 pr-3 mono font-semibold">{s.strategy}</td>
                  <td className="py-1.5 pr-3 mono">{formatInt(s.chains)}</td>
                  <td className="py-1.5 pr-3 mono">{formatFloat(s.avgEvents, 1)}</td>
                  <td className="py-1.5 pr-3 mono">{formatFloat(s.avgNodes, 1)}</td>
                  <td className="py-1.5 pr-3 mono">{formatDurationLong(s.avgDuration)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        <div className="tg-card p-4">
          <SectionTitle>Chain length distribution</SectionTitle>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={chainLengthBuckets} margin={{ top: 8, right: 8, bottom: 0, left: -16 }}>
              <CartesianGrid {...CHART_GRID_STYLE} />
              <XAxis dataKey="x" {...CHART_AXIS_STYLE} />
              <YAxis {...CHART_AXIS_STYLE} width={36} />
              <Tooltip
                {...CHART_TOOLTIP_STYLE}
                formatter={(value: any, name: any) => [
                  typeof value === 'number' && !Number.isNaN(value) ? formatInt(value) : '0',
                  name,
                ]}
              />
              <Bar dataKey="count" fill={CHART_COLORS.teal} radius={[2, 2, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="tg-card p-4">
          <SectionTitle>Chain duration distribution (seconds)</SectionTitle>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={chainDurationBuckets} margin={{ top: 8, right: 8, bottom: 0, left: -16 }}>
              <CartesianGrid {...CHART_GRID_STYLE} />
              <XAxis dataKey="x" {...CHART_AXIS_STYLE} />
              <YAxis {...CHART_AXIS_STYLE} width={36} />
              <Tooltip
                {...CHART_TOOLTIP_STYLE}
                formatter={(value: any, name: any) => [
                  typeof value === 'number' && !Number.isNaN(value) ? formatInt(value) : '0',
                  name,
                ]}
              />
              <Bar dataKey="count" fill={CHART_COLORS.violet} radius={[2, 2, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="tg-card p-4">
        <SectionTitle>Top tactics in attack chains</SectionTitle>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={tacticInChains} layout="vertical" margin={{ top: 0, right: 8, bottom: 0, left: 0 }}>
            <CartesianGrid {...CHART_GRID_STYLE} horizontal={false} />
            <XAxis type="number" {...CHART_AXIS_STYLE} />
            <YAxis type="category" dataKey="name" {...CHART_AXIS_STYLE} width={120} />
            <Tooltip
              {...CHART_TOOLTIP_STYLE}
              formatter={(value: any, name: any) => [
                typeof value === 'number' && !Number.isNaN(value) ? formatInt(value) : '0',
                name,
              ]}
            />
            <Bar isAnimationActive={false} dataKey="value" fill={CHART_COLORS.red} radius={[0, 2, 2, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Dataset analytics
// ─────────────────────────────────────────────────────────────────────────────

function DatasetAnalytics() {
  const statsRes = useGraphStats();

  const normalizationData = useMemo(() => {
    const stats = statsRes.data;
    if (!stats) return [];
    const accepted = stats.normalization?.accepted ?? 0;
    const rejected = stats.normalization?.rejected ?? 0;
    return [
      { name: 'accepted', value: accepted, color: CHART_COLORS.green },
      { name: 'rejected', value: rejected, color: CHART_COLORS.red },
    ];
  }, [statsRes.data]);

  const rejectReasons = useMemo(() => {
    const stats = statsRes.data;
    if (!stats?.normalization?.reject_reasons) return [];
    return Object.entries(stats.normalization.reject_reasons)
      .map(([k, v]) => ({ name: k, value: v }))
      .sort((a, b) => b.value - a.value);
  }, [statsRes.data]);

  const labelingModeData = useMemo(() => {
    const stats = statsRes.data;
    if (!stats) return [];
    const mal = stats.labeling?.malicious_events ?? stats.graph?.malicious_events ?? 0;
    const ben = stats.labeling?.benign_events ?? stats.graph?.benign_events ?? Math.max(0, (stats.graph?.total_events ?? 0) - mal);
    return [
      { name: 'benign', value: ben, color: CHART_COLORS.green },
      { name: 'malicious', value: mal, color: CHART_COLORS.red },
    ];
  }, [statsRes.data]);

  if (statsRes.state === 'loading') return <LoadingState label="Loading…" />;
  if (!statsRes.data) return <EmptyState title="No stats" />;
  const stats = statsRes.data;

  const norm = stats.normalization;
  const accepted = norm?.accepted ?? 0;
  const rejected = norm?.rejected ?? 0;
  const seen = norm?.seen ?? (accepted + rejected);
  const acceptanceRate = seen > 0 ? accepted / seen : 1.0;

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <MetricCard label="Normalization seen" value={formatInt(seen)} icon={Database} color="info" />
        <MetricCard label="Normalization accepted" value={formatInt(accepted)} icon={Tag} color="success" />
        <MetricCard label="Normalization rejected" value={formatInt(rejected)} icon={BarChart3} color="danger" />
        <MetricCard label="Acceptance rate" value={formatPercent(acceptanceRate, 2)} icon={TrendingUp} color="amber" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
        <div className="tg-card p-4">
          <SectionTitle>Normalization outcome</SectionTitle>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie data={normalizationData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={70}>
                {normalizationData.map((entry, i) => (
                  <Cell key={i} fill={entry.color} stroke="hsl(var(--card))" strokeWidth={2} />
                ))}
              </Pie>
              <Tooltip
                {...CHART_TOOLTIP_STYLE}
                formatter={(value: any, name: any) => [
                  typeof value === 'number' && !Number.isNaN(value) ? formatInt(value) : '0',
                  name,
                ]}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>
        <div className="tg-card p-4 lg:col-span-2">
          <SectionTitle>Rejection reasons</SectionTitle>
          {rejectReasons.length === 0 ? (
            <EmptyState title="No rejections" description="All events passed normalization" />
          ) : (
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={rejectReasons} layout="vertical" margin={{ top: 0, right: 8, bottom: 0, left: 0 }}>
                <CartesianGrid {...CHART_GRID_STYLE} horizontal={false} />
                <XAxis type="number" {...CHART_AXIS_STYLE} />
                <YAxis type="category" dataKey="name" {...CHART_AXIS_STYLE} width={140} />
                <Tooltip
                  {...CHART_TOOLTIP_STYLE}
                  formatter={(value: any, name: any) => [
                    typeof value === 'number' && !Number.isNaN(value) ? formatInt(value) : '0',
                    name,
                  ]}
                />
                <Bar isAnimationActive={false} dataKey="value" fill={CHART_COLORS.red} radius={[0, 2, 2, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      <div className="tg-card p-4">
        <SectionTitle>Label distribution (mode: <span className="mono text-[hsl(var(--purple))]">{stats.labeling.mode}</span>)</SectionTitle>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={labelingModeData} margin={{ top: 8, right: 8, bottom: 0, left: -16 }}>
            <CartesianGrid {...CHART_GRID_STYLE} />
            <XAxis dataKey="name" {...CHART_AXIS_STYLE} />
            <YAxis {...CHART_AXIS_STYLE} width={48} />
            <Tooltip
              {...CHART_TOOLTIP_STYLE}
              formatter={(value: any, name: any) => [
                typeof value === 'number' && !Number.isNaN(value) ? formatInt(value) : '0',
                name,
              ]}
            />
            <Bar isAnimationActive={false} dataKey="value" radius={[2, 2, 0, 0]}>
              {labelingModeData.map((entry, i) => (
                <Cell key={i} fill={entry.color} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {stats.labeling.mode === 'heuristic' && (
        <div className="tg-card p-4">
          <SectionTitle>Heuristic labeler indicators</SectionTitle>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-2">
            <MetricCard label="Seed indicator hits" value={formatInt(stats.labeling.seed_indicator_hits)} icon={Tag} color="info" />
            <MetricCard label="Propagated events" value={formatInt(stats.labeling.propagated_events)} icon={TrendingUp} color="amber" />
            <MetricCard label="Malicious events" value={formatInt(stats.labeling.malicious_events)} icon={BarChart3} color="danger" />
            <MetricCard label="Malicious ratio" value={formatPercent(stats.labeling.malicious_ratio, 2)} icon={TrendingUp} color="danger" />
          </div>
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared metric card
// ─────────────────────────────────────────────────────────────────────────────

function MetricCard({
  label,
  value,
  icon: Icon,
  color,
}: {
  label: string;
  value: string;
  icon: typeof BarChart3;
  color: 'info' | 'success' | 'danger' | 'amber' | 'purple' | 'teal';
}) {
  const colorClass = {
    info: 'text-[hsl(var(--info))]',
    success: 'text-[hsl(var(--success))]',
    danger: 'text-[hsl(var(--danger))]',
    amber: 'text-[hsl(var(--warning))]',
    purple: 'text-[hsl(var(--purple))]',
    teal: 'text-[hsl(var(--teal))]',
  }[color];
  return (
    <div className="tg-card p-3 flex flex-col gap-1">
      <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wide text-[hsl(var(--muted-foreground))] font-semibold">
        <Icon className={`size-3 ${colorClass}`} />
        {label}
      </div>
      <div className={`metric-value-sm font-mono ${colorClass}`}>{value}</div>
    </div>
  );
}
