'use client';

import { useMemo, useState } from 'react';
import {
  ChevronRight,
  Clock,
  GitBranch,
  ListTree,
  Network,
  Search,
  ShieldAlert,
} from 'lucide-react';
import { useDataset } from '@/lib/dataset-context';
import {
  useChain,
  useChainEvents,
  useChainSubgraph,
  useChains,
  useChainsByStrategy,
} from '@/lib/tgdetect/services/hooks';
import {
  chainSeverity,
  formatDurationLong,
  formatEpoch,
  formatEpochTime,
  formatInt,
} from '@/lib/tgdetect/formatters';
import { CHAIN_STRATEGIES } from '@/lib/tgdetect/constants';
import type { ChainStrategy } from '@/lib/tgdetect/types';
import {
  EmptyState,
  ErrorState,
  LabelPill,
  LoadingState,
  MonoId,
  NodeTypePill,
  RelationPill,
  SectionTitle,
  StrategyPill,
} from '../shared/pills';
import { TemporalGraphViz } from '../shared/temporal-graph-viz';
import type { ChainSeverity } from '@/lib/tgdetect/formatters';
import type { TGEvent } from '@/lib/tgdetect/types';

export interface ChainsPageProps {
  initialChainId?: string | null;
  onNavigate?: (page: string, ctx?: Record<string, unknown>) => void;
}

type DetailTab = 'timeline' | 'graph' | 'events' | 'evidence';

export function ChainsPage({ initialChainId, onNavigate }: ChainsPageProps) {
  const { activeDataset, activeDatasetId } = useDataset();
  const [selectedChainId, setSelectedChainId] = useState<string | null>(initialChainId ?? null);
  const [strategyFilter, setStrategyFilter] = useState<Set<ChainStrategy>>(new Set(CHAIN_STRATEGIES));
  const [search, setSearch] = useState('');
  const [tab, setTab] = useState<DetailTab>('timeline');

  const chainsRes = useChains();
  const filtered = useMemo(() => {
    const list = chainsRes.data ?? [];
    return list.filter((c) => {
      if (!strategyFilter.has(c.strategy)) return false;
      if (search && !c.chain_id.toLowerCase().includes(search.toLowerCase())) return false;
      return true;
    });
  }, [chainsRes.data, strategyFilter, search]);

  const effectiveChainId = selectedChainId ?? filtered[0]?.chain_id ?? null;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
      {/* Left: chain list */}
      <div className="space-y-3">
      <p className="text-xs text-[hsl(var(--muted-foreground))]">Groups labeled malicious events from the same process within 300 seconds. These clusters do not establish causal attack paths. This dataset has no MITRE tactic or explicit causal-parent annotations.</p>
        <div className="tg-card p-3 space-y-2">
          <SectionTitle right={<GitBranch className="size-3 text-[hsl(var(--muted-foreground))]" />}>
            <div className="flex items-center gap-2">
              <span className={`size-2 rounded-full ${activeDataset?.provenance === 'benchmark' ? 'bg-amber-400' : 'bg-emerald-400 animate-pulse'}`} />
              <span>Attack Chains · entity-time investigation</span>
            </div>
          </SectionTitle>
          <div className="text-[10px] text-[hsl(var(--muted-foreground))] font-mono">
            PARTITION: <strong className="text-[hsl(var(--primary))]">{activeDataset?.name ?? activeDatasetId ?? 'ACTIVE'}</strong>
          </div>
          <div className="relative">
            <Search className="absolute left-2 top-1/2 -translate-y-1/2 size-3 text-[hsl(var(--muted-foreground))]" />
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="filter chain_id…"
              className="w-full pl-7 pr-2 py-1 text-xs bg-[hsl(var(--background))] border border-[hsl(var(--border))] rounded font-mono"
            />
          </div>
          <div className="flex flex-wrap gap-1">
            {CHAIN_STRATEGIES.map((s) => {
              const active = strategyFilter.has(s);
              return (
                <button
                  key={s}
                  type="button"
                  onClick={() => {
                    const ns = new Set(strategyFilter);
                    if (ns.has(s)) ns.delete(s); else ns.add(s);
                    setStrategyFilter(ns);
                  }}
                  className={`px-1.5 py-0.5 text-[10px] font-mono border rounded ${
                    active ? 'border-[hsl(var(--primary)/0.6)] bg-[hsl(var(--info-bg))]' : 'border-[hsl(var(--border))] bg-[hsl(var(--card))]'
                  }`}
                >{s}</button>
              );
            })}
          </div>
        </div>
        <div className="tg-card p-0 max-h-[70vh] overflow-y-auto">
          {chainsRes.state === 'loading' ? (
            <LoadingState label="Loading chains…" />
          ) : chainsRes.state === 'failed' ? (
            <ErrorState message={`Failed: ${chainsRes.error}`} />
          ) : (chainsRes.data?.length ?? 0) === 0 ? (
            <EmptyState
              title="Attack chains absent"
              description={`No multi-hop attack chains were reconstructed for active telemetry partition "${activeDataset?.name ?? activeDatasetId}".`}
            />
          ) : filtered.length === 0 ? (
            <EmptyState title="No chains match filters" description="Adjust strategy selection or search filter." />
          ) : (
            <div className="divide-y divide-[hsl(var(--border)/0.5)]">
              {filtered.map((c) => {
                const severity = chainSeverity(c);
                const isSelected = c.chain_id === effectiveChainId;
                return (
                  <button
                    key={c.chain_id}
                    type="button"
                    onClick={() => setSelectedChainId(c.chain_id)}
                    className={`w-full text-left p-3 hover:bg-[hsl(var(--card-hover))] ${isSelected ? 'bg-[hsl(var(--info-bg))]' : ''}`}
                  >
                    <div className="flex items-center gap-2 mb-1">
                      <MonoId truncateAt={26}>{c.chain_id}</MonoId>
                      <StrategyPill strategy={c.strategy} />
                      <span className="ml-auto"><SeverityPill severity={severity} /></span>
                    </div>
                    <div className="grid grid-cols-3 gap-1 text-[10px] text-[hsl(var(--muted-foreground))] font-mono">
                      <span>{c.num_events} events</span>
                      <span>{c.num_nodes} nodes</span>
                      <span>{formatDurationLong(c.duration_s)}</span>
                    </div>
                    <div className="text-[10px] text-[hsl(var(--muted-foreground))] mt-1">
                      {c.tactic_sequence.slice(0, 4).join(' → ')}
                      {c.tactic_sequence.length > 4 ? ' …' : ''}
                    </div>
                  </button>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* Right: detail */}
      <div className="lg:col-span-2">
        <ChainDetailPanel
          chainId={effectiveChainId}
          tab={tab}
          onTabChange={setTab}
          onNavigate={onNavigate}
        />
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Severity pill (UI-only interpretation)
// ─────────────────────────────────────────────────────────────────────────────

function SeverityPill({ severity }: { severity: ChainSeverity }) {
  const cls = {
    low: 'bg-[hsl(var(--success-bg))] text-[hsl(var(--success))] border-[hsl(var(--success)/0.25)]',
    medium: 'bg-[hsl(var(--warning-bg))] text-[hsl(var(--warning))] border-[hsl(var(--warning)/0.25)]',
    high: 'bg-orange-500/10 text-orange-700 dark:text-orange-400 border-orange-500/25',
    critical: 'bg-[hsl(var(--danger-bg))] text-[hsl(var(--danger))] border-[hsl(var(--danger)/0.25)]',
  }[severity];
  return (
    <span className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-mono font-semibold border uppercase ${cls}`}>
      <ShieldAlert className="size-2.5" />
      {severity}
    </span>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Chain detail panel
// ─────────────────────────────────────────────────────────────────────────────

function ChainDetailPanel({
  chainId,
  tab,
  onTabChange,
  onNavigate,
}: {
  chainId: string | null;
  tab: DetailTab;
  onTabChange: (t: DetailTab) => void;
  onNavigate?: (page: string, ctx?: Record<string, unknown>) => void;
}) {
  const chainRes = useChain(chainId);
  const eventsRes = useChainEvents(chainId);
  const subgraphRes = useChainSubgraph(chainId);

  if (!chainId) {
    return (
      <div className="tg-card p-4 h-full">
        <SectionTitle>Chain Detail</SectionTitle>
        <EmptyState
          title="No chain selected"
          description="Select a chain on the left to inspect its timeline, subgraph, events, and evidence."
        />
      </div>
    );
  }
  if (chainRes.state === 'loading') return <LoadingState label="Loading chain…" />;
  if (!chainRes.data) return <EmptyState title="Chain not found" description={chainId} />;

  const chain = chainRes.data;
  return (
    <div className="space-y-3">
      <div className="tg-card p-4">
        <div className="flex items-center justify-between flex-wrap gap-2 mb-2">
          <SectionTitle>Chain Detail</SectionTitle>
          <div className="flex items-center gap-2">
            <StrategyPill strategy={chain.strategy} />
            <SeverityPill severity={chainSeverity(chain)} />
          </div>
        </div>
        <div className="mono text-xs break-all text-[hsl(var(--foreground))] mb-2">{chain.chain_id}</div>
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
          <MiniStat label="Events" value={formatInt(chain.num_events)} />
          <MiniStat label="Nodes" value={formatInt(chain.num_nodes)} />
          <MiniStat label="Duration" value={formatDurationLong(chain.duration_s)} />
          <MiniStat label="Start" value={formatEpochTime(chain.start_ts)} />
          <MiniStat label="End" value={formatEpochTime(chain.end_ts)} />
          <MiniStat label="Tactics" value={String(chain.tactic_sequence.length)} />
        </div>
      </div>

      <div className="bg-[hsl(var(--card))] p-1.5 rounded-lg border border-[hsl(var(--border))] flex gap-1 shadow-xs">
        {([
          { id: 'timeline', label: 'Timeline', icon: Clock },
          { id: 'graph', label: 'Subgraph', icon: Network },
          { id: 'events', label: 'Events', icon: ListTree },
          { id: 'evidence', label: 'Evidence', icon: GitBranch },
        ] as { id: DetailTab; label: string; icon: typeof Clock }[]).map((t) => {
          const Icon = t.icon;
          const isActive = tab === t.id;
          return (
            <button
              key={t.id}
              type="button"
              onClick={() => onTabChange(t.id)}
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

      <div className="tg-card p-3">
        {tab === 'timeline' && <ChainTimeline events={eventsRes.data ?? []} loading={eventsRes.state === 'loading'} onEventClick={(id) => onNavigate?.('events', { eventId: id })} />}
        {tab === 'graph' && <ChainGraph subgraph={subgraphRes.data} loading={subgraphRes.state === 'loading'} chainId={chain.chain_id} />}
        {tab === 'events' && <ChainEventsTable events={eventsRes.data ?? []} loading={eventsRes.state === 'loading'} onEventClick={(id) => onNavigate?.('events', { eventId: id })} />}
        {tab === 'evidence' && <ChainEvidence chain={chain} events={eventsRes.data ?? []} loading={eventsRes.state === 'loading'} />}
      </div>
    </div>
  );
}

function MiniStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col gap-0.5 p-2 rounded border border-[hsl(var(--border))] bg-[hsl(var(--background))]">
      <div className="text-[9px] uppercase tracking-wide text-[hsl(var(--muted-foreground))] font-semibold">{label}</div>
      <div className="text-xs font-mono font-semibold tabular-nums text-[hsl(var(--foreground))]">{value}</div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Timeline view
// ─────────────────────────────────────────────────────────────────────────────

function ChainTimeline({ events, loading, onEventClick }: { events: TGEvent[]; loading: boolean; onEventClick?: (id: string) => void }) {
  if (loading) return <LoadingState label="Loading chain events…" />;
  if (events.length === 0) return <EmptyState title="No events" />;
  const sorted = [...events].sort((a, b) => a.ts - b.ts);
  return (
    <div className="relative pl-6">
      <div className="absolute left-2 top-0 bottom-0 w-px bg-[hsl(var(--border))]" />
      <div className="space-y-3">
        {sorted.map((e, i) => (
          <div key={e.event_id} className="relative">
            <div className="absolute -left-[18px] top-1 size-3 rounded-full border-2 border-[hsl(var(--card))] bg-[hsl(var(--danger))]" />
            <button
              type="button"
              onClick={() => onEventClick?.(e.event_id)}
              className="w-full text-left p-2 rounded border border-[hsl(var(--border))] hover:bg-[hsl(var(--card-hover))] bg-[hsl(var(--card))]"
            >
              <div className="flex items-center gap-2 flex-wrap">
                <span className="mono text-[10px] text-[hsl(var(--muted-foreground))]">#{i + 1}</span>
                <span className="mono text-[10px] text-[hsl(var(--muted-foreground))]">{formatEpochTime(e.ts)}</span>
                <RelationPill relation={e.relation} />
                {e.tactics.map((t) => (
                  <span key={t} className="text-[10px] font-mono px-1.5 py-0.5 rounded border border-[hsl(var(--border))]">{t.replace(/_/g, ' ')}</span>
                ))}
                <LabelPill label={e.label} />
                <span className="ml-auto mono text-[10px] text-[hsl(var(--muted-foreground))]">{e.apt_stage}</span>
              </div>
              <div className="mt-1 flex items-center gap-1 text-[11px] mono">
                <span>{e.src_id}</span>
                <ChevronRight className="size-3 text-[hsl(var(--muted-foreground))]" />
                <span>{e.dst_id}</span>
              </div>
              <div className="mt-1 flex items-center gap-1">
                <NodeTypePill type={e.src_type} showLabel={false} />
                <NodeTypePill type={e.dst_type} showLabel={false} />
                <span className="ml-auto mono text-[10px] text-[hsl(var(--muted-foreground))]">{e.event_id}</span>
              </div>
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Graph view
// ─────────────────────────────────────────────────────────────────────────────

function ChainGraph({ subgraph, loading, chainId }: { subgraph: import('@/lib/tgdetect/types').ChainSubgraph | null; loading: boolean; chainId: string }) {
  if (loading) return <LoadingState label="Loading subgraph…" />;
  if (!subgraph) return <EmptyState title="Subgraph not available" />;
  return (
    <div className="space-y-2">
      <div className="text-[10px] text-[hsl(var(--muted-foreground))]">
        {subgraph.nodes.length} nodes · {subgraph.edges.length} ordered edges · strategy: <span className="mono">{subgraph.strategy}</span>
      </div>
      <div className="h-[60vh] border border-[hsl(var(--border))] rounded overflow-hidden bg-[hsl(var(--background))]">
        <TemporalGraphViz
          nodes={subgraph.nodes}
          edges={subgraph.edges}
          highlightChainId={chainId}
          showEdgeLabels
          maxNodes={500}
        />
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Events view
// ─────────────────────────────────────────────────────────────────────────────

function ChainEventsTable({ events, loading, onEventClick }: { events: TGEvent[]; loading: boolean; onEventClick?: (id: string) => void }) {
  if (loading) return <LoadingState label="Loading events…" />;
  if (events.length === 0) return <EmptyState title="No events" />;
  const sorted = [...events].sort((a, b) => a.ts - b.ts);
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="text-left text-[10px] uppercase tracking-wide text-[hsl(var(--muted-foreground))] border-b border-[hsl(var(--border))]">
            <th className="py-1.5 pr-3">#</th>
            <th className="py-1.5 pr-3">ts</th>
            <th className="py-1.5 pr-3">event_id</th>
            <th className="py-1.5 pr-3">relation</th>
            <th className="py-1.5 pr-3">src → dst</th>
            <th className="py-1.5 pr-3">tactic</th>
            <th className="py-1.5 pr-3">causal_parent</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((e, i) => (
            <tr
              key={e.event_id}
              className="border-b border-[hsl(var(--border)/0.5)] hover:bg-[hsl(var(--card-hover))] cursor-pointer"
              onClick={() => onEventClick?.(e.event_id)}
            >
              <td className="py-1.5 pr-3 mono text-[hsl(var(--muted-foreground))]">{i + 1}</td>
              <td className="py-1.5 pr-3 mono text-[11px] whitespace-nowrap" title={formatEpoch(e.ts)}>{formatEpochTime(e.ts)}</td>
              <td className="py-1.5 pr-3"><MonoId truncateAt={22}>{e.event_id}</MonoId></td>
              <td className="py-1.5 pr-3"><RelationPill relation={e.relation} /></td>
              <td className="py-1.5 pr-3 mono text-[11px]">
                {e.src_id} → {e.dst_id}
              </td>
              <td className="py-1.5 pr-3 text-[11px]">{e.tactics.join(', ')}</td>
              <td className="py-1.5 pr-3">
                {e.causal_parent ? <MonoId truncateAt={16}>{e.causal_parent}</MonoId> : <span className="text-[hsl(var(--muted-foreground))]">—</span>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Evidence view
// ─────────────────────────────────────────────────────────────────────────────

function ChainEvidence({
  chain,
  events,
  loading,
}: {
  chain: import('@/lib/tgdetect/types').AttackChainSummary;
  events: TGEvent[];
  loading: boolean;
}) {
  if (loading) return <LoadingState label="Loading evidence…" />;
  // Aggregate nodes by type
  const nodeTypeCount = new Map<string, number>();
  for (const id of chain.nodes) {
    const t = id.split(':')[0]?.toUpperCase() ?? 'UNKNOWN';
    nodeTypeCount.set(t, (nodeTypeCount.get(t) ?? 0) + 1);
  }
  return (
    <div className="space-y-4">
      <div>
        <div className="text-[10px] uppercase tracking-wide text-[hsl(var(--muted-foreground))] font-semibold mb-1.5">Reconstruction strategy</div>
        <div className="text-xs">{strategyDescription(chain.strategy)}</div>
      </div>
      <div>
        <div className="text-[10px] uppercase tracking-wide text-[hsl(var(--muted-foreground))] font-semibold mb-1.5">Tactic sequence</div>
        <div className="flex flex-wrap gap-1">
          {chain.tactic_sequence.map((t, i) => (
            <span key={i} className="text-[10px] font-mono px-1.5 py-0.5 rounded border border-[hsl(var(--border))]">
              {i + 1}. {t.replace(/_/g, ' ')}
            </span>
          ))}
        </div>
      </div>
      <div>
        <div className="text-[10px] uppercase tracking-wide text-[hsl(var(--muted-foreground))] font-semibold mb-1.5">APT stage sequence</div>
        <div className="flex flex-wrap gap-1">
          {chain.stage_sequence.map((s, i) => (
            <span key={i} className="text-[10px] font-mono px-1.5 py-0.5 rounded border border-[hsl(var(--border))] bg-[hsl(var(--purple-bg))] text-[hsl(var(--purple))]">
              {s || '—'}
            </span>
          ))}
        </div>
      </div>
      <div>
        <div className="text-[10px] uppercase tracking-wide text-[hsl(var(--muted-foreground))] font-semibold mb-1.5">Relation sequence</div>
        <div className="flex flex-wrap gap-1">
          {chain.relation_sequence.map((r, i) => (
            <RelationPill key={i} relation={r} />
          ))}
        </div>
      </div>
      <div>
        <div className="text-[10px] uppercase tracking-wide text-[hsl(var(--muted-foreground))] font-semibold mb-1.5">Involved nodes by type</div>
        <div className="flex flex-wrap gap-1.5">
          {Array.from(nodeTypeCount.entries()).map(([t, c]) => (
            <div key={t} className="flex items-center gap-1.5 px-2 py-1 rounded border border-[hsl(var(--border))] bg-[hsl(var(--background))]">
              <NodeTypePill type={t as import('@/lib/tgdetect/types').NodeType} />
              <span className="text-xs font-mono tabular-nums">{c}</span>
            </div>
          ))}
        </div>
      </div>
      <div>
        <div className="text-[10px] uppercase tracking-wide text-[hsl(var(--muted-foreground))] font-semibold mb-1.5">Causal links</div>
        {events.filter((e) => e.causal_parent).length === 0 ? (
          <div className="text-xs text-[hsl(var(--muted-foreground))]">No explicit causal_parent links in this chain.</div>
        ) : (
          <div className="space-y-1 max-h-60 overflow-y-auto">
            {events.filter((e) => e.causal_parent).map((e) => (
              <div key={e.event_id} className="text-[10px] mono p-1.5 rounded border border-[hsl(var(--border))] bg-[hsl(var(--background))]">
                <span className="text-[hsl(var(--muted-foreground))]">parent:</span>{' '}
                <span className="text-[hsl(var(--primary))]">{e.causal_parent}</span>{' '}
                <span className="text-[hsl(var(--muted-foreground))]">→ event:</span>{' '}
                <span>{e.event_id}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function strategyDescription(s: ChainStrategy): string {
  switch (s) {
    case 'chain_id':
      return 'Group by ground-truth chain_id field on each event. Highest-priority strategy — every malicious event with a non-null chain_id is grouped here. No minimum size filter.';
    case 'causal_parent':
      return 'Union-find over causal_parent event references. Each event whose causal_parent points to another event in the same set is joined. Chains with ≤1 event are dropped.';
    case 'entity_time':
      return 'Group ground-truth malicious events from the same source process, splitting on gaps longer than 300 seconds. These temporal clusters do not establish attack causality.';
  }
}
