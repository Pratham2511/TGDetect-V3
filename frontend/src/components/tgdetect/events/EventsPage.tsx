'use client';

import { useMemo, useState } from 'react';
import { ChevronRight, Filter, Search, X } from 'lucide-react';
import { useEvents } from '@/lib/tgdetect/services/hooks';
import { useDataset } from '@/lib/dataset-context';
import {
  epochToIso,
  eventAttrsString,
  formatEpoch,
  formatEpochTime,
  formatInt,
  labelText,
} from '@/lib/tgdetect/formatters';
import {
  NODE_TYPES,
  RELATION_TYPES,
  SYNTHETIC_TACTIC_ORDER,
} from '@/lib/tgdetect/constants';
import type {
  EventFilter,
  EventLabel,
  NodeType,
  RelationType,
  TGEvent,
} from '@/lib/tgdetect/types';
import {
  EmptyState,
  ErrorState,
  LabelPill,
  LoadingState,
  MonoId,
  NodeTypePill,
  RelationPill,
  SectionTitle,
  StatePill,
} from '../shared/pills';
import { useChain } from '@/lib/tgdetect/services/hooks';
import { useNodeEvents } from '@/lib/tgdetect/services/hooks';

export interface EventsPageProps {
  initialFilter?: Partial<EventFilter>;
  initialEventId?: string | null;
  onNavigate?: (page: string, ctx?: Record<string, unknown>) => void;
}

/**
 * The page wrapper uses a `key` derived from initialFilter + initialEventId
 * so that when navigation passes new initial state, the inner component
 * remounts and re-seeds its useState cleanly without useEffect.
 */
export function EventsPage(props: EventsPageProps) {
  const key = useMemo(() => {
    return JSON.stringify({
      labels: props.initialFilter?.labels,
      q: props.initialFilter?.event_id_query,
      ev: props.initialEventId,
    });
  }, [props.initialFilter, props.initialEventId]);
  return <EventsPageInner key={key} {...props} />;
}

function EventsPageInner({ initialFilter, initialEventId, onNavigate }: EventsPageProps) {
  const [filter, setFilter] = useState<EventFilter>({
    limit: 100,
    offset: 0,
    ...initialFilter,
  });
  const [selectedEventId, setSelectedEventId] = useState<string | null>(initialEventId ?? null);
  const [searchInput, setSearchInput] = useState(initialFilter?.event_id_query ?? '');

  const eventsRes = useEvents(filter);

  return (
    <div className="space-y-4">
      <FiltersBar
        filter={filter}
        setFilter={setFilter}
        searchInput={searchInput}
        setSearchInput={setSearchInput}
      />
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2">
          <EventsTable
            events={eventsRes.data?.items ?? []}
            total={eventsRes.data?.total ?? 0}
            loading={eventsRes.state === 'loading'}
            error={eventsRes.state === 'failed' ? eventsRes.error : null}
            selectedEventId={selectedEventId}
            onSelect={(id) => setSelectedEventId(id)}
            onNavigate={onNavigate}
            filter={filter}
            setFilter={setFilter}
          />
        </div>
        <div>
          <EventDetailPanel eventId={selectedEventId} onClose={() => setSelectedEventId(null)} onNavigate={onNavigate} />
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Filters bar
// ─────────────────────────────────────────────────────────────────────────────

function FiltersBar({
  filter,
  setFilter,
  searchInput,
  setSearchInput,
}: {
  filter: EventFilter;
  setFilter: (f: EventFilter) => void;
  searchInput: string;
  setSearchInput: (s: string) => void;
}) {
  const toggleLabel = (l: EventLabel) => {
    const labels = new Set(filter.labels ?? []);
    if (labels.has(l)) labels.delete(l); else labels.add(l);
    setFilter({ ...filter, labels: Array.from(labels), offset: 0 });
  };
  const toggleNodeType = (t: NodeType) => {
    const arr = new Set(filter.node_types ?? []);
    if (arr.has(t)) arr.delete(t); else arr.add(t);
    setFilter({ ...filter, node_types: Array.from(arr) as NodeType[], offset: 0 });
  };
  const toggleRelation = (r: RelationType) => {
    const arr = new Set((filter.relations ?? []) as string[]);
    if (arr.has(r)) arr.delete(r); else arr.add(r);
    setFilter({ ...filter, relations: Array.from(arr) as RelationType[], offset: 0 });
  };
  const toggleTactic = (t: string) => {
    const arr = new Set(filter.tactics ?? []);
    if (arr.has(t)) arr.delete(t); else arr.add(t);
    setFilter({ ...filter, tactics: Array.from(arr), offset: 0 });
  };
  return (
    <div className="tg-card p-3 space-y-2">
      <div className="flex items-center gap-2">
        <Filter className="size-3.5 text-[hsl(var(--muted-foreground))]" />
        <SectionTitle className="flex-1">Filters</SectionTitle>
        <button
          type="button"
          onClick={() => { setFilter({ limit: 100, offset: 0 }); setSearchInput(''); }}
          className="text-[10px] text-[hsl(var(--muted-foreground))] hover:text-[hsl(var(--foreground))]"
        >clear all</button>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <div className="relative flex-1 min-w-[220px]">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 size-3.5 text-[hsl(var(--muted-foreground))]" />
          <input
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') { setFilter({ ...filter, event_id_query: searchInput, offset: 0 }); } }}
            placeholder="search event_id… (press Enter)"
            className="w-full pl-8 pr-2.5 py-1.5 text-xs bg-[hsl(var(--background))] border border-[hsl(var(--border))] rounded font-mono focus:outline-hidden focus:border-[hsl(var(--primary))]"
          />
        </div>
        <div className="flex items-center gap-1.5">
          <span className="text-[10px] font-mono text-[hsl(var(--muted-foreground))] uppercase mr-1">Label:</span>
          {([0, 1] as EventLabel[]).map((l) => (
            <button
              key={l}
              type="button"
              onClick={() => toggleLabel(l)}
              className={`px-2 py-1 text-[10px] font-mono font-semibold border rounded transition-all ${
                (filter.labels ?? []).includes(l)
                  ? (l === 1 ? 'bg-[hsl(var(--danger-bg))] text-[hsl(var(--danger))] border-[hsl(var(--danger)/0.4)] shadow-xs' : 'bg-[hsl(var(--success-bg))] text-[hsl(var(--success))] border-[hsl(var(--success)/0.4)] shadow-xs')
                  : 'bg-[hsl(var(--card))] text-[hsl(var(--muted-foreground))] border-[hsl(var(--border))] hover:border-[hsl(var(--primary)/0.4)]'
              }`}
            >{l === 1 ? 'MAL' : 'BEN'} · {labelText(l)}</button>
          ))}
        </div>
      </div>

      <div className="space-y-1.5 pt-1.5 border-t border-[hsl(var(--border))]">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-[9px] uppercase tracking-wider text-[hsl(var(--muted-foreground))] font-semibold w-16 flex-shrink-0">Entity:</span>
          <div className="flex flex-wrap gap-1 flex-1">
            {NODE_TYPES.map((t) => {
              const active = (filter.node_types ?? []).includes(t);
              return (
                <button
                  key={t}
                  type="button"
                  onClick={() => toggleNodeType(t)}
                  className={`px-1.5 py-0.5 text-[10px] font-mono border rounded transition-colors ${
                    active ? 'border-[hsl(var(--primary)/0.6)] bg-[hsl(var(--info-bg))] text-[hsl(var(--foreground))] font-medium' : 'border-[hsl(var(--border))] bg-[hsl(var(--card))] text-[hsl(var(--muted-foreground))] hover:text-[hsl(var(--foreground))]'
                  }`}
                >{t}</button>
              );
            })}
          </div>
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-[9px] uppercase tracking-wider text-[hsl(var(--muted-foreground))] font-semibold w-16 flex-shrink-0">Relation:</span>
          <div className="flex flex-wrap gap-1 flex-1">
            {RELATION_TYPES.map((r) => {
              const active = (filter.relations ?? []).includes(r);
              return (
                <button
                  key={r}
                  type="button"
                  onClick={() => toggleRelation(r)}
                  className={`px-1.5 py-0.5 text-[10px] font-mono border rounded transition-colors ${
                    active ? 'border-[hsl(var(--primary)/0.6)] bg-[hsl(var(--info-bg))] text-[hsl(var(--foreground))] font-medium' : 'border-[hsl(var(--border))] bg-[hsl(var(--card))] text-[hsl(var(--muted-foreground))] hover:text-[hsl(var(--foreground))]'
                  }`}
                >{r}</button>
              );
            })}
          </div>
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-[9px] uppercase tracking-wider text-[hsl(var(--muted-foreground))] font-semibold w-16 flex-shrink-0">Tactic:</span>
          <div className="flex flex-wrap gap-1 flex-1">
            {SYNTHETIC_TACTIC_ORDER.map((t) => {
              const active = (filter.tactics ?? []).includes(t);
              return (
                <button
                  key={t}
                  type="button"
                  onClick={() => toggleTactic(t)}
                  className={`px-1.5 py-0.5 text-[10px] font-mono border rounded transition-colors ${
                    active ? 'border-[hsl(var(--primary)/0.6)] bg-[hsl(var(--info-bg))] text-[hsl(var(--foreground))] font-medium' : 'border-[hsl(var(--border))] bg-[hsl(var(--card))] text-[hsl(var(--muted-foreground))] hover:text-[hsl(var(--foreground))]'
                  }`}
                >{t.replace(/_/g, ' ')}</button>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Events table
// ─────────────────────────────────────────────────────────────────────────────

function EventsTable({
  events,
  total,
  loading,
  error,
  selectedEventId,
  onSelect,
  onNavigate,
  filter,
  setFilter,
}: {
  events: TGEvent[];
  total: number;
  loading: boolean;
  error: string | null;
  selectedEventId: string | null;
  onSelect: (id: string) => void;
  onNavigate?: (page: string, ctx?: Record<string, unknown>) => void;
  filter: EventFilter;
  setFilter: (f: EventFilter) => void;
}) {
  const { activeDataset, activeDatasetId } = useDataset();
  const pageSize = filter.limit ?? 100;
  const offset = filter.offset ?? 0;
  return (
    <div className="tg-card p-3">
      <SectionTitle
        right={
          <div className="flex items-center gap-2 text-[10px] text-[hsl(var(--muted-foreground))] font-mono">
            <span>{formatInt(total)} events</span>
            <span>·</span>
            <span>showing {offset + 1}–{Math.min(offset + pageSize, total)}</span>
            <button
              type="button"
              disabled={offset === 0}
              onClick={() => setFilter({ ...filter, offset: Math.max(0, offset - pageSize) })}
              className="px-1.5 py-0.5 border border-[hsl(var(--border))] rounded disabled:opacity-30"
            >prev</button>
            <button
              type="button"
              disabled={offset + pageSize >= total}
              onClick={() => setFilter({ ...filter, offset: offset + pageSize })}
              className="px-1.5 py-0.5 border border-[hsl(var(--border))] rounded disabled:opacity-30"
            >next</button>
          </div>
        }
      >
        <div className="flex items-center gap-2">
          <span className={`size-2 rounded-full ${activeDataset?.provenance === 'benchmark' ? 'bg-amber-400' : 'bg-emerald-400 animate-pulse'}`} />
          <span>Events</span>
          <span className="text-[10px] font-mono px-2 py-0.5 rounded border border-[hsl(var(--border))] bg-[hsl(var(--background))] text-[hsl(var(--primary))] font-semibold">
            {activeDataset?.name ?? activeDatasetId ?? 'LIVE TELEMETRY'}
          </span>
        </div>
      </SectionTitle>
      {loading ? (
        <LoadingState label="Loading events…" />
      ) : error ? (
        <ErrorState message={`Failed: ${error}`} />
      ) : total === 0 ? (
        <EmptyState
          title="No events in active telemetry"
          description={`The active dataset "${activeDataset?.name ?? activeDatasetId}" has 0 recorded events or has not yet been processed.`}
        />
      ) : events.length === 0 ? (
        <EmptyState title="No events match filters" description="Adjust label, relation, or search parameters to view events" />
      ) : (
        <div className="overflow-x-auto max-h-[70vh] overflow-y-auto">
          <table className="w-full text-xs">
            <thead className="sticky top-0 bg-[hsl(var(--card))]">
              <tr className="text-left text-[10px] uppercase tracking-wide text-[hsl(var(--muted-foreground))] border-b border-[hsl(var(--border))]">
                <th className="py-1.5 pr-3">event_id</th>
                <th className="py-1.5 pr-3">ts</th>
                <th className="py-1.5 pr-3">src → dst</th>
                <th className="py-1.5 pr-3">relation</th>
                <th className="py-1.5 pr-3">Ground truth</th><th className="py-1.5 pr-3">Score</th><th className="py-1.5 pr-3">Prediction</th>
                <th className="py-1.5 pr-3">tactics</th>
                <th className="py-1.5 pr-3">stage</th>
                <th className="py-1.5 pr-3">source</th>
                <th className="py-1.5 pr-3">chain</th>
              </tr>
            </thead>
            <tbody>
              {events.map((e) => {
                const isSelected = e.event_id === selectedEventId;
                const isMalicious = e.label === 1;
                return (
                  <tr
                    key={e.event_id}
                    className={`border-b border-[hsl(var(--border)/0.5)] cursor-pointer hover:bg-[hsl(var(--card-hover))] transition-colors ${
                      isSelected
                        ? 'bg-[hsl(var(--info-bg))] border-l-2 border-l-[hsl(var(--primary))]'
                        : isMalicious
                          ? 'border-l-2 border-l-red-500/80 bg-red-500/[0.03]'
                          : 'border-l-2 border-l-transparent'
                    }`}
                    onClick={() => onSelect(e.event_id)}
                  >
                    <td className="py-1.5 pr-3"><MonoId truncateAt={18}>{e.event_id}</MonoId></td>
                    <td className="py-1.5 pr-3 mono text-[11px] whitespace-nowrap" title={formatEpoch(e.ts)}>{formatEpochTime(e.ts)}</td>
                    <td className="py-1.5 pr-3 mono text-[11px] whitespace-nowrap">
                      <div className="flex items-center gap-1">
                        <span>{e.src_id}</span>
                        <ChevronRight className="size-2.5 text-[hsl(var(--muted-foreground))]" />
                        <span>{e.dst_id}</span>
                      </div>
                      <div className="flex items-center gap-1 mt-0.5">
                        <NodeTypePill type={e.src_type} showLabel={false} />
                        <NodeTypePill type={e.dst_type} showLabel={false} />
                      </div>
                    </td>
                    <td className="py-1.5 pr-3"><RelationPill relation={e.relation} /></td>
                    <td className="py-1.5 pr-3"><LabelPill label={e.label} /></td><td className="py-1.5 pr-3 font-mono">{typeof e.attrs.probability === "number" ? e.attrs.probability.toFixed(5) : "—"}</td><td className="py-1.5 pr-3">{e.attrs.prediction === 1 ? "Alert" : e.attrs.prediction === 0 ? "Below threshold" : "—"}</td>
                    <td className="py-1.5 pr-3 text-[11px]">{e.tactics.join(', ')}</td>
                    <td className="py-1.5 pr-3 text-[11px] text-[hsl(var(--muted-foreground))]">{e.apt_stage ?? '—'}</td>
                    <td className="py-1.5 pr-3"><span className="font-mono text-[10px] px-1.5 py-0.5 rounded border border-[hsl(var(--border))]">{e.source_tag}</span></td>
                    <td className="py-1.5 pr-3">
                      {e.chain_id ? (
                        <button
                          type="button"
                          onClick={(ev) => { ev.stopPropagation(); onNavigate?.('chains', { chainId: e.chain_id }); }}
                          className="mono text-[10px] text-[hsl(var(--primary))] hover:underline"
                          title={e.chain_id ?? ''}
                        >{e.chain_id.slice(0, 14)}…</button>
                      ) : (
                        <span className="text-[hsl(var(--muted-foreground))]">—</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Event detail panel
// ─────────────────────────────────────────────────────────────────────────────

function EventDetailPanel({
  eventId,
  onClose,
  onNavigate,
}: {
  eventId: string | null;
  onClose: () => void;
  onNavigate?: (page: string, ctx?: Record<string, unknown>) => void;
}) {
  if (!eventId) {
    return (
      <div className="tg-card p-4 h-full">
        <SectionTitle>Event Detail</SectionTitle>
        <EmptyState
          title="No event selected"
          description="Click a row in the table to inspect event details, causal links, and related chains."
        />
      </div>
    );
  }
  return <EventDetailBody eventId={eventId} onClose={onClose} onNavigate={onNavigate} />;
}

function EventDetailBody({
  eventId,
  onClose,
  onNavigate,
}: {
  eventId: string;
  onClose: () => void;
  onNavigate?: (page: string, ctx?: Record<string, unknown>) => void;
}) {
  // We don't have a direct hook for fetching a single event without re-listing,
  // but the events service exposes a `get` method — replicate it via useEvents
  // with an event_id_query filter.
  const eventsRes = useEvents({ event_id_query: eventId, limit: 1 });
  const event = eventsRes.data?.items[0];
  const chainRes = useChain(event?.chain_id ?? null);
  const srcNodeEventsRes = useNodeEvents(event?.src_id ?? null);
  const dstNodeEventsRes = useNodeEvents(event?.dst_id ?? null);

  if (eventsRes.state === 'loading') {
    return (
      <div className="tg-card p-4 h-full">
        <LoadingState label="Loading event…" />
      </div>
    );
  }
  if (!event) {
    return (
      <div className="tg-card p-4 h-full">
        <SectionTitle>Event Detail</SectionTitle>
        <EmptyState title="Event not found" description={eventId} />
      </div>
    );
  }

  const attrsString = eventAttrsString(event);
  return (
    <div className="tg-card p-4 space-y-3 h-full overflow-y-auto max-h-[80vh]">
      <div className="flex items-center justify-between">
        <SectionTitle>Event Detail</SectionTitle>
        <button onClick={onClose} className="text-[hsl(var(--muted-foreground))] hover:text-[hsl(var(--foreground))]" title="close">
          <X className="size-3.5" />
        </button>
      </div>
      <div>
        <div className="text-[10px] uppercase tracking-wide text-[hsl(var(--muted-foreground))] font-semibold">event_id</div>
        <div className="mono text-xs break-all">{event.event_id}</div>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <Field label="Timestamp">
          <div className="text-[11px] mono">{formatEpoch(event.ts)}</div>
          <div className="text-[10px] text-[hsl(var(--muted-foreground))]">{formatEpochTime(event.ts)}</div>
        </Field>
        <Field label="Label">
          <LabelPill label={event.label} />
        </Field>
        <Field label="Source node">
          <div className="mono text-[11px] break-all">{event.src_id}</div>
          <NodeTypePill type={event.src_type} />
        </Field>
        <Field label="Destination node">
          <div className="mono text-[11px] break-all">{event.dst_id}</div>
          <NodeTypePill type={event.dst_type} />
        </Field>
        <Field label="Relation">
          <RelationPill relation={event.relation} />
        </Field>
        <Field label="Source tag">
          <span className="font-mono text-[10px] px-1.5 py-0.5 rounded border border-[hsl(var(--border))]">{event.source_tag}</span>
        </Field>
        <Field label="APT stage">
          <span className="text-[11px]">{event.apt_stage ?? '—'}</span>
        </Field>
        <Field label="Tactics">
          <div className="flex flex-wrap gap-1">
            {event.tactics.map((t) => (
              <span key={t} className="text-[10px] font-mono px-1.5 py-0.5 rounded border border-[hsl(var(--border))]">{t}</span>
            ))}
          </div>
        </Field>
        <Field label="Chain ID">
          {event.chain_id ? (
            <button
              type="button"
              onClick={() => onNavigate?.('chains', { chainId: event.chain_id })}
              className="mono text-[11px] text-[hsl(var(--primary))] hover:underline break-all"
            >{event.chain_id}</button>
          ) : (
            <span className="text-[hsl(var(--muted-foreground))]">—</span>
          )}
        </Field>
        <Field label="Causal parent">
          {event.causal_parent ? (
            <button
              type="button"
              onClick={() => onNavigate?.('events', { eventId: event.causal_parent })}
              className="mono text-[11px] text-[hsl(var(--primary))] hover:underline"
            >{event.causal_parent}</button>
          ) : (
            <span className="text-[hsl(var(--muted-foreground))]">—</span>
          )}
        </Field>
      </div>
      {attrsString && (
        <div>
          <div className="text-[10px] uppercase tracking-wide text-[hsl(var(--muted-foreground))] font-semibold mb-1">attrs</div>
          <div className="mono text-[10px] p-2 rounded border border-[hsl(var(--border))] bg-[hsl(var(--background))] break-all whitespace-pre-wrap">
            {attrsString}
          </div>
        </div>
      )}
      {/* Chain context */}
      {chainRes.data && (
        <div className="border-t border-[hsl(var(--border))] pt-2">
          <div className="text-[10px] uppercase tracking-wide text-[hsl(var(--muted-foreground))] font-semibold mb-1">In chain</div>
          <div className="flex items-center gap-2">
            <span className="mono text-[11px]">{chainRes.data.chain_id}</span>
            <span className="text-[10px] text-[hsl(var(--muted-foreground))]">strategy:</span>
            <span className="mono text-[10px]">{chainRes.data.strategy}</span>
            <button
              type="button"
              onClick={() => onNavigate?.('chains', { chainId: chainRes.data!.chain_id })}
              className="text-[10px] text-[hsl(var(--primary))] hover:underline ml-auto"
            >open →</button>
          </div>
        </div>
      )}
      {/* Related events */}
      <div className="border-t border-[hsl(var(--border))] pt-2">
        <div className="text-[10px] uppercase tracking-wide text-[hsl(var(--muted-foreground))] font-semibold mb-1">
          Related events touching source node ({srcNodeEventsRes.data?.length ?? 0})
        </div>
        <div className="max-h-32 overflow-y-auto">
          {(srcNodeEventsRes.data ?? []).slice(0, 8).map((e) => (
            <button
              key={e.event_id}
              type="button"
              onClick={() => onNavigate?.('events', { eventId: e.event_id })}
              className="block w-full text-left text-[10px] mono px-1.5 py-0.5 hover:bg-[hsl(var(--card-hover))] rounded"
            >
              <span className="text-[hsl(var(--muted-foreground))]">{formatEpochTime(e.ts)}</span> · {e.relation} · {e.label === 1 ? 'MAL' : 'BEN'}
            </button>
          ))}
        </div>
      </div>
      <div className="border-t border-[hsl(var(--border))] pt-2">
        <div className="text-[10px] uppercase tracking-wide text-[hsl(var(--muted-foreground))] font-semibold mb-1">
          Related events touching destination node ({dstNodeEventsRes.data?.length ?? 0})
        </div>
        <div className="max-h-32 overflow-y-auto">
          {(dstNodeEventsRes.data ?? []).slice(0, 8).map((e) => (
            <button
              key={e.event_id}
              type="button"
              onClick={() => onNavigate?.('events', { eventId: e.event_id })}
              className="block w-full text-left text-[10px] mono px-1.5 py-0.5 hover:bg-[hsl(var(--card-hover))] rounded"
            >
              <span className="text-[hsl(var(--muted-foreground))]">{formatEpochTime(e.ts)}</span> · {e.relation} · {e.label === 1 ? 'MAL' : 'BEN'}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-0.5">
      <div className="text-[10px] uppercase tracking-wide text-[hsl(var(--muted-foreground))] font-semibold">{label}</div>
      <div>{children}</div>
    </div>
  );
}
