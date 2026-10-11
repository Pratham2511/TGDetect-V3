'use client';

/**
 * Triage rail (ORTHRUS §4.4 suspicious-node isolation).
 *
 * Ranked list of flagged nodes blending model anomaly with ground-truth
 * incidence into a single priority. Picking a row sets the focal node for the
 * attack-summary reconstruction. Includes a text filter and clear loading /
 * empty / error states.
 */

import { useMemo, useState } from 'react';
import { Search, ShieldAlert } from 'lucide-react';

import { formatInt, shortNodeId } from '@/lib/tgdetect/formatters';
import type { Suspect } from '@/lib/tgdetect/graph/types';
import type { NodeType } from '@/lib/tgdetect/types';
import { cn } from '@/lib/utils';
import { EmptyState, ErrorState, LoadingState, MonoId, NodeTypePill, SectionTitle } from '../shared/pills';
import type { AsyncState } from '@/lib/tgdetect/services/hooks';

const pct = (n: number): string => `${Math.round(n * 100)}%`;

export interface TriageRailProps {
  suspects: Suspect[];
  state: AsyncState;
  error: string | null;
  selectedId: string | null;
  onSelect: (id: string, type: NodeType) => void;
  className?: string;
}

function SuspectRow({ suspect, rank, selected, onSelect }: {
  suspect: Suspect;
  rank: number;
  selected: boolean;
  onSelect: (id: string, type: NodeType) => void;
}) {
  const priorityW = Math.round(suspect.priority * 100);
  return (
    <button
      type="button"
      role="option"
      aria-selected={selected}
      onClick={() => onSelect(suspect.id, suspect.type)}
      className={cn(
        'w-full rounded-md border px-2.5 py-2 text-left transition-colors',
        selected
          ? 'border-[hsl(var(--primary)/0.5)] bg-[hsl(var(--primary)/0.08)]'
          : 'border-transparent hover:border-[hsl(var(--border))] hover:bg-[hsl(var(--muted)/0.4)]',
      )}
    >
      <div className="flex items-center gap-2">
        <span className="mono w-5 shrink-0 text-right text-[11px] font-semibold text-[hsl(var(--muted-foreground))]">{rank}</span>
        <NodeTypePill type={suspect.type} showLabel={false} />
        <MonoId truncateAt={20} className="flex-1">{shortNodeId(suspect.id)}</MonoId>
      </div>
      <div className="mt-1.5 flex items-center gap-2 pl-7">
        <div className="h-1 flex-1 overflow-hidden rounded-full bg-[hsl(var(--muted))]">
          <div className="h-full rounded-full bg-[hsl(var(--danger))]" style={{ width: `${priorityW}%` }} />
        </div>
        <span className="mono text-[10px] text-[hsl(var(--muted-foreground))]">{pct(suspect.priority)}</span>
      </div>
      <div className="mt-1 flex items-center gap-2 pl-7 text-[10px] text-[hsl(var(--muted-foreground))]">
        <span>{suspect.anomaly != null ? `anomaly ${pct(suspect.anomaly)}` : `severity ${pct(suspect.severity)}`}</span>
        <span>·</span>
        <span>{formatInt(suspect.maliciousEvents)} mal</span>
        <span>·</span>
        <span>deg {formatInt(suspect.degree)}</span>
      </div>
    </button>
  );
}
export function TriageRail({ suspects, state, error, selectedId, onSelect, className }: TriageRailProps) {
  const [query, setQuery] = useState('');

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return suspects;
    return suspects.filter((s) => s.id.toLowerCase().includes(q));
  }, [suspects, query]);

  return (
    <div className={cn('tg-panel flex h-full flex-col overflow-hidden', className)}>
      <div className="tg-hud-header shrink-0">
        <SectionTitle>
          <span className="inline-flex items-center gap-1.5">
            <ShieldAlert className="size-3.5 text-[hsl(var(--danger))]" /> Triage
          </span>
        </SectionTitle>
        <span className="mono text-[11px] text-[hsl(var(--muted-foreground))]">{formatInt(suspects.length)}</span>
      </div>

      <div className="shrink-0 border-b border-[hsl(var(--border))] p-2">
        <div className="relative">
          <Search className="pointer-events-none absolute left-2 top-1/2 size-3.5 -translate-y-1/2 text-[hsl(var(--muted-foreground))]" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Filter suspects…"
            className="h-8 w-full rounded-md border border-[hsl(var(--border))] bg-[hsl(var(--background))] pl-7 pr-2 text-[12px] text-[hsl(var(--foreground))] outline-none focus:border-[hsl(var(--primary)/0.5)]"
          />
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-1.5">
        {state === 'loading' && <LoadingState label="Ranking suspects…" />}
        {state === 'failed' && <ErrorState message={error ?? 'Failed to load suspects'} />}
        {(state === 'empty' || (state === 'success' && suspects.length === 0)) && (
          <EmptyState icon={ShieldAlert} title="No flagged nodes" description="This dataset has no malicious-labelled nodes to triage." />
        )}
        {state === 'success' && suspects.length > 0 && filtered.length === 0 && (
          <p className="px-2 py-6 text-center text-[11px] text-[hsl(var(--muted-foreground))]">No suspects match “{query}”.</p>
        )}
        {filtered.length > 0 && (
          <div className="space-y-0.5" role="listbox" aria-label="Suspect nodes">
            {filtered.map((s, i) => (
              <SuspectRow key={s.id} suspect={s} rank={i + 1} selected={s.id === selectedId} onSelect={onSelect} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}