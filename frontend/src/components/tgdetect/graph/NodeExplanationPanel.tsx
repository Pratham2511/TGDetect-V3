'use client';

/**
 * "Why is this node flagged?" explanation panel.
 *
 * Turns a selected SummaryNode into an analyst-readable verdict: the model's
 * decision (mean anomaly vs threshold, flagged-event count) with an explicit
 * ground-truth fallback when the dataset has no scores, the three ORTHRUS
 * scores (anomaly fA, degree fD, criticality fC) as bars, structural + temporal
 * facts, and the top incident events sorted by model probability as evidence.
 */

import { useMemo } from 'react';
import { AlertTriangle, ArrowRight, Clock, GitBranch, ShieldCheck, Target } from 'lucide-react';

import { useNodeEvents } from '@/lib/tgdetect/services/hooks';
import {
  formatEpochTime,
  formatInt,
  shortNodeId,
} from '@/lib/tgdetect/formatters';
import type { AttackSummary, SummaryNode } from '@/lib/tgdetect/graph/types';
import type { TGEvent } from '@/lib/tgdetect/types';
import { cn } from '@/lib/utils';
import { LabelPill, MonoId, NodeTypePill, RelationPill, SectionTitle } from '../shared/pills';

const pct = (n: number): string => `${Math.round(n * 100)}%`;

type Tone = 'danger' | 'warning' | 'success' | 'info';
const toneBadge: Record<Tone, string> = {
  danger: 'badge-danger',
  warning: 'badge-warning',
  success: 'badge-success',
  info: 'badge-info',
};

const ROLE_LABEL: Record<SummaryNode['role'], string> = {
  focal: 'Focal detection',
  entry: 'Entry · backward trace',
  exit: 'Exit · forward trace',
  pivot: 'Pivot host',
  related: 'Related activity',
};

/** A labelled 0–1 score bar. */
function ScoreBar({ label, value, token, hint }: { label: string; value: number; token: string; hint?: string }) {
  const w = Math.round(Math.max(0, Math.min(1, value)) * 100);
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-baseline justify-between">
        <span className="text-[11px] font-medium text-[hsl(var(--foreground))]">{label}</span>
        <span className="mono text-[11px] text-[hsl(var(--muted-foreground))]">{pct(value)}</span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-[hsl(var(--muted))]">
        <div className="h-full rounded-full" style={{ width: `${w}%`, backgroundColor: `hsl(var(--${token}))` }} />
      </div>
      {hint && <span className="text-[10px] text-[hsl(var(--muted-foreground))]">{hint}</span>}
    </div>
  );
}

/** A compact labelled fact tile. */
function Fact({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="tg-subsurface flex flex-col gap-0.5 px-2.5 py-1.5">
      <span className="text-[10px] uppercase tracking-wide text-[hsl(var(--muted-foreground))]">{label}</span>
      <span className="mono text-[13px] font-semibold text-[hsl(var(--foreground))]">{value}</span>
      {sub && <span className="text-[10px] text-[hsl(var(--muted-foreground))]">{sub}</span>}
    </div>
  );
}
export interface NodeExplanationPanelProps {
  node: SummaryNode | null;
  summary: AttackSummary;
  onOpenEvents?: (nodeId: string) => void;
  className?: string;
}

interface Evidence {
  e: TGEvent;
  probability: number | null;
  prediction: 0 | 1 | null;
  threshold: number | null;
  counterpart: string;
}

export function NodeExplanationPanel({ node, summary, onOpenEvents, className }: NodeExplanationPanelProps) {
  const { data: events } = useNodeEvents(node?.id ?? null);

  const evidence: Evidence[] = useMemo(() => {
    if (!node) return [];
    const list = (events ?? []).map((e) => {
      const a = (e.attrs ?? {}) as Record<string, unknown>;
      const probability = typeof a.probability === 'number' && Number.isFinite(a.probability) ? a.probability : null;
      const prediction = a.prediction === 1 || a.prediction === 0 ? (a.prediction as 0 | 1) : null;
      const threshold = typeof a.threshold === 'number' && Number.isFinite(a.threshold) ? a.threshold : null;
      const counterpart = e.src_id === node.id ? e.dst_id : e.src_id;
      return { e, probability, prediction, threshold, counterpart };
    });
    list.sort((x, y) => (y.probability ?? -1) - (x.probability ?? -1) || y.e.label - x.e.label);
    return list;
  }, [events, node]);

  const threshold = evidence.find((x) => x.threshold != null)?.threshold ?? null;

  if (!node) {
    return (
      <div className={cn('tg-panel flex h-full flex-col items-center justify-center p-6 text-center', className)}>
        <Target className="mb-2 size-7 text-[hsl(var(--muted-foreground))] opacity-50" />
        <p className="text-sm font-semibold text-[hsl(var(--foreground))]">No node selected</p>
        <p className="mt-1 text-xs text-[hsl(var(--muted-foreground))]">
          Click any node in the attack summary to see why it was flagged and the evidence behind it.
        </p>
      </div>
    );
  }

  const hasScores = summary.stats.hasModelScores;
  let tone: Tone = 'info';
  let title = 'Context node';
  let detail = '';
  let VerdictIcon = ShieldCheck;
  if (node.flaggedEvents > 0) {
    tone = 'danger';
    VerdictIcon = AlertTriangle;
    title = 'Model-flagged anomaly';
    detail = `The model scored ${node.flaggedEvents} of ${node.incidentEvents} incident event${node.incidentEvents === 1 ? '' : 's'} above its decision threshold${threshold != null ? ` (${pct(threshold)})` : ''}, with mean anomaly ${pct(node.anomaly)}.`;
  } else if (hasScores && node.anomaly >= 0.5) {
    tone = 'warning';
    VerdictIcon = AlertTriangle;
    title = 'Elevated anomaly';
    detail = `Mean model anomaly is ${pct(node.anomaly)} across ${node.incidentEvents} incident event${node.incidentEvents === 1 ? '' : 's'} — elevated, though no single event crossed the threshold${threshold != null ? ` (${pct(threshold)})` : ''}.`;
  } else if (node.maliciousEvents > 0) {
    tone = 'danger';
    VerdictIcon = AlertTriangle;
    title = 'Ground-truth malicious';
    detail = `${hasScores ? 'The model did not flag this node, but ' : ''}${node.maliciousEvents} of ${node.incidentEvents} incident event${node.incidentEvents === 1 ? '' : 's'} are labelled malicious in the dataset.`;
  } else {
    detail = `Included as ${ROLE_LABEL[node.role].toLowerCase()}; it shares a host and timing with the focal detection but carries no direct anomaly or malicious label.`;
  }

  const temporalSub =
    node.role === 'focal' ? 'Detection anchor'
    : node.lastTs <= summary.stats.center ? 'Acted before focal (precursor)'
    : node.firstTs >= summary.stats.center ? 'Acted after focal (successor)'
    : 'Spans the detection window';
  return (
    <div className={cn('tg-panel flex h-full flex-col overflow-hidden', className)}>
      {/* header */}
      <div className="tg-hud-header shrink-0">
        <div className="flex min-w-0 items-center gap-2">
          <NodeTypePill type={node.type} />
          <MonoId truncateAt={22} className="text-[12px]">{shortNodeId(node.id)}</MonoId>
        </div>
        <span className={cn('shrink-0 rounded px-2 py-0.5 text-[10px] font-semibold', toneBadge[tone])}>
          {ROLE_LABEL[node.role]}
        </span>
      </div>

      <div className="flex-1 space-y-4 overflow-y-auto p-3.5">
        {/* verdict */}
        <div className={cn('flex items-start gap-2.5 rounded-md border p-3', toneBadge[tone])}>
          <VerdictIcon className="mt-0.5 size-4 shrink-0" />
          <div className="space-y-0.5">
            <p className="text-[13px] font-semibold">{title}</p>
            <p className="text-[11px] leading-snug opacity-90">{detail}</p>
          </div>
        </div>

        {/* ORTHRUS scores */}
        <section className="space-y-2.5">
          <SectionTitle>Reconstruction scores</SectionTitle>
          <ScoreBar label="Anomaly fA" value={node.anomaly} token="danger" hint={hasScores ? 'Mean model probability on incident edges' : 'Label-derived (no model scores)'} />
          <ScoreBar label="Degree fD" value={node.degreeScore} token="info" hint="Structural load within this summary" />
          <ScoreBar label="Criticality fC" value={node.criticality} token="purple" hint="Blended rank used to size & keep this node" />
        </section>
        {/* structural + temporal facts */}
        <section className="space-y-2">
          <SectionTitle>Facts</SectionTitle>
          <div className="grid grid-cols-2 gap-2">
            <Fact label="Incident events" value={formatInt(node.incidentEvents)} sub={`${node.inDeg} in · ${node.outDeg} out`} />
            <Fact label="Flagged / malicious" value={`${formatInt(node.flaggedEvents)} / ${formatInt(node.maliciousEvents)}`} sub="model · ground-truth" />
            <Fact label="First seen" value={formatEpochTime(node.firstTs)} />
            <Fact label="Last seen" value={formatEpochTime(node.lastTs)} />
          </div>
          <div className="tg-subsurface flex items-center gap-2 px-2.5 py-1.5">
            <Clock className="size-3.5 text-[hsl(var(--teal))]" />
            <span className="text-[11px] text-[hsl(var(--foreground))]">{temporalSub}</span>
            <span className="ml-auto inline-flex items-center gap-1 text-[10px] text-[hsl(var(--muted-foreground))]">
              <GitBranch className="size-3" /> hop {node.hop} · layer {node.layer}
            </span>
          </div>
        </section>

        {/* evidence */}
        <section className="space-y-2">
          <SectionTitle
            right={
              onOpenEvents ? (
                <button
                  type="button"
                  onClick={() => onOpenEvents(node.id)}
                  className="inline-flex items-center gap-1 text-[10px] font-medium text-[hsl(var(--primary))] hover:underline"
                >
                  Open in Events <ArrowRight className="size-3" />
                </button>
              ) : undefined
            }
          >
            Top evidence {hasScores ? '· by model probability' : '· by label'}
          </SectionTitle>
          {evidence.length === 0 ? (
            <p className="tg-subsurface px-2.5 py-2 text-[11px] text-[hsl(var(--muted-foreground))]">
              No incident events returned for this node in the current dataset.
            </p>
          ) : (
            <div className="space-y-1.5">
              {evidence.slice(0, 6).map(({ e, probability, prediction, counterpart }) => (
                <div key={e.event_id} className="tg-subsurface flex items-center gap-2 px-2.5 py-1.5">
                  <span
                    className={cn(
                      'mono w-10 shrink-0 text-right text-[11px] font-semibold',
                      prediction === 1 ? 'text-[hsl(var(--danger))]' : 'text-[hsl(var(--muted-foreground))]',
                    )}
                  >
                    {probability != null ? pct(probability) : '—'}
                  </span>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-1.5">
                      <ArrowRight className="size-3 shrink-0 text-[hsl(var(--muted-foreground))]" />
                      <MonoId truncateAt={22}>{shortNodeId(counterpart)}</MonoId>
                    </div>
                    <div className="mt-0.5 flex items-center gap-1.5">
                      <RelationPill relation={e.relation} />
                      <span className="text-[10px] text-[hsl(var(--muted-foreground))]">{formatEpochTime(e.ts)}</span>
                    </div>
                  </div>
                  <LabelPill label={e.label} />
                </div>
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}