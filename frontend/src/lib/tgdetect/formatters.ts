/**
 * TGDetect formatters — pure presentation adapters.
 *
 * These functions convert backend-conformant domain objects into
 * UI-displayable strings/numbers WITHOUT mutating the source. They live
 * separately from the domain types so the types can be serialized to the
 * backend verbatim and the formatters can be re-skinned without touching
 * data semantics.
 */

import { formatDateTime } from '@/lib/date-utils';

import {
  NODE_TYPE_META,
  RELATION_TYPE_META,
  PROCESSING_STATE_META,
} from './constants';
import type {
  AttackChainSummary,
  ChainStrategy,
  EventLabel,
  GraphNode,
  NodeType,
  ProcessingState,
  RelationType,
  TGEvent,
} from './types';

// ─────────────────────────────────────────────────────────────────────────────
// Timestamp formatting — epoch seconds → human
// ─────────────────────────────────────────────────────────────────────────────

/** Convert epoch seconds (float) to a Date-safe ISO string for date-utils. */
export function epochToIso(ts: number | null | undefined): string {
  if (ts == null || !Number.isFinite(ts)) return '';
  return new Date(ts * 1000).toISOString();
}

export function formatEpoch(ts: number | null | undefined): string {
  if (ts == null || !Number.isFinite(ts)) return 'unavailable';
  const iso = epochToIso(ts);
  return iso ? formatDateTime(iso) : 'unavailable';
}

export function formatEpochTime(ts: number | null | undefined): string {
  if (ts == null || !Number.isFinite(ts)) return 'unavailable';
  const d = new Date(ts * 1000);
  const h = String(d.getHours()).padStart(2, '0');
  const m = String(d.getMinutes()).padStart(2, '0');
  const s = String(d.getSeconds()).padStart(2, '0');
  return `${h}:${m}:${s}`;
}

export function formatEpochDate(ts: number | null | undefined): string {
  if (ts == null || !Number.isFinite(ts)) return 'unavailable';
  const d = new Date(ts * 1000);
  const y = d.getFullYear();
  const mo = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${y}-${mo}-${day}`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Duration formatting — seconds → human compact
// ─────────────────────────────────────────────────────────────────────────────

export function formatDurationShort(s: number | null | undefined): string {
  if (s == null || !Number.isFinite(s)) return 'unavailable';
  if (s < 1) return `${(s * 1000).toFixed(0)}ms`;
  if (s < 60) return `${s.toFixed(1)}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${Math.floor(s % 60)}s`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
  return `${Math.floor(s / 86400)}d ${Math.floor((s % 86400) / 3600)}h`;
}

export function formatDurationLong(s: number | null | undefined): string {
  if (s == null || !Number.isFinite(s)) return 'unavailable';
  if (s < 1) return `${(s * 1000).toFixed(0)} milliseconds`;
  if (s < 60) return `${s.toFixed(2)} seconds`;
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  if (s < 3600) return `${m}m ${sec}s`;
  const h = Math.floor(s / 3600);
  const mm = Math.floor((s % 3600) / 60);
  if (s < 86400) return `${h}h ${mm}m`;
  const d = Math.floor(s / 86400);
  const hh = Math.floor((s % 86400) / 3600);
  return `${d}d ${hh}h`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Byte formatting
// ─────────────────────────────────────────────────────────────────────────────

export function formatBytes(bytes: number | null | undefined): string {
  if (bytes == null || !Number.isFinite(bytes)) return '—';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Number formatting
// ─────────────────────────────────────────────────────────────────────────────

export function formatInt(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return '—';
  return Math.round(n).toLocaleString('en-US');
}

export function formatFloat(n: number | null | undefined, digits = 2): string {
  if (n == null || !Number.isFinite(n)) return '—';
  return n.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export function formatPercent(n: number | null | undefined, digits = 1): string {
  if (n == null || !Number.isFinite(n)) return '—';
  return `${(n * 100).toFixed(digits)}%`;
}

export function formatRatio(n: number | null | undefined, digits = 3): string {
  if (n == null || !Number.isFinite(n)) return '—';
  return n.toFixed(digits);
}

// ─────────────────────────────────────────────────────────────────────────────
// Node id helpers — strip canonical "type:" prefix for compact display
// ─────────────────────────────────────────────────────────────────────────────

export function shortNodeId(nodeId: string): string {
  const idx = nodeId.indexOf(':');
  return idx >= 0 ? nodeId.slice(idx + 1) : nodeId;
}

export function nodeTypeOf(nodeId: string): NodeType | null {
  const idx = nodeId.indexOf(':');
  if (idx < 0) return null;
  const prefix = nodeId.slice(0, idx).toUpperCase();
  // Map canonical prefixes back to NodeType enum (note UNKNOWN→"entity")
  const map: Record<string, NodeType> = {
    USER: 'USER', HOST: 'HOST', PROCESS: 'PROCESS', FILE: 'FILE',
    IP: 'IP', DOMAIN: 'DOMAIN', SOCKET: 'SOCKET', ENTITY: 'UNKNOWN',
  };
  return map[prefix] ?? null;
}

// ─────────────────────────────────────────────────────────────────────────────
// Label presentation
// ─────────────────────────────────────────────────────────────────────────────

export type LabelBadge = 'success' | 'danger' | 'gray';

export function labelBadge(label: EventLabel): LabelBadge {
  return label === 1 ? 'danger' : 'success';
}

export function labelText(label: EventLabel): string {
  return label === 1 ? 'Malicious' : 'Benign';
}

// ─────────────────────────────────────────────────────────────────────────────
// NodeType presentation
// ─────────────────────────────────────────────────────────────────────────────

export function nodeTypeLabel(t: NodeType): string {
  return NODE_TYPE_META[t].label;
}

export function nodeTypeColor(t: NodeType): string {
  return NODE_TYPE_META[t].color;
}

export function nodeTypeDescription(t: NodeType): string {
  return NODE_TYPE_META[t].description;
}

// Strip "type:" prefix — preserve raw value but match the backend canonical id.
export function nodeDisplayName(nodeId: string): { type: NodeType | null; value: string } {
  const t = nodeTypeOf(nodeId);
  return { type: t, value: shortNodeId(nodeId) };
}

// ─────────────────────────────────────────────────────────────────────────────
// Relation presentation
// ─────────────────────────────────────────────────────────────────────────────

export function relationLabel(r: RelationType | string): string {
  if (r in RELATION_TYPE_META) {
    return RELATION_TYPE_META[r as RelationType].label;
  }
  return r.replace(/_/g, ' ').toLowerCase().replace(/^\w/, (c) => c.toUpperCase());
}

export function relationCategory(r: RelationType | string) {
  if (r in RELATION_TYPE_META) {
    return RELATION_TYPE_META[r as RelationType].category;
  }
  return 'access' as const;
}

export function relationIsAttack(r: RelationType | string): boolean {
  if (r in RELATION_TYPE_META) {
    return RELATION_TYPE_META[r as RelationType].is_attack;
  }
  return false;
}

// ─────────────────────────────────────────────────────────────────────────────
// Strategy presentation
// ─────────────────────────────────────────────────────────────────────────────

export function strategyLabel(s: ChainStrategy): string {
  return s;
}

export function strategyPriority(s: ChainStrategy): number {
  const order: ChainStrategy[] = ['chain_id', 'causal_parent', 'entity_time'];
  return order.indexOf(s) + 1;
}

// ─────────────────────────────────────────────────────────────────────────────
// Processing state presentation
// ─────────────────────────────────────────────────────────────────────────────

export function processingStateLabel(s: ProcessingState): string {
  return PROCESSING_STATE_META[s].label;
}

export function processingStateBadge(s: ProcessingState) {
  return PROCESSING_STATE_META[s].badge;
}

export function processingStateDescription(s: ProcessingState): string {
  return PROCESSING_STATE_META[s].description;
}

// ─────────────────────────────────────────────────────────────────────────────
// Chain summary → derived severity (UI-only interpretation)
// ─────────────────────────────────────────────────────────────────────────────

export type ChainSeverity = 'low' | 'medium' | 'high' | 'critical';

export function chainSeverity(chain: AttackChainSummary): ChainSeverity {
  // UI-derived only. Backend never stores a severity score on chains.
  // Combines chain length + presence of attack relations + tactics count.
  const hasExfil = chain.tactic_sequence.includes('Exfiltration');
  const hasLateral = chain.tactic_sequence.includes('Lateral_Movement');
  const hasCred = chain.tactic_sequence.includes('Credential_Access');
  const attackCount = [hasExfil, hasLateral, hasCred].filter(Boolean).length;
  if (chain.num_events >= 9 && attackCount >= 2) return 'critical';
  if (chain.num_events >= 7 || attackCount >= 2) return 'high';
  if (chain.num_events >= 4 || attackCount >= 1) return 'medium';
  return 'low';
}

// ─────────────────────────────────────────────────────────────────────────────
// Node severity (UI-derived, mirrors visualize_severity_graph.py formula)
// ─────────────────────────────────────────────────────────────────────────────

export function nodeSeverityScore(node: GraphNode): number {
  // 0.55 * mal_ratio + 0.30 * log1p(malicious_events) + 0.15 * log1p(degree)
  const total = (node.in_degree ?? 0) + (node.out_degree ?? 0);
  if (total === 0) return 0;
  const mal = node.malicious_events ?? 0;
  const mal_ratio = mal / total;
  return Math.min(
    1,
    0.55 * mal_ratio + 0.30 * Math.log1p(mal) + 0.15 * Math.log1p(total)
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Event short display (for table rows)
// ─────────────────────────────────────────────────────────────────────────────

export function eventShortId(eventId: string): string {
  const parts = eventId.split('::');
  if (parts.length === 2) return parts[1];
  return eventId.length > 12 ? eventId.slice(0, 12) + '…' : eventId;
}

export function eventTacticsString(e: TGEvent): string {
  return e.tactics.join(', ');
}

export function eventAttrsString(e: TGEvent): string {
  const entries = Object.entries(e.attrs);
  if (entries.length === 0) return '';
  return entries
    .map(([k, v]) => `${k}=${typeof v === 'string' ? v : JSON.stringify(v)}`)
    .join(', ');
}

// ─────────────────────────────────────────────────────────────────────────────
// Confusion matrix helpers
// ─────────────────────────────────────────────────────────────────────────────

export interface ConfusionMatrixView {
  tn: number; fp: number; fn: number; tp: number;
  total: number;
  tpr: number; // recall / sensitivity
  fpr: number;
  tnr: number; // specificity
  ppv: number; // precision
  npv: number;
  fnr: number;
  accuracy: number;
}

export function confusionMatrixView(m: { tn: number; fp: number; fn: number; tp: number } | number[][] | unknown): ConfusionMatrixView {
  let tn = 0;
  let fp = 0;
  let fn = 0;
  let tp = 0;
  if (Array.isArray(m) && m.length >= 2) {
    tn = Number(m[0]?.[0] ?? 0);
    fp = Number(m[0]?.[1] ?? 0);
    fn = Number(m[1]?.[0] ?? 0);
    tp = Number(m[1]?.[1] ?? 0);
  } else if (m && typeof m === 'object') {
    const obj = m as Record<string, unknown>;
    tn = Number(obj.tn ?? 0);
    fp = Number(obj.fp ?? 0);
    fn = Number(obj.fn ?? 0);
    tp = Number(obj.tp ?? 0);
  }
  const total = tn + fp + fn + tp;
  const tpr = tp + fn > 0 ? tp / (tp + fn) : 0;
  const fpr = fp + tn > 0 ? fp / (fp + tn) : 0;
  const tnr = fp + tn > 0 ? tn / (fp + tn) : 0;
  const ppv = tp + fp > 0 ? tp / (tp + fp) : 0;
  const npv = tn + fn > 0 ? tn / (tn + fn) : 0;
  const fnr = 1 - tpr;
  const accuracy = total > 0 ? (tp + tn) / total : 0;
  return { tn, fp, fn, tp, total, tpr, fpr, tnr, ppv, npv, fnr, accuracy };
}

// ─────────────────────────────────────────────────────────────────────────────
// Histogram bucket formatter
// ─────────────────────────────────────────────────────────────────────────────

export function bucketize(values: (number | null | undefined)[], bucketCount: number, min?: number, max?: number): { x: string; count: number }[] {
  const valid = values.filter((v): v is number => v != null && !Number.isNaN(v));
  if (valid.length === 0) return [];
  const lo = min ?? Math.min(...valid);
  const hi = max ?? Math.max(...valid);
  if (hi <= lo) return [{ x: String(lo.toFixed(0)), count: valid.length }];
  const step = (hi - lo) / bucketCount;
  const buckets = Array.from({ length: bucketCount }, (_, i) => ({
    x: `${(lo + i * step).toFixed(0)}–${(lo + (i + 1) * step).toFixed(0)}`,
    count: 0,
  }));
  for (const v of valid) {
    let idx = Math.floor((v - lo) / step);
    if (idx >= bucketCount) idx = bucketCount - 1;
    if (idx < 0) idx = 0;
    buckets[idx].count += 1;
  }
  return buckets;
}

export function formatHistogramKey(key: string): string {
  // The backend emits histogram keys like "1", "2-3", "4-5", "6-10", "11-20", "20+".
  // Pass through as-is — these are already human-readable.
  return key;
}
