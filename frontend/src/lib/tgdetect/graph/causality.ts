/**
 * Attack-reconstruction engine (ORTHRUS §4.5, adapted for bipartite
 * process↔host OBSERVED_ON telemetry).
 *
 * Pure data layer — no React. Given a flagged focal node it fetches the
 * node's incident events (which carry the model's per-edge anomaly
 * probability), expands one hop through shared hosts, applies an adjustable
 * time window, and assembles a concise, scored attack-summary subgraph.
 *
 * Honesty note: the only relation in this dataset is OBSERVED_ON (a process
 * observed on a host). We never fabricate causal relations — "backward" and
 * "forward" tracing is inferred from temporal ordering of co-located activity,
 * exactly like the backend's entity_time chain strategy, and the UI says so.
 */

import { api } from '../api/client';
import { formatDurationLong, formatEpochTime, nodeTypeOf, shortNodeId } from '../formatters';
import type { EventLabel, GraphNode, NodeType, TGEvent } from '../types';
import type {
  AttackSummary,
  AttackSummaryStatsBlock,
  ExtractOptions,
  SummaryEdge,
  SummaryNode,
  Suspect,
  TraceDirection,
} from './types';

const clamp01 = (n: number): number => (n < 0 ? 0 : n > 1 ? 1 : n);

/** Resolve a node's type from its canonical `type:value` id, with a fallback. */
function typeOf(id: string, fallback: NodeType = 'UNKNOWN'): NodeType {
  return nodeTypeOf(id) ?? fallback;
}

/** Narrow attrs.probability / prediction / threshold without trusting shape. */
function readScore(e: TGEvent): { probability: number | null; prediction: 0 | 1 | null; threshold: number | null } {
  const a = (e.attrs ?? {}) as Record<string, unknown>;
  const probability = typeof a.probability === 'number' && Number.isFinite(a.probability) ? a.probability : null;
  const prediction = a.prediction === 1 || a.prediction === 0 ? (a.prediction as 0 | 1) : null;
  const threshold = typeof a.threshold === 'number' && Number.isFinite(a.threshold) ? a.threshold : null;
  return { probability, prediction, threshold };
}

/** Edge-level anomaly: model probability if present, else a label-derived proxy. */
function edgeAnomaly(probability: number | null, label: EventLabel): number {
  if (probability != null) return clamp01(probability);
  return label === 1 ? 0.9 : 0.1;
}

interface FetchResult {
  items: TGEvent[];
  total: number;
}

/** Fetch events incident to a node (src OR dst) within an optional window. */
async function fetchNodeEvents(
  nodeId: string,
  opts: { limit: number; startTs?: number; endTs?: number },
): Promise<FetchResult> {
  const params: Record<string, string | number> = { node_id: nodeId, limit: opts.limit };
  if (opts.startTs != null) params.start_ts = opts.startTs;
  if (opts.endTs != null) params.end_ts = opts.endTs;
  try {
    const res = await api.get<FetchResult>('/api/events', { params });
    return { items: Array.isArray(res?.items) ? res.items : [], total: res?.total ?? 0 };
  } catch {
    return { items: [], total: 0 };
  }
}
// ─────────────────────────────────────────────────────────────────────────────
// Triage (§4.4) — rank the flagged nodes so the analyst starts somewhere useful
// ─────────────────────────────────────────────────────────────────────────────

export interface AnomalyAgg {
  sum: number;
  count: number;
  max: number;
  flagged: number;
}

/** Aggregate model anomaly per endpoint from a sample of (scored) events. */
export function aggregateAnomaly(events: TGEvent[]): Map<string, AnomalyAgg> {
  const map = new Map<string, AnomalyAgg>();
  for (const e of events) {
    const { probability, prediction } = readScore(e);
    if (probability == null) continue;
    for (const id of [e.src_id, e.dst_id]) {
      const a = map.get(id) ?? { sum: 0, count: 0, max: 0, flagged: 0 };
      a.sum += probability;
      a.count += 1;
      a.max = Math.max(a.max, probability);
      a.flagged += prediction === 1 ? 1 : 0;
      map.set(id, a);
    }
  }
  return map;
}

/** Build a ranked triage list from flagged nodes + an optional anomaly sample. */
export function buildSuspects(nodes: GraphNode[], anomalyByNode?: Map<string, AnomalyAgg>): Suspect[] {
  const maxMal = Math.max(1, ...nodes.map((n) => n.malicious_events ?? 0));
  const suspects: Suspect[] = nodes.map((n) => {
    const degree = (n.in_degree ?? 0) + (n.out_degree ?? 0);
    const mal = n.malicious_events ?? 0;
    const malRatio = degree > 0 ? mal / degree : 0;
    const severity = clamp01(0.6 * malRatio + 0.4 * (Math.log1p(mal) / Math.log1p(maxMal)));
    const agg = anomalyByNode?.get(n.node_id);
    const anomaly = agg && agg.count > 0 ? clamp01(agg.sum / agg.count) : null;
    const priority = anomaly != null ? clamp01(0.6 * anomaly + 0.4 * severity) : severity;
    return {
      id: n.node_id,
      type: n.node_type,
      maliciousEvents: mal,
      incidentEvents: degree,
      degree,
      firstTs: n.first_seen_ts,
      lastTs: n.last_seen_ts,
      severity,
      anomaly,
      priority,
    };
  });
  suspects.sort((a, b) => b.priority - a.priority || b.maliciousEvents - a.maliciousEvents || b.degree - a.degree);
  return suspects;
}

// ─────────────────────────────────────────────────────────────────────────────
// Attack reconstruction (§4.5) — backward/forward tracing + concise summary
// ─────────────────────────────────────────────────────────────────────────────

/** Mutable per-node accumulator used while assembling the subgraph. */
interface NodeAcc {
  id: string;
  type: NodeType;
  hop: number;
  firstTs: number;
  lastTs: number;
  inDeg: number;
  outDeg: number;
  incident: number;
  mal: number;
  flagged: number;
  anomSum: number;
  anomCount: number;
}

/** A fetched, scored event normalised for subgraph assembly. */
interface RawEdge {
  id: string;
  src: string;
  dst: string;
  relation: string;
  ts: number;
  label: EventLabel;
  probability: number | null;
  prediction: 0 | 1 | null;
  threshold: number | null;
  anomaly: number;
}

const dirOf = (ts: number, center: number): TraceDirection =>
  ts < center ? 'backward' : ts > center ? 'forward' : 'focal';

/** Normalise a raw TGEvent into a scored RawEdge (null when it has no id). */
function toRawEdge(e: TGEvent): RawEdge | null {
  if (!e.event_id || !e.src_id || !e.dst_id) return null;
  const { probability, prediction, threshold } = readScore(e);
  return {
    id: e.event_id,
    src: e.src_id,
    dst: e.dst_id,
    relation: e.relation || 'OBSERVED_ON',
    ts: e.ts ?? 0,
    label: (e.label === 1 ? 1 : 0) as EventLabel,
    probability,
    prediction,
    threshold,
    anomaly: edgeAnomaly(probability, (e.label === 1 ? 1 : 0) as EventLabel),
  };
}
/**
 * Reconstruct a concise attack-summary subgraph around a flagged focal node.
 *
 * Anchor fetch gives the focal node's real, model-scored incident edges; we
 * expand exactly one hop through its most anomalous shared hosts, keep only
 * activity inside the time window, score every node/edge, and cap the result to
 * `maxNodes` so the analyst sees a readable summary — never the whole graph.
 */
export async function extractAttackSummary(
  focalId: string,
  focalType: NodeType,
  opts: ExtractOptions,
): Promise<AttackSummary> {
  const windowSec = Math.max(0, opts.windowSec);
  const maxNodes = Math.max(2, opts.maxNodes);
  const maxHosts = Math.max(1, opts.maxHosts);

  // 1 ── Focal incident edges (real, model-scored). This is the anchor fetch.
  const focalFetch = await fetchNodeEvents(focalId, { limit: 500 });
  const focalRaw = focalFetch.items.map(toRawEdge).filter((e): e is RawEdge => e != null);

  // 2 ── Detection center = focal's first flagged / malicious activity.
  const flaggedTs = focalRaw.filter((e) => e.prediction === 1).map((e) => e.ts);
  const malTs = focalRaw.filter((e) => e.label === 1).map((e) => e.ts);
  const allTs = focalRaw.map((e) => e.ts);
  const center =
    flaggedTs.length > 0 ? Math.min(...flaggedTs)
    : malTs.length > 0 ? Math.min(...malTs)
    : allTs.length > 0 ? Math.min(...allTs)
    : Math.floor(Date.now() / 1000);
  const windowStart = center - windowSec;
  const windowEnd = center + windowSec;
  const inWindow = (ts: number): boolean => ts >= windowStart && ts <= windowEnd;

  // 3 ── Focal edges within the window (fall back to all if the window empties).
  let focalWin = focalRaw.filter((e) => inWindow(e.ts));
  if (focalWin.length === 0) focalWin = focalRaw;

  // 4 ── Hop-1 neighbours (pivots) = the "other" endpoint of each focal edge.
  const neighborAgg = new Map<string, { count: number; anom: number }>();
  for (const e of focalWin) {
    const other = e.src === focalId ? e.dst : e.src;
    if (!other || other === focalId) continue;
    const n = neighborAgg.get(other) ?? { count: 0, anom: 0 };
    n.count += 1;
    n.anom = Math.max(n.anom, e.anomaly);
    neighborAgg.set(other, n);
  }
  const neighbors = [...neighborAgg.entries()]
    .sort((a, b) => b[1].anom - a[1].anom || b[1].count - a[1].count)
    .slice(0, maxHosts)
    .map(([id]) => id);
  const neighborSet = new Set(neighbors);

  // 5 ── Expand one hop through each pivot (brings in co-located hop-2 nodes).
  const perHostLimit = Math.max(60, Math.ceil((maxNodes * 4) / Math.max(1, neighbors.length)));
  const hopFetches = await Promise.all(
    neighbors.map((h) => fetchNodeEvents(h, { limit: perHostLimit, startTs: windowStart, endTs: windowEnd })),
  );

  // 6 ── Collect unique edges (dedup by event_id) that touch focal or a pivot.
  const edgeMap = new Map<string, RawEdge>();
  for (const e of focalWin) edgeMap.set(e.id, e);
  for (const fr of hopFetches) {
    for (const ev of fr.items) {
      const e = toRawEdge(ev);
      if (!e || !inWindow(e.ts)) continue;
      if (e.src === focalId || e.dst === focalId || neighborSet.has(e.src) || neighborSet.has(e.dst)) {
        edgeMap.set(e.id, e);
      }
    }
  }
  // 7 ── Build node accumulators + hop assignment from the collected edges.
  const acc = new Map<string, NodeAcc>();
  const touch = (id: string, hop: number): NodeAcc => {
    const existing = acc.get(id);
    if (existing) {
      if (hop < existing.hop) existing.hop = hop;
      return existing;
    }
    const a: NodeAcc = {
      id,
      type: id === focalId ? focalType : typeOf(id),
      hop,
      firstTs: Infinity,
      lastTs: -Infinity,
      inDeg: 0,
      outDeg: 0,
      incident: 0,
      mal: 0,
      flagged: 0,
      anomSum: 0,
      anomCount: 0,
    };
    acc.set(id, a);
    return a;
  };
  const hopOf = (id: string): number => (id === focalId ? 0 : neighborSet.has(id) ? 1 : 2);
  const absorb = (node: NodeAcc, e: RawEdge): void => {
    node.incident += 1;
    node.mal += e.label === 1 ? 1 : 0;
    node.flagged += e.prediction === 1 ? 1 : 0;
    if (e.probability != null) {
      node.anomSum += e.probability;
      node.anomCount += 1;
    }
    node.firstTs = Math.min(node.firstTs, e.ts);
    node.lastTs = Math.max(node.lastTs, e.ts);
  };
  const edges = [...edgeMap.values()];
  for (const e of edges) {
    const s = touch(e.src, hopOf(e.src));
    const d = touch(e.dst, hopOf(e.dst));
    s.outDeg += 1;
    d.inDeg += 1;
    absorb(s, e);
    absorb(d, e);
  }
  if (!acc.has(focalId)) touch(focalId, 0); // focal is always present

  // 8 ── Provisional criticality (anomaly + raw degree) just for the node cap.
  const provMaxDeg = Math.max(1, ...[...acc.values()].map((a) => a.inDeg + a.outDeg));
  const provScore = (a: NodeAcc): number => {
    const anomaly = a.anomCount > 0 ? a.anomSum / a.anomCount : a.mal > 0 ? 0.9 : 0.1;
    const deg = clamp01(Math.log1p(a.inDeg + a.outDeg) / Math.log1p(provMaxDeg));
    return clamp01(0.6 * anomaly + 0.4 * deg);
  };

  // 9 ── Cap: always keep focal, then the top-(maxNodes-1) by provisional score.
  const ranked = [...acc.values()]
    .filter((a) => a.id !== focalId)
    .sort((a, b) => provScore(b) - provScore(a));
  const keptIds = new Set<string>([focalId]);
  for (const a of ranked) {
    if (keptIds.size >= maxNodes) break;
    keptIds.add(a.id);
  }
  const nodeTruncated = ranked.length + 1 > keptIds.size;

  // 10 ── Keep only edges whose endpoints both survived; recompute degrees.
  for (const a of acc.values()) {
    a.inDeg = 0; a.outDeg = 0; a.incident = 0; a.mal = 0; a.flagged = 0;
    a.anomSum = 0; a.anomCount = 0; a.firstTs = Infinity; a.lastTs = -Infinity;
  }
  const keptEdges = edges.filter((e) => keptIds.has(e.src) && keptIds.has(e.dst));
  for (const e of keptEdges) {
    const s = acc.get(e.src)!;
    const d = acc.get(e.dst)!;
    s.outDeg += 1;
    d.inDeg += 1;
    absorb(s, e);
    absorb(d, e);
  }
  // 11 ── Finalise node scores + temporal layers (−hop … 0 … +hop columns).
  const kept = [...keptIds].map((id) => acc.get(id)!).filter(Boolean);
  const maxDeg = Math.max(1, ...kept.map((a) => a.inDeg + a.outDeg));
  const layerOf = (a: NodeAcc): number => {
    if (a.id === focalId) return 0;
    const t = Number.isFinite(a.firstTs) ? a.firstTs : center;
    const sign = t < center ? -1 : t > center ? 1 : 1; // ties lean forward
    return sign * a.hop;
  };
  const layered = kept.map((a) => ({ a, layer: layerOf(a) }));
  const layerVals = layered.map((l) => l.layer);
  const minLayer = Math.min(0, ...layerVals);
  const maxLayer = Math.max(0, ...layerVals);

  const summaryNodes: SummaryNode[] = layered.map(({ a, layer }) => {
    const anomaly = a.anomCount > 0 ? clamp01(a.anomSum / a.anomCount) : a.mal > 0 ? 0.9 : 0.1;
    const degreeScore = clamp01(Math.log1p(a.inDeg + a.outDeg) / Math.log1p(maxDeg));
    const criticality = clamp01(0.6 * anomaly + 0.4 * degreeScore);
    let role: SummaryNode['role'];
    if (a.id === focalId) role = 'focal';
    else if (a.hop === 1) role = 'pivot';
    else if (layer === minLayer && minLayer < 0) role = 'entry';
    else if (layer === maxLayer && maxLayer > 0) role = 'exit';
    else role = 'related';
    return {
      id: a.id,
      type: a.type,
      role,
      layer,
      hop: a.hop,
      firstTs: Number.isFinite(a.firstTs) ? a.firstTs : center,
      lastTs: Number.isFinite(a.lastTs) ? a.lastTs : center,
      inDeg: a.inDeg,
      outDeg: a.outDeg,
      incidentEvents: a.incident,
      maliciousEvents: a.mal,
      flaggedEvents: a.flagged,
      anomaly,
      degreeScore,
      criticality,
      x: 0,
      y: 0,
    };
  });
  const critById = new Map(summaryNodes.map((n) => [n.id, n.criticality]));
  // 12 ── Finalise edges with ORTHRUS criticality fC(e) = anomaly + endpoint load.
  const summaryEdges: SummaryEdge[] = keptEdges
    .map((e) => {
      const cs = critById.get(e.src) ?? 0;
      const cd = critById.get(e.dst) ?? 0;
      return {
        id: e.id,
        src: e.src,
        dst: e.dst,
        relation: e.relation,
        ts: e.ts,
        label: e.label,
        probability: e.probability,
        prediction: e.prediction,
        threshold: e.threshold,
        criticality: clamp01(0.5 * e.anomaly + 0.25 * cs + 0.25 * cd),
        direction: dirOf(e.ts, center),
      };
    })
    .sort((a, b) => a.ts - b.ts);

  // 13 ── Entry / exit nodes + the sorted layer axis for column layout.
  const entryIds = summaryNodes.filter((n) => n.role === 'entry').map((n) => n.id);
  const exitIds = summaryNodes.filter((n) => n.role === 'exit').map((n) => n.id);
  const layers = [...new Set(summaryNodes.map((n) => n.layer))].sort((a, b) => a - b);

  // 14 ── Stats block.
  const hasModelScores = summaryEdges.some((e) => e.probability != null);
  const anomVals = summaryEdges.map((e) => e.probability ?? edgeAnomaly(null, e.label));
  const anomalyMean = anomVals.length > 0 ? anomVals.reduce((s, v) => s + v, 0) / anomVals.length : 0;
  const anomalyMax = anomVals.length > 0 ? Math.max(...anomVals) : 0;
  const stats: AttackSummaryStatsBlock = {
    totalIncident: focalFetch.total,
    renderedEvents: summaryEdges.length,
    renderedNodes: summaryNodes.length,
    maliciousEvents: summaryEdges.filter((e) => e.label === 1).length,
    flaggedEvents: summaryEdges.filter((e) => e.prediction === 1).length,
    hasModelScores,
    windowStart,
    windowEnd,
    center,
    anomalyMean: clamp01(anomalyMean),
    anomalyMax: clamp01(anomalyMax),
    hostsTouched: summaryNodes.filter((n) => n.hop === 1).length,
    colocatedProcesses: summaryNodes.filter((n) => n.hop === 2).length,
  };

  const truncated =
    nodeTruncated || neighborAgg.size > neighbors.length || focalFetch.total > focalRaw.length;
  const narrative = buildNarrative(focalId, focalType, summaryNodes, stats, entryIds, exitIds);

  return {
    focalId,
    focalType,
    windowSec,
    nodes: summaryNodes,
    edges: summaryEdges,
    layers,
    entryIds,
    exitIds,
    stats,
    narrative,
    truncated,
  };
}
/** Plain-English reconstruction of the summary, one string per paragraph. */
function buildNarrative(
  focalId: string,
  focalType: NodeType,
  nodes: SummaryNode[],
  stats: AttackSummaryStatsBlock,
  entryIds: string[],
  exitIds: string[],
): string[] {
  const out: string[] = [];
  const focal = shortNodeId(focalId);
  const pct = (n: number): string => `${Math.round(n * 100)}%`;

  const verdict = stats.hasModelScores
    ? `The model scored ${stats.renderedEvents} incident event${stats.renderedEvents === 1 ? '' : 's'} around ${focal}, flagging ${stats.flaggedEvents} as anomalous (mean anomaly ${pct(stats.anomalyMean)}, peak ${pct(stats.anomalyMax)}).`
    : `This dataset carries no model scores, so ranking falls back to ground-truth labels: ${stats.maliciousEvents} of ${stats.renderedEvents} rendered events are labelled malicious.`;
  out.push(`${focalType} ${focal} is the focal detection. ${verdict}`);

  const half = Math.max(0, Math.round((stats.windowEnd - stats.windowStart) / 2));
  out.push(
    `Reconstruction centres on ${formatEpochTime(stats.center)} — ${focal}'s first ${stats.hasModelScores ? 'flagged' : 'malicious'} activity — and traces ±${formatDurationLong(half)} of co-located telemetry across ${stats.hostsTouched} host${stats.hostsTouched === 1 ? '' : 's'} and ${stats.colocatedProcesses} neighbouring process${stats.colocatedProcesses === 1 ? '' : 'es'}.`,
  );

  if (entryIds.length > 0) {
    const names = entryIds.slice(0, 3).map(shortNodeId).join(', ');
    const more = entryIds.length > 3 ? ` +${entryIds.length - 3} more` : '';
    out.push(`Backward tracing surfaces likely precursors (entry side): ${names}${more}. These acted before the focal detection on a shared host.`);
  }
  if (exitIds.length > 0) {
    const names = exitIds.slice(0, 3).map(shortNodeId).join(', ');
    const more = exitIds.length > 3 ? ` +${exitIds.length - 3} more` : '';
    out.push(`Forward tracing surfaces likely successors (exit side): ${names}${more}. These acted after the focal detection on a shared host.`);
  }

  const topPivot = nodes
    .filter((n) => n.role === 'pivot')
    .sort((a, b) => b.criticality - a.criticality)[0];
  if (topPivot) {
    out.push(`The busiest pivot is host ${shortNodeId(topPivot.id)} (${topPivot.incidentEvents} incident events, criticality ${pct(topPivot.criticality)}) — the shared vantage point linking these processes.`);
  }

  out.push(
    `Causality is inferred from the temporal ordering of activity co-located on shared hosts (the OBSERVED_ON relation), mirroring the backend's entity-time chaining — it is not a captured parent/child edge.`,
  );
  return out;
}

