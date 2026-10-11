/**
 * Attack-reconstruction engine — shared types.
 *
 * Implements an adaptation of ORTHRUS (Jiang et al., USENIX Security '25)
 * §4.5 "Attack Reconstruction" for the TGDetect bipartite process↔host
 * provenance graph (relation OBSERVED_ON). Starting from a flagged node we
 * trace backward (precursors / entry) and forward (successors / exit) within
 * an adjustable time window, assemble a *concise* attack-summary subgraph, and
 * score each node (degree fD, anomaly fA) and edge (criticality fC) so the UI
 * can rank, size, and explain the result instead of dumping the whole graph.
 */

import type { EventLabel, NodeType } from '../types';

/** Structural/temporal role of a node inside an attack summary. */
export type NodeRole = 'focal' | 'entry' | 'exit' | 'pivot' | 'related';

/** Where an edge sits relative to the focal detection time. */
export type TraceDirection = 'backward' | 'forward' | 'focal';

/** Which side of the trace the user wants emphasised. */
export type TraceMode = 'both' | 'backward' | 'forward';

export interface SummaryNode {
  id: string;
  type: NodeType;
  role: NodeRole;
  /** Causal/temporal column: negative = backward/entry side, 0 = focal, positive = forward/exit side. */
  layer: number;
  /** Hop distance from the focal node (0 focal, 1 host, 2 co-located process). */
  hop: number;
  firstTs: number;
  lastTs: number;
  /** Degrees measured *within* the extracted subgraph. */
  inDeg: number;
  outDeg: number;
  incidentEvents: number;
  maliciousEvents: number;
  /** Events the model flagged (prediction === 1) within the subgraph. */
  flaggedEvents: number;
  /** fA(u) ∈ [0,1] — mean anomaly of incident edges (model probability, label fallback). */
  anomaly: number;
  /** fD(u) ∈ [0,1] — normalised (outDeg+1)/(inDeg+1) structural score. */
  degreeScore: number;
  /** fC(u) ∈ [0,1] — blended criticality used for ranking / sizing. */
  criticality: number;
  /** Layout coordinates in virtual space (filled by layout.ts). */
  x: number;
  y: number;
}
export interface SummaryEdge {
  /** Backend event_id. */
  id: string;
  src: string;
  dst: string;
  relation: string;
  ts: number;
  label: EventLabel;
  /** Model edge-anomaly probability, or null when the dataset carries no scores. */
  probability: number | null;
  prediction: 0 | 1 | null;
  threshold: number | null;
  /** fC(e) ∈ [0,1] — ORTHRUS criticality (normalised degree + anomaly). */
  criticality: number;
  direction: TraceDirection;
}

export interface AttackSummaryStatsBlock {
  /** Incident events the backend reports for the focal node (before any cap). */
  totalIncident: number;
  renderedEvents: number;
  renderedNodes: number;
  maliciousEvents: number;
  flaggedEvents: number;
  hasModelScores: boolean;
  windowStart: number;
  windowEnd: number;
  /** Center of the trace window = focal's first flagged/malicious activity. */
  center: number;
  anomalyMean: number;
  anomalyMax: number;
  hostsTouched: number;
  colocatedProcesses: number;
}

export interface AttackSummary {
  focalId: string;
  focalType: NodeType;
  /** Half-window (seconds) applied each side of the center. */
  windowSec: number;
  nodes: SummaryNode[];
  edges: SummaryEdge[];
  /** Sorted unique layer indices present (for column layout). */
  layers: number[];
  entryIds: string[];
  exitIds: string[];
  stats: AttackSummaryStatsBlock;
  /** Plain-English reconstruction, one string per paragraph. */
  narrative: string[];
  /** True when the subgraph was capped by maxNodes / host limits. */
  truncated: boolean;
}

export interface ExtractOptions {
  /** Half-window seconds each side of the detection center. */
  windowSec: number;
  /** Hard cap on rendered nodes to keep the summary concise. */
  maxNodes: number;
  /** Max pivot hosts expanded from the focal node. */
  maxHosts: number;
}

/** A ranked triage candidate (§4.4 suspicious-node isolation). */
export interface Suspect {
  id: string;
  type: NodeType;
  maliciousEvents: number;
  incidentEvents: number;
  degree: number;
  firstTs: number;
  lastTs: number;
  /** Ground-truth incidence severity ∈ [0,1]. */
  severity: number;
  /** Mean model anomaly across sampled incident events, or null if unsampled. */
  anomaly: number | null;
  /** Blended triage priority ∈ [0,1] used for ordering. */
  priority: number;
}

