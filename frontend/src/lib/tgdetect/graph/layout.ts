/**
 * Layered-DAG layout for the attack-summary subgraph.
 *
 * Pure geometry — no React, no canvas. Nodes are placed into vertical columns
 * keyed by their temporal `layer` (negative = backward/entry side, 0 = focal,
 * positive = forward/exit side) and stacked by criticality within each column.
 * The result is a stable set of virtual coordinates the canvas renderer maps to
 * screen space; because it is deterministic there is no force simulation and no
 * per-frame layout cost (fixing the O(n²) spring sim in the old viewer).
 */

import type { AttackSummary, SummaryNode } from './types';

export interface LayoutConfig {
  /** Horizontal distance between adjacent temporal layers. */
  columnGap: number;
  /** Vertical distance between stacked nodes in a column. */
  rowGap: number;
  /** Smallest node radius (criticality 0). */
  minRadius: number;
  /** Largest node radius (criticality 1). */
  maxRadius: number;
}

export const DEFAULT_LAYOUT: LayoutConfig = {
  columnGap: 240,
  rowGap: 96,
  minRadius: 7,
  maxRadius: 22,
};

export const clamp01 = (n: number): number => (n < 0 ? 0 : n > 1 ? 1 : n);

/** Linear interpolation. */
export const lerp = (a: number, b: number, t: number): number => a + (b - a) * t;

/** Cubic ease-in-out for the layout morph animation. */
export const easeInOutCubic = (t: number): number =>
  t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

/** Node radius scales with criticality fC(u). */
export function nodeRadius(criticality: number, cfg: LayoutConfig = DEFAULT_LAYOUT): number {
  return cfg.minRadius + clamp01(criticality) * (cfg.maxRadius - cfg.minRadius);
}

export interface LayoutBounds {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
  width: number;
  height: number;
}

export interface PositionedSummary {
  nodes: SummaryNode[];
  bounds: LayoutBounds;
  /** Virtual x of each occupied layer (for drawing column guides / labels). */
  columns: { layer: number; x: number }[];
}
/**
 * Position every node in virtual space. Columns are ordered by layer; within a
 * column, nodes are stacked by criticality (most critical nearest the centre
 * line) so the eye lands on the important nodes first.
 */
export function layoutSummary(summary: AttackSummary, cfg: LayoutConfig = DEFAULT_LAYOUT): PositionedSummary {
  const byLayer = new Map<number, SummaryNode[]>();
  for (const n of summary.nodes) {
    const arr = byLayer.get(n.layer);
    if (arr) arr.push(n);
    else byLayer.set(n.layer, [n]);
  }

  const positioned: SummaryNode[] = [];
  const columns: { layer: number; x: number }[] = [];
  const sortedLayers = [...byLayer.keys()].sort((a, b) => a - b);

  for (const layer of sortedLayers) {
    const arr = byLayer.get(layer)!;
    // Stable ordering: criticality desc, then id for determinism.
    arr.sort((a, b) => b.criticality - a.criticality || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
    // Re-interleave so the most critical sits at the vertical centre.
    const ordered: SummaryNode[] = [];
    arr.forEach((node, i) => (i % 2 === 0 ? ordered.push(node) : ordered.unshift(node)));

    const x = layer * cfg.columnGap;
    columns.push({ layer, x });
    const count = ordered.length;
    ordered.forEach((node, i) => {
      const y = (i - (count - 1) / 2) * cfg.rowGap;
      positioned.push({ ...node, x, y });
    });
  }

  const pad = cfg.maxRadius + 24;
  const xs = positioned.map((p) => p.x);
  const ys = positioned.map((p) => p.y);
  const minX = (xs.length ? Math.min(...xs) : 0) - pad;
  const maxX = (xs.length ? Math.max(...xs) : 0) + pad;
  const minY = (ys.length ? Math.min(...ys) : 0) - pad;
  const maxY = (ys.length ? Math.max(...ys) : 0) + pad;

  return {
    nodes: positioned,
    columns,
    bounds: { minX, maxX, minY, maxY, width: maxX - minX, height: maxY - minY },
  };
}

/** Human label for a temporal layer column (used by the canvas header). */
export function layerLabel(layer: number): string {
  if (layer === 0) return 'Focal';
  const hop = Math.abs(layer);
  const side = layer < 0 ? 'Backward' : 'Forward';
  const tier = hop === 1 ? 'host' : 'process';
  return `${side} · ${tier}`;
}