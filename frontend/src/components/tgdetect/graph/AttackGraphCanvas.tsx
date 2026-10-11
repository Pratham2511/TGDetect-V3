'use client';

/**
 * Attack-summary canvas renderer (ORTHRUS Fig. 3 style).
 *
 * Draws the precomputed layered-DAG layout: backward/entry columns on the left,
 * the focal detection in the centre, forward/exit columns on the right. Unlike
 * the old shared viewer this does NO force simulation and runs requestAnimation-
 * Frame ONLY during the brief eased morph when the summary changes — every other
 * repaint (zoom, pan, hover, select) is a single synchronous draw. The canvas
 * backing store is resized only when its box or DPR changes.
 */

import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore, type PointerEvent as ReactPointerEvent } from 'react';
import { Crosshair, Minus, Plus, Maximize2, type LucideIcon } from 'lucide-react';

import { colorHex } from '@/lib/tgdetect/constants';
import { shortNodeId } from '@/lib/tgdetect/formatters';
import {
  DEFAULT_LAYOUT,
  easeInOutCubic,
  layerLabel,
  layoutSummary,
  lerp,
  nodeRadius,
  type PositionedSummary,
} from '@/lib/tgdetect/graph/layout';
import type { AttackSummary, SummaryNode, TraceMode } from '@/lib/tgdetect/graph/types';
import { cn } from '@/lib/utils';

// ── dark-mode (mirrors shared/temporal-graph-viz.tsx) ────────────────────────
function useIsDarkMode(): boolean {
  return useSyncExternalStore(
    (cb) => {
      const obs = new MutationObserver(cb);
      if (typeof document !== 'undefined') obs.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
      return () => obs.disconnect();
    },
    () => (typeof document !== 'undefined' ? document.documentElement.classList.contains('dark') : false),
    () => false,
  );
}

// ── small colour helpers ─────────────────────────────────────────────────────
function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace('#', '');
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
}
function rgb(c: [number, number, number], alpha = 1): string {
  return `rgba(${Math.round(c[0])}, ${Math.round(c[1])}, ${Math.round(c[2])}, ${alpha})`;
}
function mix(a: [number, number, number], b: [number, number, number], t: number): [number, number, number] {
  return [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];
}

/** Benign→malicious fill for a node, driven by anomaly fA(u). */
function anomalyFill(a: number, dark: boolean): [number, number, number] {
  const green = hexToRgb(colorHex('green', dark));
  const amber = hexToRgb(colorHex('amber', dark));
  const red = hexToRgb(colorHex('red', dark));
  return a < 0.5 ? mix(green, amber, a / 0.5) : mix(amber, red, (a - 0.5) / 0.5);
}

/** Role → stroke colour token. */
function roleToken(role: SummaryNode['role']): string {
  switch (role) {
    case 'focal': return 'red';
    case 'entry': return 'amber';
    case 'exit': return 'teal';
    case 'pivot': return 'violet';
    default: return 'gray';
  }
}
/** A floating square canvas control (zoom / fit / recentre). */
function CtrlButton({ icon: Icon, title, onClick }: { icon: LucideIcon; title: string; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={title}
      aria-label={title}
      className="flex h-8 w-8 items-center justify-center rounded-md border border-border bg-card/90 text-muted-foreground shadow-sm backdrop-blur transition-colors hover:bg-card hover:text-foreground"
    >
      <Icon className="h-4 w-4" />
    </button>
  );
}

export interface AttackGraphCanvasProps {
  summary: AttackSummary;
  selectedNodeId: string | null;
  onSelectNode: (id: string | null) => void;
  onHoverNode?: (id: string | null) => void;
  traceMode: TraceMode;
  className?: string;
}

interface XY { x: number; y: number; }

export function AttackGraphCanvas({
  summary,
  selectedNodeId,
  onSelectNode,
  onHoverNode,
  traceMode,
  className,
}: AttackGraphCanvasProps) {
  const isDark = useIsDarkMode();
  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [size, setSize] = useState({ w: 800, h: 520 });
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState<XY>({ x: 0, y: 0 });
  const [hovered, setHovered] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);

  const positioned: PositionedSummary = useMemo(() => layoutSummary(summary, DEFAULT_LAYOUT), [summary]);
  const nodeById = useMemo(() => new Map(positioned.nodes.map((n) => [n.id, n])), [positioned]);
  const center = useMemo(() => ({
    x: (positioned.bounds.minX + positioned.bounds.maxX) / 2,
    y: (positioned.bounds.minY + positioned.bounds.maxY) / 2,
  }), [positioned]);

  // Animated positions (virtual coords), updated by the morph rAF.
  const posRef = useRef<Map<string, XY>>(new Map());
  const drawRef = useRef<() => void>(() => {});

  // Mirror reactive values into refs so pointer handlers read fresh values.
  // Written in an effect (not during render) per the react-hooks/refs rule.
  const zoomRef = useRef(zoom);
  const panRef = useRef(pan);
  const sizeRef = useRef(size);
  const centerRef = useRef(center);
  const positionedRef = useRef(positioned);
  useEffect(() => {
    zoomRef.current = zoom;
    panRef.current = pan;
    sizeRef.current = size;
    centerRef.current = center;
    positionedRef.current = positioned;
  });

  const getPos = useCallback((id: string): XY => {
    return posRef.current.get(id) ?? nodeById.get(id) ?? { x: 0, y: 0 };
  }, [nodeById]);
  // ── fit-to-view ────────────────────────────────────────────────────────────
  const fitView = useCallback(() => {
    const b = positionedRef.current.bounds;
    const s = sizeRef.current;
    if (b.width <= 0 || b.height <= 0 || s.w <= 0 || s.h <= 0) {
      setZoom(1);
      setPan({ x: 0, y: 0 });
      return;
    }
    const z = Math.min(s.w / b.width, s.h / b.height) * 0.9;
    setZoom(Math.max(0.15, Math.min(2.5, z)));
    setPan({ x: 0, y: 0 });
  }, []);

  // Re-fit once per focal node, once the container size is known.
  const fittedFocalRef = useRef<string | null>(null);
  useEffect(() => {
    if (size.w <= 0) return;
    if (fittedFocalRef.current === summary.focalId) return;
    fittedFocalRef.current = summary.focalId;
    fitView();
  }, [summary.focalId, size.w, size.h, fitView]);

  // ── container resize ─────────────────────────────────────────────────────────
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      for (const e of entries) {
        const { width, height } = e.contentRect;
        if (width > 0 && height > 0) setSize({ w: Math.floor(width), h: Math.floor(height) });
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Screen → virtual-world, for hit-testing (reads live refs).
  const screenToWorld = useCallback((sx: number, sy: number): XY => {
    const z = zoomRef.current;
    const p = panRef.current;
    const s = sizeRef.current;
    const c = centerRef.current;
    return {
      x: (sx - s.w / 2 - p.x) / z + c.x,
      y: (sy - s.h / 2 - p.y) / z + c.y,
    };
  }, []);

  // Virtual-world → screen radius (clamped so nodes stay legible at any zoom).
  const screenRadius = useCallback((crit: number, z: number): number => {
    return Math.max(3.5, Math.min(32, nodeRadius(crit) * z));
  }, []);
  // ── paint (single synchronous draw; rAF only drives the morph) ───────────────
  const paint = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;

    const s = size;
    const dpr = Math.min(typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1, 2);
    const bw = Math.round(s.w * dpr);
    const bh = Math.round(s.h * dpr);
    if (canvas.width !== bw || canvas.height !== bh) {
      canvas.width = bw;
      canvas.height = bh;
    }
    canvas.style.width = `${s.w}px`;
    canvas.style.height = `${s.h}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, s.w, s.h);

    const z = zoom;
    const c = center;
    const W = (x: number): number => (x - c.x) * z + s.w / 2 + pan.x;
    const H = (y: number): number => (y - c.y) * z + s.h / 2 + pan.y;
    const fg = isDark ? 'rgba(226,232,240,0.92)' : 'rgba(30,41,59,0.92)';

    // 1 ── temporal column guides + labels.
    ctx.textAlign = 'center';
    for (const col of positioned.columns) {
      const x = W(col.x);
      ctx.strokeStyle = rgb(hexToRgb(colorHex('gray', isDark)), 0.12);
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 6]);
      ctx.beginPath();
      ctx.moveTo(x, 22);
      ctx.lineTo(x, s.h - 8);
      ctx.stroke();
      ctx.setLineDash([]);
      const tok = col.layer === 0 ? 'red' : col.layer < 0 ? 'amber' : 'teal';
      ctx.fillStyle = rgb(hexToRgb(colorHex(tok, isDark)), 0.85);
      ctx.font = '600 11px ui-sans-serif, system-ui, sans-serif';
      ctx.fillText(layerLabel(col.layer), x, 15);
    }

    // 2 ── edges (colour by trace direction, width/alpha by criticality fC(e)).
    ctx.lineCap = 'round';
    for (const e of summary.edges) {
      const a = getPos(e.src);
      const b = getPos(e.dst);
      const dim = traceMode !== 'both' && e.direction !== traceMode && e.direction !== 'focal';
      const tok = e.direction === 'backward' ? 'amber' : e.direction === 'forward' ? 'teal' : 'red';
      const col = hexToRgb(colorHex(tok, isDark));
      const alpha = dim ? 0.05 : 0.16 + 0.5 * e.criticality;
      const x1 = W(a.x), y1 = H(a.y), x2 = W(b.x), y2 = H(b.y);
      ctx.strokeStyle = rgb(col, alpha);
      ctx.lineWidth = Math.max(0.75, (0.6 + 2.6 * e.criticality) * (dim ? 0.5 : 1));
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
      if (!dim && e.criticality > 0.3) {
        const ang = Math.atan2(y2 - y1, x2 - x1);
        const hx = x2 - Math.cos(ang) * 13;
        const hy = y2 - Math.sin(ang) * 13;
        ctx.fillStyle = rgb(col, Math.min(0.95, alpha + 0.25));
        ctx.beginPath();
        ctx.moveTo(x2 - Math.cos(ang) * 2, y2 - Math.sin(ang) * 2);
        ctx.lineTo(hx - Math.cos(ang - Math.PI / 2) * 4, hy - Math.sin(ang - Math.PI / 2) * 4);
        ctx.lineTo(hx - Math.cos(ang + Math.PI / 2) * 4, hy - Math.sin(ang + Math.PI / 2) * 4);
        ctx.closePath();
        ctx.fill();
      }
    }
    // 3 ── nodes (fill = anomaly fA, stroke = role, radius = criticality fC).
    ctx.textBaseline = 'alphabetic';
    for (const n of positioned.nodes) {
      const pos = getPos(n.id);
      const x = W(pos.x);
      const y = H(pos.y);
      const r = screenRadius(n.criticality, z);
      const fill = anomalyFill(n.anomaly, isDark);
      const stroke = hexToRgb(colorHex(roleToken(n.role), isDark));
      const isSel = n.id === selectedNodeId;
      const isHover = n.id === hovered;

      if (n.role === 'focal' || isSel || isHover) {
        ctx.beginPath();
        ctx.arc(x, y, r + (isSel ? 7 : 5), 0, Math.PI * 2);
        ctx.fillStyle = rgb(stroke, isSel ? 0.24 : 0.14);
        ctx.fill();
      }
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fillStyle = rgb(fill, 0.92);
      ctx.fill();
      ctx.lineWidth = n.role === 'focal' ? 3 : isSel ? 2.5 : 1.5;
      ctx.strokeStyle = rgb(stroke, 0.95);
      ctx.stroke();

      if (r >= 9) {
        ctx.fillStyle = isDark ? 'rgba(9,13,20,0.92)' : 'rgba(255,255,255,0.96)';
        ctx.font = `700 ${Math.round(r * 0.95)}px ui-sans-serif, system-ui, sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(n.type.charAt(0), x, y + 0.5);
        ctx.textBaseline = 'alphabetic';
      }
      if (z > 0.62 || isSel || isHover) {
        const label = shortNodeId(n.id);
        const short = label.length > 20 ? `${label.slice(0, 19)}…` : label;
        ctx.font = '500 11px ui-sans-serif, system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillStyle = fg;
        ctx.fillText(short, x, y + r + 13);
      }
    }
  }, [size, zoom, pan, center, positioned, summary, isDark, traceMode, selectedNodeId, hovered, getPos, screenRadius]);

  // Keep the morph's draw callback current and repaint when a visual input changes.
  useEffect(() => {
    drawRef.current = paint;
    paint();
  }, [paint]);
  // ── morph animation: ease node positions from their old spot (or the focal) ──
  useEffect(() => {
    const targets = new Map(positioned.nodes.map((n) => [n.id, { x: n.x, y: n.y } as XY]));
    const focalTarget = targets.get(summary.focalId) ?? { x: 0, y: 0 };
    const prev = posRef.current;
    const starts = new Map<string, XY>();
    for (const n of positioned.nodes) starts.set(n.id, prev.get(n.id) ?? focalTarget);

    const dur = 540;
    const t0 = typeof performance !== 'undefined' ? performance.now() : Date.now();
    let raf = 0;
    const step = (now: number): void => {
      const t = Math.min(1, (now - t0) / dur);
      const e = easeInOutCubic(t);
      const m = new Map<string, XY>();
      for (const n of positioned.nodes) {
        const sPos = starts.get(n.id)!;
        const tg = targets.get(n.id)!;
        m.set(n.id, { x: lerp(sPos.x, tg.x, e), y: lerp(sPos.y, tg.y, e) });
      }
      posRef.current = m;
      drawRef.current();
      if (t < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [positioned, summary.focalId]);

  // ── hit-testing (screen space) ───────────────────────────────────────────────
  const nodeAt = useCallback((sx: number, sy: number): string | null => {
    const z = zoomRef.current, p = panRef.current, s = sizeRef.current, c = centerRef.current;
    let best: string | null = null;
    let bestD = Infinity;
    for (const n of positionedRef.current.nodes) {
      const pos = posRef.current.get(n.id) ?? { x: n.x, y: n.y };
      const x = (pos.x - c.x) * z + s.w / 2 + p.x;
      const y = (pos.y - c.y) * z + s.h / 2 + p.y;
      const r = Math.max(6, Math.min(32, nodeRadius(n.criticality) * z)) + 3;
      const dx = sx - x, dy = sy - y, d = dx * dx + dy * dy;
      if (d <= r * r && d < bestD) {
        bestD = d;
        best = n.id;
      }
    }
    return best;
  }, []);

  // ── wheel zoom-around-cursor (native listener so we can preventDefault) ───────
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const onWheel = (ev: WheelEvent): void => {
      ev.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const sx = ev.clientX - rect.left;
      const sy = ev.clientY - rect.top;
      const before = screenToWorld(sx, sy);
      const factor = Math.exp(-ev.deltaY * 0.0015);
      const z = Math.max(0.1, Math.min(4, zoomRef.current * factor));
      const s = sizeRef.current, c = centerRef.current;
      setZoom(z);
      setPan({ x: sx - (before.x - c.x) * z - s.w / 2, y: sy - (before.y - c.y) * z - s.h / 2 });
    };
    canvas.addEventListener('wheel', onWheel, { passive: false });
    return () => canvas.removeEventListener('wheel', onWheel);
  }, [screenToWorld]);
  // ── pointer drag / hover / click ─────────────────────────────────────────────
  const dragRef = useRef({ active: false, sx: 0, sy: 0, panX: 0, panY: 0, moved: 0 });

  const onPointerDown = (ev: ReactPointerEvent<HTMLCanvasElement>): void => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.setPointerCapture(ev.pointerId);
    dragRef.current = { active: true, sx: ev.clientX, sy: ev.clientY, panX: panRef.current.x, panY: panRef.current.y, moved: 0 };
    setDragging(true);
  };
  const onPointerMove = (ev: ReactPointerEvent<HTMLCanvasElement>): void => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const d = dragRef.current;
    if (d.active) {
      const dx = ev.clientX - d.sx;
      const dy = ev.clientY - d.sy;
      d.moved = Math.abs(dx) + Math.abs(dy);
      setPan({ x: d.panX + dx, y: d.panY + dy });
    } else {
      const id = nodeAt(ev.clientX - rect.left, ev.clientY - rect.top);
      if (id !== hovered) {
        setHovered(id);
        onHoverNode?.(id);
      }
    }
  };
  const onPointerUp = (ev: ReactPointerEvent<HTMLCanvasElement>): void => {
    const canvas = canvasRef.current;
    if (canvas) {
      try { canvas.releasePointerCapture(ev.pointerId); } catch { /* ignore */ }
    }
    const d = dragRef.current;
    dragRef.current = { ...d, active: false };
    setDragging(false);
    if (d.active && d.moved < 5 && canvas) {
      const rect = canvas.getBoundingClientRect();
      onSelectNode(nodeAt(ev.clientX - rect.left, ev.clientY - rect.top));
    }
  };
  const onPointerLeave = (): void => {
    if (hovered) {
      setHovered(null);
      onHoverNode?.(null);
    }
  };

  const zoomBy = useCallback((factor: number): void => {
    const s = sizeRef.current, c = centerRef.current;
    const before = screenToWorld(s.w / 2, s.h / 2);
    const z = Math.max(0.1, Math.min(4, zoomRef.current * factor));
    setZoom(z);
    setPan({ x: -(before.x - c.x) * z, y: -(before.y - c.y) * z });
  }, [screenToWorld]);
  const cursor = dragging ? 'grabbing' : hovered ? 'pointer' : 'grab';
  const legend: { role: SummaryNode['role']; label: string }[] = [
    { role: 'focal', label: 'Focal detection' },
    { role: 'entry', label: 'Entry (backward)' },
    { role: 'exit', label: 'Exit (forward)' },
    { role: 'pivot', label: 'Pivot host' },
    { role: 'related', label: 'Related' },
  ];

  return (
    <div
      ref={containerRef}
      className={cn('relative h-full w-full overflow-hidden rounded-xl border border-border bg-background/40', className)}
    >
      <canvas
        ref={canvasRef}
        className="absolute inset-0 block touch-none"
        style={{ cursor }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={onPointerLeave}
      />

      {/* zoom / view controls */}
      <div className="absolute right-3 top-3 flex flex-col gap-1.5">
        <CtrlButton icon={Plus} title="Zoom in" onClick={() => zoomBy(1.25)} />
        <CtrlButton icon={Minus} title="Zoom out" onClick={() => zoomBy(0.8)} />
        <CtrlButton icon={Maximize2} title="Fit to view" onClick={fitView} />
        <CtrlButton icon={Crosshair} title="Re-centre" onClick={() => setPan({ x: 0, y: 0 })} />
      </div>

      {/* zoom readout */}
      <div className="absolute right-3 bottom-3 rounded-md border border-border bg-card/85 px-2 py-1 text-[11px] font-medium text-muted-foreground backdrop-blur">
        {Math.round(zoom * 100)}%
      </div>

      {/* role legend + honesty disclosure */}
      <div className="absolute left-3 bottom-3 max-w-[16rem] rounded-lg border border-border bg-card/85 p-2.5 text-[11px] leading-tight backdrop-blur">
        <div className="mb-1.5 flex flex-wrap gap-x-3 gap-y-1">
          {legend.map(({ role, label }) => (
            <span key={role} className="inline-flex items-center gap-1.5 text-muted-foreground">
              <span
                className="inline-block h-2.5 w-2.5 rounded-full"
                style={{ backgroundColor: colorHex(roleToken(role), isDark) }}
              />
              {label}
            </span>
          ))}
        </div>
        <p className="text-[10px] text-muted-foreground/80">
          Fill = model anomaly · size = criticality. Direction is inferred from
          co-located timing (OBSERVED_ON), not a captured causal edge.
        </p>
      </div>

      {summary.truncated && (
        <div className="absolute left-3 top-3 rounded-md border border-amber-500/40 bg-amber-500/10 px-2 py-1 text-[11px] font-medium text-amber-600 backdrop-blur dark:text-amber-400">
          Summary capped — showing the most critical {summary.stats.renderedNodes} nodes
        </div>
      )}
    </div>
  );
}