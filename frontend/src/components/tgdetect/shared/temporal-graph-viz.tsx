'use client';

import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react';
import { NODE_TYPE_META } from '@/lib/tgdetect/constants';
import { colorHex } from '@/lib/tgdetect/constants';
import { relationIsAttack, shortNodeId } from '@/lib/tgdetect/formatters';
import type { GraphEdge, GraphNode, NodeType } from '@/lib/tgdetect/types';
import type { ChainSubgraph } from '@/lib/tgdetect/types';

function subscribeDarkMode(cb: () => void): () => void {
  const obs = new MutationObserver(cb);
  obs.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
  return () => obs.disconnect();
}

function getDarkSnapshot(): boolean {
  return typeof document !== 'undefined'
    ? document.documentElement.classList.contains('dark')
    : false;
}

function getDarkServerSnapshot(): boolean {
  return false;
}

function useIsDarkMode(): boolean {
  return useSyncExternalStore(subscribeDarkMode, getDarkSnapshot, getDarkServerSnapshot);
}

interface VisNode {
  id: string;
  type: NodeType;
  x: number;
  y: number;
  vx: number;
  vy: number;
  radius: number;
  severity: number;
  degree: number;
}

interface VisEdge {
  src: string;
  dst: string;
  relation: string;
  label: number;
  ts: number;
}

export interface GraphSelection {
  nodeId?: string;
  edgeId?: string;
}

interface TemporalGraphVizProps {
  nodes: GraphNode[] | ChainSubgraph['nodes'];
  edges: GraphEdge[] | ChainSubgraph['edges'];
  highlightChainId?: string | null;
  selectedNodeId?: string;
  onSelectNode?: (nodeId: string | null) => void;
  showEdgeLabels?: boolean;
  maliciousOnly?: boolean;
  maxNodes?: number;
  className?: string;
}

export function TemporalGraphViz({
  nodes,
  edges,
  highlightChainId,
  selectedNodeId,
  onSelectNode,
  showEdgeLabels = false,
  maliciousOnly = false,
  maxNodes = 300,
  className = '',
}: TemporalGraphVizProps) {
  const isDark = useIsDarkMode();
  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [size, setSize] = useState({ w: 800, h: 500 });
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);

  // Zoom and Pan state
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const isDraggingRef = useRef(false);
  const dragStartRef = useRef({ x: 0, y: 0 });

  // Flow animation frame count
  const frameRef = useRef(0);

  // ResizeObserver
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      for (const e of entries) {
        const { width, height } = e.contentRect;
        if (width > 0 && height > 0) {
          setSize({ w: Math.floor(width), h: Math.floor(height) });
        }
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Prepare nodes & edges with threat-first priority sampling
  const vis = useMemo(() => {
    const nodeMap = new Map<string, VisNode>();
    const inDegrees = new Map<string, number>();
    const outDegrees = new Map<string, number>();
    const edgeList = Array.isArray(edges) ? edges : [];

    for (const e of edgeList) {
      const src = "src_id" in e ? e.src_id : (e as any).source;
      const dst = "dst_id" in e ? e.dst_id : (e as any).target;
      if (!src || !dst) continue;
      outDegrees.set(src, (outDegrees.get(src) ?? 0) + 1);
      inDegrees.set(dst, (inDegrees.get(dst) ?? 0) + 1);
    }

    const nodeList = Array.isArray(nodes) ? [...nodes] : [];
    // Sort candidate nodes to prioritize security incidents, malicious activity, and high-connectivity hubs
    nodeList.sort((a, b) => {
      const aId = "node_id" in a ? a.node_id : (a as any).id;
      const bId = "node_id" in b ? b.node_id : (b as any).id;
      const aMal = ("malicious_events" in a ? (a as any).malicious_events ?? 0 : 0);
      const bMal = ("malicious_events" in b ? (b as any).malicious_events ?? 0 : 0);
      const aDeg = (inDegrees.get(aId) ?? 0) + (outDegrees.get(aId) ?? 0);
      const bDeg = (inDegrees.get(bId) ?? 0) + (outDegrees.get(bId) ?? 0);
      return (bMal * 10000 + bDeg) - (aMal * 10000 + aDeg);
    });

    const sampleNodes = nodeList.slice(0, maxNodes);
    const nodeArr: VisNode[] = [];
    for (const n of sampleNodes) {
      const id = "node_id" in n ? n.node_id : (n as any).id;
      const type = ("node_type" in n ? n.node_type : (n as any).type) as NodeType;
      const deg = (inDegrees.get(id) ?? 0) + (outDegrees.get(id) ?? 0);
      const malCount = "malicious_events" in n ? (n as any).malicious_events ?? 0 : 0;
      const isMaliciousNode = malCount > 0;
      const baseR = isMaliciousNode ? 14 : Math.min(18, Math.max(8, 7 + Math.log2(deg + 1) * 3));
      const vn: VisNode = {
        id,
        type: type || "UNKNOWN",
        x: size.w / 2 + (Math.random() - 0.5) * Math.min(size.w * 0.6, 400),
        y: size.h / 2 + (Math.random() - 0.5) * Math.min(size.h * 0.6, 300),
        vx: 0,
        vy: 0,
        radius: baseR,
        severity: isMaliciousNode ? 1.0 : 0.0,
        degree: deg,
      };
      nodeMap.set(id, vn);
      nodeArr.push(vn);
    }

    // Connect edges whose endpoints are both in the sampled node set
    const candidateEdges: VisEdge[] = [];
    for (const e of edgeList) {
      const src = "src_id" in e ? e.src_id : (e as any).source;
      const dst = "dst_id" in e ? e.dst_id : (e as any).target;
      if (!src || !dst) continue;
      if (nodeMap.has(src) && nodeMap.has(dst)) {
        candidateEdges.push({
          src,
          dst,
          relation: "relation" in e ? e.relation : "GENERIC",
          label: "label" in e ? e.label : 0,
          ts: "ts" in e ? e.ts : 0,
        });
      }
    }

    // Sort edges so malicious flows are rendered first, cap at maxEdges to guarantee 60 FPS
    const maxEdges = Math.min(600, maxNodes * 3);
    candidateEdges.sort((a, b) => b.label - a.label);
    const edgeArr = candidateEdges.slice(0, maxEdges);

    return { nodeMap, nodeArr, edgeArr };
  }, [nodes, edges, maxNodes, size.w, size.h]);

  // Main canvas render pass
  const renderCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = size.w * dpr;
    canvas.height = size.h * dpr;
    canvas.style.width = `${size.w}px`;
    canvas.style.height = `${size.h}px`;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size.w, size.h);

    const cx = size.w / 2 + pan.x;
    const cy = size.h / 2 + pan.y;

    // 1. Draw Radar Range Rings & Tactical Grid Coordinates
    ctx.save();
    const ringColor = isDark ? 'rgba(56, 189, 248, 0.04)' : 'rgba(7, 54, 66, 0.045)';
    const gridColor = isDark ? 'rgba(56, 189, 248, 0.02)' : 'rgba(7, 54, 66, 0.025)';
    const textColor = isDark ? 'rgba(148, 163, 184, 0.35)' : 'rgba(88, 110, 117, 0.4)';

    ctx.strokeStyle = ringColor;
    ctx.lineWidth = 1;
    const maxRadius = Math.max(size.w, size.h) * 1.2;
    for (let r = 90 * zoom; r < maxRadius; r += 110 * zoom) {
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.stroke();

      // Range ring distance marker
      ctx.fillStyle = textColor;
      ctx.font = '9px monospace';
      ctx.fillText(`${Math.round(r / zoom)} px`, cx + r + 4, cy - 3);
    }

    // Coordinate crosshairs
    ctx.strokeStyle = gridColor;
    ctx.beginPath();
    ctx.moveTo(cx, 0);
    ctx.lineTo(cx, size.h);
    ctx.moveTo(0, cy);
    ctx.lineTo(size.w, cy);
    ctx.stroke();

    ctx.restore();

    // 2. Draw Edges with Flow Pulses
    const { nodeMap, edgeArr } = vis;
    const selected = selectedNodeId ?? null;
    const frame = frameRef.current;

    for (let i = 0; i < edgeArr.length; i++) {
      const e = edgeArr[i];
      const a = nodeMap.get(e.src);
      const b = nodeMap.get(e.dst);
      if (!a || !b) continue;

      const ax = cx + (a.x - size.w / 2) * zoom;
      const ay = cy + (a.y - size.h / 2) * zoom;
      const bx = cx + (b.x - size.w / 2) * zoom;
      const by = cy + (b.y - size.h / 2) * zoom;

      const isMalicious = e.label === 1;
      if (maliciousOnly && !isMalicious) continue;

      const isAttack = relationIsAttack(e.relation);
      const isSelected = selected && (selected === a.id || selected === b.id);

      ctx.save();
      let strokeColor = isDark ? '#1e293b' : '#dfd7c2';
      let strokeWidth = 1.2 * zoom;

      if (isMalicious) {
        strokeColor = isDark ? '#f43f5e' : '#dc322f';
        strokeWidth = (isAttack ? 2.8 : 2.0) * zoom;
        if (isDark) {
          ctx.shadowColor = 'rgba(244, 63, 94, 0.7)';
          ctx.shadowBlur = 8;
        }
      } else if (isSelected) {
        strokeColor = isDark ? '#38bdf8' : '#2aa198';
        strokeWidth = 2.2 * zoom;
      }

      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = strokeWidth;
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.stroke();

      // Draw directional arrow
      const angle = Math.atan2(by - ay, bx - ax);
      const arrowDist = b.radius * zoom + 6;
      const arrowX = bx - Math.cos(angle) * arrowDist;
      const arrowY = by - Math.sin(angle) * arrowDist;
      const headLen = 7 * zoom;

      ctx.fillStyle = strokeColor;
      ctx.beginPath();
      ctx.moveTo(arrowX, arrowY);
      ctx.lineTo(arrowX - headLen * Math.cos(angle - Math.PI / 6), arrowY - headLen * Math.sin(angle - Math.PI / 6));
      ctx.lineTo(arrowX - headLen * Math.cos(angle + Math.PI / 6), arrowY - headLen * Math.sin(angle + Math.PI / 6));
      ctx.closePath();
      ctx.fill();

      // Flow pulse packet moving along edge
      const pulseProgress = ((frame * 0.015 + i * 0.3) % 1);
      const pulseX = ax + (bx - ax) * pulseProgress;
      const pulseY = ay + (by - ay) * pulseProgress;
      ctx.beginPath();
      ctx.arc(pulseX, pulseY, (isMalicious ? 3 : 2) * zoom, 0, Math.PI * 2);
      ctx.fillStyle = isMalicious ? (isDark ? '#f43f5e' : '#dc322f') : (isDark ? '#38bdf8' : '#2aa198');
      ctx.fill();

      // Edge labels
      if (showEdgeLabels) {
        const mx = (ax + bx) / 2;
        const my = (ay + by) / 2;
        ctx.fillStyle = isDark ? '#94a3b8' : '#586e75';
        ctx.font = `${Math.max(9, 10 * zoom)}px monospace`;
        ctx.fillText(e.relation, mx + 5, my - 5);
      }

      ctx.restore();
    }

    // 3. Draw Nodes
    for (const n of vis.nodeArr) {
      const nx = cx + (n.x - size.w / 2) * zoom;
      const ny = cy + (n.y - size.h / 2) * zoom;
      const nr = n.radius * zoom;

      const isSelected = selected === n.id;
      const isHovered = hoveredNodeId === n.id;
      const isMalicious = n.severity > 0.8;

      ctx.save();

      // Selected or Threat Luminous Halo
      if (isSelected || isMalicious) {
        ctx.beginPath();
        ctx.arc(nx, ny, nr + 6 * zoom, 0, Math.PI * 2);
        ctx.strokeStyle = isMalicious
          ? (isDark ? 'rgba(244, 63, 94, 0.4)' : 'rgba(220, 50, 47, 0.35)')
          : (isDark ? 'rgba(56, 189, 248, 0.4)' : 'rgba(42, 161, 152, 0.35)');
        ctx.lineWidth = 1.5;
        ctx.stroke();

        if (isSelected) {
          // Tactical corner reticle around selected node
          const boxS = nr + 10 * zoom;
          ctx.strokeStyle = isDark ? '#38bdf8' : '#2aa198';
          ctx.lineWidth = 1.5;
          ctx.strokeRect(nx - boxS, ny - boxS, boxS * 2, boxS * 2);
        }
      }

      // Node Body
      let fillColor = isDark ? '#111724' : '#fffcf4';
      let strokeColor = isDark ? '#38bdf8' : '#268bd2';

      if (isMalicious) {
        fillColor = isDark ? 'rgba(244, 63, 94, 0.2)' : 'rgba(220, 50, 47, 0.15)';
        strokeColor = isDark ? '#f43f5e' : '#dc322f';
      } else if (n.severity > 0.4) {
        fillColor = isDark ? 'rgba(246, 163, 32, 0.15)' : 'rgba(181, 137, 0, 0.12)';
        strokeColor = isDark ? '#f6a320' : '#b58900';
      }

      ctx.beginPath();
      ctx.arc(nx, ny, nr, 0, Math.PI * 2);
      ctx.fillStyle = fillColor;
      ctx.fill();
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = (isSelected ? 2.5 : 1.5) * zoom;
      ctx.stroke();

      // Inner icon / dot
      ctx.beginPath();
      ctx.arc(nx, ny, Math.max(2, nr * 0.3), 0, Math.PI * 2);
      ctx.fillStyle = strokeColor;
      ctx.fill();

      // Node text label
      ctx.fillStyle = isDark ? '#f8fafc' : '#073642';
      ctx.font = `${Math.max(10, 11 * zoom)}px 'JetBrains Mono', monospace`;
      ctx.textAlign = 'center';
      const labelText = shortNodeId(n.id);
      ctx.fillText(labelText, nx, ny + nr + 14 * zoom);

      // Threat tag if malicious
      if (isMalicious) {
        ctx.fillStyle = isDark ? '#f43f5e' : '#dc322f';
        ctx.font = '9px monospace';
        ctx.fillText('MALICIOUS', nx, ny - nr - 6 * zoom);
      }

      ctx.restore();
    }
  }, [vis, size, zoom, pan, isDark, showEdgeLabels, maliciousOnly, selectedNodeId, hoveredNodeId]);

  // Force simulation loop with cooling schedule & idle optimization
  useEffect(() => {
    let raf: number;
    const arr = vis.nodeArr;
    const edgeArr = vis.edgeArr;
    if (arr.length === 0) return;

    let alpha = 1.0;
    const alphaMin = 0.005;
    const alphaDecay = 0.985;

    const tick = () => {
      frameRef.current++;

      if (alpha > alphaMin) {
        // Repulsion between nodes
        for (let i = 0; i < arr.length; i++) {
          for (let j = i + 1; j < arr.length; j++) {
            const a = arr[i];
            const b = arr[j];
            const dx = b.x - a.x;
            const dy = b.y - a.y;
            const d2 = dx * dx + dy * dy + 0.1;
            const dist = Math.sqrt(d2);
            const targetDist = a.radius + b.radius + 100;
            if (dist < targetDist) {
              const force = ((targetDist - dist) / dist) * 0.08 * alpha;
              a.vx -= dx * force;
              a.vy -= dy * force;
              b.vx += dx * force;
              b.vy += dy * force;
            }
          }
        }

        // Spring attraction along edges
        for (let i = 0; i < edgeArr.length; i++) {
          const e = edgeArr[i];
          const a = vis.nodeMap.get(e.src);
          const b = vis.nodeMap.get(e.dst);
          if (!a || !b) continue;
          const dx = b.x - a.x;
          const dy = b.y - a.y;
          const dist = Math.sqrt(dx * dx + dy * dy) || 1;
          const idealDist = 180;
          const force = (dist - idealDist) * 0.003 * alpha;
          const fx = (dx / dist) * force;
          const fy = (dy / dist) * force;
          a.vx += fx;
          a.vy += fy;
          b.vx -= fx;
          b.vy -= fy;
        }

        // Center gravity & damping
        const cx = size.w / 2;
        const cy = size.h / 2;
        for (let i = 0; i < arr.length; i++) {
          const n = arr[i];
          n.vx += (cx - n.x) * 0.003 * alpha;
          n.vy += (cy - n.y) * 0.003 * alpha;
          n.vx *= 0.85;
          n.vy *= 0.85;
          n.x += n.vx;
          n.y += n.vy;
        }

        alpha *= alphaDecay;
      }

      renderCanvas();
      raf = requestAnimationFrame(tick);
    };

    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [vis, size, zoom, pan, isDark, showEdgeLabels, maliciousOnly, selectedNodeId, hoveredNodeId]);

  // Interactive mouse handlers: Click, Drag to Pan, Wheel to Zoom
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    isDraggingRef.current = true;
    dragStartRef.current = { x: e.clientX - pan.x, y: e.clientY - pan.y };
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (isDraggingRef.current) {
      setPan({
        x: e.clientX - dragStartRef.current.x,
        y: e.clientY - dragStartRef.current.y,
      });
      return;
    }

    // Hover detection
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    const cx = size.w / 2 + pan.x;
    const cy = size.h / 2 + pan.y;

    let found: string | null = null;
    for (const n of vis.nodeArr) {
      const nx = cx + (n.x - size.w / 2) * zoom;
      const ny = cy + (n.y - size.h / 2) * zoom;
      const dist = Math.hypot(mx - nx, my - ny);
      if (dist <= (n.radius + 6) * zoom) {
        found = n.id;
        break;
      }
    }
    setHoveredNodeId(found);
  };

  const handleMouseUp = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const wasDragging = Math.hypot(e.clientX - dragStartRef.current.x - pan.x, e.clientY - dragStartRef.current.y - pan.y) > 4;
    isDraggingRef.current = false;

    if (!wasDragging) {
      // Click selection
      if (hoveredNodeId) {
        if (onSelectNode) onSelectNode(hoveredNodeId === selectedNodeId ? null : hoveredNodeId);
      } else {
        if (onSelectNode) onSelectNode(null);
      }
    }
  };

  const handleWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const delta = e.deltaY < 0 ? 1.1 : 0.9;
    setZoom((z) => Math.min(2.5, Math.max(0.4, z * delta)));
  };

  const resetView = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
    if (onSelectNode) onSelectNode(null);
  };

  return (
    <div ref={containerRef} className={`relative w-full h-full min-h-[480px] select-none ${className}`}>
      <canvas
        ref={canvasRef}
        data-testid="temporal-graph"
        data-node-count={vis.nodeArr.length}
        data-edge-count={vis.edgeArr.length}
        aria-label="Recorded temporal graph"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onWheel={handleWheel}
        className="w-full h-full cursor-grab active:cursor-grabbing block"
      />

      {/* Floating Canvas Controls */}
      <div className="absolute bottom-4 right-4 flex items-center gap-1.5 bg-[hsl(var(--card)/0.85)] backdrop-blur-xs border border-[hsl(var(--border))] rounded-md p-1 shadow-lg">
        <button
          type="button"
          onClick={() => setZoom((z) => Math.min(2.5, z * 1.2))}
          className="size-7 rounded hover:bg-[hsl(var(--background))] flex items-center justify-center font-mono text-xs font-bold text-[hsl(var(--foreground))]"
          title="Zoom In"
        >
          +
        </button>
        <button
          type="button"
          onClick={() => setZoom((z) => Math.max(0.4, z * 0.8))}
          className="size-7 rounded hover:bg-[hsl(var(--background))] flex items-center justify-center font-mono text-xs font-bold text-[hsl(var(--foreground))]"
          title="Zoom Out"
        >
          -
        </button>
        <button
          type="button"
          onClick={resetView}
          className="px-2 h-7 rounded hover:bg-[hsl(var(--background))] flex items-center justify-center font-mono text-[10px] text-[hsl(var(--muted-foreground))] hover:text-[hsl(var(--foreground))]"
          title="Reset View"
        >
          RESET
        </button>
      </div>

      {/* Interactive Legend & Coordinates Indicator */}
      <div className="absolute bottom-4 left-4 pointer-events-none flex items-center gap-3 text-[10px] font-mono text-[hsl(var(--muted-foreground))] bg-[hsl(var(--card)/0.7)] backdrop-blur-xs px-2.5 py-1 rounded border border-[hsl(var(--border)/0.5)]">
        <span className="flex items-center gap-1.5">
          <span className="size-2 rounded-full bg-emerald-500" />
          <span>UNFLAGGED / UNKNOWN</span>
        </span>
        <span className="flex items-center gap-1.5">
          <span className="size-2 rounded-full bg-rose-500" />
          <span>MALICIOUS</span>
        </span>
        <span className="text-[hsl(var(--primary))] pl-1 border-l border-[hsl(var(--border))]">
          ZOOM: {Math.round(zoom * 100)}% · {vis.nodeArr.length} nodes / {vis.edgeArr.length} edges displayed
        </span>
      </div>
    </div>
  );
}
