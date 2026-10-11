"use client";

import { useEffect, useId, useMemo, useRef, useState } from "react";
import {
  ArrowDownToLine,
  ArrowLeft,
  ArrowRight,
  ChevronLeft,
  ChevronRight,
  FileText,
  Focus,
  GitBranch,
  Globe,
  Layers,
  Loader2,
  Pause,
  Play,
  RotateCcw,
  Search,
  Terminal,
  TriangleAlert,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import ui from "./investigation.module.css";

export type Entity = {
  id: number;
  score: number;
  alert: boolean;
  attributes: Record<string, unknown>;
};
export type Observation = {
  id: string;
  split: string;
  idx: number;
  src: number;
  dst: number;
  time: number;
  relation_name: string;
  relation: number;
  evidence: number | null;
  loss: number | null;
};
type Graph = { nodes: Entity[]; events: Observation[] };
type Path = {
  id: string;
  events: Observation[];
  evidence_sum: number;
  anchor_step: number;
};
type Construction = {
  anchor: Observation;
  paths: Path[];
  nodes: Entity[];
  search_pruned: boolean;
  labels_used: boolean;
  ranking: string;
  rule: string;
  interpretation: string;
};
const title = (n: Entity) =>
  String(
    n.attributes.path ||
      n.attributes.cmd_line ||
      n.attributes.remote_ip ||
      `${n.attributes.type || "entity"} #${n.id}`,
  );
const short = (text: string) =>
  text.length > 25 ? `…${text.slice(-24)}` : text;
const clock = (t: number) => new Date(t * 1000).toISOString().slice(11, 23);
const kind = (n: Entity) => String(n.attributes.type || "entity").toLowerCase();
function EntityIcon({
  node,
  ...props
}: {
  node: Entity;
  x?: number;
  y?: number;
  size?: number;
}) {
  const Icon = kind(node).includes("subject")
    ? Terminal
    : kind(node).includes("file")
      ? FileText
      : Globe;
  return <Icon {...props} size={props.size || 18} />;
}
function save(data: unknown) {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = "cadets-constructed-chain.json";
  a.click();
  URL.revokeObjectURL(url);
}

export function AttackGraph({
  chain,
  seed,
  select,
}: {
  chain: Graph;
  seed: number;
  select: (e: Observation) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const [zoom, setZoom] = useState(1);
  const [focused, setFocused] = useState(seed);
  const [selected, setSelected] = useState<Observation | null>(null);
  const [moved, setMoved] = useState<Record<number, { x: number; y: number }>>(
    {},
  );
  const drag = useRef<{
    node: number;
    x: number;
    y: number;
    px: number;
    py: number;
  } | null>(null);
  const svg = useRef<SVGSVGElement>(null);
  const marker = useId().replaceAll(":", "");
  const ranked = useMemo(() => {
    const weight = new Map<number, number>();
    for (const e of chain.events)
      for (const n of [e.src, e.dst])
        weight.set(n, Math.max(weight.get(n) || 0, e.evidence || 0));
    return [...chain.nodes].sort((a, b) =>
      a.id === seed
        ? -1
        : b.id === seed
          ? 1
          : (weight.get(b.id) || 0) - (weight.get(a.id) || 0) || a.id - b.id,
    );
  }, [chain, seed]);
  // Keep the collapsed view connected: a high-scoring two-hop entity must
  // bring its connecting entity into view before it can be displayed.
  const shown: Entity[] = ranked.length ? [ranked[0]] : [];
  const targetCount = expanded ? ranked.length : 6;
  while (shown.length < targetCount) {
    const visible = new Set(shown.map((n) => n.id));
    const next = ranked.find(
      (n) =>
        !visible.has(n.id) &&
        chain.events.some(
          (e) =>
            (e.src === n.id && visible.has(e.dst)) ||
            (e.dst === n.id && visible.has(e.src)),
        ),
    );
    if (!next) break;
    shown.push(next);
  }
  const ids = new Set(shown.map((n) => n.id));
  const grouped = new Map<
    string,
    { event: Observation; count: number; relations: Set<string> }
  >();
  for (const e of chain.events)
    if (ids.has(e.src) && ids.has(e.dst)) {
      const key = `${e.src}:${e.dst}`;
      const previous = grouped.get(key);
      grouped.set(key, {
        event:
          !previous || (e.evidence || 0) > (previous.event.evidence || 0)
            ? e
            : previous.event,
        count: (previous?.count || 0) + 1,
        relations: new Set([...(previous?.relations || []), e.relation_name]),
      });
    }
  const links = [...grouped.values()];
  const distance = new Map<number, number>([[seed, 0]]);
  for (let iteration = 0; iteration < shown.length; iteration++)
    for (const { event: e } of links) {
      if (distance.has(e.src) && !distance.has(e.dst))
        distance.set(e.dst, distance.get(e.src)! + 1);
      if (distance.has(e.dst) && !distance.has(e.src))
        distance.set(e.src, distance.get(e.dst)! + 1);
    }
  const lanes = new Map<number, Entity[]>();
  for (const n of shown) {
    const incoming = chain.events.some((e) => e.src === n.id && e.dst === seed);
    const lane =
      n.id === seed ? 0 : incoming ? -1 : Math.min(distance.get(n.id) || 1, 2);
    lanes.set(lane, [...(lanes.get(lane) || []), n]);
  }
  const height = Math.max(
    390,
    Math.max(...[...lanes.values()].map((list) => list.length)) * 98 + 110,
  );
  const laneKeys = [...lanes.keys()].sort((a, b) => a - b);
  const width = Math.max(760, laneKeys.length * 275 + 50);
  const positions = new Map(
    shown.map((n) => {
      const lane = [...lanes.entries()].find(([, list]) => list.includes(n))!;
      const list = lane[1];
      return [
        n.id,
        moved[n.id] || {
          x: 35 + laneKeys.indexOf(lane[0]) * 275,
          y: 85 + (list.indexOf(n) + 0.5) * ((height - 130) / list.length),
        },
      ];
    }),
  );
  function point(e: React.PointerEvent<SVGSVGElement>) {
    const matrix = svg.current?.getScreenCTM()?.inverse();
    return matrix
      ? new DOMPoint(e.clientX, e.clientY).matrixTransform(matrix)
      : null;
  }
  return (
    <div className={ui.graphShell}>
      <div className={ui.graphToolbar}>
        <span>
          <Layers size={14} />
          Entity relationships{" "}
          <b>
            {shown.length} / {ranked.length}
          </b>
        </span>
        <div>
          <button
            aria-label="Zoom out"
            onClick={() => setZoom(Math.max(0.6, zoom - 0.15))}
          >
            <ZoomOut size={15} />
          </button>
          <button
            aria-label="Fit graph"
            onClick={() => {
              setZoom(0.9);
              setMoved({});
            }}
          >
            <Focus size={15} />
          </button>
          <button
            aria-label="Zoom in"
            onClick={() => setZoom(Math.min(1.8, zoom + 0.15))}
          >
            <ZoomIn size={15} />
          </button>
          <button onClick={() => setExpanded(!expanded)}>
            {expanded ? "Focus view" : "Show all entities"}
          </button>
        </div>
      </div>
      <div
        className={ui.graphViewport}
        role="region"
        aria-label="Candidate temporal chain graph"
        tabIndex={0}
      >
        <svg
          ref={svg}
          viewBox={`0 0 ${width} ${height}`}
          width={width * zoom}
          height={height * zoom}
          role="img"
          aria-label="Directed entity graph. Drag cards to arrange; select a connection to inspect evidence."
          onPointerMove={(e) => {
            const currentDrag = drag.current;
            if (!currentDrag) return;
            const p = point(e);
            if (p)
              setMoved((old) => ({
                ...old,
                [currentDrag.node]: {
                  x: Math.max(
                    15,
                    Math.min(
                      width - 220,
                      currentDrag.x + p.x - currentDrag.px,
                    ),
                  ),
                  y: Math.max(
                    60,
                    Math.min(
                      height - 45,
                      currentDrag.y + p.y - currentDrag.py,
                    ),
                  ),
                },
              }));
          }}
          onPointerUp={() => {
            drag.current = null;
          }}
          onPointerCancel={() => {
            drag.current = null;
          }}
        >
          <defs>
            <marker
              id={marker}
              viewBox="0 0 10 10"
              refX="9"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto"
            >
              <path d="M0 0L10 5L0 10z" fill="var(--graph-edge)" />
            </marker>
            <pattern
              id={`${marker}-dots`}
              width="24"
              height="24"
              patternUnits="userSpaceOnUse"
            >
              <circle cx="1" cy="1" r=".7" fill="var(--graph-pattern)" />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill={`url(#${marker}-dots)`} />
          {[
            [-1, "INCOMING CONNECTIONS"],
            [0, "ALERT ENTITY"],
            [1, "CONNECTED ENTITIES"],
            [2, "EXTENDED CONTEXT"],
          ]
            .filter(([lane]) => laneKeys.includes(Number(lane)))
            .map(([lane, label]) => (
              <text
                key={lane}
                x={35 + laneKeys.indexOf(Number(lane)) * 275}
                y="32"
                className={ui.laneLabel}
              >
                {label}
              </text>
            ))}
          {links.map(({ event: e, count, relations }) => {
            const a = positions.get(e.src)!;
            const b = positions.get(e.dst)!;
            const active = focused === e.src || focused === e.dst;
            const isSelected =
              selected?.src === e.src && selected?.dst === e.dst;
            const right = b.x >= a.x;
            const sx = a.x + (right ? 214 : 0),
              tx = b.x + (right ? 0 : 214);
            const d =
              e.src === e.dst
                ? `M${a.x + 160},${a.y - 36}C${a.x + 230},${a.y - 100} ${a.x - 20},${a.y - 100} ${a.x + 40},${a.y - 36}`
                : `M${sx},${a.y}C${(sx + tx) / 2},${a.y} ${(sx + tx) / 2},${b.y} ${tx},${b.y}`;
            return (
              <g
                key={`${e.src}-${e.dst}`}
                opacity={active ? 1 : 0.25}
                className={ui.edgeGroup}
              >
                <path
                  d={d}
                  className={`${ui.edge} ${isSelected ? ui.selectedEdge : ""}`}
                  markerEnd={`url(#${marker})`}
                />
                <path
                  d={d}
                  className={ui.edgeHit}
                  role="button"
                  tabIndex={0}
                  aria-label={`Inspect ${e.id} ${e.relation_name}, ${count} grouped events`}
                  onClick={() => {
                    setSelected(e);
                    select(e);
                  }}
                  onKeyDown={(key) => {
                    if (key.key === "Enter" || key.key === " ") {
                      key.preventDefault();
                      setSelected(e);
                      select(e);
                    }
                  }}
                >
                  <title>
                    {[...relations].join(", ")} · {count} observations ·
                    strongest event evidence {(e.evidence || 0).toFixed(2)}
                  </title>
                </path>
                {active && e.src !== e.dst && (
                  <g
                    className={ui.edgeLabel}
                    transform={`translate(${(sx + tx) / 2 - 39},${(a.y + b.y) / 2 - 10})`}
                  >
                    <rect width="78" height="20" rx="7" />
                    <text x="39" y="13" textAnchor="middle">
                      {relations.size > 1
                        ? `${count} events`
                        : `${e.relation_name}${count > 1 ? ` ×${count}` : ""}`}
                    </text>
                  </g>
                )}
              </g>
            );
          })}
          {shown.map((n) => {
            const p = positions.get(n.id)!;
            return (
              <g
                key={n.id}
                transform={`translate(${p.x},${p.y - 36})`}
                className={`${ui.entityCard} ${n.id === seed ? ui.seedCard : ""} ${focused === n.id ? ui.focusCard : ""}`}
                role="button"
                tabIndex={0}
                aria-label={`Focus entity ${title(n)}`}
                onClick={() => setFocused(n.id)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    setFocused(n.id);
                  }
                }}
                onPointerDown={(e) => {
                  setFocused(n.id);
                  const matrix = svg.current?.getScreenCTM()?.inverse();
                  if (!matrix) return;
                  const q = new DOMPoint(e.clientX, e.clientY).matrixTransform(
                    matrix,
                  );
                  drag.current = {
                    node: n.id,
                    x: p.x,
                    y: p.y,
                    px: q.x,
                    py: q.y,
                  };
                  svg.current?.setPointerCapture(e.pointerId);
                }}
              >
                <rect width="214" height="72" rx="12" />
                <rect
                  x="12"
                  y="17"
                  width="35"
                  height="35"
                  rx="9"
                  className={ui.iconTile}
                />
                <EntityIcon node={n} x={20} y={25} size={19} />
                <text x="58" y="27" className={ui.entityTitle}>
                  {short(title(n))}
                </text>
                <text x="58" y="46" className={ui.entityMeta}>
                  {kind(n)} · #{n.id}
                </text>
                <circle
                  cx="196"
                  cy="17"
                  r="3"
                  fill={
                    n.id === seed ? "var(--app-accent)" : n.alert ? "var(--app-danger)" : "var(--app-success)"
                  }
                />
                <title>
                  {title(n)} · node evidence {n.score.toFixed(3)}. Drag to
                  reposition.
                </title>
              </g>
            );
          })}
        </svg>
      </div>
      <div className={ui.graphHint}>
        <span>
          <i /> Selected alert
        </span>
        <span>Arrows = recorded direction · repeated events bundled</span>
        <span>Drag cards · scroll to pan</span>
      </div>
      {selected && (
        <div className={ui.connectionSummary}>
          <GitBranch size={17} />
          <span>
            <b>{selected.relation_name}</b> #{selected.src} → #{selected.dst}
          </span>
          <span>
            Evidence <b>{selected.evidence?.toFixed(3)}</b>
          </span>
        </div>
      )}
    </div>
  );
}

export function AttackChainBuilder({
  node,
  anchors,
  select,
}: {
  node: number;
  anchors: Observation[];
  select: (e: Observation) => void;
}) {
  const [anchor, setAnchor] = useState(anchors[0]?.idx);
  const [direction, setDirection] = useState("both");
  const [depth, setDepth] = useState(5);
  const [data, setData] = useState<Construction | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [revision, setRevision] = useState(0);
  const [pathIndex, setPathIndex] = useState(0);
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [reverse, setReverse] = useState(false);
  const [speed, setSpeed] = useState(1200);
  const [reduceMotion, setReduceMotion] = useState(false);
  const pathCanvas = useRef<HTMLDivElement>(null);
  const requestKey = JSON.stringify([node, anchor, direction, depth, revision]);
  const [activeRequest, setActiveRequest] = useState(requestKey);
  // Reset the playback with its input change before rendering stale chain steps.
  if (activeRequest !== requestKey) {
    setActiveRequest(requestKey);
    setLoading(true);
    setData(null);
    setError("");
    setPlaying(false);
    setStep(0);
    setPathIndex(0);
    setReverse(false);
  }
  useEffect(() => {
    const media = window.matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => setReduceMotion(media.matches);
    update();
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);
  useEffect(() => {
    const controller = new AbortController();
    fetch(
      `/api/cadets/nodes/${node}/construction?anchor_index=${anchor}&direction=${direction}&max_steps=${depth}`,
      { signal: controller.signal },
    )
      .then(async (r) => {
        if (!r.ok) {
          const body = await r.json();
          throw new Error(body.detail || "Chain construction failed");
        }
        return r.json();
      })
      .then((result) => { if (!controller.signal.aborted) setData(result); })
      .catch((e) => {
        if (!controller.signal.aborted) setError(e.message);
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [node, anchor, direction, depth, revision]);
  const path = data?.paths[pathIndex];
  useEffect(() => {
    const canvas = pathCanvas.current;
    const active = canvas?.querySelector<HTMLElement>(
      `[data-chain-step="${step}"]`,
    );
    if (!canvas || !active) return;
    const left =
      active.getBoundingClientRect().left -
      canvas.getBoundingClientRect().left +
      canvas.scrollLeft;
    canvas.scrollTo({
      left: Math.max(0, left - 22),
      behavior: reduceMotion ? "auto" : "smooth",
    });
  }, [step, path, reduceMotion]);
  useEffect(() => {
    if (!playing || !path) return;
    const timer = window.setTimeout(() => {
      const next = step + (reverse ? -1 : 1);
      if (next < 0 || next >= path.events.length) {
        setPlaying(false);
        return;
      }
      setStep(next);
      select(path.events[next]);
    }, speed);
    return () => window.clearTimeout(timer);
  }, [playing, path, step, reverse, speed, select]);
  function jump(next: number) {
    if (!path) return;
    setPlaying(false);
    setStep(next);
    select(path.events[next]);
  }
  function start(backward: boolean) {
    if (!path) return;
    setReverse(backward);
    const first = backward ? path.events.length - 1 : 0;
    setStep(first);
    select(path.events[first]);
    setPlaying(!reduceMotion);
  }
  const current = path?.events[step];
  const entities = new Map(data?.nodes.map((n) => [n.id, n]) || []);
  return (
    <section className={ui.builder} aria-label="Attack chain construction">
      <div className={ui.builderHeading}>
        <span className={ui.builderIcon}>
          <GitBranch size={20} />
        </span>
        <div>
          <p>CONSTRUCT · TRACE · EXPLAIN</p>
          <h2>Attack-chain construction</h2>
        </div>
        <span className={ui.hypothesis}>Candidate paths</span>
      </div>
      <p className={ui.builderIntro}>
        Start with an anomalous event. Connect recorded events through the same
        entity, in time order. Replay a path to see each connection.
      </p>
      <div className={ui.builderControls}>
        <label>
          Starting evidence
          <select
            aria-label="Starting evidence"
            value={anchor}
            onChange={(e) => setAnchor(Number(e.target.value))}
          >
            {anchors.map((e, i) => (
              <option key={e.id} value={e.idx}>
                #{i + 1} · {e.relation_name} · {clock(e.time)}
              </option>
            ))}
          </select>
        </label>
        <label>
          Trace direction
          <select
            aria-label="Trace direction"
            value={direction}
            onChange={(e) => setDirection(e.target.value)}
          >
            <option value="both">Before + after</option>
            <option value="backward">Backtrace only</option>
            <option value="forward">Forward only</option>
          </select>
        </label>
        <label>
          Max events
          <select
            aria-label="Maximum chain events"
            value={depth}
            onChange={(e) => setDepth(Number(e.target.value))}
          >
            {[3, 5, 7].map((n) => (
              <option key={n}>{n}</option>
            ))}
          </select>
        </label>
        <button onClick={() => setRevision(revision + 1)} disabled={loading}>
          <Search size={15} />
          Construct chains
        </button>
      </div>
      {loading && (
        <div role="status" className={ui.state}>
          <Loader2 size={18} className={ui.spin} />
          Connecting observed events…
        </div>
      )}
      {error && (
        <div role="alert" className={ui.state}>
          <TriangleAlert size={18} />
          {error}
        </div>
      )}
      {data && path && (
        <>
          <div
            className={ui.pathTabs}
            role="group"
            aria-label="Constructed candidate paths"
          >
            {data.paths.map((p, i) => (
              <button
                key={p.id}
                aria-pressed={pathIndex === i}
                onClick={() => {
                  setPathIndex(i);
                  setStep(0);
                  setPlaying(false);
                  setReverse(false);
                  select(p.events[0]);
                }}
              >
                <GitBranch size={13} />
                Path {i + 1}
                <span>{p.events.length} events</span>
              </button>
            ))}
          </div>
          <div className={ui.playback}>
            <div>
              <button
                aria-label={playing ? "Pause replay" : "Play chain"}
                onClick={() => {
                  if (playing) {
                    setPlaying(false);
                    return;
                  }
                  const exhausted = reverse
                    ? step === 0
                    : step === path.events.length - 1;
                  const next = exhausted
                    ? reverse
                      ? path.events.length - 1
                      : 0
                    : step;
                  setStep(next);
                  select(path.events[next]);
                  setPlaying(true);
                }}
                disabled={(reduceMotion && !playing) || path.events.length < 2}
              >
                {playing ? <Pause size={16} /> : <Play size={16} />}
              </button>
              <button
                aria-label="Backtrace chain"
                onClick={() => start(true)}
                disabled={reduceMotion || path.events.length < 2}
              >
                <ArrowLeft size={16} />
              </button>
              <button
                aria-label="Previous chain event"
                disabled={step === 0}
                onClick={() => jump(step - 1)}
              >
                <ChevronLeft size={16} />
              </button>
              <button
                aria-label="Next chain event"
                disabled={step === path.events.length - 1}
                onClick={() => jump(step + 1)}
              >
                <ChevronRight size={16} />
              </button>
              <button
                aria-label="Reset chain replay"
                onClick={() => {
                  setReverse(false);
                  jump(0);
                }}
              >
                <RotateCcw size={14} />
              </button>
            </div>
            <span aria-live="polite">
              {playing && reverse ? "Backtracing" : "Event"}{" "}
              <b>
                {step + 1} / {path.events.length}
              </b>
            </span>
            <label>
              Speed
              <select
                aria-label="Replay speed"
                value={speed}
                onChange={(e) => setSpeed(Number(e.target.value))}
              >
                <option value={2000}>Slow</option>
                <option value={1200}>Normal</option>
                <option value={600}>Fast</option>
              </select>
            </label>
            <button
              aria-label="Export constructed chain"
              onClick={() => save({ ...data, selected_path: path.id })}
            >
              <ArrowDownToLine size={15} />
              Export
            </button>
          </div>
          <div
            ref={pathCanvas}
            className={`${ui.pathCanvas} ${playing ? ui.playing : ""} ${reduceMotion ? ui.noMotion : ""}`}
            aria-label="Constructed event path"
          >
            <div className={ui.pathStrip}>
              {path.events.map((e, i) => (
                <div
                  key={e.id}
                  data-chain-step={i}
                  className={`${ui.pathStep} ${i === step ? ui.activeStep : ""} ${i < step ? ui.completeStep : ""}`}
                >
                  <button
                    aria-label={`Inspect chain event ${i + 1} ${e.relation_name}`}
                    aria-current={i === step ? "step" : undefined}
                    onClick={() => jump(i)}
                  >
                    <span className={ui.stepNumber}>{i + 1}</span>
                    <span className={ui.pathEntity}>
                      {entities.has(e.src) ? (
                        <EntityIcon node={entities.get(e.src)!} />
                      ) : (
                        <Terminal size={18} />
                      )}
                      <b>
                        {short(
                          entities.has(e.src)
                            ? title(entities.get(e.src)!)
                            : `#${e.src}`,
                        )}
                      </b>
                      <small>#{e.src}</small>
                    </span>
                    <span className={ui.pathRelation}>
                      <i />
                      <ArrowRight size={17} />
                      <b>{e.relation_name}</b>
                      {i === path.anchor_step && <em>STARTING EVIDENCE</em>}
                    </span>
                    <span className={ui.pathEntity}>
                      {entities.has(e.dst) ? (
                        <EntityIcon node={entities.get(e.dst)!} />
                      ) : (
                        <FileText size={18} />
                      )}
                      <b>
                        {short(
                          entities.has(e.dst)
                            ? title(entities.get(e.dst)!)
                            : `#${e.dst}`,
                        )}
                      </b>
                      <small>#{e.dst}</small>
                    </span>
                    <time>{clock(e.time)} UTC</time>
                  </button>
                  {i < path.events.length - 1 && (
                    <div className={ui.join}>
                      <span>same entity #{e.dst}</span>
                      <ArrowRight size={17} />
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
          <input
            className={ui.scrubber}
            type="range"
            aria-label="Chain event position"
            min="0"
            max={path.events.length - 1}
            value={step}
            onChange={(e) => jump(Number(e.target.value))}
            disabled={path.events.length < 2}
          />
          {current && (
            <div className={ui.story} aria-live="polite">
              <span className={ui.storyIndex}>
                {String(step + 1).padStart(2, "0")}
              </span>
              <div>
                <h3>
                  {current.relation_name}:{" "}
                  {short(
                    entities.has(current.src)
                      ? title(entities.get(current.src)!)
                      : `#${current.src}`,
                  )}{" "}
                  →{" "}
                  {short(
                    entities.has(current.dst)
                      ? title(entities.get(current.dst)!)
                      : `#${current.dst}`,
                  )}
                </h3>
                <p>
                  {clock(current.time)} UTC · calibrated event evidence{" "}
                  <b>{current.evidence?.toFixed(3)}</b>
                  {step === path.anchor_step
                    ? " · selected starting evidence"
                    : step < path.anchor_step
                      ? " · recorded before the starting evidence"
                      : " · recorded after the starting evidence"}
                </p>
                <p>
                  {step > 0
                    ? `Linked because the previous event ends at entity #${current.src}, where this event starts. `
                    : "The first recorded event in this candidate path. "}
                  This explains the connection; it does not establish malicious
                  intent.
                </p>
              </div>
            </div>
          )}
          <div className={ui.constructionFoot}>
            <span>
              {data.paths.length} candidates · endpoint + time checks ·
              benchmark labels excluded
            </span>
            <span>
              Evidence sum {path.evidence_sum.toFixed(2)}{" "}
              <small>(ranking only)</small>
            </span>
          </div>
          {path.events.length === 1 && (
            <p className={ui.boundNote}>
              No compatible extension was found within the search bounds. The
              starting event is shown on its own; additional attack stages are
              not invented.
            </p>
          )}
          {data.search_pruned && (
            <p className={ui.boundNote}>
              Bounded search: high-branching alternatives were pruned. These are
              candidate paths, not an exhaustive reconstruction.
            </p>
          )}
          {reduceMotion && (
            <p className={ui.boundNote}>
              Reduced motion is enabled. Use the step buttons or scrubber to
              trace the chain.
            </p>
          )}
        </>
      )}
    </section>
  );
}
