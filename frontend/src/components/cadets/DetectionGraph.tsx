"use client";

import {
  ArrowDownToLine,
  ArrowRight,
  CircleCheck,
  Crosshair,
  Database,
  FileText,
  Focus,
  Globe2,
  ZoomIn,
  ZoomOut,
  Loader2,
  Network,
  RotateCcw,
  Search,
  ShieldAlert,
  SlidersHorizontal,
  Terminal,
  X,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import styles from "./detection.module.css";
import { useTheme } from "@/lib/theme-context";
import { graphStyles } from "./graph-theme";

type Entity = {
  id: number;
  score: number;
  alert: boolean;
  benchmark_member: boolean | null;
  ip: string | null;
  prediction: string;
  attributes: Record<string, unknown>;
};

type Event = {
  id: string;
  src: number;
  dst: number;
  time: number;
  relation_name: string;
  evidence: number;
};

type Dataset = {
  id: string;
  name: string;
  status: string;
  stage?: string;
  progress: number;
  events?: number;
  nodes?: number;
  error?: string;
  scoring?: string;
};

type EntityOptions = {
  items: { label: string; count: number; alert: number }[];
  available_kinds: string[];
  truncated: boolean;
};

type Graph = {
  dataset: Dataset;
  nodes: Entity[];
  events: Event[];
  counts: { predicted_malicious: number; predicted_benign: number };
  threshold: number;
  scope: string;
  ip_scope: string;
  truncated: boolean;
  seeds: number[];
};

type EdgeBundle = {
  src: number;
  dst: number;
  count: number;
  evidence: number;
  relations: string[];
};

async function readJSON<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : `Request failed (${response.status})`,
    );
  }
  return data;
}

function entityLabel(node: Entity) {
  return String(
    node.ip ||
      node.attributes.cmd_line ||
      node.attributes.path ||
      node.attributes.host ||
      node.attributes.hostname ||
      node.attributes.name ||
      `${node.attributes.type || "entity"} #${node.id}`,
  );
}

function shortLabel(node: Entity) {
  const text = entityLabel(node);
  return text.length <= 28 ? text : `…${text.slice(-25)}`;
}

function entityKind(node: Entity) {
  const raw = String(node.attributes.type || "").toLowerCase();
  if (node.ip || raw.includes("netflow") || raw.includes("socket")) return "network";
  if (raw.includes("file")) return "file";
  if (raw.includes("host")) return "host";
  if (raw.includes("subject") || raw.includes("process")) return "process";
  return "entity";
}

function EntityGlyph({ node, size = 18 }: { node: Entity; size?: number }) {
  const kind = entityKind(node);
  if (kind === "network") return <Globe2 size={size} />;
  if (kind === "file") return <FileText size={size} />;
  if (kind === "host") return <Database size={size} />;
  return <Terminal size={size} />;
}

function bundleEdges(events: Event[]) {
  const grouped = new Map<string, EdgeBundle>();
  for (const event of events) {
    const key = `${event.src}:${event.dst}`;
    const old = grouped.get(key);
    if (!old) {
      grouped.set(key, {
        src: event.src,
        dst: event.dst,
        count: 1,
        evidence: event.evidence || 0,
        relations: [event.relation_name],
      });
      continue;
    }
    old.count += 1;
    old.evidence = Math.max(old.evidence, event.evidence || 0);
    if (!old.relations.includes(event.relation_name)) old.relations.push(event.relation_name);
  }
  return [...grouped.values()];
}

let fcoseRegistered = false;

export function DetectionGraph({ investigate }: { investigate: (node: number) => void }) {
  const { theme } = useTheme();
  const graphContainer = useRef<HTMLDivElement>(null);
  const cyRef = useRef<any>(null);
  const selectedRef = useRef<number | null>(null);

  const [data, setData] = useState<Graph | null>(null);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [optionsResponse, setOptionsResponse] = useState<{ key: string; data: EntityOptions } | null>(null);

  const [dataset, setDataset] = useState("cadets");
  const [kind, setKind] = useState("ip");
  const [value, setValue] = useState("");
  const [filter, setFilter] = useState("all");
  const [hops, setHops] = useState(2);

  // Selection is inspection-only. Anchor is the explicit request to refetch a focused graph.
  const [selected, setSelected] = useState<number | null>(null);
  const [anchor, setAnchor] = useState<number | null>(null);

  const [search, setSearch] = useState("");
  const [threatsOnly, setThreatsOnly] = useState(false);
  const [settledGraphRequest, setSettledGraphRequest] = useState("");
  const [error, setError] = useState("");
  const [graphReady, setGraphReady] = useState(false);

  const [uploading, setUploading] = useState(false);
  const [upload, setUpload] = useState<Dataset | null>(null);
  const [uploadError, setUploadError] = useState("");

  const optionsKey = JSON.stringify([dataset, kind, filter]);
  const options = optionsResponse?.key === optionsKey ? optionsResponse.data : null;
  const graphRequest = JSON.stringify([dataset, kind, value, filter, anchor, hops]);
  const loading = settledGraphRequest !== graphRequest;

  const shown = data?.nodes || [];
  const visibleEvents = useMemo(() => {
    if (!data) return [];
    const ids = new Set(data.nodes.map((node) => node.id));
    return data.events.filter((event) => ids.has(event.src) && ids.has(event.dst));
  }, [data]);
  const edges = useMemo(() => bundleEdges(visibleEvents), [visibleEvents]);
  const selectedNode = useMemo(
    () => shown.find((node) => node.id === selected) || null,
    [shown, selected],
  );
  const selectedDegree = useMemo(() => {
    if (selected === null) return 0;
    const neighbors = new Set<number>();
    for (const event of visibleEvents) {
      if (event.src === selected) neighbors.add(event.dst);
      if (event.dst === selected) neighbors.add(event.src);
    }
    return neighbors.size;
  }, [selected, visibleEvents]);

  useEffect(() => {
    readJSON<{ items: Dataset[] }>("/api/graph-datasets")
      .then((result) => {
        setDatasets(result.items);
        const pending = result.items.find((item) => ["queued", "processing"].includes(item.status));
        if (pending) setUpload(pending);
      })
      .catch((reason) => setError(reason.message));
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    const params = new URLSearchParams({ kind, q: "", prediction: filter });
    readJSON<EntityOptions>(`/api/graph-datasets/${dataset}/entities?${params}`, {
      signal: controller.signal,
    })
      .then((result) => {
        if (controller.signal.aborted) return;
        if (!result.available_kinds.includes(kind) && result.available_kinds.length) {
          setKind(result.available_kinds[0]);
          return;
        }
        setOptionsResponse({ key: optionsKey, data: result });
      })
      .catch((reason) => {
        if (!controller.signal.aborted) setError(reason.message);
      });
    return () => controller.abort();
  }, [dataset, kind, filter, optionsKey]);

  useEffect(() => {
    const controller = new AbortController();
    const params = new URLSearchParams({
      kind,
      value,
      prediction: filter,
      limit: "16",
      hops: String(hops),
    });
    if (anchor !== null) params.set("node", String(anchor));

    readJSON<Graph>(`/api/graph-datasets/${dataset}/graph?${params}`, {
      signal: controller.signal,
    })
      .then((result) => {
        if (controller.signal.aborted) return;
        setError("");
        setData(result);
        setSelected((current) =>
          current !== null && result.nodes.some((node) => node.id === current) ? current : null,
        );
      })
      .catch((reason) => {
        if (!controller.signal.aborted) setError(reason.message);
      })
      .finally(() => {
        if (!controller.signal.aborted) setSettledGraphRequest(graphRequest);
      });
    return () => controller.abort();
  }, [dataset, kind, value, filter, anchor, hops, graphRequest]);

  useEffect(() => {
    if (!upload || !["queued", "processing"].includes(upload.status)) return;
    const controller = new AbortController();
    const timer = window.setInterval(() => {
      readJSON<Dataset>(`/api/graph-datasets/${upload.id}`, { signal: controller.signal })
        .then((result) => {
          setUpload(result);
          if (result.status === "ready") {
            setDatasets((old) => [...old.filter((item) => item.id !== result.id), result]);
            setDataset(result.id);
            setAnchor(null);
            setSelected(null);
          }
          if (result.status === "failed") setUploadError(result.error || "Scoring failed");
        })
        .catch((reason) => {
          if (!controller.signal.aborted) setUploadError(reason.message);
        });
    }, 1000);
    return () => {
      window.clearInterval(timer);
      controller.abort();
    };
  }, [upload?.id, upload?.status]);

  const applyPersistentSelection = useCallback((cy: any) => {
    cy.elements().removeClass("dim neighbor activeEdge selectedNode");
    const selectedId = selectedRef.current;
    if (selectedId === null) return;
    const node = cy.getElementById(String(selectedId));
    if (!node || node.empty()) return;
    cy.elements().addClass("dim");
    node.removeClass("dim").addClass("selectedNode");
    node.neighborhood("node").removeClass("dim").addClass("neighbor");
    node.connectedEdges().removeClass("dim").addClass("activeEdge");
  }, []);

  useEffect(() => {
    if (!graphContainer.current || !data) return;
    let disposed = false;
    let instance: any = null;
    let resizeObserver: ResizeObserver | null = null;

    const start = async () => {
      setGraphReady(false);
      const cytoscapeModule = await import("cytoscape");
      const fcoseModule = await import("cytoscape-fcose");
      if (disposed || !graphContainer.current) return;
      const cytoscape = cytoscapeModule.default;
      const fcose = (fcoseModule as { default?: unknown }).default || fcoseModule;
      if (!fcoseRegistered) {
        cytoscape.use(fcose as any);
        fcoseRegistered = true;
      }

      const elements = [
        ...data.nodes.map((node) => ({
          data: {
            id: String(node.id),
            label: shortLabel(node),
            fullLabel: entityLabel(node),
            score: node.score,
            alert: node.alert ? 1 : 0,
            kind: entityKind(node),
          },
          classes: `entity ${entityKind(node)} ${node.alert ? "threat" : "benign"}`,
        })),
        ...edges.map((edge, index) => ({
          data: {
            id: `edge-${edge.src}-${edge.dst}-${index}`,
            source: String(edge.src),
            target: String(edge.dst),
            label: `${edge.relations.slice(0, 2).join(" / ")}${edge.count > 1 ? ` ×${edge.count}` : ""}`,
            evidence: edge.evidence,
          },
        })),
      ];

      instance = cytoscape({
        container: graphContainer.current,
        elements,
        minZoom: 0.35,
        maxZoom: 3.0,
        boxSelectionEnabled: false,
        style: graphStyles() as any,
      });
      cyRef.current = instance;
      resizeObserver = new ResizeObserver(() => {
        if (!disposed) instance.resize();
      });
      resizeObserver.observe(graphContainer.current);
      /* TGDETECT_RUNTIME_EDGE_CLASSIFICATION */
      instance.edges().forEach((edge: any) => {
        const sourceThreat = Boolean(edge.source().data("alert"));
        const targetThreat = Boolean(edge.target().data("alert"));

        edge.removeClass("attackEdge benignEdge mixedEdge");

        if (sourceThreat && targetThreat) {
          edge.addClass("attackEdge");
        } else if (!sourceThreat && !targetThreat) {
          edge.addClass("benignEdge");
        } else {
          edge.addClass("mixedEdge");
        }
      });


      instance.layout({
        name: "fcose",
        quality: "proof",
        randomize: true,
        animate: false,
        fit: true,
        padding: 55,
        nodeDimensionsIncludeLabels: true,
        nodeRepulsion: () => 4500,
        idealEdgeLength: () => 88,
        edgeElasticity: () => 0.55,
        gravity: 0.68,
        numIter: 2200,
        tile: true,
        tilingPaddingVertical: 18,
        tilingPaddingHorizontal: 18,
      } as any).run();

      let lastTap = { id: "", at: 0 };
      const selectNodeOnly = (nodeId: number) => {
        selectedRef.current = nodeId;
        setSelected(nodeId);
        applyPersistentSelection(instance);
      };

      instance.on("tap", "node", (event: any) => {
        const nodeId = Number(event.target.id());
        const now = Date.now();
        const isDouble = lastTap.id === String(nodeId) && now - lastTap.at < 360;
        selectNodeOnly(nodeId);
        if (isDouble) {
          setValue("");
          setAnchor(nodeId);
          lastTap = { id: "", at: 0 };
          return;
        }
        lastTap = { id: String(nodeId), at: now };
      });

      instance.on("tap", (event: any) => {
        if (event.target !== instance) return;
        selectedRef.current = null;
        setSelected(null);
        instance.elements().removeClass("dim neighbor activeEdge selectedNode");
      });

      instance.on("mouseover", "node", (event: any) => {
        if (selectedRef.current !== null) return;
        const node = event.target;
        instance.elements().addClass("dim");
        node.removeClass("dim");
        node.neighborhood("node").removeClass("dim").addClass("neighbor");
        node.connectedEdges().removeClass("dim").addClass("activeEdge");
      });

      instance.on("mouseout", "node", () => {
        if (selectedRef.current === null) {
          instance.elements().removeClass("dim neighbor activeEdge");
          return;
        }
        applyPersistentSelection(instance);
      });

      window.setTimeout(() => {
        if (!disposed && instance) {
          instance.fit(instance.elements(":visible"), 48);
          setGraphReady(true);
          applyPersistentSelection(instance);
        }
      }, 50);
    };

    void start().catch((reason) => {
      if (!disposed) setError(reason instanceof Error ? reason.message : "Unable to render the graph.");
    });
    return () => {
      disposed = true;
      resizeObserver?.disconnect();
      setGraphReady(false);
      if (instance) instance.destroy();
      if (cyRef.current === instance) cyRef.current = null;
    };
  }, [data, edges, applyPersistentSelection]);

  useEffect(() => {
    // Repaint in place after the theme class changes; preserve positions, zoom, and focus.
    let disposed = false;
    const repaint = () => { if (!disposed && cyRef.current) cyRef.current.style(graphStyles()).update(); };
    const frame = requestAnimationFrame(repaint);
    void document.fonts.ready.then(repaint);
    return () => { disposed = true; cancelAnimationFrame(frame); };
  }, [theme, graphReady]);

  useEffect(() => {
    selectedRef.current = selected;
    if (cyRef.current) applyPersistentSelection(cyRef.current);
  }, [selected, applyPersistentSelection]);

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    cy.batch(() => {
      cy.nodes().removeClass("filteredOut");
      cy.edges().removeClass("filteredOut");
      if (!threatsOnly) return;
      cy.nodes().forEach((node: any) => {
        if (!Boolean(node.data("alert"))) node.addClass("filteredOut");
      });
      cy.edges().forEach((edge: any) => {
        if (edge.source().hasClass("filteredOut") || edge.target().hasClass("filteredOut")) {
          edge.addClass("filteredOut");
        }
      });
    });
  }, [threatsOnly, data, graphReady]);

  function fitGraph() {
    const cy = cyRef.current;
    if (!cy) return;
    cy.animate({ fit: { eles: cy.elements(":visible"), padding: 80 } }, { duration: 260 });
  }

  function zoomGraph(direction: "in" | "out") {
    const cy = cyRef.current;
    if (!cy) return;

    const factor = direction === "in" ? 1.38 : 0.72;
    const next = Math.max(
      cy.minZoom(),
      Math.min(cy.maxZoom(), cy.zoom() * factor),
    );

    cy.animate({ zoom: next }, { duration: 150 });
  }

  function centerOn(nodeId: number) {
    const cy = cyRef.current;
    if (!cy) return;
    const node = cy.getElementById(String(nodeId));
    if (!node || node.empty()) return;
    cy.animate(
      { center: { eles: node }, zoom: Math.max(1.05, Math.min(1.45, cy.zoom())) },
      { duration: 420 },
    );
  }

  function locateCurrentGraph() {
    const query = search.trim().toLowerCase();
    if (!query) return;
    const match = shown.find((node) => entityLabel(node).toLowerCase().includes(query));
    if (!match) {
      setError(`No visible entity matches "${search}". Change start entity or context depth.`);
      return;
    }
    setError("");
    selectedRef.current = match.id;
    setSelected(match.id);
    centerOn(match.id);
  }

  function resetFocus() {
    setAnchor(null);
    setValue("");
    selectedRef.current = null;
    setSelected(null);
    setThreatsOnly(false);
  }

  function focusNeighborhood(nodeId: number) {
    setValue("");
    setAnchor(nodeId);
    selectedRef.current = nodeId;
    setSelected(nodeId);
  }

  function changeDataset(id: string) {
    setDataset(id);
    setKind("ip");
    setValue("");
    setFilter("all");
    setAnchor(null);
    selectedRef.current = null;
    setSelected(null);
    setThreatsOnly(false);
  }

  async function uploadFile(file: File) {
    setUploading(true);
    setUploadError("");
    setUpload(null);
    try {
      const result = await readJSON<Dataset>(
        `/api/graph-datasets?name=${encodeURIComponent(file.name)}`,
        {
          method: "POST",
          body: file,
          headers: { "Content-Type": "application/octet-stream" },
        },
      );
      setUpload(result);
    } catch (reason) {
      setUploadError(reason instanceof Error ? reason.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  function exportGraph() {
    if (!data) return;
    const url = URL.createObjectURL(
      new Blob(
        [JSON.stringify({ ...data, selected_node: selected, focus_node: anchor, threats_only: threatsOnly }, null, 2)],
        { type: "application/json" },
      ),
    );
    const link = document.createElement("a");
    link.href = url;
    link.download = "tgdetect-presentation-graph.json";
    link.click();
    URL.revokeObjectURL(url);
  }

  return (
    <section className={styles.workspace} aria-label="Detection graph workspace">
      <div className={styles.heading}>
        <div>
          <p>Entity relationships</p>
          <h2>Relationship map</h2>
          <span>Select to inspect. Hover to reveal context. Double-click only when you want to refocus.</span>
        </div>
        <button onClick={exportGraph} disabled={!data || loading}>
          <ArrowDownToLine size={15} /> Export graph
        </button>
      </div>

      <div className={styles.presentationStats}>
        <div><ShieldAlert size={18} /><span>Flagged entities</span><strong>{data?.counts.predicted_malicious ?? "—"}</strong></div>
        <div><CircleCheck size={18} /><span>Benign context</span><strong>{data?.counts.predicted_benign ?? "—"}</strong></div>
        <div><Network size={18} /><span>Relationships</span><strong>{edges.length || "—"}</strong></div>
        <div><Crosshair size={18} /><span>Graph focus</span><strong>{anchor !== null ? `#${anchor}` : "Overview"}</strong></div>
      </div>

      <div className={styles.presentationControls}>
        <label>Dataset<select value={dataset} onChange={(e) => changeDataset(e.target.value)}>{datasets.filter((item) => item.status === "ready").map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label>
        <label>Start from<select value={value} onChange={(e) => { setValue(e.target.value); setAnchor(null); selectedRef.current = null; setSelected(null); }}><option value="">Highest-risk entity</option>{options?.items.slice(0, 25).map((item) => <option key={item.label} value={item.label}>{item.label}</option>)}</select></label>
        <label>Entity type<select value={kind} onChange={(e) => { setKind(e.target.value); setValue(""); setAnchor(null); selectedRef.current = null; setSelected(null); }}>{[...new Set([kind, ...(options?.available_kinds || [])])].map((k) => <option key={k} value={k}>{k === "ip" ? "IP addresses" : k === "host" ? "Hosts" : k === "process" ? "Processes" : k === "file" ? "Files" : "All entities"}</option>)}</select></label>
        <label>Starting class<select value={filter} onChange={(e) => { setFilter(e.target.value); setValue(""); setAnchor(null); selectedRef.current = null; setSelected(null); }}><option value="all">All classes</option><option value="malicious">Threat first</option><option value="benign">Benign first</option></select></label>
      </div>

      <div className={styles.graphStage}>
        <div className={styles.graphHud}>
          <form className={styles.graphSearch} onSubmit={(e) => { e.preventDefault(); locateCurrentGraph(); }}>
            <Search size={15} />
            <input aria-label="Find an entity in the visible graph" value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Find visible entity" />
            <button type="submit">Locate</button>
          </form>
          <div className={styles.hudActions}>
            <label className={styles.hopControl}><SlidersHorizontal size={14} /> Context<select value={hops} onChange={(e) => setHops(Number(e.target.value))}><option value={1}>1 hop</option><option value={2}>2 hops</option><option value={3}>3 hops</option></select></label>
            <button type="button" className={threatsOnly ? styles.toggleActive : ""} onClick={() => setThreatsOnly((old) => !old)} aria-pressed={threatsOnly}><ShieldAlert size={14} /> Threats only</button>
            <button
              type="button"
              onClick={() => zoomGraph("out")}
              disabled={!graphReady || loading}
              aria-label="Zoom graph out"
            >
              <ZoomOut size={14} />
              Zoom out
            </button>

            <button
              type="button"
              onClick={() => zoomGraph("in")}
              disabled={!graphReady || loading}
              aria-label="Zoom graph in"
            >
              <ZoomIn size={14} />
              Zoom in
            </button>

            <button type="button" onClick={fitGraph} disabled={!graphReady || loading}><Focus size={14} /> Fit</button>
            <button type="button" onClick={resetFocus} disabled={anchor === null && selected === null}><RotateCcw size={14} /> Reset focus</button>
          </div>
        </div>

        <div className={styles.graphViewportShell}>
          {loading ? (
            <div className={styles.graphLoading} role="status"><Loader2 size={22} className={styles.graphSpinner} />Loading entity relationships…</div>
          ) : error && !data ? (
            <div className={styles.graphLoading} role="alert">{error}</div>
          ) : (
            <div ref={graphContainer} className={styles.cyGraph} aria-label="Interactive TGDetect entity relationship graph" />
          )}

          {!loading && error && data && (
            <div className={styles.inlineGraphError} role="alert">{error}<button type="button" onClick={() => setError("")} aria-label="Dismiss graph message"><X size={13} /></button></div>
          )}

          {!loading && data && shown.length === 0 && <div className={styles.graphEmpty}>No entities match the current graph scope.</div>}

          {selectedNode && data && (
            <aside className={styles.floatingInspector} aria-label="Selected entity details">
              <div className={styles.inspectorTopline}>
                <span className={selectedNode.alert ? styles.inspectorThreatIcon : styles.inspectorBenignIcon}><EntityGlyph node={selectedNode} size={18} /></span>
                <span className={styles.inspectorKicker}>Entity #{selectedNode.id}</span>
                <button type="button" aria-label="Close entity inspector" onClick={() => { selectedRef.current = null; setSelected(null); }}><X size={16} /></button>
              </div>

              <div className={styles.inspectorIdentity}>
                <div><h3>{shortLabel(selectedNode)}</h3><p>{entityKind(selectedNode).toUpperCase()}</p></div>
                <span className={selectedNode.alert ? styles.threatBadge : styles.benignBadge}>{selectedNode.alert ? "Flagged" : "Benign"}</span>
              </div>

              <div className={styles.scoreBlock}>
                <div><span>Evidence score</span><strong>{selectedNode.score.toFixed(3)}</strong></div>
                <div><span>Threshold</span><strong>{data.threshold.toFixed(3)}</strong></div>
                <div><span>Neighbors</span><strong>{selectedDegree}</strong></div>
              </div>

              <div className={styles.inspectorFacts}>
                <div><span>Remote IP</span><b>{selectedNode.ip || "Not recorded"}</b></div>
                <div><span>Prediction</span><b>{selectedNode.alert ? "Malicious" : "Benign"}</b></div>
                <div><span>Current focus</span><b>{anchor === selectedNode.id ? "Focused" : "Overview graph"}</b></div>
              </div>

              <div className={styles.inspectorActions}>
                <button type="button" onClick={() => focusNeighborhood(selectedNode.id)}><Crosshair size={15} />Focus neighborhood</button>
                {dataset === "cadets" && <button type="button" className={styles.inspectorPrimary} onClick={() => investigate(selectedNode.id)}>Investigate entity<ArrowRight size={15} /></button>}
              </div>

              <details className={styles.inspectorDetails}><summary>Observed attributes</summary><pre>{JSON.stringify(selectedNode.attributes, null, 2)}</pre></details>
            </aside>
          )}
        </div>

        <div className={styles.graphLegend}>
          <span><i className={styles.legendThreat} />Flagged</span>
          <span><i className={styles.legendBenign} />Benign</span>
          <span>
            <i className={styles.legendAttackEdge} />
            Link between flagged entities
          </span>
          <span><i className={styles.legendSelected} />Selected</span>
          <span>Hover = reveal neighbors</span><span>Click = inspect</span><span>Double-click = focus neighborhood</span>
        </div>
      </div>

      <p className={styles.presentationNote}>{anchor !== null ? `Focused on entity #${anchor}.` : "Showing the selected dataset neighborhood."} Flagged and benign classes are model predictions. Lines represent recorded relationships.</p>

      <details className={styles.dataTools}>
        <summary><Database size={14} />Dataset tools</summary>
        <div>
          <label className={styles.uploadButton}>{uploading ? "Uploading…" : "Upload logs / dataset"}<input type="file" accept=".json,.jsonl,.ndjson,.csv" disabled={uploading || (!!upload && ["queued", "processing"].includes(upload.status))} onChange={(e) => { const file = e.target.files?.[0]; if (file) void uploadFile(file); e.target.value = ""; }} /></label>
          {upload && <div className={styles.uploadStatus}><span>{upload.name} · {upload.stage || upload.status}</span><progress max={1} value={upload.progress} /></div>}
          {uploadError && <p className={styles.uploadError}>{uploadError}</p>}
        </div>
        {data?.dataset.scoring && <p className={styles.presentationNote}>{data.dataset.scoring}</p>}
      </details>
    </section>
  );
}
