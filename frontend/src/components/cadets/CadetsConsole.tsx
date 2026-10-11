"use client";

import { useEffect, useState } from "react";
import {
  Activity,
  ArrowDownToLine,
  ArrowRight,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Cpu,
  Database,
  GitBranch,
  LayoutDashboard,
  ListFilter,
  Loader2,
  Network,
  Play,
  Search,
  Sparkles,
  TriangleAlert,
} from "lucide-react";
import styles from "./console.module.css";
import { AttackGraph, AttackChainBuilder } from "./AttackInvestigation";
import { DetectionGraph } from "./DetectionGraph";
import { BrandMark } from "./BrandMark";
import { ThemeControl } from "./ThemeControl";
import { AmbientBackdrop } from "./AmbientBackdrop";

type Node = {
  id: number;
  score: number;
  alert: boolean;
  benchmark_member: boolean;
  attributes: Record<string, unknown>;
};
type Event = {
  id: string;
  split: string;
  idx: number;
  src: number;
  dst: number;
  time: number;
  relation: number;
  relation_name: string;
  loss: number | null;
  evidence: number | null;
};
type Summary = {
  dataset: string;
  model: string;
  model_config?: {
    event_dim?: number;
    window?: number;
    gat_layers?: number;
    short_slots?: number;
    short_readout?: string;
    temporal?: string;
    num_relations?: number;
    pair_head?: boolean;
    token_config?: { two_hash?: boolean };
  };
  calibration?: { window_seconds?: number };
  total_events: number;
  events: Record<string, number>;
  scored_nodes: number;
  alert_nodes: number;
  metrics: Record<string, number>;
  threshold: number;
  epochs: number;
  batch_size: number;
  seed: number;
  parameter_count: number;
  checkpoint_epoch: number;
  checkpoint_sha256: string;
  validation_ce: number;
  test_ce: number;
  limitations: string[];
  unseen_ground_truth: number[];
};
type Explanation = {
  node: Node;
  window_start: number;
  contributors: Event[];
  reconstructed_score: number;
  single_event_channel: number;
  repeated_event_channel: number;
  threshold: number;
  rule: string;
  meaning: string;
  limitations: string;
};
type Chain = {
  nodes: Node[];
  events: Event[];
  truncated: boolean;
  interpretation: string;
  strategy: string;
  labels_used: boolean;
};
type Replay = {
  events: number;
  device: string;
  setup_seconds: number;
  batch_ms: number[];
  max_absolute_error_vs_saved_gpu_scores: number | null;
  scope: string;
  scores: {
    idx: number;
    relation_ce: number;
    pair_anomaly: number;
    combined_loss: number;
  }[];
};
const number = (n: number) => new Intl.NumberFormat("en-US").format(n);
const pct = (n: number) => `${(n * 100).toFixed(2)}%`;
const time = (n: number) =>
  new Date(n * 1000).toLocaleString("en-GB", {
    timeZone: "UTC",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
const name = (n: Node) =>
  String(
    n.attributes.path ||
      n.attributes.cmd_line ||
      n.attributes.remote_ip ||
      `${n.attributes.type || "Entity"} #${n.id}`,
  );

async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`/api/cadets${path}`, options);
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(
      typeof error.detail === "string"
        ? error.detail
        : `Request failed (${response.status})`,
    );
  }
  return response.json();
}
function useResource<T>(path: string | null): { data?: T; error?: string; loading: boolean } {
  const [state, setState] = useState<{
    path: string | null;
    data?: T;
    error?: string;
    loading: boolean;
  }>({ path: null, loading: false });
  useEffect(() => {
    if (!path) return;
    const controller = new AbortController();
    api<T>(path, { signal: controller.signal })
      .then((data) => { if (!controller.signal.aborted) setState({ path, data, loading: false }); })
      .catch((error) => {
        if (!controller.signal.aborted)
          setState({ path, error: error.message, loading: false });
      });
    return () => controller.abort();
  }, [path]);
  return path === null ? { loading: false } : state.path === path ? state : { loading: true };
}
function Notice({ loading, error }: { loading: boolean; error?: string }) {
  if (error)
    return (
      <div role="alert" className={styles.error}>
        <TriangleAlert size={18} />
        {error}
      </div>
    );
  if (loading)
    return (
      <div role="status" className={styles.loading}>
        <Loader2 size={18} className={styles.spin} />
        Loading workspace data…
      </div>
    );
  return null;
}
function download(data: unknown, filename: string) {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }),
  );
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

export function CadetsConsole() {
  const [tab, setTab] = useState("overview");
  useEffect(() => { window.scrollTo({ top: 0, behavior: "instant" }); }, [tab]);
  const summary = useResource<Summary>("/summary");
  const [chosenNode, setSelectedNode] = useState<number | null>(null);
  const [nodePage, setNodePage] = useState(0);
  const nodes = useResource<{ total: number; items: Node[] }>(
    `/nodes?limit=12&offset=${nodePage * 12}`,
  );
  const selectedNode = chosenNode ?? nodes.data?.items[0]?.id ?? null;
  const [hops, setHops] = useState(2);
  const explanation = useResource<Explanation>(
    selectedNode === null || tab !== "investigation"
      ? null
      : `/nodes/${selectedNode}/explanation`,
  );
  const chain = useResource<Chain>(
    selectedNode === null || tab !== "investigation"
      ? null
      : `/nodes/${selectedNode}/chain?hops=${hops}&limit=60`,
  );
  const [split, setSplit] = useState("test");
  const [eventPage, setEventPage] = useState(0);
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState("");
  const [selectedEvent, setSelectedEvent] = useState<Event | null>(null);
  const events = useResource<{ items: Event[]; total: number }>(
    tab !== "events"
      ? null
      : `/events?split=${split}&limit=20&offset=${eventPage * 20}${filter ? `&node=${filter}` : ""}`,
  );
  const [replay, setReplay] = useState<Replay | null>(null);
  const [replaying, setReplaying] = useState(false);
  const [replayError, setReplayError] = useState("");
  const [benchmark, setBenchmark] = useState(false);
  const s = summary.data;
  const config = s?.model_config;
  const configured = (value: number | undefined) => value === undefined ? "Not exposed" : number(value);
  const calibrationWindow = s?.calibration?.window_seconds;
  async function runReplay() {
    setReplaying(true);
    setReplayError("");
    try {
      setReplay(
        await api<Replay>("/replay", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ split: "test", batches: 1 }),
        }),
      );
    } catch (e) {
      setReplayError(e instanceof Error ? e.message : "Replay failed");
    } finally {
      setReplaying(false);
    }
  }
  const tabs = [
    { id: "overview", label: "Overview", icon: LayoutDashboard },
    { id: "detection", label: "Detection graph", icon: Network },
    { id: "investigation", label: "Investigate alerts", icon: GitBranch },
    { id: "events", label: "Event explorer", icon: Database },
    { id: "model", label: "Model & validation", icon: Cpu },
  ];
  return (
    <div className={styles.console}>
      <AmbientBackdrop />
      <aside className={styles.sidebar}>
        <a className={styles.brand} href="/">
          <span className={styles.brandIcon}>
            <BrandMark size={35} />
          </span>
          <span>
            TGDetect<small>Investigation workspace</small>
          </span>
        </a>
        <div className={styles.workspace}>
          <span className={styles.dot} />
          {s?.dataset || "Dataset"}<small>{s?.model || "Connecting to model…"}</small>
        </div>
        <p className={styles.navLabel}>Workspace</p>
        <nav aria-label="Workspace navigation">
          {tabs.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              aria-current={tab === id ? "page" : undefined}
              className={tab === id ? styles.navActive : ""}
              onClick={() => setTab(id)}
            >
              <Icon size={19} />
              {label}
              {id === "investigation" && s && (
                <span className={styles.navCount}>{number(s.alert_nodes)}</span>
              )}
            </button>
          ))}
        </nav>
        <div className={styles.sidebarBottom}>
          <span className={styles.localPill}><Database size={14} />Local workspace</span>
          <p>
            All event times are shown in UTC.
          </p>
        </div>
      </aside>
      <main className={styles.main}>
        <header className={styles.topbar}>
          <div>
            <span className={styles.breadcrumb}>Workspace / </span>
            {tabs.find((t) => t.id === tab)?.label}
          </div>
          <div className={styles.topbarActions}>
          <span className={styles.status}>
            <span className={styles.dot} />
            {s ? "Dataset ready" : summary.error ? "Connection error" : "Connecting"}
          </span>
          <ThemeControl />
          </div>
        </header>
        <div className={styles.content}>
          <div className={styles.heading}>
            <div>
              <p className={styles.eyebrow}>Workspace / {tab === "detection" ? "Relationships" : s?.dataset || "Dataset"}</p>
              <h1>
                {tab === "overview"
                  ? "Overview"
                  : tab === "detection"
                    ? "Detection graph"
                    : tab === "investigation"
                      ? "Alert investigation"
                      : tab === "events"
                        ? "Event explorer"
                        : "Model & validation"}
              </h1>
              <p>
                {tab === "overview"
                  ? "Review detection results and move from an alert to its evidence."
                  : tab === "detection"
                    ? "Inspect flagged and benign entities across recorded relationships."
                    : tab === "investigation"
                      ? "Trace contributing events and reconstruct candidate attack chains."
                      : tab === "events"
                        ? "Find recorded events, inspect their scores, and follow entity activity."
                        : "Review the active checkpoint, evaluation results, and replay verification."}
              </p>
            </div>
            <span className={styles.datasetBadge}>
              <Database size={16} />
              {tab === "detection"
                ? "Log datasets"
                : s?.dataset || "Loading dataset…"}
            </span>
          </div>
          <Notice loading={summary.loading} error={summary.error} />
          {s && (
            <>
              {tab === "overview" && (
                <>
                  <div className={styles.stats}>
                    <Metric
                      label="Prepared events"
                      value={number(s.total_events)}
                      sub="Full train, validation and test corpus"
                      icon={<Database size={19} />}
                    />
                    <Metric
                      label="Scored test entities"
                      value={number(s.scored_nodes)}
                      sub="Node-level evaluation universe"
                      icon={<Network size={19} />}
                    />
                    <Metric
                      label="Predicted alerts"
                      value={number(s.alert_nodes)}
                      sub={`Evidence score > ${s.threshold.toFixed(3)}`}
                      icon={<TriangleAlert size={19} />}
                    />
                    <Metric
                      label="Checkpoint"
                      value={`Epoch ${s.checkpoint_epoch} / ${s.epochs}`}
                      sub={`${number(s.parameter_count)} trainable parameters`}
                      icon={<CheckCircle2 size={19} />}
                    />
                  </div>
                  <div className={styles.quickActions}>
                    <div><span className={styles.quickActionIcon}><GitBranch size={22} /></span><div><h2>Ready for review</h2><p>{number(s.alert_nodes)} entities exceed the detection threshold. Review the evidence before deciding.</p></div></div>
                    <div className={styles.quickActionButtons}>
                      <button className={styles.secondary} onClick={() => setTab("detection")}><Network size={16} />Open graph</button>
                      <button className={styles.primary} onClick={() => setTab("investigation")}>Review alerts<ArrowRight size={16} /></button>
                    </div>
                  </div>
                  <div className={styles.overviewGrid}>
                    <section className={styles.panel}>
                      <PanelTitle
                        eyebrow="HELD-OUT TEST • NODE LEVEL"
                        title="Detection quality"
                      />
                      <div className={styles.quality}>
                        {[
                          ["ROC AUC", s.metrics.roc_auc],
                          ["Average precision", s.metrics.pr_ap],
                          ["Precision", s.metrics.precision],
                          ["Recall", s.metrics.recall],
                          ["F1 score", s.metrics.f1],
                        ].map(([label, value]) => (
                          <div key={String(label)}>
                            <span>{label}</span>
                            <strong>{pct(Number(value))}</strong>
                            <div className={styles.track}>
                              <i style={{ width: pct(Number(value)) }} />
                            </div>
                          </div>
                        ))}
                      </div>
                      <div className={styles.caution}>
                        <TriangleAlert size={18} />
                        <p>
                          <strong>High false-alert burden.</strong> ROC AUC does
                          not establish good alert precision. This checkpoint
                          needs analyst review and further accuracy work.
                        </p>
                      </div>
                    </section>
                    <section className={styles.panel}>
                      <PanelTitle
                        eyebrow="DATA COVERAGE"
                        title="Dataset coverage"
                      />
                      <div className={styles.splitBar}>
                        {Object.entries(s.events).map(([k, v]) => (
                          <span
                            key={k}
                            title={`${k}: ${number(v)}`}
                            style={{ flex: v }}
                          />
                        ))}
                      </div>
                      {Object.entries(s.events).map(([k, v]) => (
                        <div className={styles.splitRow} key={k}>
                          <span>
                            {k === "val"
                              ? "Validation"
                              : k === "test"
                                ? "Held-out test"
                                : "Training"}
                          </span>
                          <strong>{number(v)}</strong>
                          <small>{pct(v / s.total_events)}</small>
                        </div>
                      ))}
                      <div className={styles.note}>
                        Ground truth is entity-level. Event rows do not carry
                        independently verified attack labels.
                      </div>
                      <button
                        className={styles.secondary}
                        onClick={() => setTab("events")}
                      >
                        Browse telemetry <ArrowRight size={15} />
                      </button>
                    </section>
                  </div>
                </>
              )}
              {tab === "detection" && (
                <DetectionGraph
                  investigate={(node) => {
                    setSelectedNode(node);
                    setSelectedEvent(null);
                    setTab("investigation");
                  }}
                />
              )}
              {tab === "investigation" && (
                <div className={styles.investigationGrid}>
                  <section className={`${styles.panel} ${styles.queue}`}>
                    <PanelTitle
                      eyebrow="PREDICTED ALERTS"
                      title="Triage queue"
                    />
                    <p className={styles.muted}>
                      Ranked by calibrated node evidence.
                    </p>
                    <Notice loading={nodes.loading} error={nodes.error} />
                    <div className={styles.nodeList}>
                      {nodes.data?.items.map((n) => (
                        <button
                          key={n.id}
                          aria-pressed={selectedNode === n.id}
                          className={
                            selectedNode === n.id ? styles.selectedNode : ""
                          }
                          onClick={() => {
                            setSelectedNode(n.id);
                            setSelectedEvent(null);
                          }}
                        >
                          <span className={styles.entityIcon}>
                            <Network size={17} />
                          </span>
                          <span className={styles.entityText}>
                            <strong title={name(n)}>{name(n)}</strong>
                            <small>
                              #{n.id} · {String(n.attributes.type || "entity")}
                            </small>
                          </span>
                          <b>{n.score.toFixed(2)}</b>
                        </button>
                      ))}
                    </div>
                    <Pager
                      page={nodePage}
                      total={nodes.data?.total || 0}
                      size={12}
                      change={(page) => { setSelectedNode(selectedNode); setNodePage(page); }}
                    />
                  </section>
                  <div className={styles.investigationMain}>
                    <section className={styles.panel}>
                      <div className={styles.panelHeading}>
                        <PanelTitle
                          eyebrow="TEMPORAL RECONSTRUCTION"
                          title="Candidate attack chain"
                        />
                        <label className={styles.control}>
                          Depth
                          <select
                            value={hops}
                            onChange={(e) => setHops(Number(e.target.value))}
                          >
                            <option value={1}>1 hop</option>
                            <option value={2}>2 hops</option>
                            <option value={3}>3 hops</option>
                          </select>
                        </label>
                      </div>
                      <Notice loading={chain.loading} error={chain.error} />
                      {chain.data && (
                        <>
                          <AttackGraph
                            key={`${selectedNode}:${hops}`}
                            chain={chain.data}
                            seed={selectedNode!}
                            select={setSelectedEvent}
                          />
                          <div className={styles.graphFooter}>
                            <span>
                              <i />
                              Seed alert
                            </span>
                            <span>
                              {chain.data.nodes.length} entities ·{" "}
                              {chain.data.events.length} observed events
                            </span>
                            <button
                              className={styles.textButton}
                              onClick={() =>
                                download(
                                  chain.data,
                                  `cadets-chain-${selectedNode}.json`,
                                )
                              }
                            >
                              <ArrowDownToLine size={14} />
                              Export
                            </button>
                          </div>
                          <p className={styles.note}>
                            {chain.data.interpretation}
                            {chain.data.truncated &&
                              " Display bounded to 60 events; additional neighbors were omitted."}
                          </p>
                        </>
                      )}
                    </section>
                    {explanation.data && (
                      <AttackChainBuilder
                        key={selectedNode}
                        node={selectedNode!}
                        anchors={explanation.data.contributors}
                        select={setSelectedEvent}
                      />
                    )}
                    <section className={styles.panel}>
                      <div className={styles.panelHeading}>
                        <PanelTitle
                          eyebrow="EXPLAINABLE EVIDENCE"
                          title="Why this entity was flagged"
                        />
                        <Sparkles size={20} className={styles.teal} />
                      </div>
                      <Notice
                        loading={explanation.loading}
                        error={explanation.error}
                      />
                      {explanation.data && (
                        <>
                          <div className={styles.entityProfile}>
                            <strong>{name(explanation.data.node)}</strong>
                            <span>
                              Entity #{explanation.data.node.id} ·{" "}
                              {String(
                                explanation.data.node.attributes.type ||
                                  "unknown type",
                              )}
                            </span>
                            <details>
                              <summary>Observed entity attributes</summary>
                              <pre>
                                {JSON.stringify(
                                  explanation.data.node.attributes,
                                  null,
                                  2,
                                )}
                              </pre>
                            </details>
                          </div>
                          <div className={styles.explanationScore}>
                            <div>
                              <span>Node evidence</span>
                              <strong>
                                {explanation.data.reconstructed_score.toFixed(
                                  4,
                                )}
                              </strong>
                            </div>
                            <ArrowRight size={22} />
                            <div>
                              <span>Alert threshold</span>
                              <strong>{s.threshold.toFixed(4)}</strong>
                            </div>
                            <span className={styles.alertPill}>
                              {explanation.data.node.alert
                                ? "Above threshold"
                                : "At or below threshold"}
                            </span>
                          </div>
                          <div className={styles.evidenceVisual}>
                            <p>
                              <strong>
                                {explanation.data.repeated_event_channel >
                                explanation.data.single_event_channel
                                  ? "Repeated unusual events"
                                  : "One unusually strong event"}
                              </strong>{" "}
                              contributed to this{" "}
                              {explanation.data.node.alert
                                ? "alert"
                                : "benign prediction"}
                              . The evidence score is{" "}
                              <strong>
                                {(
                                  explanation.data.reconstructed_score /
                                  s.threshold
                                ).toFixed(2)}
                                ×
                              </strong>{" "}
                              the validation threshold in this {calibrationWindow ? `${number(calibrationWindow / 60)}-minute` : "calibration"} window.
                            </p>
                            <div className={styles.evidenceRuler}>
                              <i
                                style={{
                                  width: `${(90 * explanation.data.reconstructed_score) / Math.max(s.threshold, explanation.data.reconstructed_score)}%`,
                                }}
                              />
                              <span
                                style={{
                                  left: `${(90 * s.threshold) / Math.max(s.threshold, explanation.data.reconstructed_score)}%`,
                                }}
                              >
                                <small>Alert threshold</small>
                              </span>
                            </div>
                            <small>
                              Observed evidence{" "}
                              {explanation.data.reconstructed_score.toFixed(4)}{" "}
                              · threshold {s.threshold.toFixed(4)}
                            </small>
                          </div>
                          <p className={styles.muted}>
                            {explanation.data.rule}
                          </p>
                          <div className={styles.channels}>
                            <span>
                              Single-event channel{" "}
                              <b>
                                {explanation.data.single_event_channel.toFixed(
                                  4,
                                )}
                              </b>
                            </span>
                            <span>
                              Repeated-evidence channel{" "}
                              <b>
                                {explanation.data.repeated_event_channel.toFixed(
                                  4,
                                )}
                              </b>
                            </span>
                          </div>
                          <h3 className={styles.sectionLabel}>
                            TOP CONTRIBUTORS ·{" "}
                            {time(explanation.data.window_start)} UTC
                          </h3>
                          {explanation.data.contributors.map((e, i) => (
                            <button
                              className={styles.contributor}
                              key={e.id}
                              onClick={() => setSelectedEvent(e)}
                            >
                              <span>{i + 1}</span>
                              <div>
                                <strong>{e.relation_name}</strong>
                                <small>
                                  #{e.src} → #{e.dst} · {time(e.time)}
                                </small>
                              </div>
                              <b>{e.evidence?.toFixed(3)}</b>
                            </button>
                          ))}
                          <p className={styles.note}>
                            {explanation.data.limitations}
                          </p>
                          <label className={styles.checkbox}>
                            <input
                              type="checkbox"
                              checked={benchmark}
                              onChange={(e) => setBenchmark(e.target.checked)}
                            />
                            Show benchmark membership (excluded from
                            reconstruction)
                          </label>
                          {benchmark && (
                            <p className={styles.note}>
                              Seed entity:{" "}
                              {explanation.data.node.benchmark_member
                                ? "listed in benchmark ground truth"
                                : "not listed in benchmark ground truth"}
                              . This does not label every incident event.
                            </p>
                          )}
                        </>
                      )}
                    </section>
                    {selectedEvent && (
                      <EventDetail
                        event={selectedEvent}
                        close={() => setSelectedEvent(null)}
                      />
                    )}
                    {chain.data && (
                      <section className={styles.panel}>
                        <PanelTitle
                          eyebrow="OBSERVED TIMELINE • UTC"
                          title="Chronological event trail"
                        />
                        <div className={styles.timeline}>
                          {chain.data.events.map((e) => (
                            <button
                              key={e.id}
                              onClick={() => setSelectedEvent(e)}
                            >
                              <time>{time(e.time)}</time>
                              <span className={styles.timelineDot} />
                              <div>
                                <strong>{e.relation_name}</strong>
                                <small>
                                  #{e.src} → #{e.dst}
                                </small>
                              </div>
                              <b>{e.evidence?.toFixed(2)}</b>
                            </button>
                          ))}
                        </div>
                        {!chain.data.events.length && (
                          <p className={styles.note}>
                            No connected events in this time bound.
                          </p>
                        )}
                      </section>
                    )}
                  </div>
                </div>
              )}
              {tab === "events" && (
                <section className={styles.panel}>
                  <div className={styles.panelHeading}>
                    <PanelTitle
                      eyebrow="INDEXED TELEMETRY"
                      title="Event explorer"
                    />
                    <span className={styles.subtlePill}>
                      {number(events.data?.total || 0)} matching events
                    </span>
                  </div>
                  <form
                    className={styles.filters}
                    onSubmit={(e) => {
                      e.preventDefault();
                      setFilter(search);
                      setEventPage(0);
                      setSelectedEvent(null);
                    }}
                  >
                    <label className={styles.control}>
                      Split
                      <select
                        aria-label="Dataset split"
                        value={split}
                        onChange={(e) => {
                          setSplit(e.target.value);
                          setEventPage(0);
                          setSelectedEvent(null);
                        }}
                      >
                        <option value="train">Training</option>
                        <option value="val">Validation</option>
                        <option value="test">Test</option>
                      </select>
                    </label>
                    <label className={styles.search}>
                      <Search size={17} />
                      <input
                        aria-label="Filter by node ID"
                        placeholder="Filter by entity ID…"
                        value={search}
                        inputMode="numeric"
                        pattern="[0-9]*"
                        onChange={(e) => setSearch(e.target.value)}
                      />
                    </label>
                    <button className={styles.secondary} type="submit">
                      <ListFilter size={16} />
                      Apply
                    </button>
                    {filter && (
                      <button
                        type="button"
                        className={styles.textButton}
                        onClick={() => {
                          setSearch("");
                          setFilter("");
                          setEventPage(0);
                        }}
                      >
                        Clear
                      </button>
                    )}
                  </form>
                  <Notice loading={events.loading} error={events.error} />
                  <div className={styles.tableWrap}>
                    <table>
                      <thead>
                        <tr>
                          <th>Event / UTC</th>
                          <th>Source entity</th>
                          <th>Relation</th>
                          <th>Destination entity</th>
                          <th>Combined loss</th>
                          <th>Evidence</th>
                        </tr>
                      </thead>
                      <tbody>
                        {events.data?.items.map((e) => (
                          <tr key={e.id}>
                            <td>
                              <button
                                aria-label={e.id}
                                className={styles.eventButton}
                                onClick={() => setSelectedEvent(e)}
                              >
                                {e.id}
                                <small>{time(e.time)}</small>
                              </button>
                            </td>
                            <td>#{e.src}</td>
                            <td>
                              <span className={styles.relation}>
                                {e.relation_name}
                              </span>
                            </td>
                            <td>#{e.dst}</td>
                            <td>
                              {e.loss === null
                                ? "Not scored"
                                : e.loss.toFixed(4)}
                            </td>
                            <td>{e.evidence?.toFixed(4) ?? "—"}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  {events.data && !events.data.items.length && (
                    <p className={styles.note}>
                      No matching events. Try another entity or split.
                    </p>
                  )}
                  <Pager
                    page={eventPage}
                    total={events.data?.total || 0}
                    size={20}
                    change={setEventPage}
                  />
                  <p className={styles.note}>
                    Training events were not scored in the saved evaluation.
                    Validation evidence uses the fitted reference distribution
                    and is shown for inspection, not as independent validation
                    performance.
                  </p>
                  {selectedEvent && (
                    <EventDetail
                      event={selectedEvent}
                      close={() => setSelectedEvent(null)}
                    />
                  )}
                </section>
              )}
              {tab === "model" && (
                <>
                  <div className={styles.modelGrid}>
                    <section className={styles.panel}>
                      <PanelTitle
                        eyebrow={s.model}
                        title="Model architecture"
                      />
                      <div className={styles.architecture}>
                        {[
                          [
                            "Typed key–value encoder",
                            config?.event_dim === undefined ? "Identity dimensions not exposed" : `${config.token_config?.two_hash ? "Two-hash" : "Typed"} embeddings → ${number(config.event_dim)}D identities`,
                          ],
                          [
                            "Balanced endpoint history",
                            config?.window === undefined ? "Window metadata not exposed" : `${number(config.window)}-event causal window · current query included`,
                          ],
                          [
                            config?.gat_layers === undefined ? "Event-line GAT layers" : `${number(config.gat_layers)} event-line GAT layers`,
                            "Shared-entity causal neighborhood",
                          ],
                          [
                            "Temporal encoder",
                            `${config?.temporal?.toUpperCase() || "Not exposed"} · causal event representations`,
                          ],
                          [
                            "Selective short memory",
                            `${configured(config?.short_slots)} compressed slots per endpoint · ${config?.short_readout?.toUpperCase() || "Not exposed"} readout`,
                          ],
                          [
                            "Affine node-memory scan",
                            "Fast / slow endpoint states · pre-event reads",
                          ],
                          [
                            "Relation + pair heads",
                            `${configured(config?.num_relations)} relations${config?.pair_head ? " + pair-plausibility evidence" : ""}`,
                          ],
                          [
                            "Validation calibration",
                            calibrationWindow ? `${number(calibrationWindow / 60)}-minute node windows · calibrated evidence` : "Calibration window metadata not exposed",
                          ],
                        ].map(([title, body], i) => (
                          <div key={title}>
                            <span>{String(i + 1).padStart(2, "0")}</span>
                            <div>
                              <strong>{title}</strong>
                              <p>{body}</p>
                            </div>
                            {i < 7 && <ArrowRight size={16} />}
                          </div>
                        ))}
                      </div>
                    </section>
                    <section className={styles.panel}>
                      <PanelTitle
                        eyebrow="REPRODUCIBILITY"
                        title="Checkpoint & evaluation"
                      />
                      <dl className={styles.facts}>
                        {[
                          ["Epochs", s.epochs],
                          ["Selected checkpoint", s.checkpoint_epoch],
                          ["Batch size", s.batch_size],
                          ["Seed", s.seed],
                          ["Parameters", number(s.parameter_count)],
                          [
                            "Validation relation CE",
                            s.validation_ce.toFixed(6),
                          ],
                          ["Test relation CE", s.test_ce.toFixed(6)],
                          ["False positives", number(s.metrics.fp)],
                          ["True positives", s.metrics.tp],
                          ["False negatives", s.metrics.fn],
                          [
                            "Unscored truth entities",
                            s.unseen_ground_truth.length,
                          ],
                        ].map(([k, v]) => (
                          <div key={k}>
                            <dt>{k}</dt>
                            <dd>{v}</dd>
                          </div>
                        ))}
                      </dl>
                      <h3 className={styles.sectionLabel}>
                        VERIFIED CHECKPOINT SHA-256
                      </h3>
                      <code className={styles.hash}>{s.checkpoint_sha256}</code>
                      <button
                        className={styles.secondary}
                        onClick={() => download(s, "tgdetect-summary.json")}
                      >
                        <ArrowDownToLine size={15} />
                        Export summary
                      </button>
                    </section>
                  </div>
                  <section className={styles.panel}>
                    <div className={styles.panelHeading}>
                      <PanelTitle
                        eyebrow="CHECKPOINT VERIFICATION"
                        title="Verify event scoring"
                      />
                      <button
                        className={styles.primary}
                        disabled={replaying}
                        onClick={runReplay}
                      >
                        {replaying ? (
                          <Loader2 className={styles.spin} size={16} />
                        ) : (
                          <Play size={16} />
                        )}
                        {replaying
                          ? "Running chronological replay…"
                          : `Run ${number(s.batch_size)} test events`}
                      </button>
                    </div>
                    <p className={styles.muted}>
                      Runs the active trained checkpoint on CPU from empty
                      state. Each replay is isolated and keeps the original
                      evaluation batch boundaries. Initial encoder
                      precomputation may take a moment.
                    </p>
                    <Notice loading={replaying} error={replayError} />
                    {replay && (
                      <>
                        <div className={styles.replayResults}>
                          <Metric
                            label="Events processed"
                            value={number(replay.events)}
                            sub={replay.device}
                            icon={<Activity size={18} />}
                          />
                          <Metric
                            label="Forward latency"
                            value={`${replay.batch_ms[0].toFixed(1)} ms`}
                            sub="Local CPU batch · setup excluded"
                            icon={<Cpu size={18} />}
                          />
                          <Metric
                            label="Saved GPU score difference"
                            value={
                              replay.max_absolute_error_vs_saved_gpu_scores?.toExponential(
                                2,
                              ) || "N/A"
                            }
                            sub="Maximum absolute error on replayed prefix"
                            icon={<CheckCircle2 size={18} />}
                          />
                        </div>
                        <p className={styles.note}>
                          Setup: {replay.setup_seconds.toFixed(2)} s.{" "}
                          {replay.scope} These timings are not the cloud L4
                          latency benchmark.
                        </p>
                        <div className={styles.tableWrap}>
                          <table>
                            <thead>
                              <tr>
                                <th>Event</th>
                                <th>Relation CE</th>
                                <th>Pair anomaly</th>
                                <th>Combined loss</th>
                              </tr>
                            </thead>
                            <tbody>
                              {replay.scores.slice(0, 5).map((x) => (
                                <tr key={x.idx}>
                                  <td>test:{x.idx}</td>
                                  <td>{x.relation_ce.toFixed(5)}</td>
                                  <td>{x.pair_anomaly.toFixed(5)}</td>
                                  <td>{x.combined_loss.toFixed(5)}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </>
                    )}
                  </section>
                  <section className={styles.panel}>
                    <PanelTitle
                      eyebrow="INTERPRETATION"
                      title="Evaluation limitations"
                    />
                    {s.limitations.map((l) => (
                      <p className={styles.limit} key={l}>
                        <TriangleAlert size={16} />
                        {l}
                      </p>
                    ))}
                  </section>
                </>
              )}
            </>
          )}
          <footer className={styles.footer}>
            <span>
              <BrandMark size={19} />
              TGDetect
            </span>
            <span>
              {tab === "detection"
                ? "Graph source: selected log dataset"
                : "CADETS E3 / April 2018 · Times shown in UTC"}
            </span>
          </footer>
        </div>
      </main>
    </div>
  );
}

function Metric({
  label,
  value,
  sub,
  icon,
}: {
  label: string;
  value: string;
  sub: string;
  icon: React.ReactNode;
}) {
  return (
    <div className={styles.metric}>
      <div>
        <span>{label}</span>
        {icon}
      </div>
      <strong>{value}</strong>
      <small>{sub}</small>
    </div>
  );
}
function PanelTitle({ eyebrow, title }: { eyebrow: string; title: string }) {
  return (
    <div>
      <p className={styles.panelEyebrow}>{eyebrow}</p>
      <h2 className={styles.panelTitle}>{title}</h2>
    </div>
  );
}
function Pager({
  page,
  total,
  size,
  change,
}: {
  page: number;
  total: number;
  size: number;
  change: (p: number) => void;
}) {
  return (
    <div className={styles.pager}>
      <span>
        {total
          ? `${number(page * size + 1)}–${number(Math.min((page + 1) * size, total))} of ${number(total)}`
          : "0 results"}
      </span>
      <button
        aria-label="Previous page"
        disabled={page === 0}
        onClick={() => change(page - 1)}
      >
        <ChevronLeft size={17} />
      </button>
      <button
        aria-label="Next page"
        disabled={(page + 1) * size >= total}
        onClick={() => change(page + 1)}
      >
        <ChevronRight size={17} />
      </button>
    </div>
  );
}
function EventDetail({ event, close }: { event: Event; close: () => void }) {
  const detail = useResource<
    Event & {
      src_attributes: Record<string, unknown>;
      dst_attributes: Record<string, unknown>;
      explanation: string;
    }
  >(`/events/${event.split}/${event.idx}`);
  return (
    <section className={styles.eventDetail} aria-label="Selected event">
      <div className={styles.panelHeading}>
        <PanelTitle eyebrow="SELECTED OBSERVED EVENT" title={event.id} />
        <button className={styles.textButton} onClick={close}>
          Close
        </button>
      </div>
      <div className={styles.eventEndpoints}>
        <span>#{event.src}</span>
        <span>
          <ArrowRight size={20} />
          {event.relation_name}
        </span>
        <span>#{event.dst}</span>
      </div>
      <p>
        {time(event.time)} UTC · Combined loss:{" "}
        {event.loss?.toFixed(5) ?? "not scored"} · Calibrated evidence:{" "}
        {event.evidence?.toFixed(5) ?? "not scored"}
      </p>
      <Notice loading={detail.loading} error={detail.error} />
      {detail.data && (
        <>
          <div className={styles.attributeGrid}>
            {(["src", "dst"] as const).map((role) => (
              <div key={role}>
                <h3>{role === "src" ? "Source" : "Destination"} attributes</h3>
                <dl>
                  {Object.entries(detail.data?.[`${role}_attributes`] ?? {}).map(
                    ([key, value]) => (
                      <div key={key}>
                        <dt>{key}</dt>
                        <dd>
                          {value === null
                            ? "Not recorded"
                            : typeof value === "object"
                              ? JSON.stringify(value)
                              : String(value)}
                        </dd>
                      </div>
                    ),
                  )}
                </dl>
              </div>
            ))}
          </div>
          <p className={styles.note}>{detail.data.explanation}</p>
        </>
      )}
      <p className={styles.note}>
        An anomaly score is not a probability of attack. Direction is preserved
        from the prepared provenance record.
      </p>
    </section>
  );
}
