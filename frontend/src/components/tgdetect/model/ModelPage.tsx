'use client';

import { useState } from 'react';
import {
  Activity,
  Brain,
  GitGraph,
  Layers,
  LineChart as LineChartIcon,
  PlayCircle,
  Repeat,
  Target,
  Zap,
} from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import {
  useEvaluationRun,
  useEvaluationRuns,
  usePredictions,
  useSnapshotMeta,
  useSnapshots,
  useTGNNConfig,
  useTGNNSummary,
  useTrainingRun,
} from '@/lib/tgdetect/services/hooks';
import { LiveInference, MatchedComparison } from './ModalPanels';
import { useModel } from '@/lib/model-context';
import {
  formatDurationLong,
  formatFloat,
  formatInt,
  formatPercent,
} from '@/lib/tgdetect/formatters';
import { CHART_AXIS_STYLE, CHART_COLORS, CHART_GRID_STYLE, CHART_TOOLTIP_STYLE } from '@/lib/tgdetect/chart-constants';
import {
  EmptyState,
  ErrorState,
  LabelPill,
  LoadingState,
  MonoId,
  NodeTypePill,
  SectionTitle,
  StatePill,
} from '../shared/pills';
import { confusionMatrixView } from '@/lib/tgdetect/formatters';
import type { EvalSplit } from '@/lib/tgdetect/types';

type ModelTab = 'architecture' | 'snapshots' | 'training' | 'evaluation' | 'comparison';

export function ModelPage() {
  const [tab, setTab] = useState<ModelTab>(() => {
    if (typeof window !== 'undefined') {
      const params = new URLSearchParams(window.location.search);
      const sub = params.get('sub') || window.location.hash.replace('#', '');
      if (sub === 'architecture' || sub === 'snapshots' || sub === 'training' || sub === 'evaluation' || sub === 'comparison') {
        return sub as ModelTab;
      }
    }
    return 'architecture';
  });
  const { activeModel, activeModelId, setActiveModelId, models } = useModel();

  return (
    <div className="space-y-3">
      {/* Authoritative Single Production Model Banner */}
      <div className="tg-card p-3.5 flex items-center justify-between gap-3 flex-wrap bg-gradient-to-r from-[hsl(var(--card))] via-[hsl(var(--card))] to-cyan-500/5 border border-cyan-500/30 shadow-[0_0_15px_rgba(0,242,254,0.06)]">
        <div className="flex items-center gap-3">
          <div className="size-9 rounded-md bg-cyan-500/15 border border-cyan-500/30 flex items-center justify-center text-cyan-400 font-mono font-bold text-sm shadow-[0_0_10px_rgba(0,242,254,0.15)]">
            E
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-sm font-bold text-[hsl(var(--foreground))] tracking-wide">
                {activeModel?.name ?? 'Loading checkpoint…'}
              </span>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded border border-indigo-500/40 bg-indigo-500/15 text-indigo-300 font-semibold uppercase">
                SYSMON EVENT CLASSIFICATION
              </span>
              <span className="inline-flex items-center gap-1.5 text-[10px] font-mono px-2 py-0.5 rounded border border-emerald-500/40 bg-emerald-500/15 text-emerald-400 font-semibold">
                <span className="size-1.5 rounded-full bg-emerald-400 animate-pulse" />
                PRODUCTION BENCHMARK
              </span>
            </div>
            <div className="text-[11px] text-[hsl(var(--muted-foreground))] font-mono mt-0.5 flex items-center gap-2 flex-wrap">
              <span>model_id: <strong className="text-cyan-400 font-semibold">{activeModelId}</strong></span>
              <span>•</span>
              <span>checkpoint: <span className="text-[hsl(var(--foreground))]">{activeModel?.checkpoint}</span></span>
              <span>•</span>
              <span>trainable_parameters: <strong className="text-cyan-400 font-semibold">{formatInt(activeModel?.trainable_parameters ?? 0)}</strong></span>
              <span>•</span>
              <span>benchmark: <span className="text-[hsl(var(--foreground))]">{activeModel?.dataset_name}</span></span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <select aria-label="Active model" value={activeModelId} onChange={e => setActiveModelId(e.target.value)} className="bg-[hsl(var(--background))] border border-[hsl(var(--border))] rounded p-2 text-xs">{models.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}</select>
          <div className="text-[10px] font-mono px-3 py-1 rounded bg-[hsl(var(--background))] border border-[hsl(var(--border))] text-cyan-400/90 font-medium">
            Matched Modal checkpoints
          </div>
        </div>
      </div>

      <div className="bg-[hsl(var(--card))] p-1.5 rounded-lg border border-[hsl(var(--border))] flex flex-wrap gap-1 shadow-xs">
        {([
          { id: 'architecture', label: 'Architecture', icon: Brain },
          { id: 'snapshots', label: 'Live inference', icon: Layers },
          { id: 'training', label: 'Saved training', icon: LineChartIcon },
          { id: 'evaluation', label: 'Evaluation', icon: Target },
          { id: 'comparison', label: 'Matched comparison', icon: Target },
        ] as { id: ModelTab; label: string; icon: typeof Brain }[]).map((t) => {
          const Icon = t.icon;
          const isActive = tab === t.id;
          return (
            <button
              key={t.id}
              type="button"
              onClick={() => setTab(t.id)}
              className={`px-3 py-1.5 text-xs rounded transition-all flex items-center gap-1.5 ${
                isActive
                  ? 'bg-[hsl(var(--primary)/0.15)] text-[hsl(var(--primary))] border border-[hsl(var(--primary)/0.3)] font-semibold shadow-xs'
                  : 'text-[hsl(var(--muted-foreground))] hover:text-[hsl(var(--foreground))] hover:bg-[hsl(var(--card-hover))]'
              }`}
            >
              <Icon className="size-3.5" />
              {t.label}
            </button>
          );
        })}
      </div>

      {tab === 'architecture' && <ArchitecturePanel modelId={activeModelId} />}
      {tab === 'snapshots' && <LiveInference key={activeModelId} modelId={activeModelId} />}
      {tab === 'comparison' && <MatchedComparison />}
      {tab === 'training' && <TrainingPanel modelId={activeModelId} />}
      {tab === 'evaluation' && <EvaluationPanel modelId={activeModelId} />}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Architecture
// ─────────────────────────────────────────────────────────────────────────────

function ArchitecturePanel({ modelId }: { modelId: string }) {
  const configRes = useTGNNConfig(modelId);
  const summaryRes = useTGNNSummary(modelId);
  if (configRes.state === 'loading' || summaryRes.state === 'loading') return <LoadingState label="Loading model config…" />;
  if (!configRes.data || !summaryRes.data) return <EmptyState title="No model config" />;
  const cfg = configRes.data;
  const summary = summaryRes.data;
  return (
    <div className="space-y-3">
      <div className="tg-card p-4">
        <SectionTitle right={<span className="text-[10px] font-mono px-1.5 py-0.5 rounded border border-[hsl(var(--purple)/0.4)] bg-[hsl(var(--purple-bg))] text-[hsl(var(--purple))]">{summary.architecture}</span>}>
          TGNN Architecture
        </SectionTitle>
        <div className="text-[11px] text-[hsl(var(--muted-foreground))] mt-1">
          {summary.gnn_operator} · {summary.temporal_aggregator} {summary.total_parameters ? ` · ${formatInt(summary.total_parameters)} total parameters` : ''}{summary.bn_running_stats ? ` (+${summary.bn_running_stats} BN buffers = ${formatInt(summary.total_state_dict_elements ?? 0)} state_dict elements)` : ''} · loss: <span className="mono">{summary.loss}</span>
        </div>
        <div className="grid grid-cols-3 gap-2 mt-3">
          <Attention flag={summary.has_attention} label="Self-attention" />
          <Attention flag={summary.has_transformer} label="Transformer block" />
          <Attention flag={summary.has_llm} label="LLM component" />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        {/* Architecture diagram */}
        <div className="tg-card p-4">
          <SectionTitle right={<GitGraph className="size-3 text-[hsl(var(--muted-foreground))]" />}>Forward pass</SectionTitle>
          <div className="mt-3 space-y-1.5">
            <FlowNode index={1} label="Causal event ego window" sub={`x: [batch, 5, ${cfg.in_channels}] · valid mask · elapsed ages`} color="info" />
            <FlowArrow />
            <FlowNode index={2} label={cfg.gnn_type ?? "Graph attention"} sub={`${cfg.num_gnn_layers} GAT layers · ${cfg.hidden_channels} hidden dimensions · temporal age encoding`} color="purple" />
            <FlowArrow />
            <FlowNode index={3} label="GRU ()" sub={`${cfg.num_rnn_layers} layer(s) · input=${cfg.hidden_channels} → hidden=${cfg.out_channels}`} color="teal" />
            <FlowArrow />
            <div className={`grid ${summary.output_heads && summary.output_heads.length > 2 ? 'grid-cols-3' : 'grid-cols-2'} gap-2`}>
              {(summary.output_heads ?? ['node_classifier', 'snapshot_classifier']).map((head: string, idx: number) => (
                <FlowNode
                  key={head}
                  index={4 + idx}
                  label={head.replace(/_/g, ' ')}
                  sub="Hidden representation → 2 logits → malicious probability"
                  color={head === 'node_classifier' ? 'danger' : head === 'snapshot_classifier' ? 'amber' : 'purple'}
                />
              ))}
            </div>
            <div className="text-[10px] text-[hsl(var(--muted-foreground))] mt-2">
              Output: <span className="mono">{(summary.output_heads ?? ['node_classifier', 'snapshot_classifier']).map((h: string) => `${h.replace('_classifier', '')}_logits`).join(' + ')}</span> · dynamic head projection
            </div>
          </div>
        </div>

        {/* Hyperparameters */}
        <div className="tg-card p-4">
          <SectionTitle right={<Activity className="size-3 text-[hsl(var(--muted-foreground))]" />}>Hyperparameters</SectionTitle>
          <div className="grid grid-cols-2 gap-3 mt-3">
            <Hyperparam label="in_channels" value={cfg.in_channels} hint="64 hashed categorical + 8 scaled numeric features" />
            <Hyperparam label="edge_dim" value={cfg.edge_dim} hint="No separate edge feature vector" />
            <Hyperparam label="hidden_channels" value={cfg.hidden_channels} />
            <Hyperparam label="out_channels" value={cfg.out_channels} hint="Event hidden dimensions" />
            <Hyperparam label="num_gnn_layers" value={cfg.num_gnn_layers} hint="Graph attention layers" />
            <Hyperparam label="num_rnn_layers" value={cfg.num_rnn_layers} hint="History GRU resets for each event window" />
            <Hyperparam label="dropout" value={cfg.dropout} />
            <Hyperparam label="node_types" value={cfg.node_types ?? '—'} hint="Process and host entities" />
            <Hyperparam label="num_relations" value={cfg.num_relations ?? '—'} hint="Recorded process-to-host incidence" />
            <Hyperparam label="total_params" value={formatInt(summary.total_parameters ?? 0)} hint={`${formatInt(summary.total_parameters ?? 0)} total parameters`} />
            <Hyperparam label="trainable_params" value={formatInt(summary.trainable_parameters ?? 0)} hint={`${formatInt(summary.trainable_parameters ?? 0)} trainable weights & biases`} />
            <Hyperparam label="bn_buffers" value={summary.bn_running_stats ?? 0} hint="BatchNorm running-stat buffers" />
            <Hyperparam label="state_dict_elems" value={formatInt(summary.total_state_dict_elements ?? 0)} hint={`${formatInt(summary.total_state_dict_elements ?? 0)} state_dict elements`} />
          </div>
        </div>
      </div>
    </div>
  );
}

function Attention({ flag, label }: { flag: boolean; label: string }) {
  return (
    <div className={`p-2 rounded border text-center ${flag ? 'border-[hsl(var(--purple)/0.4)] bg-[hsl(var(--purple-bg))]' : 'border-[hsl(var(--border))] bg-[hsl(var(--background))]'}`}>
      <div className="text-[10px] font-semibold" style={{ color: flag ? 'hsl(var(--purple))' : 'hsl(var(--muted-foreground))' }}>
        {flag ? 'PRESENT' : 'absent'}
      </div>
      <div className="text-[9px] text-[hsl(var(--muted-foreground))]">{label}</div>
    </div>
  );
}

function FlowNode({ index, label, sub, color }: { index: number; label: string; sub: string; color: 'info' | 'purple' | 'teal' | 'danger' | 'amber' }) {
  const colorClass = {
    info: 'border-[hsl(var(--info)/0.4)] bg-[hsl(var(--info-bg))] text-[hsl(var(--info))]',
    purple: 'border-[hsl(var(--purple)/0.4)] bg-[hsl(var(--purple-bg))] text-[hsl(var(--purple))]',
    teal: 'border-[hsl(var(--teal)/0.4)] bg-[hsl(173_55%/95%)] dark:bg-[hsl(173_58%/8%)] text-[hsl(var(--teal))]',
    danger: 'border-[hsl(var(--danger)/0.4)] bg-[hsl(var(--danger-bg))] text-[hsl(var(--danger))]',
    amber: 'border-[hsl(var(--warning)/0.4)] bg-[hsl(var(--warning-bg))] text-[hsl(var(--warning))]',
  }[color];
  return (
    <div className={`p-2 rounded border ${colorClass}`}>
      <div className="flex items-center gap-2">
        <span className="size-5 rounded-full bg-[hsl(var(--card))] text-[hsl(var(--foreground))] flex items-center justify-center text-[10px] mono font-bold">{index}</span>
        <span className="text-[11px] font-mono font-semibold">{label}</span>
      </div>
      <div className="text-[10px] mono mt-1 ml-7 opacity-80">{sub}</div>
    </div>
  );
}

function FlowArrow() {
  return <div className="ml-3 w-px h-3 bg-[hsl(var(--border))] " />;
}

function Hyperparam({ label, value, hint }: { label: string; value: string | number; hint?: string }) {
  return (
    <div className="flex flex-col gap-0.5 p-2 rounded border border-[hsl(var(--border))] bg-[hsl(var(--background))]">
      <div className="text-[9px] uppercase tracking-wide text-[hsl(var(--muted-foreground))] font-semibold mono">{label}</div>
      <div className="text-sm mono font-semibold tabular-nums text-[hsl(var(--foreground))]">{value}</div>
      {hint && <div className="text-[9px] text-[hsl(var(--muted-foreground))]">{hint}</div>}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Snapshots
// ─────────────────────────────────────────────────────────────────────────────

function SnapshotsPanel({ modelId }: { modelId?: string } = {}) {
  const metaRes = useSnapshotMeta();
  const snapsRes = useSnapshots(120);
  if (metaRes.state === 'loading' || snapsRes.state === 'loading') return <LoadingState label="Loading snapshots…" />;
  if (!metaRes.data) return <EmptyState title="No snapshot metadata" />;
  const meta = metaRes.data;
  const snaps = snapsRes.data ?? [];
  return (
    <div className="space-y-3">
      <div className="tg-card p-4">
        <SectionTitle>Snapshot configuration</SectionTitle>
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3 mt-3">
          <Hyperparam label="window_size_s" value={meta.window_size_s} />
          <Hyperparam label="stride_s" value={meta.stride_s} />
          <Hyperparam label="num_snapshots" value={formatInt(meta.num_snapshots)} />
          <Hyperparam label="node_feature_mode" value={meta.node_feature_mode} />
          <Hyperparam label="edge_feature_mode" value={meta.edge_feature_mode} />
          <Hyperparam label="node_feature_dim" value={meta.node_feature_dim} hint="8 one-hot type + 4 numerical" />
          <Hyperparam label="edge_feature_dim" value={meta.edge_feature_dim} hint="14 relation one-hot + 1 rel ts" />
          <Hyperparam label="num_node_types" value={meta.num_node_types} />
          <Hyperparam label="num_relations" value={meta.num_relations} />
        </div>
      </div>
      <div className="tg-card p-4">
        <SectionTitle right={<div className="text-[10px] text-[hsl(var(--muted-foreground))]">showing first {snaps.length} / {formatInt(meta.num_snapshots)} snapshots</div>}>
          Snapshot window activity
        </SectionTitle>
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={snaps.map((s) => ({ idx: s.index ?? (s as any).sequence ?? 0, nodes: s.num_nodes, edges: s.num_edges, mal_edges: s.num_malicious_edges ?? 0 }))} margin={{ top: 8, right: 8, bottom: 0, left: -16 }}>
            <CartesianGrid {...CHART_GRID_STYLE} />
            <XAxis dataKey="idx" {...CHART_AXIS_STYLE} />
            <YAxis {...CHART_AXIS_STYLE} width={36} />
            <Tooltip
              {...CHART_TOOLTIP_STYLE}
              formatter={(value: any, name: any) => [
                typeof value === 'number' && !Number.isNaN(value) ? formatInt(value) : '0',
                name,
              ]}
            />
            <Bar dataKey="edges" fill={CHART_COLORS.cyan} radius={[2, 2, 0, 0]} />
            <Bar dataKey="mal_edges" fill={CHART_COLORS.red} radius={[2, 2, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
        <div className="flex gap-3 text-[10px] mt-1">
          <span className="flex items-center gap-1"><span className="size-2 rounded-full" style={{ background: CHART_COLORS.cyan }} />edges per snapshot</span>
          <span className="flex items-center gap-1"><span className="size-2 rounded-full" style={{ background: CHART_COLORS.red }} />malicious edges</span>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Training
// ─────────────────────────────────────────────────────────────────────────────

function TrainingPanel({ modelId }: { modelId: string }) {
  const runRes = useTrainingRun(modelId);
  if (runRes.state === 'loading') return <LoadingState label="Loading training run…" />;
  if (!runRes.data) return <EmptyState title="No training run" />;
  const run = runRes.data;
  const history = run.history;
  return (
    <div className="space-y-3">
      <div className="tg-card p-4">
        <div className="flex items-center justify-between flex-wrap gap-2 mb-2">
          <SectionTitle>Training Run</SectionTitle>
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded border border-[hsl(var(--purple)/0.4)] bg-[hsl(var(--purple-bg))] text-[hsl(var(--purple))]">{run.id}</span>
            <StatePill state={run.state === 'completed' ? 'completed' : run.state === 'failed' ? 'failed' : 'building_graph'} />
          </div>
        </div>
        <div className="text-[10px] text-[hsl(var(--muted-foreground))] mono">
          dataset: {run.dataset_id} · checkpoint: <span className="text-[hsl(var(--foreground))]">{run.checkpoint_path ?? 'Not recorded'}</span>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3 mt-3">
          <Hyperparam label="epochs" value={run.total_epochs != null ? `${run.current_epoch ?? run.total_epochs}/${run.total_epochs}` : 'Not recorded'} />
          <Hyperparam label="best_epoch" value={run.best_epoch ?? '—'} hint={run.best_metric ? `by ${run.best_metric}` : undefined} />
          <Hyperparam label="best_score" value={run.best_score != null ? formatFloat(run.best_score, 4) : 'Not recorded'} />
          <Hyperparam label="best_f1" value={run.best_val_f1 != null ? formatFloat(run.best_val_f1, 4) : 'Not recorded'} />
          <Hyperparam label="threshold" value={run.threshold != null ? formatFloat(run.threshold, 4) : 'Not recorded'} hint="tuned on val set" />
          <Hyperparam label="elapsed" value={run.elapsed_s != null ? formatDurationLong(run.elapsed_s) : 'Not recorded'} />
        </div>

        <div className="border-t border-[hsl(var(--border)/0.5)] pt-3 mt-3">
          <div className="text-[10px] uppercase tracking-wider text-[hsl(var(--muted-foreground))] font-semibold mb-2">
            Recorded Hyperparameters
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
            <Hyperparam label="learning_rate" value={run.config?.lr != null ? run.config.lr : '—'} />
            <Hyperparam label="batch_size" value={run.config?.batch_size != null ? run.config.batch_size : '—'} />
            <Hyperparam label="hidden_channels" value={run.config?.hidden_channels != null ? run.config.hidden_channels : '—'} />
            <Hyperparam label="out_channels" value={run.config?.out_channels != null ? run.config.out_channels : '—'} />
            <Hyperparam label="gnn_layers" value={run.config?.gnn_layers != null ? run.config.gnn_layers : '—'} />
            <Hyperparam label="rnn_layers" value={run.config?.rnn_layers != null ? run.config.rnn_layers : '—'} />
            <Hyperparam label="dropout" value={run.config?.dropout != null ? run.config.dropout : '—'} />
            <Hyperparam label="window_size" value={run.config?.window_size != null ? run.config.window_size : '—'} hint="snapshots/seq" />
            <Hyperparam label="val_ratio" value={run.config?.val_ratio != null ? formatPercent(run.config.val_ratio) : '—'} />
            <Hyperparam label="test_ratio" value={run.config?.test_ratio != null ? formatPercent(run.config.test_ratio) : '—'} />
            <Hyperparam label="seed" value={run.config?.seed != null ? run.config.seed : '—'} />
            <Hyperparam label="weight_decay" value={run.config?.weight_decay != null ? run.config.weight_decay : '—'} hint={run.config?.weight_decay == null ? 'Not recorded' : undefined} />
          </div>
        </div>
      </div>

      <div className="tg-card p-4">
        <SectionTitle right={<div className="flex gap-3 text-[10px]">
          <span className="flex items-center gap-1"><span className="size-2 rounded-full" style={{ background: CHART_COLORS.cyan }} />train_loss</span>
          <span className="flex items-center gap-1"><span className="size-2 rounded-full" style={{ background: CHART_COLORS.violet }} />val_loss</span>
          <span className="flex items-center gap-1"><span className="size-2 rounded-full" style={{ background: CHART_COLORS.red }} />threshold</span>
        </div>}>
          Loss & threshold per epoch
        </SectionTitle>
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={history} margin={{ top: 8, right: 8, bottom: 0, left: -16 }}>
            <CartesianGrid {...CHART_GRID_STYLE} />
            <XAxis dataKey="epoch" {...CHART_AXIS_STYLE} />
            <YAxis {...CHART_AXIS_STYLE} width={36} />
            <Tooltip
              {...CHART_TOOLTIP_STYLE}
              formatter={(value: any, name: any) => [
                typeof value === 'number' && !Number.isNaN(value) ? formatFloat(value, 4) : '—',
                name,
              ]}
            />
            <Line isAnimationActive={false} type="monotone" dataKey="train_loss" stroke={CHART_COLORS.cyan} strokeWidth={1.5} dot={false} />
            <Line isAnimationActive={false} type="monotone" dataKey="loss" stroke={CHART_COLORS.violet} strokeWidth={1.5} dot={false} />
            <Line isAnimationActive={false} type="monotone" dataKey="threshold" stroke={CHART_COLORS.red} strokeWidth={1.5} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        <div className="tg-card p-4">
          <SectionTitle>AUC per epoch</SectionTitle>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={history} margin={{ top: 8, right: 8, bottom: 0, left: -16 }}>
              <CartesianGrid {...CHART_GRID_STYLE} />
              <XAxis dataKey="epoch" {...CHART_AXIS_STYLE} />
              <YAxis {...CHART_AXIS_STYLE} width={36} domain={[0, 1]} />
              <Tooltip
                {...CHART_TOOLTIP_STYLE}
                formatter={(value: any, name: any) => [
                  typeof value === 'number' && !Number.isNaN(value) ? formatFloat(value, 4) : '—',
                  name,
                ]}
              />
              <Line isAnimationActive={false} type="monotone" dataKey="auc_roc" stroke={CHART_COLORS.cyan} strokeWidth={1.5} dot={false} />
              <Line isAnimationActive={false} type="monotone" dataKey="auc_pr" stroke={CHART_COLORS.green} strokeWidth={1.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="tg-card p-4">
          <SectionTitle>Precision / Recall / F1 per epoch</SectionTitle>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={history} margin={{ top: 8, right: 8, bottom: 0, left: -16 }}>
              <CartesianGrid {...CHART_GRID_STYLE} />
              <XAxis dataKey="epoch" {...CHART_AXIS_STYLE} />
              <YAxis {...CHART_AXIS_STYLE} width={36} domain={[0, 1]} />
              <Tooltip
                {...CHART_TOOLTIP_STYLE}
                formatter={(value: any, name: any) => [
                  typeof value === 'number' && !Number.isNaN(value) ? formatFloat(value, 4) : '—',
                  name,
                ]}
              />
              <Line isAnimationActive={false} type="monotone" dataKey="precision" stroke={CHART_COLORS.amber} strokeWidth={1.5} dot={false} />
              <Line isAnimationActive={false} type="monotone" dataKey="recall" stroke={CHART_COLORS.teal} strokeWidth={1.5} dot={false} />
              <Line isAnimationActive={false} type="monotone" dataKey="f1" stroke={CHART_COLORS.violet} strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="tg-card p-4">
        <SectionTitle>Per-epoch history (best epoch highlighted)</SectionTitle>
        <div className="overflow-x-auto mt-2">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-left text-[10px] uppercase tracking-wide text-[hsl(var(--muted-foreground))] border-b border-[hsl(var(--border))]">
                <th className="py-1.5 pr-3">epoch</th>
                <th className="py-1.5 pr-3">train_loss</th>
                <th className="py-1.5 pr-3">val_loss</th>
                <th className="py-1.5 pr-3">auc_roc</th>
                <th className="py-1.5 pr-3">auc_pr</th>
                <th className="py-1.5 pr-3">precision</th>
                <th className="py-1.5 pr-3">recall</th>
                <th className="py-1.5 pr-3">f1</th>
                <th className="py-1.5 pr-3">threshold</th>
              </tr>
            </thead>
            <tbody>
              {history.map((e) => {
                const isBest = e.epoch === run.best_epoch;
                return (
                  <tr key={e.epoch} className={`border-b border-[hsl(var(--border)/0.5)] ${isBest ? 'bg-[hsl(var(--info-bg))]' : ''}`}>
                    <td className="py-1.5 pr-3 mono font-semibold">
                      {e.epoch}{isBest && <span className="ml-1 text-[10px] text-[hsl(var(--primary))]">★ best</span>}
                    </td>
                    <td className="py-1.5 pr-3 mono">{formatFloat(e.train_loss, 4)}</td>
                    <td className="py-1.5 pr-3 mono">{formatFloat(e.loss, 4)}</td>
                    <td className="py-1.5 pr-3 mono">{e.auc_roc === null ? '—' : formatFloat(e.auc_roc, 4)}</td>
                    <td className="py-1.5 pr-3 mono">{e.auc_pr === null ? '—' : formatFloat(e.auc_pr, 4)}</td>
                    <td className="py-1.5 pr-3 mono">{formatFloat(e.precision, 4)}</td>
                    <td className="py-1.5 pr-3 mono">{formatFloat(e.recall, 4)}</td>
                    <td className="py-1.5 pr-3 mono font-semibold">{formatFloat(e.f1, 4)}</td>
                    <td className="py-1.5 pr-3 mono">{e.threshold != null ? formatFloat(e.threshold, 4) : '—'}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Evaluation
// ─────────────────────────────────────────────────────────────────────────────

function EvaluationPanel({ modelId }: { modelId: string }) {
  const [split, setSplit] = useState<EvalSplit>('test');
  const evalRes = useEvaluationRun(split, undefined, modelId);
  const predsRes = usePredictions(split, modelId);

  if (evalRes.state === 'loading') return <LoadingState label="Loading model evaluation…" />;

  const run = evalRes.data;
  const m = run?.metrics;
  const cm = m?.confusion_matrix ? confusionMatrixView(m.confusion_matrix) : null;

  return (
    <div className="space-y-3">
      <div className="tg-card p-3 flex items-center gap-3 flex-wrap">
        <SectionTitle className="flex-1">
          Evaluation · {modelId} ({run?.target?.toUpperCase() ?? 'TARGET'})
        </SectionTitle>
        <div className="flex gap-1">
          {(['train', 'val', 'test'] as EvalSplit[]).map((s) => {
            const isSelected = split === s;
            return (
              <button
                key={s}
                type="button"
                onClick={() => setSplit(s)}
                className={`px-2 py-1 text-[10px] font-mono border rounded transition-colors ${
                  isSelected
                    ? 'border-[hsl(var(--primary)/0.6)] bg-[hsl(var(--info-bg))] font-semibold text-[hsl(var(--primary))]'
                    : 'border-[hsl(var(--border))] bg-[hsl(var(--card))] text-[hsl(var(--muted-foreground))]'
                }`}
              >
                {s}
              </button>
            );
          })}
        </div>
      </div>

      {!run || !m ? (
        <div className="tg-card p-6">
          <EmptyState
            title={`No evaluation artifact for split="${split}"`}
            description={`Evaluation metrics for "${split}" were not computed or are not present in models/checkpoints. Only splits with physical evaluation artifacts (e.g. test) are loaded.`}
          />
        </div>
      ) : (
        <>
          <div className="tg-card p-4">
            <SectionTitle right={<div className="text-[10px] text-[hsl(var(--muted-foreground))]">{formatInt(m.num_samples)} samples · {formatInt(m.num_positive)} positive · {formatInt(m.num_negative)} negative</div>}>
              Metrics · split={split}
            </SectionTitle>
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3 mt-3">
              <Hyperparam label="threshold" value={formatFloat(m.threshold, 4)} />
              <Hyperparam label="accuracy" value={formatFloat(m.accuracy, 4)} />
              <Hyperparam label="precision" value={formatFloat(m.precision, 4)} />
              <Hyperparam label="recall" value={formatFloat(m.recall, 4)} />
              <Hyperparam label="f1" value={formatFloat(m.f1, 4)} />
              <Hyperparam label="auc_roc" value={m.auc_roc === null ? 'null' : formatFloat(m.auc_roc, 4)} hint="null = one-class split" />
              <Hyperparam label="auc_pr" value={m.auc_pr === null ? 'null' : formatFloat(m.auc_pr, 4)} hint="null = one-class split" />
            </div>
            <div className="mt-2 text-[10px] text-[hsl(var(--muted-foreground))]">
              predictions: <span className="mono">{run.predictions_path ?? '—'}</span> · checkpoint: <span className="mono">{run.checkpoint_path}</span>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
            <div className="tg-card p-4">
              <SectionTitle>Confusion Matrix</SectionTitle>
              {cm ? (
                <>
                  <div className="grid grid-cols-2 gap-2 mt-3 max-w-md mx-auto">
                    <CmCell label="TN" value={cm.tn} color="success" />
                    <CmCell label="FP" value={cm.fp} color="danger" />
                    <CmCell label="FN" value={cm.fn} color="warning" />
                    <CmCell label="TP" value={cm.tp} color="info" />
                  </div>
                  <div className="mt-3 grid grid-cols-2 gap-2 text-[10px] font-mono">
                    <div>TPR (recall): <span className="text-[hsl(var(--info))]">{formatFloat(cm.tpr, 4)}</span></div>
                    <div>FPR: <span className="text-[hsl(var(--danger))]">{formatFloat(cm.fpr, 4)}</span></div>
                    <div>TNR (specificity): <span className="text-[hsl(var(--success))]">{formatFloat(cm.tnr, 4)}</span></div>
                    <div>FNR: <span className="text-[hsl(var(--warning))]">{formatFloat(cm.fnr, 4)}</span></div>
                    <div>PPV (precision): <span className="text-[hsl(var(--info))]">{formatFloat(cm.ppv, 4)}</span></div>
                    <div>NPV: <span className="text-[hsl(var(--success))]">{formatFloat(cm.npv, 4)}</span></div>
                  </div>
                </>
              ) : (
                <div className="p-4 rounded border border-[hsl(var(--border))] bg-[hsl(var(--background))] text-center my-4">
                  <div className="text-xs text-[hsl(var(--muted-foreground))] mb-1 font-mono">
                    Aggregate confusion matrix not computed in raw artifact
                  </div>
                  <div className="text-[11px] text-[hsl(var(--foreground))] font-medium">
                    Model evaluated across threshold operating points:
                  </div>
                  <div className="mt-2 text-[10px] font-mono text-[hsl(var(--muted-foreground))] space-y-0.5">
                    <div>Recall @ 1% FPR: <span className="text-emerald-400 font-semibold">{m.recall_at_1pct_fpr != null ? `${(m.recall_at_1pct_fpr * 100).toFixed(2)}%` : '—'}</span></div>
                    <div>Threshold @ 1% FPR: <span className="text-[hsl(var(--foreground))]">{m.threshold_at_1pct_fpr != null ? formatFloat(m.threshold_at_1pct_fpr, 6) : '—'}</span></div>
                    <div>Optimal Threshold: <span className="text-[hsl(var(--foreground))]">{m.threshold_best != null ? formatFloat(m.threshold_best, 6) : '—'}</span></div>
                  </div>
                </div>
              )}
            </div>

            <div className="tg-card p-4">
              <SectionTitle right={<Target className="size-3 text-[hsl(var(--muted-foreground))]" />}>Sample predictions</SectionTitle>
              {predsRes.state === 'loading' ? (
                <LoadingState label="Loading predictions…" />
              ) : (predsRes.data ?? []).length === 0 ? (
                <EmptyState title="No predictions" />
              ) : (
                <div className="overflow-x-auto mt-2 max-h-80 overflow-y-auto">
                  <table className="w-full text-xs">
                    <thead className="sticky top-0 bg-[hsl(var(--card))]">
                      <tr className="text-left text-[10px] uppercase tracking-wide text-[hsl(var(--muted-foreground))] border-b border-[hsl(var(--border))]">
                        <th className="py-1.5 pr-3">seq</th>
                        <th className="py-1.5 pr-3">node_id</th>
                        <th className="py-1.5 pr-3">prob</th>
                        <th className="py-1.5 pr-3">pred</th>
                        <th className="py-1.5 pr-3">truth</th>
                        <th className="py-1.5 pr-3">snap</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(predsRes.data ?? []).slice(0, 50).map((p, i) => (
                        <tr key={i} className="border-b border-[hsl(var(--border)/0.5)]">
                          <td className="py-1.5 pr-3 mono text-[10px]">{p.sequence}</td>
                          <td className="py-1.5 pr-3"><MonoId truncateAt={20}>{p.node_id}</MonoId></td>
                          <td className="py-1.5 pr-3 mono text-[10px]">{formatFloat(p.probability, 4)}</td>
                          <td className="py-1.5 pr-3"><LabelPill label={p.prediction} /></td>
                          <td className="py-1.5 pr-3"><LabelPill label={p.ground_truth} /></td>
                          <td className="py-1.5 pr-3 mono text-[10px]">{p.snapshot_label}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function CmCell({ label, value, color }: { label: string; value: number; color: 'success' | 'danger' | 'warning' | 'info' }) {
  const cls = {
    success: 'border-[hsl(var(--success)/0.4)] bg-[hsl(var(--success-bg))] text-[hsl(var(--success))]',
    danger: 'border-[hsl(var(--danger)/0.4)] bg-[hsl(var(--danger-bg))] text-[hsl(var(--danger))]',
    warning: 'border-[hsl(var(--warning)/0.4)] bg-[hsl(var(--warning-bg))] text-[hsl(var(--warning))]',
    info: 'border-[hsl(var(--info)/0.4)] bg-[hsl(var(--info-bg))] text-[hsl(var(--info))]',
  }[color];
  return (
    <div className={`p-4 rounded border ${cls} text-center`}>
      <div className="text-[10px] uppercase tracking-wide font-semibold">{label}</div>
      <div className="text-2xl font-mono font-bold tabular-nums mt-1">{formatInt(value)}</div>
    </div>
  );
}
