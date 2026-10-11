'use client';

import { useMemo, useState } from 'react';
import { Download, FileText, FolderOpen, Table2 } from 'lucide-react';
import { useArtifacts, useJobs } from '@/lib/tgdetect/services/hooks';
import { useDataset } from '@/lib/dataset-context';
import {
  EmptyState,
  ErrorState,
  LoadingState,
  MonoId,
  SectionTitle,
} from '../shared/pills';
import {
  formatBytes,
  formatInt,
} from '@/lib/tgdetect/formatters';
import type { ArtifactKind, ArtifactMeta } from '@/lib/tgdetect/types';
import { API_BASE_URL } from '@/lib/tgdetect/api/client';

const ARTIFACT_LABELS: Record<ArtifactKind, string> = {
  events: 'events.json',
  edges: 'edges.json',
  nodes: 'nodes.json',
  chains_summary: 'chains_summary.json',
  graph_stats: 'graph_stats.json',
  subgraph: 'subgraphs/ (JSON dir)',
};

const ARTIFACT_DESCRIPTIONS: Record<ArtifactKind, string> = {
  events: 'Full event table — one row per TGEvent. Includes recorded metadata.',
  edges: 'Thin edge projection — 9 columns. Derived directly from recorded events.',
  nodes: 'Node metadata — 7 columns. Computed from actual recorded relationships.',
  chains_summary: 'Chain-level summary — 12 columns. One row per reconstructed chain using entity-time grouping.',
  graph_stats: 'Pipeline summary JSON — graph, normalization, labeling, attacks, outputs.',
  subgraph: 'Per-chain subgraph JSON files. One file per chain under <graphs_dir>/chains/<chain_id>.json',
};

/**
 * Outer wrapper uses `key` based on initialJobId so the inner component
 * remounts when navigation passes a new initial job id.
 */
export function ArtifactsPage({ initialJobId }: { initialJobId?: string | null }) {
  const key = initialJobId ?? '__none__';
  return <ArtifactsPageInner key={key} initialJobId={initialJobId} />;
}

function ArtifactsPageInner({ initialJobId }: { initialJobId?: string | null }) {
  const { activeDataset, activeDatasetId } = useDataset();
  const jobsRes = useJobs();
  const [selectedJobId, setSelectedJobId] = useState<string | null>(initialJobId ?? null);
  const [selectedKind, setSelectedKind] = useState<ArtifactKind | null>(null);

  // Auto-select job matching active dataset if none was explicitly picked
  const activeDatasetJob = useMemo(() => {
    return (jobsRes.data ?? []).find((j) => j.dataset_id === activeDatasetId);
  }, [jobsRes.data, activeDatasetId]);

  const effectiveJobId = selectedJobId ?? activeDatasetJob?.id ?? jobsRes.data?.[0]?.id ?? null;

  const artifactsRes = useArtifacts(effectiveJobId);

  const effectiveKind = selectedKind ?? artifactsRes.data?.[0]?.kind ?? null;

  const selectedArtifact = useMemo(() => {
    return artifactsRes.data?.find((a) => a.kind === effectiveKind) ?? null;
  }, [artifactsRes.data, effectiveKind]);

  return (
    <div className="space-y-4">
      {/* Job picker */}
      <div className="tg-card p-3">
        <div className="flex items-center justify-between gap-2 mb-2 flex-wrap">
          <SectionTitle right={<FolderOpen className="size-3 text-[hsl(var(--muted-foreground))]" />}>
            <div className="flex items-center gap-2">
              <span className={`size-2 rounded-full ${activeDataset?.provenance === 'benchmark' ? 'bg-amber-400' : 'bg-emerald-400 animate-pulse'}`} />
              <span>Inspect Graph Artifacts</span>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded border border-[hsl(var(--border))] bg-[hsl(var(--background))] text-[hsl(var(--primary))] font-semibold">
                {activeDataset?.name ?? activeDatasetId ?? 'ACTIVE DATASET'}
              </span>
            </div>
          </SectionTitle>
          <span className="text-[10px] font-mono text-[hsl(var(--muted-foreground))]">
            PARQUET / JSON STORAGE DECK
          </span>
        </div>
        <div className="flex flex-wrap gap-2 mt-2">
          {(jobsRes.data ?? []).map((job) => {
            const isActiveJob = job.dataset_id === activeDatasetId;
            return (
              <button
                key={job.id}
                type="button"
                onClick={() => { setSelectedJobId(job.id); setSelectedKind(null); }}
                className={`px-2.5 py-1 text-[10px] font-mono border rounded flex items-center gap-1.5 transition-colors ${
                  effectiveJobId === job.id
                    ? 'border-[hsl(var(--primary)/0.6)] bg-[hsl(var(--info-bg))] font-semibold'
                    : 'border-[hsl(var(--border))] bg-[hsl(var(--card))] hover:border-[hsl(var(--primary)/0.4)]'
                }`}
                title={job.dataset_name}
              >
                {isActiveJob && <span className="size-1.5 rounded-full bg-emerald-400 animate-pulse" />}
                <span>{job.id}</span>
                <span className="text-[hsl(var(--muted-foreground))]">·</span>
                <span>{job.dataset_name}</span>
                {isActiveJob && (
                  <span className="text-[8px] uppercase tracking-wider px-1 py-0.2 rounded bg-emerald-500/15 text-emerald-400 font-bold ml-1">
                    ACTIVE
                  </span>
                )}
              </button>
            );
          })}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-3">
        {/* Artifact list */}
        <div className="lg:col-span-1">
          <div className="tg-card p-3">
            <SectionTitle>Artifacts</SectionTitle>
            {artifactsRes.state === 'loading' ? (
              <LoadingState label="Loading artifacts…" />
            ) : artifactsRes.state === 'failed' ? (
              <ErrorState message={artifactsRes.error ?? 'Failed'} />
            ) : (artifactsRes.data ?? []).length === 0 ? (
              <EmptyState title="No artifacts" description="Select a job to inspect its outputs" />
            ) : (
              <div className="mt-2 space-y-1">
                {(artifactsRes.data ?? []).map((a) => (
                  <button
                    key={a.kind}
                    type="button"
                    onClick={() => setSelectedKind(a.kind)}
                    className={`w-full text-left p-2 rounded border ${
                      effectiveKind === a.kind
                        ? 'border-[hsl(var(--primary)/0.6)] bg-[hsl(var(--info-bg))]'
                        : 'border-[hsl(var(--border))] bg-[hsl(var(--card))] hover:border-[hsl(var(--primary)/0.35)]'
                    }`}
                  >
                    <div className="flex items-center gap-1.5">
                      <FileText className="size-3 text-[hsl(var(--muted-foreground))]" />
                      <span className="text-[11px] mono font-semibold text-[hsl(var(--foreground))]">{ARTIFACT_LABELS[a.kind]}</span>
                    </div>
                    <div className="text-[10px] text-[hsl(var(--muted-foreground))] mt-0.5 font-mono">
                      {a.format} · {a.size_bytes == null ? 'Streamed export' : formatBytes(a.size_bytes)}{a.row_count !== null ? ` · ${formatInt(a.row_count)} rows` : ''}
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Artifact detail */}
        <div className="lg:col-span-3">
          <ArtifactDetail artifact={selectedArtifact} />
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Artifact detail panel
// ─────────────────────────────────────────────────────────────────────────────

function ArtifactDetail({ artifact }: { artifact: ArtifactMeta | null }) {
  if (!artifact) {
    return (
      <div className="tg-card p-4 h-full">
        <SectionTitle>Artifact Detail</SectionTitle>
        <EmptyState
          title="No artifact selected"
          description="Pick an artifact on the left to inspect its schema, row count, and preview."
          icon={Table2}
        />
      </div>
    );
  }
  return (
    <div className="space-y-3">
      <div className="tg-card p-4">
        <div className="flex items-center justify-between gap-2 mb-2">
          <SectionTitle>Artifact Detail</SectionTitle>
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded border border-[hsl(var(--border))]">{artifact.format}</span>
            <button
              type="button"
              onClick={() => {
                if (typeof window !== 'undefined') {
                  window.open(`${API_BASE_URL}${artifact.path}`, '_blank');
                }
              }}
              className="px-2 py-1 text-[10px] font-mono border border-[hsl(var(--primary)/0.4)] text-[hsl(var(--primary))] rounded flex items-center gap-1 hover:bg-[hsl(var(--info-bg))] transition-colors cursor-pointer"
              title={`Download ${artifact.kind} file`}
            >
              <Download className="size-3" />
              export
            </button>
          </div>
        </div>
        <div className="mono text-xs text-[hsl(var(--foreground))] break-all">{artifact.path}</div>
        <div className="text-[10px] text-[hsl(var(--muted-foreground))] mt-1">{ARTIFACT_DESCRIPTIONS[artifact.kind]}</div>
        <div className="grid grid-cols-3 gap-3 mt-3">
          <MiniField label="Format" value={artifact.format} />
          <MiniField label="Size" value={artifact.size_bytes == null ? 'Determined on download' : formatBytes(artifact.size_bytes)} />
          <MiniField label="Rows" value={artifact.row_count !== null ? formatInt(artifact.row_count) : '—'} />
        </div>
      </div>

      <div className="tg-card p-4">
        <SectionTitle right={<div className="text-[10px] text-[hsl(var(--muted-foreground))]">{artifact.schema.length} fields</div>}>
          Schema
        </SectionTitle>
        <div className="overflow-x-auto mt-2">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-left text-[10px] uppercase tracking-wide text-[hsl(var(--muted-foreground))] border-b border-[hsl(var(--border))]">
                <th className="py-1.5 pr-3">name</th>
                <th className="py-1.5 pr-3">type</th>
                <th className="py-1.5 pr-3">nullable</th>
                <th className="py-1.5 pr-3">description</th>
              </tr>
            </thead>
            <tbody>
              {artifact.schema.map((f) => (
                <tr key={f.name} className="border-b border-[hsl(var(--border)/0.5)]">
                  <td className="py-1.5 pr-3 mono font-semibold text-[hsl(var(--primary))]">{f.name}</td>
                  <td className="py-1.5 pr-3 mono text-[10px]">{f.type}</td>
                  <td className="py-1.5 pr-3 mono text-[10px]">
                    {f.nullable ? (
                      <span className="text-[hsl(var(--warning))]">yes</span>
                    ) : (
                      <span className="text-[hsl(var(--success))]">no</span>
                    )}
                  </td>
                  <td className="py-1.5 pr-3 text-[10px] text-[hsl(var(--muted-foreground))]">{f.description}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="tg-card p-4">
        <SectionTitle right={<div className="text-[10px] text-[hsl(var(--muted-foreground))]">{artifact.preview.length} preview rows</div>}>
          Preview
        </SectionTitle>
        <div className="overflow-x-auto mt-2">
          {artifact.preview.length === 0 ? (
            <EmptyState title="No preview rows" />
          ) : (
            <table className="w-full text-xs">
              <thead>
                <tr className="text-left text-[10px] uppercase tracking-wide text-[hsl(var(--muted-foreground))] border-b border-[hsl(var(--border))]">
                  {Object.keys(artifact.preview[0]).map((k) => (
                    <th key={k} className="py-1.5 pr-3 mono font-semibold">{k}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {artifact.preview.map((row, i) => (
                  <tr key={i} className="border-b border-[hsl(var(--border)/0.5)]">
                    {Object.entries(row).map(([k, v]) => (
                      <td key={k} className="py-1.5 pr-3 mono text-[10px] align-top max-w-xs truncate" title={String(v)}>
                        {formatPreviewCell(v)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}

function MiniField({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col gap-0.5">
      <div className="text-[9px] uppercase tracking-wide text-[hsl(var(--muted-foreground))] font-semibold">{label}</div>
      <div className="text-xs mono font-semibold text-[hsl(var(--foreground))]">{value}</div>
    </div>
  );
}

function formatPreviewCell(v: unknown): string {
  if (v === null || v === undefined) return '—';
  if (typeof v === 'number') return String(v);
  if (typeof v === 'string') return v.length > 80 ? v.slice(0, 80) + '…' : v;
  if (Array.isArray(v)) return `[${v.length} items]`;
  try {
    return JSON.stringify(v).slice(0, 80);
  } catch {
    return String(v);
  }
}
