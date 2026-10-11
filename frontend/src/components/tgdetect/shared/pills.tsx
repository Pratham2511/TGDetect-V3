'use client';

import { TGDetectLoader } from './TGDetectLoader';

import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';
import { NODE_TYPE_META, RELATION_TYPE_META } from '@/lib/tgdetect/constants';
import {
  labelText,
  processingStateLabel,
  relationLabel,
} from '@/lib/tgdetect/formatters';
import type {
  ChainStrategy,
  EventLabel,
  NodeType,
  ProcessingState,
  RelationType,
} from '@/lib/tgdetect/types';
import type { LucideIcon } from 'lucide-react';

// ─────────────────────────────────────────────────────────────────────────────
// Label pill — 0 = benign (green), 1 = malicious (red)
// ─────────────────────────────────────────────────────────────────────────────

export function LabelPill({ label, className }: { label: EventLabel | null; className?: string }) {
  if (label == null) return <span className={cn('text-[10px] text-[hsl(var(--muted-foreground))]', className)}>UNKNOWN · No ground truth</span>;
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono font-semibold border',
        label === 1
          ? 'bg-[hsl(var(--danger-bg))] text-[hsl(var(--danger))] border-[hsl(var(--danger)/0.25)]'
          : 'bg-[hsl(var(--success-bg))] text-[hsl(var(--success))] border-[hsl(var(--success)/0.25)]',
        className,
      )}
    >
      {label === 1 ? 'MAL' : 'BEN'} · {labelText(label)}
    </span>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Node type pill — shows icon + label
// ─────────────────────────────────────────────────────────────────────────────

const NODE_COLOR_CLASSES: Record<string, { bg: string; fg: string; border: string }> = {
  cyan: { bg: 'bg-[hsl(var(--info-bg))]', fg: 'text-[hsl(var(--info))]', border: 'border-[hsl(var(--info)/0.25)]' },
  amber: { bg: 'bg-[hsl(var(--warning-bg))]', fg: 'text-[hsl(var(--warning))]', border: 'border-[hsl(var(--warning)/0.25)]' },
  violet: { bg: 'bg-[hsl(var(--purple-bg))]', fg: 'text-[hsl(var(--purple))]', border: 'border-[hsl(var(--purple)/0.25)]' },
  green: { bg: 'bg-[hsl(var(--success-bg))]', fg: 'text-[hsl(var(--success))]', border: 'border-[hsl(var(--success)/0.25)]' },
  blue: { bg: 'bg-[hsl(var(--info-bg))]', fg: 'text-[hsl(var(--info))]', border: 'border-[hsl(var(--info)/0.25)]' },
  pink: { bg: 'bg-pink-500/10', fg: 'text-pink-600 dark:text-pink-400', border: 'border-pink-500/25' },
  teal: { bg: 'bg-[hsl(173_55%/95%)] dark:bg-[hsl(173_58%/8%)]', fg: 'text-[hsl(var(--teal))]', border: 'border-[hsl(var(--teal)/0.25)]' },
  gray: { bg: 'bg-[hsl(var(--muted))]', fg: 'text-[hsl(var(--muted-foreground))]', border: 'border-[hsl(var(--border))]' },
};

export function NodeTypePill({
  type,
  icon: Icon,
  showIcon = true,
  showLabel = true,
  className,
}: {
  type: NodeType;
  icon?: LucideIcon;
  showIcon?: boolean;
  showLabel?: boolean;
  className?: string;
}) {
  const meta = NODE_TYPE_META[type];
  const Icon2 = Icon ?? meta.icon;
  const c = NODE_COLOR_CLASSES[meta.color] ?? NODE_COLOR_CLASSES.gray;
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium border',
        c.bg, c.fg, c.border,
        className,
      )}
      title={meta.description}
    >
      {showIcon && <Icon2 className="size-3" />}
      {showLabel && <span>{meta.label}</span>}
    </span>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Relation type pill
// ─────────────────────────────────────────────────────────────────────────────

const RELATION_CATEGORY_CLASSES: Record<string, { bg: string; fg: string; border: string }> = {
  access: { bg: 'bg-[hsl(var(--info-bg))]', fg: 'text-[hsl(var(--info))]', border: 'border-[hsl(var(--info)/0.25)]' },
  execution: { bg: 'bg-[hsl(var(--purple-bg))]', fg: 'text-[hsl(var(--purple))]', border: 'border-[hsl(var(--purple)/0.25)]' },
  filesystem: { bg: 'bg-[hsl(var(--success-bg))]', fg: 'text-[hsl(var(--success))]', border: 'border-[hsl(var(--success)/0.25)]' },
  network: { bg: 'bg-[hsl(173_55%/95%)] dark:bg-[hsl(173_58%/8%)]', fg: 'text-[hsl(var(--teal))]', border: 'border-[hsl(var(--teal)/0.25)]' },
  attack: { bg: 'bg-[hsl(var(--danger-bg))]', fg: 'text-[hsl(var(--danger))]', border: 'border-[hsl(var(--danger)/0.25)]' },
};

export function RelationPill({
  relation,
  className,
}: {
  relation: RelationType | string;
  className?: string;
}) {
  const meta = relation in RELATION_TYPE_META
    ? RELATION_TYPE_META[relation as RelationType]
    : { label: relation, category: 'access' as const, is_attack: false };
  const c = RELATION_CATEGORY_CLASSES[meta.category] ?? RELATION_CATEGORY_CLASSES.access;
  return (
    <span
      className={cn(
        'inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-mono font-semibold border uppercase tracking-wide',
        c.bg, c.fg, c.border,
        className,
      )}
      title={meta.is_attack ? 'Attack relation' : meta.label}
    >
      {relationLabel(relation)}
    </span>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Strategy pill
// ─────────────────────────────────────────────────────────────────────────────

const STRATEGY_CLASSES: Record<ChainStrategy, { bg: string; fg: string; border: string }> = {
  chain_id: { bg: 'bg-[hsl(var(--info-bg))]', fg: 'text-[hsl(var(--info))]', border: 'border-[hsl(var(--info)/0.25)]' },
  causal_parent: { bg: 'bg-[hsl(var(--purple-bg))]', fg: 'text-[hsl(var(--purple))]', border: 'border-[hsl(var(--purple)/0.25)]' },
  entity_time: { bg: 'bg-[hsl(173_55%/95%)] dark:bg-[hsl(173_58%/8%)]', fg: 'text-[hsl(var(--teal))]', border: 'border-[hsl(var(--teal)/0.25)]' },
};

export function StrategyPill({ strategy, className }: { strategy: ChainStrategy; className?: string }) {
  const c = STRATEGY_CLASSES[strategy];
  return (
    <span
      className={cn(
        'inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-mono font-semibold border uppercase tracking-wide',
        c.bg, c.fg, c.border,
        className,
      )}
    >
      {strategy}
    </span>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Processing state pill
// ─────────────────────────────────────────────────────────────────────────────

const STATE_BADGE_CLASSES: Record<string, string> = {
  success: 'badge-success',
  warning: 'badge-warning',
  danger: 'badge-danger',
  info: 'badge-info',
  purple: 'badge-purple',
  gray: 'bg-[hsl(var(--muted))] text-[hsl(var(--muted-foreground))] border border-[hsl(var(--border))]',
};

export function StatePill({ state, className }: { state: ProcessingState; className?: string }) {
  // Map state → badge class via PROCESSING_STATE_META badge token
  // We import lazily to avoid cycle.
  const badgeClass = stateBadgeClass(state);
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono font-semibold border',
        badgeClass,
        className,
      )}
    >
      {processingStateLabel(state)}
    </span>
  );
}

function stateBadgeClass(state: ProcessingState): string {
  const map: Record<ProcessingState, string> = {
    idle: STATE_BADGE_CLASSES.gray,
    queued: STATE_BADGE_CLASSES.gray,
    parsing: STATE_BADGE_CLASSES.info,
    normalizing: STATE_BADGE_CLASSES.info,
    labeling: STATE_BADGE_CLASSES.info,
    building_graph: STATE_BADGE_CLASSES.info,
    exporting: STATE_BADGE_CLASSES.info,
    reconstructing_chains: STATE_BADGE_CLASSES.purple,
    completed: STATE_BADGE_CLASSES.success,
    failed: STATE_BADGE_CLASSES.danger,
  };
  return map[state];
}

// ─────────────────────────────────────────────────────────────────────────────
// MonoId — for event IDs, node IDs, chain IDs
// ─────────────────────────────────────────────────────────────────────────────

export function MonoId({
  children,
  className,
  title,
  truncateAt = 24,
}: {
  children: string;
  className?: string;
  title?: string;
  truncateAt?: number;
}) {
  const display = children.length > truncateAt ? children.slice(0, truncateAt - 1) + '…' : children;
  return (
    <span
      className={cn('mono text-[11px] text-[hsl(var(--foreground))]', className)}
      title={title ?? children}
    >
      {display}
    </span>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Tag-like Badge
// ─────────────────────────────────────────────────────────────────────────────

export function MiniTag({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <Badge
      variant="secondary"
      className={cn('text-[10px] font-mono py-0 px-1.5 h-4', className)}
    >
      {children}
    </Badge>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Stat block — compact metric tile
// ─────────────────────────────────────────────────────────────────────────────

export function StatBlock({
  label,
  value,
  hint,
  icon: Icon,
  color = 'info',
  className,
}: {
  label: string;
  value: string | number;
  hint?: string;
  icon?: LucideIcon;
  color?: 'info' | 'success' | 'warning' | 'danger' | 'purple' | 'teal' | 'gray';
  className?: string;
}) {
  const colorClass = {
    info: 'text-[hsl(var(--info))]',
    success: 'text-[hsl(var(--success))]',
    warning: 'text-[hsl(var(--warning))]',
    danger: 'text-[hsl(var(--danger))]',
    purple: 'text-[hsl(var(--purple))]',
    teal: 'text-[hsl(var(--teal))]',
    gray: 'text-[hsl(var(--muted-foreground))]',
  }[color];
  return (
    <div className={cn('tg-card p-3 flex flex-col gap-1', className)}>
      <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wide text-[hsl(var(--muted-foreground))] font-semibold">
        {Icon && <Icon className={cn('size-3', colorClass)} />}
        <span>{label}</span>
      </div>
      <div className={cn('metric-value-sm font-mono', colorClass)}>
        {value}
      </div>
      {hint && (
        <div className="text-[10px] text-[hsl(var(--muted-foreground))]">{hint}</div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Section title — small uppercase label
// ─────────────────────────────────────────────────────────────────────────────

export function SectionTitle({
  children,
  className,
  right,
}: {
  children: React.ReactNode;
  className?: string;
  right?: React.ReactNode;
}) {
  return (
    <div className={cn('flex items-center justify-between gap-2', className)}>
      <h3 className="section-title">{children}</h3>
      {right && <div>{right}</div>}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Empty / Loading / Error state shells
// ─────────────────────────────────────────────────────────────────────────────

export function EmptyState({
  title,
  description,
  icon: Icon,
  action,
  className,
}: {
  title: string;
  description?: string;
  icon?: LucideIcon;
  action?: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn('flex flex-col items-center justify-center py-12 px-6 text-center', className)}>
      {Icon && <Icon className="size-8 text-[hsl(var(--muted-foreground))] mb-2 opacity-60" />}
      <div className="text-sm font-semibold text-[hsl(var(--foreground))]">{title}</div>
      {description && (
        <div className="text-xs text-[hsl(var(--muted-foreground))] mt-1 max-w-md">{description}</div>
      )}
      {action && <div className="mt-3">{action}</div>}
    </div>
  );
}

export function LoadingState({ label = 'Streaming telemetry & temporal graph...', className }: { label?: string; className?: string }) {
  return (
    <div className={cn('py-8', className)}>
      <TGDetectLoader message={label} />
    </div>
  );
}

export function ErrorState({
  message,
  className,
}: {
  message: string;
  className?: string;
}) {
  return (
    <div className={cn('flex items-center justify-center py-12 text-xs text-[hsl(var(--danger))]', className)}>
      {message}
    </div>
  );
}
