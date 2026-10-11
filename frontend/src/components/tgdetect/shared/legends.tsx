'use client';

import { NODE_TYPE_META, NODE_TYPES, RELATION_TYPE_META, RELATION_TYPES } from '@/lib/tgdetect/constants';
import { cn } from '@/lib/utils';
import type { NodeType, RelationType } from '@/lib/tgdetect/types';

// ─────────────────────────────────────────────────────────────────────────────
// Node legend — full list of 8 node types with color dots
// ─────────────────────────────────────────────────────────────────────────────

const NODE_LEGEND_COLOR: Record<string, string> = {
  cyan: 'bg-[hsl(var(--info))]',
  amber: 'bg-[hsl(var(--warning))]',
  violet: 'bg-[hsl(var(--purple))]',
  green: 'bg-[hsl(var(--success))]',
  blue: 'bg-[hsl(var(--info))]',
  pink: 'bg-pink-500',
  teal: 'bg-[hsl(var(--teal))]',
  gray: 'bg-[hsl(var(--muted-foreground))]',
};

export function NodeLegend({
  counts,
  active,
  onToggle,
  className,
  compact = false,
}: {
  counts?: Partial<Record<NodeType, number>>;
  active?: Set<NodeType>;
  onToggle?: (t: NodeType) => void;
  className?: string;
  compact?: boolean;
}) {
  return (
    <div className={cn('flex flex-wrap gap-1.5', className)}>
      {NODE_TYPES.map((t) => {
        const meta = NODE_TYPE_META[t];
        const Icon = meta.icon;
        const isActive = !active || active.has(t);
        const count = counts?.[t];
        return (
          <button
            key={t}
            type="button"
            disabled={!onToggle}
            onClick={() => onToggle?.(t)}
            className={cn(
              'inline-flex items-center gap-1.5 px-2 py-0.5 rounded border text-[10px] font-mono transition-all',
              isActive
                ? 'border-[hsl(var(--border))] bg-[hsl(var(--card))] shadow-xs text-[hsl(var(--foreground))]'
                : 'border-dashed border-[hsl(var(--border))] bg-[hsl(var(--background))] opacity-50 text-[hsl(var(--muted-foreground))]',
              onToggle && 'hover:border-[hsl(var(--primary)/0.5)] cursor-pointer',
            )}
            title={meta.description}
          >
            <span className={cn('size-2 rounded-full', NODE_LEGEND_COLOR[meta.color])} />
            <Icon className="size-3 text-[hsl(var(--muted-foreground))]" />
            <span className="text-[hsl(var(--foreground))]">{meta.label}</span>
            {count !== undefined && (
              <span className="text-[hsl(var(--muted-foreground))] tabular-nums font-semibold">{count}</span>
            )}
            {!compact && count === undefined && (
              <span className="text-[hsl(var(--muted-foreground))]">{meta.id_prefix}:</span>
            )}
          </button>
        );
      })}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Relation legend — full list of 14 relation types
// ─────────────────────────────────────────────────────────────────────────────

const RELATION_LEGEND_COLOR: Record<string, string> = {
  access: 'bg-[hsl(var(--info))]',
  execution: 'bg-[hsl(var(--purple))]',
  filesystem: 'bg-[hsl(var(--success))]',
  network: 'bg-[hsl(var(--teal))]',
  attack: 'bg-[hsl(var(--danger))]',
};

export function RelationLegend({
  counts,
  active,
  onToggle,
  className,
}: {
  counts?: Partial<Record<RelationType, number>>;
  active?: Set<RelationType>;
  onToggle?: (r: RelationType) => void;
  className?: string;
}) {
  return (
    <div className={cn('flex flex-wrap gap-1.5', className)}>
      {RELATION_TYPES.map((r) => {
        const meta = RELATION_TYPE_META[r];
        const isActive = !active || active.has(r);
        const count = counts?.[r];
        return (
          <button
            key={r}
            type="button"
            disabled={!onToggle}
            onClick={() => onToggle?.(r)}
            className={cn(
              'inline-flex items-center gap-1.5 px-2 py-0.5 rounded border text-[10px] font-mono transition-all',
              isActive
                ? (meta.is_attack ? 'border-red-500/30 bg-red-500/5 text-[hsl(var(--foreground))]' : 'border-[hsl(var(--border))] bg-[hsl(var(--card))] shadow-xs text-[hsl(var(--foreground))]')
                : 'border-dashed border-[hsl(var(--border))] bg-[hsl(var(--background))] opacity-50 text-[hsl(var(--muted-foreground))]',
              onToggle && 'hover:border-[hsl(var(--primary)/0.5)] cursor-pointer',
            )}
            title={meta.description}
          >
            <span className={cn('size-2 rounded-full', RELATION_LEGEND_COLOR[meta.category])} />
            <span className="text-[hsl(var(--foreground))]">{meta.label}</span>
            {count !== undefined && (
              <span className="text-[hsl(var(--muted-foreground))] tabular-nums font-semibold">{count}</span>
            )}
          </button>
        );
      })}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Mini stat row — used in dashboard tiles
// ─────────────────────────────────────────────────────────────────────────────

export function MiniStat({
  label,
  value,
  color,
  className,
}: {
  label: string;
  value: string | number;
  color?: string;
  className?: string;
}) {
  return (
    <div className={cn('flex flex-col gap-0.5', className)}>
      <div className="text-[9px] uppercase tracking-wide text-[hsl(var(--muted-foreground))] font-semibold">
        {label}
      </div>
      <div className="text-sm font-mono font-semibold tabular-nums" style={color ? { color } : undefined}>
        {value}
      </div>
    </div>
  );
}
