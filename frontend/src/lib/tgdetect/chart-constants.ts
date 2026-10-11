/** Shared chart styling constants — kept DRY across all chart usages. */

export const CHART_TOOLTIP_STYLE = {
  contentStyle: {
    background: 'hsl(var(--card))',
    border: '1px solid hsl(var(--border))',
    borderRadius: '6px',
    fontSize: '11px',
    color: 'hsl(var(--foreground))',
    padding: '6px 8px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
  },
  labelStyle: { color: 'hsl(var(--muted-foreground))', fontSize: '10px', fontWeight: 600 },
  itemStyle: { color: 'hsl(var(--foreground))' },
} as const;

export const CHART_GRID_STYLE = {
  strokeDasharray: '3 3',
  stroke: 'hsl(var(--border))',
  opacity: 0.5,
} as const;

export const CHART_AXIS_STYLE = {
  tick: { fill: 'hsl(var(--muted-foreground))', fontSize: 10, fontFamily: 'ui-monospace, monospace' },
  axisLine: { stroke: 'hsl(var(--border))' },
  tickLine: false,
} as const;

export const CHART_COLORS = {
  cyan: 'hsl(var(--chart-1))',
  violet: 'hsl(var(--chart-2))',
  green: 'hsl(var(--chart-3))',
  amber: 'hsl(var(--chart-4))',
  red: 'hsl(var(--chart-5))',
  teal: 'hsl(var(--chart-6))',
} as const;

/** Map a color token string (matching `NODE_TYPE_META.color`) to a chart hex. */
export function tokenToChartHex(token: string): string {
  const m: Record<string, string> = {
    cyan: CHART_COLORS.cyan,
    blue: CHART_COLORS.cyan,
    amber: CHART_COLORS.amber,
    violet: CHART_COLORS.violet,
    green: CHART_COLORS.green,
    red: CHART_COLORS.red,
    teal: CHART_COLORS.teal,
    pink: '#ec4899',
    gray: '#7d95ae',
  };
  return m[token] ?? CHART_COLORS.cyan;
}
