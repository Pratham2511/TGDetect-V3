'use client';

import { useCallback, useMemo, useState, useSyncExternalStore } from 'react';
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Boxes,
  Database,
  GitBranch,
  LayoutDashboard,
  ListTree,
  Menu,
  Moon,
  Network,
  RefreshCw,
  Sun,
  X,
  type LucideIcon,
} from 'lucide-react';
import { BrandMark } from '@/components/cadets/BrandMark';
import { OverviewPage } from '@/components/tgdetect/overview/OverviewPage';
import { EventsPage } from '@/components/tgdetect/events/EventsPage';
import { GraphPage } from '@/components/tgdetect/graph/GraphPage';
import { ChainsPage } from '@/components/tgdetect/chains/ChainsPage';
import { DatasetsPage } from '@/components/tgdetect/datasets/DatasetsPage';
import { ArtifactsPage } from '@/components/tgdetect/artifacts/ArtifactsPage';
import { ModelPage } from '@/components/tgdetect/model/ModelPage';
import { AnalyticsPage } from '@/components/tgdetect/analytics/AnalyticsPage';
import { useTheme } from '@/lib/theme-context';
import { useModel } from '@/lib/model-context';
import { useDataset } from '@/lib/dataset-context';
import { useGraphStats } from '@/lib/tgdetect/services/hooks';
import { formatInt } from '@/lib/tgdetect/formatters';
import { API_BASE_URL } from '@/lib/tgdetect/api/client';
import { ChevronDown, Cpu, ShieldCheck } from 'lucide-react';

interface NavItem {
  id: string;
  label: string;
  icon: LucideIcon;
  description: string;
  color: string;
}

const NAV_ITEMS: NavItem[] = [
  { id: 'overview', label: 'Overview', icon: LayoutDashboard, description: 'Backend pipeline summary', color: 'cyan' },
  { id: 'events', label: 'Events', icon: ListTree, description: 'TGEvent investigation', color: 'cyan' },
  { id: 'graph', label: 'Graph', icon: Network, description: 'Temporal heterogeneous graph', color: 'cyan' },
  { id: 'chains', label: 'Attack Chains', icon: GitBranch, description: 'Chain reconstruction', color: 'teal' },
  { id: 'datasets', label: 'Datasets', icon: Database, description: 'Processing & jobs', color: 'amber' },
  { id: 'artifacts', label: 'Artifacts', icon: Boxes, description: 'Parquet / JSON outputs', color: 'violet' },
  { id: 'model', label: 'TGNN / Model', icon: Activity, description: 'GAT + causal node memory', color: 'purple' },
  { id: 'analytics', label: 'Analytics', icon: BarChart3, description: 'Event / graph / attack / dataset', color: 'teal' },
];

interface PageCtx {
  // For events page
  eventId?: string;
  labelFilter?: 0 | 1;
  nodeId?: string;
  // For chains page
  chainId?: string;
  // For artifacts page
  jobId?: string;
}

function readPage() {
  const params = new URLSearchParams(window.location.search);
  const page = params.get('page') || window.location.hash.replace('#', '');
  return NAV_ITEMS.some(item => item.id === page) ? page : 'overview';
}

function subscribePage(onChange: () => void) {
  const events = ['popstate', 'hashchange', 'tgdetect-legacy-navigation'];
  events.forEach(event => window.addEventListener(event, onChange));
  return () => events.forEach(event => window.removeEventListener(event, onChange));
}

export default function Home() {
  const activePage = useSyncExternalStore(subscribePage, readPage, () => 'overview');
  const [pageCtx, setPageCtx] = useState<PageCtx>({});
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const theme = useTheme();
  const {
    activeDatasetId,
    activeDataset,
    isSwitching,
    switchingMessage,
    targetDatasetName,
  } = useDataset();
  const {
    models,
    activeModelId,
    activeModel,
    setActiveModelId,
    apiHealth,
    healthError,
    refreshHealth,
  } = useModel();
  const statsRes = useGraphStats();

  const navigate = useCallback((page: string, ctx?: Record<string, unknown>) => {
    setPageCtx((ctx as PageCtx) ?? {});
    if (typeof window !== 'undefined') {
      const url = new URL(window.location.href);
      url.searchParams.set('page', page);
      window.history.pushState({}, '', url.toString());
      window.dispatchEvent(new Event('tgdetect-legacy-navigation'));
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  }, []);

  const activeItem = useMemo(() => NAV_ITEMS.find((i) => i.id === activePage) ?? NAV_ITEMS[0], [activePage]);

  return (
    <div className="min-h-screen flex flex-col bg-[hsl(var(--background))] text-[hsl(var(--foreground))] relative overflow-x-hidden">
      <div className="flex flex-1">
        {/* Sidebar */}
        <aside
          id="tour-sidebar"
          className="sidebar-shell w-60 flex-shrink-0 hidden md:flex flex-col"
        >
          {/* Logo / Title */}
          <div className="p-3.5 border-b border-[hsl(var(--sidebar-border))] bg-[hsl(var(--sidebar-bg))]">
            <div className="flex items-center gap-2.5">
              <div className="size-7 rounded bg-[hsl(var(--primary)/0.2)] border border-[hsl(var(--primary)/0.4)] flex items-center justify-center flex-shrink-0">
                <BrandMark size={24} />
              </div>
              <div className="flex flex-col min-w-0">
                <div className="flex items-center gap-1.5">
                  <span className="text-xs font-bold text-[hsl(var(--sidebar-logo-text))] tracking-tight">TGDetect</span>
                  <span className="text-[9px] font-mono px-1 py-0.2 rounded bg-sky-500/10 text-sky-400 border border-sky-500/20">v1.0</span>
                </div>
                <span className="text-[9px] uppercase tracking-wider text-[hsl(var(--sidebar-logo-sub))] font-mono truncate">
                  Temporal Graph Security
                </span>
              </div>
            </div>
          </div>

          {/* Nav */}
          <nav className="flex-1 overflow-y-auto py-2.5 px-2 space-y-0.5">
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon;
              const isActive = activePage === item.id;
              return (
                <button
                  key={item.id}
                  id={`nav-${item.id}`}
                  type="button"
                  onClick={() => navigate(item.id)}
                  className={`w-full text-left px-2.5 py-2 rounded-md flex items-center gap-2.5 transition-all text-xs ${
                    isActive
                      ? 'bg-[hsl(var(--sidebar-item-active))] text-[hsl(var(--sidebar-text-active))] font-semibold shadow-xs border-l-2 border-[hsl(var(--sidebar-text-active))] pl-2'
                      : 'text-[hsl(var(--sidebar-text))] hover:bg-[hsl(var(--sidebar-item-hover))] hover:text-[hsl(var(--sidebar-text-active))]'
                  }`}
                >
                  <Icon
                    className={`size-4 flex-shrink-0 ${
                      isActive ? 'text-[hsl(var(--sidebar-text-active))]' : 'text-[hsl(var(--sidebar-text))] opacity-80'
                    }`}
                  />
                  <div className="flex flex-col min-w-0">
                    <span className="truncate leading-none">{item.label}</span>
                    <span className="text-[9px] text-[hsl(var(--sidebar-logo-sub))] truncate mt-0.5">
                      {item.description}
                    </span>
                  </div>
                </button>
              );
            })}
          </nav>

          {/* Sidebar footer — backend contract & pipeline telemetry */}
          <div className="p-3 border-t border-[hsl(var(--sidebar-border))] space-y-2 bg-[hsl(var(--sidebar-bg))]">
            <div className="flex items-center justify-between text-[9px] uppercase tracking-wider text-[hsl(var(--sidebar-logo-sub))] font-mono">
              <span>Artifact-backed API</span>
              <span className="text-[8px] text-emerald-400 font-bold">{apiHealth === 'connected' ? 'CONNECTED' : 'OFFLINE'}</span>
            </div>
            <div className="space-y-1 text-[10px] font-mono">
              <div className="flex items-center justify-between">
                <span className="text-[hsl(var(--sidebar-text))]">graph_builder</span>
                <span className="px-1.5 py-0.2 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-[9px]">
                  72-feature schema
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-[hsl(var(--sidebar-text))]">TGNN</span>
                <span className="px-1.5 py-0.2 rounded bg-purple-500/10 text-purple-300 border border-purple-500/20 text-[9px]">
                  Causal GAT ({activeModel?.target?.toUpperCase() ?? 'NODE'})
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-[hsl(var(--sidebar-text))]">Params</span>
                <span className="px-1.5 py-0.2 rounded bg-cyan-500/10 text-cyan-300 border border-cyan-500/20 text-[9px]">
                  {formatInt(activeModel?.trainable_parameters ?? 0)}
                </span>
              </div>
            </div>
            {statsRes.data && (
              <div className="text-[9px] text-[hsl(var(--sidebar-logo-sub))] pt-1.5 border-t border-[hsl(var(--sidebar-border)/0.5)] font-mono flex items-center justify-between">
                <span>{formatInt(statsRes.data.graph.total_events)} events</span>
                <span>{formatInt(statsRes.data.attacks.total_chains)} chains</span>
              </div>
            )}
          </div>
        </aside>

        {/* Main */}
        <main className="flex-1 flex flex-col min-w-0">
          {/* Header */}
          <header className="header-shell sticky top-0 z-10 px-5 py-2.5 flex items-center justify-between gap-3 shadow-xs">
            <div className="flex items-center gap-2.5 min-w-0">
              <button
                type="button"
                onClick={() => setMobileMenuOpen((prev) => !prev)}
                className="md:hidden size-7 rounded border border-[hsl(var(--border))] bg-[hsl(var(--card))] flex items-center justify-center text-[hsl(var(--foreground))] hover:bg-[hsl(var(--card-hover))] transition-colors flex-shrink-0"
                aria-label="Toggle mobile menu"
              >
                {mobileMenuOpen ? <X className="size-3.5" /> : <Menu className="size-3.5" />}
              </button>
              <activeItem.icon className="size-4 text-[hsl(var(--primary))] flex-shrink-0" />
              <div className="flex items-center gap-2 min-w-0">
                <h1 className="text-xs font-semibold uppercase tracking-wider text-[hsl(var(--foreground))]">{activeItem.label}</h1>
                <span className="text-[10px] text-[hsl(var(--muted-foreground))]">/</span>
                <span className="text-xs text-[hsl(var(--muted-foreground))] truncate">{activeItem.description}</span>
              </div>
            </div>
            <div className="flex items-center gap-2 flex-wrap">
              {/* Health status badge */}
              {apiHealth === 'connected' ? (
                <span className="inline-flex items-center gap-1.5 text-[10px] font-mono px-2 py-0.5 rounded border border-emerald-500/30 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400" title="Connected to TGDetect FastAPI backend">
                  <span className="size-1.5 rounded-full bg-emerald-500" />
                  <span>CONNECTED</span>
                </span>
              ) : apiHealth === 'offline' ? (
                <span className="inline-flex items-center gap-1.5 text-[10px] font-mono px-2 py-0.5 rounded border border-rose-500/30 bg-rose-500/10 text-rose-600 dark:text-rose-400" title="Cannot reach backend">
                  <span className="size-1.5 rounded-full bg-rose-500 animate-pulse" />
                  <span>OFFLINE</span>
                </span>
              ) : (
                <span className="inline-flex items-center gap-1.5 text-[10px] font-mono px-2 py-0.5 rounded border border-[hsl(var(--border))] bg-[hsl(var(--card))] text-[hsl(var(--muted-foreground))]">
                  <span className="size-1.5 rounded-full bg-amber-500 animate-pulse" />
                  <span>CONNECTING</span>
                </span>
              )}

              {/* Persistent Dataset Provenance Indicator (Live vs Benchmark) */}
              {activeDatasetId === 'dedale-host-memory' ? (
                <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-md border border-amber-500/30 bg-amber-500/10" title="Authoritative Benchmark Reference Partition">
                  <span className="text-amber-500 text-xs">◈</span>
                  <div className="flex flex-col text-left leading-tight">
                    <span className="text-[9px] uppercase tracking-wider text-amber-500 font-bold font-mono">BENCHMARK DATASET</span>
                    <span className="text-[11px] font-mono font-semibold text-[hsl(var(--foreground))] truncate max-w-[200px]">
                      {activeDataset?.name || 'Loading dataset…'}
                    </span>
                  </div>
                  {statsRes.data && (
                    <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-400 border border-amber-500/30 font-semibold">
                      {formatInt(statsRes.data.graph.total_events)} EVENTS
                    </span>
                  )}
                </div>
              ) : (
                <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-md border border-emerald-500/40 bg-emerald-500/10 shadow-[0_0_12px_rgba(16,185,129,0.12)]" title="Live User Ingested Telemetry Partition">
                  <span className="size-2 rounded-full bg-emerald-500 animate-pulse shrink-0" />
                  <div className="flex flex-col text-left leading-tight">
                    <span className="text-[9px] uppercase tracking-wider text-emerald-500 font-bold font-mono">LIVE DATASET</span>
                    <span className="text-[11px] font-mono font-semibold text-[hsl(var(--foreground))] truncate max-w-[200px]">
                      {activeDataset?.name || activeDatasetId}
                    </span>
                  </div>
                  {statsRes.data && (
                    <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 font-semibold">
                      {formatInt(statsRes.data.graph.total_events)} EVT • {formatInt(statsRes.data.graph.total_nodes)} NODES
                    </span>
                  )}
                </div>
              )}

              <select aria-label="Global active model" value={activeModelId} onChange={e => setActiveModelId(e.target.value)} className="max-w-48 bg-[hsl(var(--background))] border border-[hsl(var(--border))] rounded px-2 py-1 text-xs">{models.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}</select>
              {/* Authoritative Single Production Model Badge (Clearly distinct from live dataset) */}
              <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-md border border-cyan-500/30 bg-cyan-500/10 shadow-[0_0_12px_rgba(0,242,254,0.08)]" title="Authoritative Single Production Model">
                <Cpu className="size-3.5 text-cyan-400 flex-shrink-0 animate-pulse" />
                <div className="flex flex-col text-left leading-tight">
                  <div className="flex items-center gap-1.5">
                    <span className="text-[11px] font-mono font-bold tracking-wide text-cyan-400 dark:text-cyan-300">
                      {activeModel?.name ?? 'Loading model…'}
                    </span>
                    <span className="text-[8px] font-mono px-1 py-0.2 rounded bg-indigo-500/20 text-indigo-400 dark:text-indigo-300 border border-indigo-500/40 font-semibold uppercase">
                      {activeModel?.target?.toUpperCase() ?? 'EDGE'}
                    </span>
                  </div>
                  <span className="text-[9px] font-mono text-[hsl(var(--muted-foreground))]">
                    MODEL BENCHMARK · {activeModel?.trainable_parameters ? `${activeModel.trainable_parameters.toLocaleString()} params` : 'LOADED'}
                  </span>
                </div>
              </div>

              <button
                type="button"
                onClick={theme.toggleTheme}
                className="size-7 rounded border border-[hsl(var(--border))] bg-[hsl(var(--card))] flex items-center justify-center hover:bg-[hsl(var(--card-hover))] transition-colors"
                title={theme.theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
                aria-label="Toggle theme"
              >
                {theme.theme === 'dark' ? (
                  <Sun className="size-3.5 text-[hsl(var(--warning))]" />
                ) : (
                  <Moon className="size-3.5 text-[hsl(var(--primary))]" />
                )}
              </button>
            </div>
          </header>

            {/* Mobile Navigation Drawer */}
            {mobileMenuOpen && (
              <div className="md:hidden border-b border-[hsl(var(--sidebar-border))] bg-[hsl(var(--sidebar-bg))] p-3 space-y-1 animate-in fade-in slide-in-from-top-2 duration-150">
                {NAV_ITEMS.map((item) => {
                  const Icon = item.icon;
                  const isActive = activePage === item.id;
                  return (
                    <button
                      key={item.id}
                      type="button"
                      onClick={() => {
                        navigate(item.id);
                        setMobileMenuOpen(false);
                      }}
                      className={`w-full text-left px-3 py-2 rounded-md flex items-center gap-2.5 transition-all text-xs ${
                        isActive
                          ? 'bg-[hsl(var(--sidebar-item-active))] text-[hsl(var(--sidebar-text-active))] font-semibold border-l-2 border-[hsl(var(--sidebar-text-active))]'
                          : 'text-[hsl(var(--sidebar-text))] hover:bg-[hsl(var(--sidebar-item-hover))] hover:text-[hsl(var(--sidebar-text-active))]'
                      }`}
                    >
                      <Icon className={`size-4 flex-shrink-0 ${isActive ? 'text-[hsl(var(--sidebar-text-active))]' : 'opacity-70'}`} />
                      <div className="flex flex-col min-w-0">
                        <span className="truncate leading-none font-medium">{item.label}</span>
                        <span className="text-[9px] text-[hsl(var(--sidebar-logo-sub))] truncate mt-0.5">{item.description}</span>
                      </div>
                    </button>
                  );
                })}
              </div>
            )}

          {/* Page content */}
          <div className="flex-1 p-4 overflow-x-hidden">
            {apiHealth === 'offline' ? (
              <div className="max-w-md mx-auto my-12 p-6 rounded border border-rose-500/30 bg-[hsl(var(--card))] shadow-sm text-center">
                <div className="size-10 rounded-full bg-rose-500/10 border border-rose-500/30 text-rose-500 flex items-center justify-center mx-auto mb-3">
                  <AlertTriangle className="size-5" />
                </div>
                <h2 className="text-sm font-semibold tracking-wider uppercase text-rose-600 dark:text-rose-400 mb-1">
                  TGDetect API OFFLINE
                </h2>
                <p className="text-xs text-[hsl(var(--muted-foreground))] mb-4 font-mono">
                  Could not connect to {API_BASE_URL}
                </p>
                {healthError && (
                  <p className="text-[11px] font-mono text-[hsl(var(--muted-foreground))] bg-[hsl(var(--background))] p-2.5 rounded mb-4 break-all text-left">
                    {healthError}
                  </p>
                )}
                <button
                  type="button"
                  onClick={() => {
                    refreshHealth();
                    statsRes.reload();
                  }}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded bg-[hsl(var(--primary))] text-white text-xs font-medium hover:opacity-90 transition-opacity"
                >
                  <RefreshCw className="size-3.5" />
                  <span>Retry Connection</span>
                </button>
              </div>
            ) : (
              <>
                {activePage === 'overview' && <OverviewPage onNavigate={navigate} />}
                {activePage === 'events' && (
                  <EventsPage
                    initialFilter={
                      pageCtx.labelFilter !== undefined
                        ? { labels: [pageCtx.labelFilter] }
                        : pageCtx.nodeId
                          ? {}
                          : undefined
                    }
                    initialEventId={pageCtx.eventId ?? null}
                    onNavigate={navigate}
                  />
                )}
                {activePage === 'graph' && <GraphPage onNavigate={navigate} />}
                {activePage === 'chains' && <ChainsPage initialChainId={pageCtx.chainId ?? null} onNavigate={navigate} />}
                {activePage === 'datasets' && <DatasetsPage onNavigate={navigate} />}
                {activePage === 'artifacts' && <ArtifactsPage initialJobId={pageCtx.jobId ?? null} />}
                {activePage === 'model' && <ModelPage />}
                {activePage === 'analytics' && <AnalyticsPage />}
              </>
            )}
          </div>
        </main>
      </div>

      {/* Tactical Dataset Change Transition Overlay */}
      {isSwitching && (
        <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-xs flex items-center justify-center animate-in fade-in duration-150">
          <div className="tg-panel p-6 max-w-md w-full mx-4 border border-[hsl(var(--primary)/0.4)] shadow-2xl text-center space-y-4">
            <div className="size-12 rounded-full bg-[hsl(var(--primary)/0.15)] border border-[hsl(var(--primary)/0.4)] flex items-center justify-center mx-auto text-[hsl(var(--primary))] animate-spin">
              <RefreshCw className="size-6" />
            </div>
            <div className="space-y-1">
              <div className="text-[10px] uppercase tracking-widest font-mono text-[hsl(var(--primary))] font-bold">
                ACTIVE DATASET CHANGED
              </div>
              <h3 className="text-sm font-bold text-[hsl(var(--foreground))] font-mono">
                {targetDatasetName}
              </h3>
              <p className="text-xs font-mono text-[hsl(var(--muted-foreground))] mt-2">
                {switchingMessage || 'Synchronizing telemetry views, graph topology, and event stream...'}
              </p>
            </div>
            <div className="h-1.5 w-full bg-[hsl(var(--muted)/0.3)] rounded-full overflow-hidden">
              <div className="h-full bg-[hsl(var(--primary))] animate-pulse w-4/5 rounded-full" />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
