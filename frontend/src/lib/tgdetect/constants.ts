/**
 * TGDetect constants — backend-exact enums + UI presentation metadata.
 *
 * The enum VALUES here must match the backend Python enums verbatim
 * (case-sensitive). The display metadata (label, icon, color token) is
 * UI-only and lives here so components can reference it consistently.
 */

import {
  Activity,
  Cpu,
  FileText,
  Globe,
  HardDrive,
  HelpCircle,
  Network,
  User,
  type LucideIcon,
} from 'lucide-react';

import type {
  ChainStrategy,
  DatasetKind,
  LabelMode,
  NodeType,
  ProcessingState,
  RelationType,
} from './types';

// ─────────────────────────────────────────────────────────────────────────────
// Node types (graph_builder/schema.py:17)
// ─────────────────────────────────────────────────────────────────────────────

export const NODE_TYPES: NodeType[] = [
  'USER',
  'HOST',
  'PROCESS',
  'FILE',
  'IP',
  'DOMAIN',
  'SOCKET',
  'UNKNOWN',
];

export interface NodeTypeMeta {
  type: NodeType;
  label: string;
  icon: LucideIcon;
  /** Color token name — resolved via CSS vars in components. */
  color: 'cyan' | 'amber' | 'violet' | 'green' | 'blue' | 'pink' | 'teal' | 'gray';
  description: string;
  /** Canonical id prefix from `NODE_TYPE_PREFIX` (UNKNOWN → "entity"). */
  id_prefix: string;
}

export const NODE_TYPE_META: Record<NodeType, NodeTypeMeta> = {
  USER: {
    type: 'USER',
    label: 'User',
    icon: User,
    color: 'cyan',
    description: 'User account / principal',
    id_prefix: 'user',
  },
  HOST: {
    type: 'HOST',
    label: 'Host',
    icon: HardDrive,
    color: 'amber',
    description: 'Endpoint / machine',
    id_prefix: 'host',
  },
  PROCESS: {
    type: 'PROCESS',
    label: 'Process',
    icon: Cpu,
    color: 'violet',
    description: 'Executable / process image',
    id_prefix: 'process',
  },
  FILE: {
    type: 'FILE',
    label: 'File',
    icon: FileText,
    color: 'green',
    description: 'Filesystem object',
    id_prefix: 'file',
  },
  IP: {
    type: 'IP',
    label: 'IP',
    icon: Network,
    color: 'blue',
    description: 'Network-layer address',
    id_prefix: 'ip',
  },
  DOMAIN: {
    type: 'DOMAIN',
    label: 'Domain',
    icon: Globe,
    color: 'pink',
    description: 'DNS hostname',
    id_prefix: 'domain',
  },
  SOCKET: {
    type: 'SOCKET',
    label: 'Socket',
    icon: Activity,
    color: 'teal',
    description: 'Network socket / port endpoint',
    id_prefix: 'socket',
  },
  UNKNOWN: {
    type: 'UNKNOWN',
    label: 'Unknown',
    icon: HelpCircle,
    color: 'gray',
    description: 'Unresolved entity',
    id_prefix: 'entity',
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// Relation types (graph_builder/schema.py:28)
// ─────────────────────────────────────────────────────────────────────────────

export const RELATION_TYPES: RelationType[] = [
  'OBSERVED_ON',
  'LOGON',
  'EXECUTES',
  'READS',
  'WRITES',
  'DELETES',
  'CONNECTS_TO',
  'AUTHENTICATES_TO',
  'NETWORK_FLOW',
  'EXPLOIT',
  'LATERAL_MOVE',
  'EXFILTRATE',
  'DISCOVER',
  'IMPACT',
  'GENERIC',
];

export type RelationCategory = 'access' | 'execution' | 'filesystem' | 'network' | 'attack';

export interface RelationTypeMeta {
  relation: RelationType;
  label: string;
  category: RelationCategory;
  /** True for malicious-leaning relations (EXPLOIT, LATERAL_MOVE, etc.). */
  is_attack: boolean;
  description: string;
}

export const RELATION_TYPE_META: Record<RelationType, RelationTypeMeta> = {
  OBSERVED_ON: {relation: 'OBSERVED_ON', label: 'Observed on', category: 'access', is_attack: false, description: 'Recorded process event on its host; does not establish a causal network connection'},
  LOGON: { relation: 'LOGON', label: 'Logon', category: 'access', is_attack: false, description: 'User logs on to host' },
  EXECUTES: { relation: 'EXECUTES', label: 'Executes', category: 'execution', is_attack: false, description: 'Process executes child process' },
  READS: { relation: 'READS', label: 'Reads', category: 'filesystem', is_attack: false, description: 'Process reads file' },
  WRITES: { relation: 'WRITES', label: 'Writes', category: 'filesystem', is_attack: false, description: 'Process writes file' },
  DELETES: { relation: 'DELETES', label: 'Deletes', category: 'filesystem', is_attack: false, description: 'Process deletes file' },
  CONNECTS_TO: { relation: 'CONNECTS_TO', label: 'Connects to', category: 'network', is_attack: false, description: 'Process / IP connects to network endpoint' },
  AUTHENTICATES_TO: { relation: 'AUTHENTICATES_TO', label: 'Authenticates to', category: 'access', is_attack: false, description: 'Cross-host authentication' },
  NETWORK_FLOW: { relation: 'NETWORK_FLOW', label: 'Network flow', category: 'network', is_attack: false, description: 'Generic network traffic' },
  EXPLOIT: { relation: 'EXPLOIT', label: 'Exploit', category: 'attack', is_attack: true, description: 'Active exploitation' },
  LATERAL_MOVE: { relation: 'LATERAL_MOVE', label: 'Lateral move', category: 'attack', is_attack: true, description: 'Adversary pivots between hosts' },
  EXFILTRATE: { relation: 'EXFILTRATE', label: 'Exfiltrate', category: 'attack', is_attack: true, description: 'Data exfiltration' },
  DISCOVER: { relation: 'DISCOVER', label: 'Discover', category: 'attack', is_attack: true, description: 'Environment discovery' },
  IMPACT: { relation: 'IMPACT', label: 'Impact', category: 'attack', is_attack: true, description: 'Destruction / denial of service' },
  GENERIC: { relation: 'GENERIC', label: 'Generic', category: 'access', is_attack: false, description: 'Unclassified relation' },
};

// ─────────────────────────────────────────────────────────────────────────────
// Label modes (scripts/build_graph.py --label-mode)
// ─────────────────────────────────────────────────────────────────────────────

export interface LabelModeMeta {
  mode: LabelMode;
  label: string;
  description: string;
  requires_label_arg: boolean;
}

export const LABEL_MODE_META: Record<LabelMode, LabelModeMeta> = {
  parser: {
    mode: 'parser',
    label: 'Parser',
    description: 'Use dataset/metadata-provided labels (default)',
    requires_label_arg: false,
  },
  force: {
    mode: 'force',
    label: 'Force',
    description: 'Force every event to --label {0|1}',
    requires_label_arg: true,
  },
  heuristic: {
    mode: 'heuristic',
    label: 'Heuristic',
    description: 'ATT&CK indicators + entity-time propagation',
    requires_label_arg: false,
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// Chain strategies (graph_builder/attack_tracker.py)
// ─────────────────────────────────────────────────────────────────────────────

export const CHAIN_STRATEGIES: ChainStrategy[] = ['chain_id', 'causal_parent', 'entity_time'];

export interface ChainStrategyMeta {
  strategy: ChainStrategy;
  label: string;
  priority: number; // 1=highest
  description: string;
  /** Prefix the strategy uses for synthesized chain_ids. */
  id_prefix: string | null;
  /** Minimum event count required to keep a chain. */
  min_events: number;
}

export const CHAIN_STRATEGY_META: Record<ChainStrategy, ChainStrategyMeta> = {
  chain_id: {
    strategy: 'chain_id',
    label: 'chain_id',
    priority: 1,
    description: 'Group by ground-truth chain_id field (highest priority)',
    id_prefix: null,
    min_events: 1,
  },
  causal_parent: {
    strategy: 'causal_parent',
    label: 'causal_parent',
    priority: 2,
    description: 'Union-find over causal_parent event references',
    id_prefix: 'causal_',
    min_events: 2,
  },
  entity_time: {
    strategy: 'entity_time',
    label: 'entity_time',
    priority: 3,
    description: 'BFS through shared entity within temporal window',
    id_prefix: 'inferred_',
    min_events: 2,
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// Dataset kinds (graph_builder/parsers.py PARSERS registry)
// ─────────────────────────────────────────────────────────────────────────────

export const DATASET_KINDS: DatasetKind[] = ['dedale'];

export interface DatasetKindMeta {
  kind: DatasetKind;
  label: string;
  description: string;
  supports_metadata_dir: boolean;
  raw_format: string;
  example_input: string;
}

export const DATASET_KIND_META: Record<DatasetKind, DatasetKindMeta> = {
  dedale: {kind: 'dedale', label: 'DEDALE Sysmon', description: 'Windows Sysmon / Winlogbeat events using the trained 72-feature pipeline', supports_metadata_dir: false, raw_format: 'JSON / JSONL', example_input: 'sysmon.jsonl'},
  ctu13: {
    kind: 'ctu13',
    label: 'CTU-13 NetFlow',
    description: 'CTU-13 botnet capture dataset formatted as unidirectional NetFlow records',
    supports_metadata_dir: false,
    raw_format: 'CSV / TSV / NetFlow',
    example_input: 'capture20110818.binetflow',
  },
  synthetic: {
    kind: 'synthetic',
    label: 'Synthetic JSONL',
    description: 'Pre-typed synthetic attack-chain JSONL',
    supports_metadata_dir: false,
    raw_format: 'JSONL',
    example_input: 'events.jsonl',
  },
  mordor: {
    kind: 'mordor',
    label: 'Host Telemetry (Sysmon JSONL)',
    description: 'OTRF Security-Datasets Sysmon+Security event logs',
    supports_metadata_dir: true,
    raw_format: 'JSON / JSONL / .gz / .zip / .tar.gz',
    example_input: 'mordor/*.json',
  },
  synthetic_demo: {
    kind: 'synthetic_demo',
    label: 'Synthetic Demo',
    description: 'Demonstration simulation dataset generated via StreamingGraphBuilder',
    supports_metadata_dir: false,
    raw_format: 'Parquet / Demo Simulation',
    example_input: 'scripts/init_demo_data.py',
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// Processing states — mirror backend pipeline stages
// ─────────────────────────────────────────────────────────────────────────────

export const PROCESSING_STATES: ProcessingState[] = [
  'idle',
  'queued',
  'parsing',
  'normalizing',
  'labeling',
  'building_graph',
  'exporting',
  'reconstructing_chains',
  'completed',
  'failed',
];

export interface ProcessingStateMeta {
  state: ProcessingState;
  label: string;
  /** Tailwind color token for badges. */
  badge: 'success' | 'warning' | 'danger' | 'info' | 'purple' | 'gray';
  description: string;
  /** True if the state represents a running stage. */
  is_active: boolean;
  /** True if the job has reached a terminal state. */
  is_terminal: boolean;
}

export const PROCESSING_STATE_META: Record<ProcessingState, ProcessingStateMeta> = {
  idle: { state: 'idle', label: 'Idle', badge: 'gray', description: 'Job created, not yet queued', is_active: false, is_terminal: false },
  queued: { state: 'queued', label: 'Queued', badge: 'gray', description: 'Waiting for worker', is_active: false, is_terminal: false },
  parsing: { state: 'parsing', label: 'Parsing', badge: 'info', description: 'Dataset-specific parser → TGEvent stream', is_active: true, is_terminal: false },
  normalizing: { state: 'normalizing', label: 'Normalizing', badge: 'info', description: 'Canonical entity typing, ts coercion', is_active: true, is_terminal: false },
  labeling: { state: 'labeling', label: 'Labeling', badge: 'info', description: 'Apply label_mode (parser/force/heuristic)', is_active: true, is_terminal: false },
  building_graph: { state: 'building_graph', label: 'Building Graph', badge: 'info', description: 'StreamingGraphBuilder ingest', is_active: true, is_terminal: false },
  exporting: { state: 'exporting', label: 'Exporting', badge: 'info', description: 'Write events/edges/nodes.parquet', is_active: true, is_terminal: false },
  reconstructing_chains: { state: 'reconstructing_chains', label: 'Reconstructing Chains', badge: 'purple', description: 'AttackTracker.build_chains + subgraphs', is_active: true, is_terminal: false },
  completed: { state: 'completed', label: 'Completed', badge: 'success', description: 'All artifacts produced', is_active: false, is_terminal: true },
  failed: { state: 'failed', label: 'Failed', badge: 'danger', description: 'Job failed; see error', is_active: false, is_terminal: true },
};

/** Ordered list of active stages (for progress visualization). */
export const PROCESSING_STAGE_ORDER: ProcessingState[] = [
  'parsing',
  'normalizing',
  'labeling',
  'building_graph',
  'exporting',
  'reconstructing_chains',
];

// ─────────────────────────────────────────────────────────────────────────────
// MITRE ATT&CK — synthetic-source tactic → relation map (parsers.py:98)
// ─────────────────────────────────────────────────────────────────────────────

export const TACTIC_TO_RELATION: Record<string, RelationType> = {
  Initial_Access: 'EXPLOIT',
  Execution: 'EXECUTES',
  Persistence: 'WRITES',
  Privilege_Escalation: 'EXPLOIT',
  Defense_Evasion: 'DELETES',
  Credential_Access: 'READS',
  Discovery: 'DISCOVER',
  Lateral_Movement: 'LATERAL_MOVE',
  Collection: 'READS',
  C2: 'CONNECTS_TO',
  Exfiltration: 'EXFILTRATE',
  Impact: 'IMPACT',
};

/** Stable, ordered list of tactics as the synthetic source emits them. */
export const SYNTHETIC_TACTIC_ORDER: string[] = Object.keys(TACTIC_TO_RELATION);

// ─────────────────────────────────────────────────────────────────────────────
// Heuristic labeler indicator buckets (graph_builder/labeler.py)
// ─────────────────────────────────────────────────────────────────────────────

export const HEURISTIC_SUSPICIOUS_PROCESSES = [
  'mimikatz', 'psexec', 'procdump', 'wce.exe', 'lazagne', 'rubeus',
  'sharphound', 'bloodhound', 'seatbelt', 'koadic', 'covenant',
  'powersploit', 'invoke-', 'empire', 'meterpreter',
  'cobaltstrike', 'beacon.exe', 'nc.exe', 'ncat.exe', 'plink.exe', 'winpeas',
] as const;

export const HEURISTIC_LOLBINS = [
  'rundll32.exe', 'regsvr32.exe', 'mshta.exe', 'wmic.exe', 'certutil.exe',
  'bitsadmin.exe', 'installutil.exe', 'msbuild.exe', 'cmstp.exe', 'odbcconf.exe',
  'control.exe', 'msiexec.exe', 'schtasks.exe', 'at.exe', 'sc.exe',
  'vssadmin.exe', 'wbadmin.exe', 'bcdedit.exe',
] as const;

export const HEURISTIC_SUSPICIOUS_EVENT_IDS = [
  '1102', '4625', '4648', '4672',
  '4697', '4698', '4699', '4700', '4701', '4702',
  '4720', '4728', '4732', '4756', '7045',
  '8', '10', '25',
];

export const HEURISTIC_SUSPICIOUS_RELATIONS: RelationType[] = [
  'EXPLOIT', 'LATERAL_MOVE', 'EXFILTRATE', 'IMPACT',
];

// ─────────────────────────────────────────────────────────────────────────────
// Selectable evaluation splits (scripts/evaluate_tgnn.py --split)
// ─────────────────────────────────────────────────────────────────────────────

export const EVAL_SPLITS = ['train', 'val', 'test'] as const;

// ─────────────────────────────────────────────────────────────────────────────
// Color tokens — semantic mapping per user spec
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Semantic color tokens mandated by the project spec:
 *   blue/cyan  = graph / data / information
 *   green      = healthy / benign / success
 *   amber      = warning / analysis
 *   red        = malicious / danger
 *   purple     = model / ML
 *   teal       = attack-chain / temporal investigation
 */
export const SEMANTIC_COLOR = {
  data: 'cyan',
  benign: 'green',
  warning: 'amber',
  malicious: 'red',
  model: 'purple',
  chain: 'teal',
} as const;

/** Hex values for canvas / SVG rendering (must match CSS vars in globals.css). */
export function colorHex(token: string, dark: boolean): string {
  const map: Record<string, [string, string]> = {
    cyan: ['#0F8CB5', '#38BDF8'],
    blue: ['#0F8CB5', '#38BDF8'],
    amber: ['#E8930A', '#F6A320'],
    violet: ['#7C5DC7', '#A78BFA'],
    green: ['#1E9E50', '#24C75F'],
    red: ['#E83333', '#F24040'],
    teal: ['#29A898', '#35C9B8'],
    pink: ['#DB2777', '#F472B6'],
    gray: ['#667085', '#7D95AE'],
  };
  const pair = map[token] ?? map.gray;
  return dark ? pair[1] : pair[0];
}
