'use client';

import React, { useState, useEffect } from 'react';
import { Network, ShieldCheck, Activity, Cpu } from 'lucide-react';

interface TGDetectLoaderProps {
  /** Optional custom status message */
  message?: string;
  /** If true, renders a full-page introductory initialization sequence */
  fullScreen?: boolean;
  /** Callback when loading sequence finishes */
  onComplete?: () => void;
}

const PHASES = [
  'INITIALIZING TELEMETRY & EVENT CONTRACTS',
  'CONSTRUCTING TEMPORAL GRAPH SNAPSHOTS',
  'ALIGNING SPATIOTEMPORAL GRU TENSORS',
  'READY // MODAL CHECKPOINT INFERENCE',
];

export function TGDetectLoader({ message, fullScreen = false, onComplete }: TGDetectLoaderProps) {
  const [phaseIndex, setPhaseIndex] = useState(0);
  const [isFading, setIsFading] = useState(false);

  useEffect(() => {
    if (!fullScreen) return;

    // Fast sequential phases
    const t1 = setTimeout(() => setPhaseIndex(1), 350);
    const t2 = setTimeout(() => setPhaseIndex(2), 700);
    const t3 = setTimeout(() => setPhaseIndex(3), 1050);
    const t4 = setTimeout(() => {
      setIsFading(true);
    }, 1350);
    const t5 = setTimeout(() => {
      if (onComplete) onComplete();
    }, 1600);

    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
      clearTimeout(t3);
      clearTimeout(t4);
      clearTimeout(t5);
    };
  }, [fullScreen, onComplete]);

  if (fullScreen) {
    return (
      <div
        onClick={() => onComplete && onComplete()}
        className={`fixed inset-0 z-50 flex flex-col items-center justify-center bg-[hsl(var(--background))] transition-opacity duration-300 select-none cursor-pointer ${
          isFading ? 'opacity-0 pointer-events-none' : 'opacity-100'
        }`}
      >
        {/* Central Hexagonal Temporal Graph Pulse */}
        <div className="relative size-20 mb-6">
          <svg viewBox="0 0 80 80" fill="none" className="w-full h-full">
            {/* Hexagon outline */}
            <polygon
              points="40,10 70,26 70,58 40,74 10,58 10,26"
              stroke="hsl(var(--primary))"
              strokeWidth="2"
              strokeDasharray="240"
              className="animate-pulse"
            />
            {/* Center Core */}
            <circle cx="40" cy="42" r="6" fill="hsl(var(--primary))" className="animate-ping opacity-75" />
            <circle cx="40" cy="42" r="5" fill="hsl(var(--primary))" />
            {/* Peripheral Nodes */}
            <circle cx="40" cy="10" r="3.5" fill="hsl(var(--warning))" opacity={phaseIndex >= 1 ? 1 : 0.2} />
            <circle cx="70" cy="26" r="3.5" fill="hsl(var(--success))" opacity={phaseIndex >= 2 ? 1 : 0.2} />
            <circle cx="70" cy="58" r="3.5" fill="hsl(var(--info))" opacity={phaseIndex >= 2 ? 1 : 0.2} />
            <circle cx="40" cy="74" r="3.5" fill="hsl(var(--danger))" opacity={phaseIndex >= 3 ? 1 : 0.2} />
            <circle cx="10" cy="58" r="3.5" fill="hsl(var(--purple))" opacity={phaseIndex >= 1 ? 1 : 0.2} />
            <circle cx="10" cy="26" r="3.5" fill="hsl(var(--teal))" opacity={phaseIndex >= 1 ? 1 : 0.2} />
            {/* Inner Spatiotemporal Edges */}
            <line x1="40" y1="42" x2="40" y2="10" stroke="hsl(var(--primary))" strokeWidth="1" strokeOpacity="0.4" />
            <line x1="40" y1="42" x2="70" y2="26" stroke="hsl(var(--primary))" strokeWidth="1" strokeOpacity="0.4" />
            <line x1="40" y1="42" x2="70" y2="58" stroke="hsl(var(--primary))" strokeWidth="1" strokeOpacity="0.4" />
            <line x1="40" y1="42" x2="40" y2="74" stroke="hsl(var(--primary))" strokeWidth="1" strokeOpacity="0.4" />
            <line x1="40" y1="42" x2="10" y2="58" stroke="hsl(var(--primary))" strokeWidth="1" strokeOpacity="0.4" />
            <line x1="40" y1="42" x2="10" y2="26" stroke="hsl(var(--primary))" strokeWidth="1" strokeOpacity="0.4" />
          </svg>
        </div>

        {/* Title */}
        <div className="flex items-center gap-2 mb-2">
          <span className="text-sm font-bold tracking-widest uppercase font-mono text-[hsl(var(--foreground))]">
            TGDETECT // COMMAND CENTER
          </span>
          <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
            v1.0
          </span>
        </div>

        {/* Phase Readout */}
        <div className="h-6 flex items-center">
          <span className="text-xs font-mono text-[hsl(var(--primary))] tracking-wider">
            {message || PHASES[phaseIndex]}
          </span>
        </div>

        {/* Progress Bar */}
        <div className="w-64 h-1 bg-[hsl(var(--border))] rounded-full mt-4 overflow-hidden">
          <div
            className="h-full bg-[hsl(var(--primary))] transition-all duration-300 ease-out"
            style={{ width: `${((phaseIndex + 1) / PHASES.length) * 100}%` }}
          />
        </div>

        <div className="absolute bottom-8 text-[10px] font-mono text-[hsl(var(--muted-foreground))] opacity-60">
          CLICK ANYWHERE TO BYPASS
        </div>
      </div>
    );
  }

  // Compact inline loader for widgets, panels, and tables
  return (
    <div className="flex flex-col items-center justify-center p-8 text-center space-y-3">
      <div className="relative size-8">
        <svg viewBox="0 0 40 40" fill="none" className="w-full h-full animate-spin duration-1000">
          <polygon
            points="20,4 34,12 34,28 20,36 6,28 6,12"
            stroke="hsl(var(--primary))"
            strokeWidth="2"
            strokeDasharray="90"
            strokeDashoffset="30"
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="size-2 rounded-full bg-[hsl(var(--primary))] animate-ping opacity-75" />
        </div>
      </div>
      <div className="text-xs font-mono text-[hsl(var(--muted-foreground))] tracking-wider">
        {message || 'STREAMING TELEMETRY & TENSOR FLOWS...'}
      </div>
    </div>
  );
}
