'use client';

import { useEffect, useRef } from 'react';
import { useTheme } from '@/lib/theme-context';

interface Point {
  x: number;
  y: number;
  vx: number;
  vy: number;
  radius: number;
}

interface Pulse {
  sourceIdx: number;
  targetIdx: number;
  progress: number;
  speed: number;
}

export function BackgroundMotion() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const { theme } = useTheme();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Check prefers-reduced-motion
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reducedMotion) return;

    let animFrame: number;
    let width = (canvas.width = window.innerWidth);
    let height = (canvas.height = window.innerHeight);

    const handleResize = () => {
      if (!canvas) return;
      width = canvas.width = window.innerWidth;
      height = canvas.height = window.innerHeight;
    };
    window.addEventListener('resize', handleResize);

    // Discrete nodes
    const nodeCount = Math.min(36, Math.max(18, Math.floor(width / 50)));
    const nodes: Point[] = [];
    for (let i = 0; i < nodeCount; i++) {
      nodes.push({
        x: Math.random() * width,
        y: Math.random() * height,
        vx: (Math.random() - 0.5) * 0.25,
        vy: (Math.random() - 0.5) * 0.25,
        radius: Math.random() * 1.5 + 1.2,
      });
    }

    // Telemetry pulse signals
    const pulses: Pulse[] = [];
    let lastPulseTime = Date.now();

    let lastFrameTime = 0;
    const fpsInterval = 1000 / 30; // 30 FPS cap for maximum power efficiency

    const isDark = theme === 'dark';
    const nodeColor = isDark ? 'rgba(56, 189, 248, 0.22)' : 'rgba(38, 139, 210, 0.18)';
    const edgeBaseColor = isDark ? '56, 189, 248' : '38, 139, 210';
    const pulseColor = isDark ? 'rgba(0, 242, 254, 0.7)' : 'rgba(42, 161, 152, 0.6)';

    const render = (currentTime: number) => {
      animFrame = requestAnimationFrame(render);

      const elapsed = currentTime - lastFrameTime;
      if (elapsed < fpsInterval) return;
      lastFrameTime = currentTime - (elapsed % fpsInterval);

      ctx.clearRect(0, 0, width, height);

      // Move nodes
      for (let i = 0; i < nodes.length; i++) {
        const n = nodes[i];
        n.x += n.vx;
        n.y += n.vy;

        if (n.x < 0) n.x = width;
        else if (n.x > width) n.x = 0;
        if (n.y < 0) n.y = height;
        else if (n.y > height) n.y = 0;

        // Draw node
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.radius, 0, Math.PI * 2);
        ctx.fillStyle = nodeColor;
        ctx.fill();
      }

      // Draw connecting edges
      const maxDist = 160;
      const validConnections: [number, number][] = [];

      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const dx = nodes[i].x - nodes[j].x;
          const dy = nodes[i].y - nodes[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);

          if (dist < maxDist) {
            const alpha = (1 - dist / maxDist) * (isDark ? 0.06 : 0.05);
            ctx.beginPath();
            ctx.moveTo(nodes[i].x, nodes[i].y);
            ctx.lineTo(nodes[j].x, nodes[j].y);
            ctx.strokeStyle = `rgba(${edgeBaseColor}, ${alpha})`;
            ctx.lineWidth = 1;
            ctx.stroke();

            validConnections.push([i, j]);
          }
        }
      }

      // Periodically spawn telemetry pulse
      const now = Date.now();
      if (now - lastPulseTime > 2500 && validConnections.length > 0 && pulses.length < 3) {
        lastPulseTime = now;
        const [sourceIdx, targetIdx] = validConnections[Math.floor(Math.random() * validConnections.length)];
        pulses.push({
          sourceIdx,
          targetIdx,
          progress: 0,
          speed: 0.015 + Math.random() * 0.015,
        });
      }

      // Draw telemetry pulses
      for (let p = pulses.length - 1; p >= 0; p--) {
        const pulse = pulses[p];
        pulse.progress += pulse.speed;

        if (pulse.progress >= 1) {
          pulses.splice(p, 1);
          continue;
        }

        const s = nodes[pulse.sourceIdx];
        const t = nodes[pulse.targetIdx];
        if (!s || !t) {
          pulses.splice(p, 1);
          continue;
        }

        const px = s.x + (t.x - s.x) * pulse.progress;
        const py = s.y + (t.y - s.y) * pulse.progress;

        ctx.beginPath();
        ctx.arc(px, py, 1.8, 0, Math.PI * 2);
        ctx.fillStyle = pulseColor;
        ctx.fill();
      }
    };

    animFrame = requestAnimationFrame(render);

    return () => {
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(animFrame);
    };
  }, [theme]);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      className="fixed inset-0 pointer-events-none z-0 opacity-80"
    />
  );
}
