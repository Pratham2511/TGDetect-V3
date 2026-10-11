'use client';

import { useEffect, useRef, useState } from 'react';
import { api, apiClient } from '@/lib/tgdetect/api/client';

export function MatchedComparison() {
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState('');
  const [operatingPoint, setOperatingPoint] = useState('test_val_threshold');
  useEffect(() => { api.get('/api/comparison').then(setData).catch(e => setError(e.message)); }, []);
  if (error) return <div role="alert" className="tg-card p-4">{error}</div>;
  if (!data) return <div className="tg-card p-4">Loading saved matched results…</div>;
  const groups = new Map<string, any[]>();
  for (const row of data.results) groups.set(row.arm, [...(groups.get(row.arm) || []), row]);
  const fields = ['precision','recall','f1','pr_ap','roc_auc','fp','fn'];
  return <div className="tg-card p-4 space-y-4">
    <div className="flex justify-between flex-wrap gap-3">
      <h2 className="font-semibold">Matched comparison · {data.results.length} saved runs</h2>
      <select aria-label="Comparison threshold" value={operatingPoint} onChange={e => setOperatingPoint(e.target.value)} className="bg-[hsl(var(--background))] border rounded p-2">
        <option value="test_val_threshold">Validation-selected threshold</option><option value="test_default_0_5">Fixed threshold 0.5</option>
      </select>
    </div>
    <p className="text-xs text-[hsl(var(--muted-foreground))]">Mean test metrics across three seeds. Each arm received the same additional training, warm start, data, and replay schedule. PR-AP is average precision. These are saved benchmark results.</p>
    <div className="overflow-x-auto"><table className="w-full text-xs text-left"><thead><tr><th className="p-2">Model</th><th>Seeds</th>{fields.map(f => <th key={f} className="p-2">{f.toUpperCase().replace('PR_AP','PR-AP')}</th>)}</tr></thead>
      <tbody>{Array.from(groups).map(([arm, rows]) => <tr key={arm} className="border-t border-[hsl(var(--border))]"><td className="p-2 font-mono">{arm}</td><td>{rows.length}</td>{fields.map(f => { const value=rows.reduce((s,r) => s+r[operatingPoint][f],0)/rows.length;return <td key={f} className="p-2 font-mono">{value.toFixed(f==='fp'||f==='fn'?1:4)}</td>; })}</tr>)}</tbody>
    </table></div>
    <details><summary className="cursor-pointer text-xs">Per-seed results and checkpoint checksums</summary><div className="overflow-auto mt-3"><table className="w-full text-xs"><thead><tr><th>Arm</th><th>Seed</th><th>Threshold</th><th>Best epoch</th><th>Validation AP</th><th>Checkpoint SHA-256</th></tr></thead><tbody>{data.results.map((r:any) => <tr key={`${r.arm}:${r.seed}`}><td>{r.arm}</td><td>{r.seed}</td><td>{r[operatingPoint].threshold.toFixed(5)}</td><td>{r.best_epoch}</td><td>{r.best_val_pr_ap.toFixed(4)}</td><td className="font-mono" title={r.checkpoint_sha256}>{r.checkpoint_sha256.slice(0,16)}…</td></tr>)}</tbody></table></div></details>
  </div>;
}

export function LiveInference({modelId}:{modelId:string}) {
  const [result,setResult]=useState<any>(null);
  const [error,setError]=useState('');
  const [busy,setBusy]=useState(false);
  const [hasSession,setHasSession]=useState(false);
  const session=useRef<string|null>(null);
  const generation=useRef(0);
  useEffect(() => {
    generation.current++;
    return () => {
      generation.current++;
      const sid=session.current;session.current=null;
      if(sid) apiClient(`/api/session/${sid}`,{method:'DELETE'}).catch(() => {});
    };
  },[modelId]);
  async function run() {
    const current=generation.current;
    setBusy(true);setError('');
    try {
      if(!session.current) {
        const created:any=await api.post('/api/sessions',{model_id:modelId,source:'recorded'});
        if(current!==generation.current) { await apiClient(`/api/session/${created.id}`,{method:'DELETE'});return; }
        session.current=created.id;
        setHasSession(true);
      }
      const next=await api.post(`/api/session/${session.current}/next?count=64`);
      if(current===generation.current) setResult(next);
    } catch(e:any) { if(current===generation.current) setError(e.message); }
    finally {setBusy(false);}
  }
  return <div className="tg-card p-4 space-y-4">
    <h2 className="font-semibold">Live checkpoint inference · causal event windows</h2>
    <p className="text-xs text-[hsl(var(--muted-foreground))]">Loads the selected saved checkpoint and processes the recording chronologically with isolated node memory. Each event uses itself and up to four strictly earlier incident events. Start at event 0 to preserve memory history. Upload new Sysmon JSON in Datasets.</p>
    <button disabled={busy||!modelId||result?.finished} onClick={run} className="border rounded px-4 py-2 bg-cyan-500/10 text-cyan-400 disabled:opacity-50">{busy?'Running checkpoint…':hasSession?'Process next 64 events':'Load checkpoint and process 64 events'}</button>
    {error&&<p role="alert" className="text-red-400">{error}</p>}
    {result&&<><p className="text-xs font-mono">Position: {result.position} · inference: {result.inference_ms?.toFixed(1)} ms · checkpoint: {result.checkpoint_sha256}</p><div className="overflow-x-auto"><table className="w-full text-xs text-left"><thead><tr>{['Event','Host','Score','Prediction','Ground truth','Admission','Fast weight','Slow weight'].map(x=><th className="p-2" key={x}>{x}</th>)}</tr></thead><tbody>{result.events.map((e:any)=><tr key={e.index} className="border-t border-[hsl(var(--border))]"><td className="p-2">{e.index}</td><td>{e.host||e.dst}</td><td>{e.score.toFixed(5)}</td><td>{e.alert?'Alert':'Below threshold'}</td><td>{e.label===null?'Unknown':e.label===1?'Malicious':'Benign'}</td><td>{e.memory.admit?.toFixed(4)??'—'}</td><td>{e.memory.effective_fast_weight?.toFixed(4)??'—'}</td><td>{e.memory.effective_slow_weight?.toFixed(6)??'—'}</td></tr>)}</tbody></table></div></>}
  </div>;
}
