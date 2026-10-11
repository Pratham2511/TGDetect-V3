"""Inference preprocessing copied from research/dedale_data.py."""
import hashlib
import math

FEAT_DIM = 72
HASH_BUCKETS = 64

def _host(e):
    w = e.get("winlog") or {}
    h = (e.get("agent") or {}).get("name") or (e.get("host") or {}).get("name") or w.get("computer_name") or ""
    return str(h).split(".")[0].upper()


def event_key(e):
    """Primary join key: (host, channel, record_id). Falls back to (host, ts, code, sequence)
    when record_id is absent, so labels still match if the export lacks record ids."""
    w = e.get("winlog") or {}
    ev = e.get("event") or {}
    rid = w.get("record_id")
    if rid not in (None, ""):
        return (_host(e), str(w.get("channel") or ""), "rid", str(rid))
    return (_host(e), "ts", str(e.get("@timestamp") or ""), str(ev.get("code") or w.get("event_id") or ""),
            str(ev.get("sequence") or w.get("record_id") or ""))


def _bucket(token):
    h = hashlib.sha256(token.encode("utf-8", "replace")).digest()
    return int.from_bytes(h[:8], "big") % HASH_BUCKETS, (1.0 if h[8] & 1 else -1.0)


def _hour(ts):
    try:
        return int(ts[11:13])
    except (TypeError, ValueError, IndexError):
        return 0


def features(e):
    import numpy as np
    x = np.zeros(FEAT_DIM, dtype=np.float32)
    w = e.get("winlog") or {}
    ed = w.get("event_data") or {}
    ev = e.get("event") or {}
    proc = e.get("process") or {}
    tokens = [("code", str(ev.get("code") or w.get("event_id") or "")),
              ("prov", str(ev.get("provider") or w.get("provider_name") or "")),
              ("act", str(ev.get("action") or "")), ("chan", str(w.get("channel") or "")),
              ("mod", str(ev.get("module") or "")), ("cat", ",".join(ev.get("category") or [])),
              ("img", str(ed.get("Image") or proc.get("executable") or "")),
              ("pimg", str(ed.get("ParentImage") or "")),
              ("ofn", str(ed.get("OriginalFileName") or proc.get("name") or "")),
              ("tgt", str(ed.get("TargetFilename") or ed.get("TargetObject") or "")),
              ("usr", str(ed.get("User") or (e.get("user") or {}).get("name") or ""))]
    for name, val in tokens:
        if val:
            b, s = _bucket(name + "=" + val)
            x[b] += s
    hour = _hour(e.get("@timestamp") or "")
    x[64] = np.log1p(len(str(e.get("message") or "")))
    try:
        x[65] = np.log1p(max(float(ev.get("code") or 0), 0))
    except (TypeError, ValueError):
        x[65] = 0.0
    x[66] = math.sin(2 * math.pi * hour / 24); x[67] = math.cos(2 * math.pi * hour / 24)
    x[68] = 1.0 if ed else 0.0
    x[69] = np.log1p(len(ed))
    x[70] = np.log1p(len(str(ed.get("CommandLine") or "")))
    x[71] = np.log1p(len(str(ed.get("ParentCommandLine") or "")))
    return x


def node_ids(e):
    """Two endpoints per event for the causal graph: the acting process and its host.
    Recent events sharing either node form the temporal ego-history."""
    w = e.get("winlog") or {}
    ed = w.get("event_data") or {}
    host = _host(e)
    proc = ed.get("ProcessGuid") or ed.get("ProcessId") or (e.get("process") or {}).get("pid") \
        or (e.get("event") or {}).get("provider") or "proc"
    return f"{host}|proc|{proc}", f"{host}|host"

def _epoch(ts):
    from datetime import datetime, timezone
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


