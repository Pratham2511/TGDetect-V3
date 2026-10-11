"""Copy checksum-verified existing CADETS inputs between authorized Modal profiles.

Run download with MODAL_PROFILE=prathampansare25, upload with
MODAL_PROFILE=fukatkaamsathi. Only prepared data is copied, no training launched.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import modal

ROOT = Path('artifacts/sliding_node_mem_e3')
SOURCE = Path('artifacts/orthrus_e3_study/prepared/manifest.json')
VOLUME = 'tg-detect-sliding-node-mem-e3-v1'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()


def inputs():
    m = json.loads(SOURCE.read_text())
    paths = set(p for split in m['files'].values() for p in split)
    paths.update(m['graph_files']['test'])
    paths.update(m[k] for k in ('node_attributes', 'ground_truth_nodes', 'attack_nodes'))
    prefix = str(Path(m['node_attributes']).parent.parent) + '/'
    return m, prefix, sorted(paths)


def local(path, prefix):
    relative = path.removeprefix(prefix)
    old = Path('artifacts/orthrus_e3_study/local_evaluation_data') / relative
    return old if old.is_file() else ROOT / 'data' / relative


def download():
    m, prefix, paths = inputs()
    volume = modal.Volume.from_name('tg-detect-orthrus-e3-v1')
    ROOT.mkdir(parents=True, exist_ok=True)
    def fetch(p):
        dest = local(p, prefix)
        if dest.exists() and sha(dest) == m['file_sha256'][p]: return dest.stat().st_size
        dest.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(5):
            try:
                tmp = dest.with_suffix(dest.suffix + '.part')
                with tmp.open('wb') as f:
                    for chunk in volume.read_file('/' + p.removeprefix(prefix)): f.write(chunk)
                if sha(tmp) != m['file_sha256'][p]: raise ValueError('Checksum mismatch: ' + p)
                tmp.replace(dest)
                return dest.stat().st_size
            except Exception:
                if attempt == 4: raise
                time.sleep(attempt + 1)
    total = 0
    with ThreadPoolExecutor(max_workers=12) as pool:
        for i, future in enumerate(as_completed([pool.submit(fetch, p) for p in paths]), 1):
            total += future.result()
            if i % 50 == 0 or i == len(paths):
                status = dict(stage='download',files=i,total_files=len(paths),bytes=total)
                (ROOT/'transfer.json').write_text(json.dumps(status)); print(status, flush=True)


def upload():
    m, prefix, paths = inputs()
    # Remap source volume's internal mount paths into the destination /study mount.
    m['file_sha256'] = {p: m['file_sha256'][p] for p in paths}
    m['graph_files'] = {'test': m['graph_files']['test']}
    copied = json.loads(json.dumps(m).replace(prefix, '/study/'))
    dest = ROOT/'manifest.json'; dest.write_text(json.dumps(copied, indent=2))
    volume = modal.Volume.from_name(VOLUME, create_if_missing=True)
    with volume.batch_upload(force=True) as upload:
        for p in paths:
            source = local(p, prefix)
            if not source.is_file() or sha(source) != m['file_sha256'][p]:
                raise ValueError('Missing/unverified local input: ' + str(source))
            upload.put_file(source, '/' + p.removeprefix(prefix))
        upload.put_file(dest, '/prepared/manifest.json')
    (ROOT/'transfer.json').write_text(json.dumps(dict(stage='complete',files=len(paths),volume=VOLUME)))
    print('Uploaded verified CADETS dataset to', VOLUME, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['download', 'upload'])
    globals()[p.parse_args().action]()
