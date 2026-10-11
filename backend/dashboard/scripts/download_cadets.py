"""Download the exact trained V3 bundle; no GPU container is started.

Requires the existing Modal account/profile and local Modal Python package.
Then run: python -m dashboard.scripts.prepare_cadets
"""
from concurrent.futures import ThreadPoolExecutor
import modal
from dashboard.cadets import DATA, ROOT, digest, CHECKPOINT_SHA


def download():
    volume = modal.Volume.from_name('tg-detect-sliding-node-mem-e3-v1')
    prefix = 'v3_vs_orthrus_20261010'
    source = prefix + '/prepared_v3_local'
    jobs = []
    for entry in volume.listdir(source, recursive=True):
        if entry.type.name == 'FILE':
            relative = entry.path.lstrip('/').removeprefix(source+'/')
            jobs.append((entry.path, DATA/'prepared'/relative))
    jobs += [(prefix+'/V3/'+remote, DATA/local) for remote,local in (
        ('best.pt','best.pt'),('result.json','result.json'),('val_edges.bin','val_edges.bin'),('test_edges.bin','test_edges.bin'))]
    jobs.append(('prepared/node_attributes.json',DATA/'node_attributes.json'))
    def fetch(job):
        remote, path = job
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name+'.part')
        with partial.open('wb') as f:
            volume.read_file_into_fileobj(remote, f)
        partial.replace(path)
        print(path.relative_to(DATA), flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch,jobs))
    assert digest(DATA/'best.pt') == CHECKPOINT_SHA, 'Wrong V3 checkpoint'
    source_manifest = ROOT/'artifacts/v3_vs_orthrus/source_manifest.json'
    if source_manifest.exists():
        (DATA/'source_manifest.json').write_bytes(source_manifest.read_bytes())
    print('V3 downloaded; run python -m dashboard.scripts.prepare_cadets', flush=True)


if __name__ == '__main__':
    download()
