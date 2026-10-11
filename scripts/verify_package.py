"""Verify source and release-data checksums; --code-only works without data."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify(code_only=False):
    files = json.loads((ROOT / 'PACKAGE_MANIFEST.json').read_text())['files']
    if not code_only:
        files = {**files, **json.loads((ROOT / 'DATA_MANIFEST.json').read_text())['files']}
    failures = []
    for name, entry in files.items():
        path = ROOT / name
        if not path.is_file():
            failures.append(name + ': missing')
            continue
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                digest.update(block)
        if path.stat().st_size != entry['bytes'] or digest.hexdigest() != entry['sha256']:
            failures.append(name + ': changed')
    if failures:
        raise SystemExit('\n'.join(failures))
    print(f'Verified {len(files)} source' + (' files.' if code_only else ' and data files.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-only', action='store_true')
    verify(parser.parse_args().code_only)
