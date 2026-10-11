"""Install the checksum-verified CADETS/V3 release bundle into backend/."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def checksum(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def install(archive=None):
    record = json.loads((ROOT / 'DATA_BUNDLE.json').read_text())
    expected = json.loads((ROOT / 'DATA_MANIFEST.json').read_text())['files']
    prefix = 'backend/dashboard/data/cadets_v3/'
    if not expected or any(not name.startswith(prefix) or '..' in PurePosixPath(name).parts for name in expected):
        raise RuntimeError('Invalid data manifest')
    if all((ROOT / name).is_file() and checksum(ROOT / name) == entry['sha256']
           for name, entry in expected.items()):
        print('Local CADETS/V3 files already verified.')
        return
    cache = ROOT / '.cache'
    cache.mkdir(exist_ok=True)
    if archive is None:
        archive = cache / 'cadets-v3-local-data.tar.gz'
        if not archive.exists() or checksum(archive) != record['sha256']:
            partial = archive.with_suffix('.part')
            request = urllib.request.Request(record['url'], headers={'User-Agent': 'TGDetect-V3-data-installer'})
            print('Downloading CADETS/V3 data bundle...', flush=True)
            with urllib.request.urlopen(request, timeout=120) as response, partial.open('wb') as output:
                shutil.copyfileobj(response, output, 4 * 1024 * 1024)
            if partial.stat().st_size != record['bytes'] or checksum(partial) != record['sha256']:
                raise RuntimeError('Data archive checksum mismatch')
            partial.replace(archive)
    archive = Path(archive)
    if archive.stat().st_size != record['bytes'] or checksum(archive) != record['sha256']:
        raise RuntimeError('Data archive checksum mismatch')
    # The original archive keeps its original dashboard/data/ paths. Only
    # allowlisted regular files are mapped into the new backend/ directory.
    # Verify every staged file before publishing any of them.
    with tempfile.TemporaryDirectory(prefix='cadets-install-', dir=cache) as staging:
        staging = Path(staging)
        with tarfile.open(archive, 'r:gz') as bundle:
            members = bundle.getmembers()
            seen = set()
            for member in members:
                path = PurePosixPath(member.name)
                if path.is_absolute() or '..' in path.parts or not (member.isfile() or member.isdir()):
                    raise RuntimeError('Unsafe archive entry: ' + member.name)
                if str(path) != 'dashboard/data/cadets_v3' and not str(path).startswith('dashboard/data/cadets_v3/'):
                    raise RuntimeError('Unexpected archive entry: ' + member.name)
                if member.isdir():
                    continue
                name = 'backend/' + str(path)
                if name not in expected or name in seen or member.size != expected[name]['bytes']:
                    raise RuntimeError('Unexpected or duplicate data file: ' + member.name)
                seen.add(name)
            if seen != set(expected):
                raise RuntimeError('Archive does not contain the complete data manifest')
            for member in members:
                if not member.isfile():
                    continue
                name = 'backend/' + str(PurePosixPath(member.name))
                output = staging / name
                output.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(member) as source, output.open('wb') as target:
                    shutil.copyfileobj(source, target, 4 * 1024 * 1024)
                if checksum(output) != expected[name]['sha256']:
                    raise RuntimeError('Extracted data checksum mismatch: ' + name)
        for name in sorted(expected):
            target = ROOT / name
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging / name, target)
    print('Verified and installed CADETS E3/V3 data: 5,537,964 events.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Install an already-downloaded archive')
    install(parser.parse_args().archive)
