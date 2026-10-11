"""Refresh the source inventory after an intentional change; excludes local data."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    names = subprocess.check_output(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=ROOT
    ).decode().split('\0')
    files = {}
    for name in sorted(set(names) - {'', 'PACKAGE_MANIFEST.json'}):
        path = ROOT / name
        if path.is_symlink():
            raise SystemExit('Do not package symbolic links: ' + name)
        files[name] = {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    record = {'scope': 'TGDetect V3 source; release data tracked separately in DATA_MANIFEST.json', 'files': files}
    (ROOT / 'PACKAGE_MANIFEST.json').write_text(json.dumps(record, indent=2) + '\n')
    print(f'Recorded {len(files)} source files.')
