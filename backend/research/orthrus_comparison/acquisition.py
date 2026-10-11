"""Download only CADETS from the authors' public, uncompressed tar bundle.

HTTP ranges skip unrelated OpTC/THEIA dumps. Every response must honor its
requested range; a partial transfer resumes at its verified contiguous length.
"""
import hashlib
import json
from pathlib import Path
import re
import tarfile
import time

import requests

DRIVE_ID = '1i7CkK20p21aBp3HGw46o-Uy31JpPC_Yx'
URL = f'https://drive.usercontent.google.com/download?id={DRIVE_ID}&export=download&confirm=t'


def response(start, end):
    r = requests.get(URL, headers={'Range':f'bytes={start}-{end}'},
                     stream=True, timeout=(30,90))
    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', r.headers.get('Content-Range',''))
    if r.status_code != 206 or not match or tuple(map(int,match.groups()[:2])) != (start,end):
        details = (r.status_code,r.headers.get('Content-Type'),r.headers.get('Content-Range'))
        r.close()
        raise RuntimeError(f'Public bundle did not honor requested byte range: {details}')
    return r, int(match.group(3))


def members():
    offset=0; total=None
    while total is None or offset+512 <= total:
        with response(offset,offset+511)[0] as r:
            header=r.content
            total=int(r.headers['Content-Range'].split('/')[-1])
        if len(header)!=512: raise RuntimeError('Truncated tar header')
        if header == bytes(512): return
        item=tarfile.TarInfo.frombuf(header,'utf-8','surrogateescape')
        if item.type in (tarfile.GNUTYPE_LONGNAME,tarfile.XHDTYPE,tarfile.XGLTYPE):
            raise ValueError('Extended tar headers require explicit handling')
        yield item,offset+512,total
        offset += 512+((item.size+511)//512)*512


def acquire(output, progress=None):
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    source=root/'dataset_source.json'
    if source.exists():
        saved=json.loads(source.read_text())
        if len(saved)==1:
            dest=root/Path(saved[0]['file']).name
            if dest.exists():
                digest=hashlib.sha256()
                with dest.open('rb') as f:
                    for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
                if dest.stat().st_size!=saved[0]['bytes'] or digest.hexdigest()!=saved[0]['sha256']:
                    raise ValueError('Stored dataset size/checksum mismatch')
                saved[0]['file']=str(dest)
                return saved
    found=[]
    for member,start,total in members():
        if not member.isfile() or 'cadets' not in member.name.lower() or not member.name.endswith('.dump'):
            continue
        dest=root/Path(member.name).name
        partial=dest.with_suffix(dest.suffix+'.part')
        if dest.exists() and dest.stat().st_size != member.size:
            raise ValueError('Existing dump has unexpected size')
        if not dest.exists():
            while not partial.exists() or partial.stat().st_size < member.size:
                written=partial.stat().st_size if partial.exists() else 0
                end=min(start+member.size-1,start+written+16*1024*1024-1)
                for attempt in range(5):
                    try:
                        with response(start+written,end)[0] as r, partial.open('ab') as f:
                            for block in r.iter_content(1024*1024): f.write(block)
                        break
                    except (requests.RequestException,RuntimeError):
                        if attempt==4: raise
                        # Partial bytes are valid contiguous response data.
                        written=partial.stat().st_size if partial.exists() else 0
                        if start+written>end: break
                        time.sleep(min(2**attempt,8))
                if progress: progress(file=dest.name,bytes=partial.stat().st_size,total=member.size)
            if partial.stat().st_size != member.size: raise ValueError('Incorrect dump length')
            partial.replace(dest)
        digest=hashlib.sha256()
        with dest.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
        found.append(dict(file=str(dest),bytes=member.size,sha256=digest.hexdigest(),
                          tar_offset=start,bundle_bytes=total,drive_id=DRIVE_ID))
    if len(found)!=1: raise ValueError(f'Expected one CADETS dump; found {len(found)}')
    (root/'dataset_source.json').write_text(json.dumps(found,indent=2))
    return found
