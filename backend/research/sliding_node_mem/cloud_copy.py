"""Copy prepared data directly between the two user-authorized workspaces.

Source credentials stay in the local Modal client. Temporary read-only signed
block URLs are passed in memory to the destination importer, never written/logged.
"""
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import modal
from .transfer import inputs


async def sign():
    from modal_proto import api_pb2
    from modal._utils.async_utils import synchronizer
    manifest, prefix, paths = inputs()
    # Execute on Modal's own async loop; its gRPC transport belongs to that loop.
    volume = synchronizer._translate_in(modal.Volume.from_name('tg-detect-orthrus-e3-v1'))
    await volume.hydrate()
    limit = asyncio.Semaphore(16)
    async def one(path):
        async with limit:
            response = await volume._client.stub.VolumeGetFile2(api_pb2.VolumeGetFile2Request(
                volume_id=volume.object_id, path='/'+path.removeprefix(prefix)))
            return dict(path='/study/'+path.removeprefix(prefix),
                        sha256=manifest['file_sha256'][path], urls=list(response.get_urls))
    entries = await asyncio.gather(*(one(p) for p in paths))
    manifest['file_sha256'] = {p: manifest['file_sha256'][p] for p in paths}
    manifest['graph_files'] = {'test': manifest['graph_files']['test']}
    manifest = json.loads(json.dumps(manifest).replace(prefix, '/study/'))
    return dict(manifest=manifest, entries=entries)


def copy():
    result = subprocess.run([sys.executable, '-m', 'sliding_node_mem.cloud_copy', '--sign'],
                            env=dict(os.environ, MODAL_PROFILE='prathampansare25'),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    data = json.loads(result.stdout)
    fn = modal.Function.from_name('tg-detect-sliding-node-mem-e3', 'import_data')
    measured = fn.remote(data['manifest'], data['entries'])
    root = Path('artifacts/sliding_node_mem_e3'); root.mkdir(exist_ok=True, parents=True)
    (root/'transfer.json').write_text(json.dumps(dict(stage='complete', method='verified cloud copy', **measured)))
    (root/'manifest.json').write_text(json.dumps(data['manifest'],indent=2))
    print(json.dumps(measured), flush=True)


if __name__ == '__main__':
    if '--sign' in sys.argv:
        from modal._utils.async_utils import synchronizer
        print(json.dumps(synchronizer.create_blocking(sign)()))
    else: copy()
