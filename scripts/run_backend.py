"""Start the unchanged dashboard API from the reorganized backend directory."""
import argparse
import os
from pathlib import Path
import sys

import uvicorn

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--reload', action='store_true')
    args = parser.parse_args()
    backend = Path(__file__).resolve().parents[1] / 'backend'
    os.chdir(backend)
    sys.path.insert(0, str(backend))
    uvicorn.run('dashboard.backend:app', host=args.host, port=args.port, reload=args.reload)
