"""Allow research tests to import scripts and the application from either directory."""
import sys
from pathlib import Path

RESEARCH = Path(__file__).resolve().parent
# Frozen experiment source bundles duplicate test names; run only live tests.
collect_ignore = ['artifacts', 'data', 'frontend_legacy']
for directory in (RESEARCH.parent, RESEARCH):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
