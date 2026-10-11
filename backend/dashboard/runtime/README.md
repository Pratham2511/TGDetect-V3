# Dashboard inference modules

These model definitions are copies of the research versions at the time of the
folder reorganization. Imports use this package so backend inference can run
without importing research scripts. Preserve parameter names and tensor shapes
when changing these files: existing checkpoints depend on them.

`preprocessing.py` preserves the feature extraction and event identity functions
from `research/dedale_data.py`, without the Modal dataset preparation application.
The original research modules remain in `research/` for training and reproducibility.
