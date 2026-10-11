.PHONY: data backend frontend build test verify
PYTHON ?= python

data:
	$(PYTHON) scripts/download_local_data.py

backend:
	$(PYTHON) scripts/run_backend.py

frontend:
	npm --prefix frontend run dev

build:
	npm --prefix frontend run build

test:
	$(PYTHON) -m pytest backend/research/test_sliding_v3.py backend/research/test_orthrus_universal.py backend/research/test_cadets_dashboard.py backend/research/test_graph_uploads.py -q

verify:
	$(PYTHON) scripts/verify_package.py
