PYTHON ?= python
.DEFAULT_GOAL := help
.PHONY: help test catalog source-check install-dev
help:
	@echo "catalog: source-backed classes; test: CPU software checks only"
	@echo "source-check SOURCE=ID DATA_ROOT=PATH: read-only pinned-file verification"
catalog:
	$(PYTHON) -m vllm_hust_workload_data.catalog research-list
test:
	$(PYTHON) -m pytest -q
source-check:
	$(PYTHON) -m vllm_hust_workload_data.catalog verify $(SOURCE) --data-root "$(DATA_ROOT)"
install-dev:
	$(PYTHON) -m pip install -e ".[dev,data]"
