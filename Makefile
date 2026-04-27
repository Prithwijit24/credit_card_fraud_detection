PYTHON ?= python
PIP ?= $(PYTHON) -m pip

.PHONY: install install-dev lint test train stream produce format

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev,stream]"

lint:
	ruff check .
	mypy src

test:
	pytest

train:
	$(PYTHON) scripts/train_model.py --config configs/base.yaml

stream:
	$(PYTHON) scripts/run_streaming_job.py --config configs/base.yaml

produce:
	$(PYTHON) scripts/produce_events.py --config configs/base.yaml

