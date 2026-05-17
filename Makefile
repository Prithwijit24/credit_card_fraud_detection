PYTHON ?= python
PIP ?= $(PYTHON) -m pip

.PHONY: install install-dev install-prod lint test train stream produce api ui docker-up k8s-apply

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev,stream,api,ui]"

install-prod:
	$(PIP) install -e ".[prod]"

lint:
	ruff check .
	mypy src

test:
	pytest

train:
	fraud train --config base

stream:
	fraud stream --config base

produce:
	fraud produce --config base

api:
	fraud api --config base --host 0.0.0.0 --port 8000

ui:
	streamlit run scripts/run_ui.py --server.port 8501

docker-up:
	docker compose up --build

k8s-apply:
	kubectl apply -f k8s/
