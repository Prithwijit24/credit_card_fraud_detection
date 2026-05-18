PYTHON ?= python
UV ?= uv

.PHONY: install install-dev install-prod build clean lint test train features stream produce api ui docker-up docker-down k8s-apply

install:
	$(UV) sync --active

install-dev:
	$(UV) sync --active --extra dev --extra stream --extra api --extra ui

install-prod:
	$(UV) sync --active --extra prod

build:
	$(UV) build

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache __pycache__

lint:
	ruff check .
	mypy src

test:
	pytest

train:
	$(UV) run fraud train --config base

features:
	$(UV) run fraud train --config base

stream:
	$(UV) run fraud stream --config base

produce:
	$(UV) run fraud produce --config base

api:
	$(UV) run fraud api --config base --host 0.0.0.0 --port 8000

ui:
	$(UV) run streamlit run scripts/run_ui.py --server.port 8501

docker-up:
	docker compose up --build

docker-down:
	docker compose down

k8s-apply:
	kubectl apply -f k8s/
