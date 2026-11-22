# Everyday commands. Run `make help` to list them.
PY := .venv/bin/python

.PHONY: help setup run run-fake test tour loadtest reset

help:
	@echo "make setup      create .venv and install dependencies"
	@echo "make run        start the service on :8000 (needs Ollama for new questions)"
	@echo "make run-fake   start the service with a fake model: no Ollama needed"
	@echo "make test       run the test suite (no Ollama needed)"
	@echo "make tour       labelled hit/miss tour against a running service"
	@echo "make loadtest   concurrency scenarios against a slow fake model"
	@echo "make reset      restore the cache to the curated corpus (service must be running)"

setup:
	python3 -m venv .venv
	$(PY) -m pip install -r requirements.txt

# RESET_TOKEN is set so `make reset` works out of the box. Override it with
# RESET_TOKEN=something make run.
run:
	cd src && RESET_TOKEN=$${RESET_TOKEN:-dev-token} ../$(PY) -m uvicorn api:app --port 8000

run-fake:
	cd src && LLM_BACKEND=fake RESET_TOKEN=$${RESET_TOKEN:-dev-token} ../$(PY) -m uvicorn api:app --port 8000

test:
	$(PY) -m pytest tests/ -q

tour:
	$(PY) scripts/try_queries.py

loadtest:
	$(PY) scripts/loadtest.py

reset:
	$(PY) scripts/reset_cache.py --token $${RESET_TOKEN:-dev-token}
