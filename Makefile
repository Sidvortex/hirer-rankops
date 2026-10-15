# Local workflow (global Python, no venv). Docker targets at the bottom.
PY ?= python
PERIOD ?= 2
export PREFECT_LOGGING_LEVEL ?= WARNING

.PHONY: install data train promote retrain monitor serve replay simulate rollback mlflow-ui test lint clean \
        up up-jobs bootstrap down logs

install:            ## install Python deps globally (EndeavourOS / Arch needs --break-system-packages)
	$(PY) -m pip install --break-system-packages -r requirements.txt

data:               ## generate periods 0..PERIOD
	$(PY) -m rankops.data $$(seq 0 $(PERIOD))

train:              ## train a challenger on the window ending at PERIOD
	$(PY) -m rankops.train --period $(PERIOD)

promote:            ## champion/challenger gate on PERIOD's holdout
	$(PY) -m rankops.promote --period $(PERIOD)

retrain:            ## full Prefect retrain flow (ingest, validate, train, gate, reload)
	$(PY) flows/pipeline.py retrain --period $(PERIOD)

monitor:            ## drift check on logged traffic; retrains if needed
	$(PY) flows/pipeline.py monitor --period $(PERIOD)

serve:              ## run the API on :8000
	uvicorn rankops.serve:app --port 8000 --reload

replay:             ## send PERIOD's jobs to the running API as traffic
	$(PY) -m rankops.replay $(PERIOD)

simulate:           ## the full drift story, periods 0..9 -> reports/timeline.{csv,png}
	$(PY) scripts/simulate.py --fresh

rollback:           ## point champion back at the previous champion
	$(PY) -m rankops.promote --rollback

mlflow-ui:          ## browse runs and the registry on :5000
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

test:
	$(PY) -m pytest

lint:
	ruff check rankops flows scripts tests

clean:              ## wipe local registry, data, logs and reports
	rm -rf mlflow.db mlartifacts data logs reports

# ---- Docker ----
up:                 ## MLflow + API + Prometheus + Grafana
	docker compose up -d --build
bootstrap:          ## first data + first model, inside the stack
	docker compose run --rm bootstrap
up-jobs:            ## also start the scheduled monitor/retrain worker
	docker compose --profile jobs up -d
down:
	docker compose --profile jobs down
logs:
	docker compose logs -f api pipeline

help:
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'
