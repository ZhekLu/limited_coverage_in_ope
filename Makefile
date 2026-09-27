UV ?= $(if $(wildcard .venv/bin/uv),.venv/bin/uv,uv)
export UV_CACHE_DIR ?= $(CURDIR)/.cache/uv
CONFIG ?= configs/study.json
RUN ?= results/study

.PHONY: help setup lock format lint typecheck test check smoke benchmark study support run resume plots report build reproduce

help: ## Display available workflows
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "%-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Install exact dependencies from uv.lock into .venv
	$(UV) sync --locked --python 3.12.7

lock: ## Intentionally refresh dependency resolution; commit uv.lock
	$(UV) lock

format: ## Format source and tests
	$(UV) run --locked ruff format src tests
	$(UV) run --locked ruff check --fix src tests

lint: ## Check formatting and lint rules
	$(UV) run --locked ruff format --check src tests
	$(UV) run --locked ruff check src tests

typecheck: ## Strict static type checking
	$(UV) run --locked mypy

test: ## Scientific and pipeline regression tests
	$(UV) run --locked pytest --cov=limited_ope

check: lint typecheck test ## Run all development checks

smoke: ## Run a small end-to-end study and render its figures
	$(UV) run --locked limited-ope run --config configs/smoke.json
	$(UV) run --locked limited-ope plot --run results/smoke

benchmark: ## Benchmark the full pipeline on a fixed pilot grid
	$(UV) run --locked limited-ope run --config configs/benchmark.json

study: ## Run the preregistered 750-dataset study and figures
	$(UV) run --locked limited-ope run --config configs/study.json
	$(UV) run --locked limited-ope plot --run results/study

support: ## Run only the support-violation experiment
	$(UV) run --locked limited-ope run --config configs/support.json
	$(UV) run --locked limited-ope plot --run results/support

run: ## Run a custom CONFIG
	$(UV) run --locked limited-ope run --config $(CONFIG)

resume: ## Resume matching CONFIG after interruption
	$(UV) run --locked limited-ope run --config $(CONFIG) --resume

plots: ## Rebuild figures independently from saved RUN
	$(UV) run --locked limited-ope plot --run $(RUN)

report: ## Rebuild tables and research report from saved RUN
	$(UV) run --locked limited-ope summarize --run $(RUN)

build: ## Build wheel and source archive
	$(UV) build

reproduce: check study ## Validate code and reproduce the complete study
