# secai — developer entry points.
#
# On Windows without GNU make, use the equivalent shim:  .\make.ps1 <target>

.DEFAULT_GOAL := help
SHELL := /bin/sh

UV      ?= uv
COMPOSE ?= docker compose
PY      ?= $(UV) run
JOB     ?=

.PHONY: help setup up up-core down restart ps logs test test-unit test-integration \
        pull-llm up-llm down-llm lint fmt typecheck check eval langfuse-auth \
        secrets env-check env-restore clean

help: ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk -F':.*?## ' '{printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# --------------------------------------------------------------- environment
setup: ## Create the venv, install deps and dev tools, install pre-commit hooks
	$(UV) sync --all-groups
	@[ -f .env ] || (cp .env.example .env && echo "created .env from .env.example — fill in the secrets")
	-$(PY) pre-commit install

secrets: ## Print freshly generated values for the Langfuse server secrets
	@$(PY) python scripts/gen_secrets.py

env-check: ## Check .env still matches the running stack
	$(PY) python scripts/restore_env.py --check

env-restore: ## Rebuild .env from the running containers (secrets live only there)
	$(PY) python scripts/restore_env.py

langfuse-auth: ## Print LANGFUSE_AUTH (base64 of the key pair) from .env
	@$(PY) python scripts/langfuse_auth.py

# -------------------------------------------------------------------- docker
up: ## Start the full stack (includes vLLM; needs an NVIDIA GPU)
	$(COMPOSE) --profile llm up -d --wait

up-core: ## Start everything except vLLM (no GPU required)
	$(COMPOSE) up -d --wait postgres langfuse-web langfuse-worker otel-collector

pull-llm: ## Pull the pinned vLLM image (large; run in the foreground)
	$(COMPOSE) --profile llm pull vllm

up-llm: ## Start vLLM on its own, on top of a running core stack
	$(COMPOSE) --profile llm up -d --wait vllm

down-llm: ## Stop vLLM, releasing the GPU and its RAM
	$(COMPOSE) --profile llm rm -sf vllm

down: ## Stop the stack (volumes are preserved)
	$(COMPOSE) down

restart: down up ## Restart the full stack

ps: ## Show service status
	$(COMPOSE) ps

logs: ## Tail logs (SERVICE=<name> to narrow)
	$(COMPOSE) logs -f --tail=100 $(SERVICE)

# --------------------------------------------------------------------- tests
test: ## Run the unit suite with coverage (no network)
	$(PY) pytest tests/unit --cov --cov-report=term-missing --cov-report=xml

test-unit: ## Run the unit suite only
	$(PY) pytest tests/unit

test-integration: ## Run integration tests against the running stack
	$(PY) pytest tests/integration -m integration

# ------------------------------------------------------------------- quality
lint: ## Lint with ruff
	$(PY) ruff check src tests scripts
	$(PY) ruff format --check src tests scripts

fmt: ## Auto-format and auto-fix
	$(PY) ruff format src tests scripts
	$(PY) ruff check --fix src tests scripts

typecheck: ## Type-check src with mypy --strict
	$(PY) mypy

check: lint typecheck test ## Everything CI runs

# ---------------------------------------------------------------------- eval
eval: ## Run a job's gold-set evaluation:  make eval JOB=04
	@if [ -z "$(JOB)" ]; then echo "usage: make eval JOB=<id>"; exit 2; fi
	$(PY) python -m secai.eval.run --job $(JOB)

# --------------------------------------------------------------------- misc
clean: ## Remove caches and build artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov build dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
