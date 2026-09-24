# secai — Securitised Products AI Platform

Self-hosted platform that automates document-heavy workflows for securitised
products (RMBS, CMBS, ABS, CLO). Open-weight LLMs are served with **vLLM**;
every job is traced end to end with **OpenTelemetry** and exported to
**Langfuse**.

Two rules shape the whole design:

- **The LLM never computes financial numbers.** It extracts, classifies and
  drafts. All arithmetic, cash flows, test calculations and reconciliations are
  deterministic Python.
- **Every LLM extraction is schema-constrained** (Pydantic + vLLM guided
  decoding). Free-text parsing of model output is not allowed.

Work is tracked job by job in [`PROGRESS.md`](PROGRESS.md); the working rules
live in [`CLAUDE.md`](CLAUDE.md).

---

## Quick start

```bash
# 1. install the toolchain (uv manages the venv and the Python version)
uv sync --all-groups          # or: make setup   /   .\make.ps1 setup

# 2. create your local env file and fill in the secrets
cp .env.example .env
make secrets                  # prints fresh values for the Langfuse secrets
#   paste them into .env, then set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY
make langfuse-auth            # prints LANGFUSE_AUTH; paste that into .env too

# 3. bring the stack up
make up                       # full stack, including vLLM (needs an NVIDIA GPU)
make up-core                  # everything except vLLM

# 4. verify
make test
make lint typecheck
```

On Windows, GNU `make` is usually absent. Use the shim, which mirrors every
target one-for-one:

```powershell
.\make.ps1 setup
.\make.ps1 up-core
.\make.ps1 test
.\make.ps1 eval -Job 04
```

---

## Make targets

| Target | What it does |
|---|---|
| `setup` | Create the venv, install deps and dev tools, install pre-commit hooks |
| `secrets` | Print freshly generated Langfuse server secrets for `.env` |
| `langfuse-auth` | Print `LANGFUSE_AUTH` (base64 of the key pair) from `.env` |
| `up` / `up-core` | Start the stack, with / without vLLM |
| `down` / `restart` / `ps` / `logs` | Lifecycle and inspection |
| `test` | Unit suite with coverage (no network) |
| `test-unit` / `test-integration` | Narrower suites |
| `lint` / `fmt` / `typecheck` / `check` | ruff, ruff --fix, `mypy --strict`, all three |
| `eval JOB=<id>` | Gold-set evaluation for a job (harness lands in JOB-05) |
| `clean` | Remove caches and build artefacts |

---

## Services and ports

| Service | Port | Notes |
|---|---|---|
| Langfuse UI | http://localhost:3000 | Self-hosted v3; org/project/keys auto-provisioned on first boot |
| vLLM (OpenAI-compatible) | http://localhost:8000/v1 | `--enable-prefix-caching`, OTLP traces → Collector |
| OTel Collector | 4317 (gRPC), 4318 (HTTP), 13133 (health) | gRPC for vLLM, HTTP for the app |
| Application Postgres | 5433 | `postgresql+psycopg://secai:secai@localhost:5433/secai` |
| MinIO console | http://localhost:9091 | Langfuse event storage |

Langfuse's own Postgres, ClickHouse and Redis are internal to the compose
network and are not published.

### Trace topology

```
App (FastAPI / jobs)                vLLM server
  │ Langfuse Python SDK v4             │ --otlp-traces-endpoint
  │ (OTel-native)                      │ (gRPC → collector)
  │ OTLP/HTTP                          ▼
  └──────────────► OTel Collector ◄────┘
                        │ otlphttp exporter
                        ▼
              Langfuse  /api/public/otel
```

The Langfuse OTLP endpoint accepts **HTTP only** (protobuf or JSON), never
gRPC — which is precisely why vLLM's gRPC exporter is routed through the
Collector. Raw OTLP exporters must send `x-langfuse-ingestion-version: 4`, or
ingestion can lag by up to 15 minutes on Langfuse v4.

---

## Repository layout

```
src/secai/
  config.py        pydantic-settings; every env var the platform reads
  telemetry/       OTel + Langfuse setup, decorators, PII masking   (JOB-01)
  llm/             vLLM client, guided decoding, retries            (JOB-02)
  parsing/         PDF/Excel → ordered blocks + tables              (JOB-03)
  schemas/         Pydantic models (deal terms, servicer reports…)
  jobs/            one package per business use case
  validation/      deterministic checks & reconciliation
  eval/            eval harness, metrics, reports                   (JOB-05)
  api/             FastAPI routes
data/gold/         labelled gold sets (synthetic or public data only)
tests/             unit / integration / fixtures
```

---

## Configuration

All configuration comes from environment variables, loaded through
`secai.config.Settings` (see `.env.example` for the full list). Secrets are
never hard-coded and `.env` is git-ignored.

| Group | Prefix | Covers |
|---|---|---|
| App | `SECAI_` | environment, log level, guardrail switches |
| Langfuse | `LANGFUSE_` | keys, host, derived OTLP endpoint and Basic auth |
| OpenTelemetry | `OTEL_` | service name, OTLP endpoint/protocol, resource attrs |
| vLLM | `VLLM_` | base URL, model, router model, timeouts, retries |
| Database | `SECAI_DB_` | Postgres URL and pool settings |
| Storage | `SECAI_STORAGE_` | document, gold-set and report directories |
| PII | `SECAI_PII_` | redaction token and the masked field list |

Tests set `SECAI_ENV_FILE` to a non-existent path so they never pick up a
developer's local `.env`.

### Data protection

No real deal or borrower data goes in the repo, tests or fixtures — synthetic
or public sources only (e.g. EDGAR ABS-EE filings), anonymised. Borrower-level
fields are masked before they reach span attributes, and the Collector deletes
the same attribute keys as defence in depth (`otel-collector-config.yaml`).

---

## Job log

### JOB-00 — Foundation & Infrastructure

Project skeleton, tooling and the local stack.

**What it gives you**

- `pyproject.toml` (uv-managed), ruff + `mypy --strict` + pytest configuration
- `Makefile` and the `make.ps1` Windows shim: `setup up up-core down test lint typecheck eval …`
- `docker-compose.yml`: vLLM, application Postgres, OTel Collector and the full
  self-hosted Langfuse v3 stack (web, worker, Postgres, ClickHouse, Redis, MinIO),
  each with a healthcheck so `docker compose up --wait` is meaningful
- `otel-collector-config.yaml`: gRPC + HTTP receivers, PII attribute scrubbing,
  OTLP/HTTP export to Langfuse
- `src/secai/config.py`: the settings tree, plus `.env.example`
- `secai` CLI (`version`, `config`) and the `scripts/` helpers behind
  `make secrets` / `make langfuse-auth`

**How to run it**: the Quick start above. **Metrics**: none — this job has no
evaluation; it is verified by `make up` health checks plus `make lint typecheck test`.

**Model sizing.** `VLLM_MODEL` defaults to `Qwen/Qwen2.5-7B-Instruct-AWQ`. The
production target from the spec (`Qwen2.5-32B-Instruct-AWQ`) needs ≳24 GB of
VRAM; on a small dev GPU set a smaller model (e.g. `Qwen/Qwen2.5-1.5B-Instruct`)
and lower `VLLM_MAX_MODEL_LEN`. Nothing in the code assumes a particular model.
