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

Work proceeds one job at a time against an internal specification and work log,
which are kept outside this repository. Each job ships code, tests, an
evaluation against a gold set where one is defined, and traces verified in
Langfuse before it is considered done.

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
| `env-check` / `env-restore` | Check `.env` against the running stack, or rebuild it from the containers |
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

## Verifying the codebase

Run this before starting any new job. It is the same sequence CI runs, and
everything must be green before new work begins.

```bash
# 0. dependencies match the lockfile
uv sync --all-groups

# 1. static checks — fast, no services needed
make lint            # .\make.ps1 lint        ruff check + format --check
make typecheck       # .\make.ps1 typecheck   mypy --strict on src/

# 2. unit tests + coverage — no network, no Docker
make test            # .\make.ps1 test

# 3. bring the stack up and confirm every service is healthy
make up-core         # .\make.ps1 up-core     (add vLLM with `make up`)
make ps              # .\make.ps1 ps          all services should say (healthy)

# 4. end-to-end telemetry check against live Langfuse
uv run python scripts/telemetry_smoke.py
#    --with-llm additionally makes a real vLLM call

# 5. integration tests (needs the stack up)
make test-integration
```

`make check` chains steps 1–2 in one go.

### What "green" means

| Step | Expected |
|---|---|
| `lint` | `All checks passed!` and `N files already formatted` |
| `typecheck` | `Success: no issues found in N source files` |
| `test` | all pass, **coverage ≥ 85% for every new module** |
| `up-core` / `ps` | all 8 services `(healthy)` |
| smoke script | prints a `trace_id`; the trace resolves in Langfuse |

Never start a new job on a red tree: fix that first and nothing else. And never
mark a job done with skipped or xfail tests unless the reason is recorded in the
work log.

### Verifying a trace actually landed

The smoke script printing a trace ID only means the span was *sent*. To confirm
Langfuse stored it — and that masking held — query the API rather than trusting
the script:

```bash
AUTH=$(grep '^LANGFUSE_AUTH=' .env | cut -d= -f2)
curl -s -H "Authorization: Basic $AUTH" \
  "http://localhost:3000/api/public/traces/<trace_id>" | python -m json.tool
```

Check that borrower fields read `[REDACTED]` and that deal terms — tranche
balances, thresholds, coupons — are still present and unaltered. A masker that
eats tranche balances passes its own unit tests and silently destroys the data
JOB-04 is graded on; that regression has happened once already.

---

## Langfuse credentials

The key pair is **not fetched from Langfuse — you declare it and Langfuse
creates it.** `docker-compose.yml` passes `LANGFUSE_PUBLIC_KEY` and
`LANGFUSE_SECRET_KEY` to `langfuse-web` as `LANGFUSE_INIT_PROJECT_*`, and on
first boot Langfuse provisions the org, project, UI user and that exact pair.
So the keys in `.env` work the moment the stack starts.

`LANGFUSE_AUTH` is just `base64("public:secret")`, used by the OTel Collector's
`Authorization: Basic` header. `make langfuse-auth` derives it.

```bash
cp .env.example .env
make secrets          # generates SALT, ENCRYPTION_KEY, NEXTAUTH_SECRET, passwords
                      #   paste the output into .env
#   then pick any key pair, e.g.:
#     LANGFUSE_PUBLIC_KEY=pk-lf-secai-dev
#     LANGFUSE_SECRET_KEY=sk-lf-<random hex>
#   and set the UI login:
#     LANGFUSE_INIT_USER_EMAIL=you@example.com
#     LANGFUSE_INIT_USER_PASSWORD=<something real>
make langfuse-auth    # prints LANGFUSE_AUTH; paste into .env
make up-core
```

The **UI login** at http://localhost:3000 is whatever you set in
`LANGFUSE_INIT_USER_EMAIL` / `LANGFUSE_INIT_USER_PASSWORD`. Both are required —
there is deliberately no default, so a weak password never ships in the repo.

Three things to know:

- **`LANGFUSE_INIT_*` only applies to an empty database.** Change the key pair
  later and Langfuse ignores it; rotate keys in the UI or wipe the volume.
- **`ENCRYPTION_KEY` is load-bearing.** Change it and Langfuse can no longer
  decrypt what is already in its database.
- **The UI password cannot be changed via `.env` after first boot.** Editing
  `LANGFUSE_INIT_USER_PASSWORD` on an existing database does nothing; change it
  in the UI.

### If `.env` is lost or reset

`.env` is git-ignored and holds secrets that exist nowhere else. While the
containers are still running, every value can be recovered from them:

```bash
make env-check      # does .env still match the running stack?
make env-restore    # rebuild .env from the containers
```

A stale `.env` fails loudly — `docker compose config` reports
`required variable LANGFUSE_AUTH is missing a value` and `make up` refuses to
start. Once the containers are removed the secrets are gone for good, so keep a
copy of `.env` somewhere safe.

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

### JOB-02 — LLM Client & Guided Decoding

One entry point for every LLM call, so the guarantees hold everywhere rather
than at each call site.

```python
from pydantic import BaseModel
from secai.llm import LLMClient

class Tranche(BaseModel):
    original_balance: int
    coupon_margin: float

with LLMClient() as client:
    tranche = client.generate_structured(
        [{"role": "user", "content": prompt}], Tranche
    )   # a validated Tranche, or an exception - never a string to parse
```

**What it guarantees**

- **Schema-constrained output.** The Pydantic model becomes a `guided_json`
  payload, so vLLM constrains the sampler itself; the result is then validated
  again on the way back. Free-text parsing of model output is not allowed.
- **Retries only where they help.** Connection faults, timeouts, 5xx and 429
  retry with exponential backoff. A 4xx or a schema violation does not — the
  same request produces the same failure, so retrying burns tokens for nothing.
  The raw output is kept on the exception for triage.
- **Every call is a Langfuse generation** with model, parameters, token usage
  and latency — recorded even when validation fails, because a failed call
  still cost tokens.
- **Trace context propagates.** `traceparent` is injected per request so vLLM's
  server spans nest under the application trace.

**Prompts** resolve from Langfuse Prompt Management, falling back to
`prompts/<name>.md` in the repo when Langfuse is unreachable — an outage must
not take extraction down. Local versions are content-addressed
(`local-<hash>`), so an edit is visible in the trace.

**Configuration**: `VLLM_MODEL`, `VLLM_TIMEOUT_S`, `VLLM_MAX_RETRIES`,
`VLLM_TEMPERATURE` (0.0 by default — extraction should be reproducible),
`VLLM_MAX_TOKENS`.

**Running the acceptance test**: `make test-integration` with the stack up. It
makes 20 consecutive live calls and asserts every one returns a valid object.

> **Note:** the OpenAI SDK is built on `httpx2` while this codebase uses
> `httpx`, so trace context is injected via `extra_headers` rather than a custom
> `http_client`, and unit tests mock at the OpenAI client seam rather than with
> `respx`.
