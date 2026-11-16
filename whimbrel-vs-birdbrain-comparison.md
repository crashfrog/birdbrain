# Whimbrel 0.4.1 vs birdbrain 0.1.0 — Comparative Analysis

## Research Method

Whimbrel 0.4.1 was inspected directly from the installed package at
`/nfs/software/apps/whimbrel/0.4.1` on the **Reedling2** HPC head node
(SSH from the birdbrain development workstation). birdbrain 0.1.0
was inspected from the local working copy at `/Users/Justin.Payne/Projects/birdbrain`,
including the Reedling-specific deployment artifacts (`birdbrain-reedling.toml`,
`vllm_job_reedling.sh.j2`, `deploy_to_reedling.sh`, `finish_deployment.sh`).

## Executive Summary

| Dimension | birdbrain 0.1.0 | Whimbrel 0.4.1 |
|---|---|---|
| **Author / org** | Justin Payne (crashfrog) | Kranti Konganti (FDA/HHS) |
| **Backend engine** | vLLM (via Singularity SIF) | SGLang (via venv at `/opt/whimbrel/.sglang-venv`) |
| **Launch mode** | Slurm-only (sbatch) | Dual-mode: `popen` (single-node) or `slurm` (multi-node) |
| **Job lifecycle** | Proactive rolling rotation — always N+1 jobs running, swap before Slurm kills | Reactive/on-demand — auto-start on request, idle-shutdown after 3600s |
| **Process topology** | Single process: FastAPI gateway + asyncio lifecycle/health tasks | Three layers: daemon (Unix-socket RPC) + proxy (FastAPI uvicorn) + SGLang processes; nginx on head node |
| **State storage** | JSON files per backend pool (`state-<pool>.json`) | SQLite DB (schema v9, WAL mode, retry-wrapped connexions) |
| **Deployment** | User-level: `pip install`, login node, no root | System-installed: root required, systemd units, nginx, logrotate |
| **Install flow** | Copy TOML, customize paths | Interactive `install-prereqs` wizard with node discovery (ping), venv setup, artifact generation |
| **Multi-model** | Pre-configured `[[backends]]` pools, each always-on with independent rotation | Allow-list (`allowed_models.yaml`) based lazy auto-start; one instance per repo_id |
| **Auth** | File-based keystore (JSON, fcntl-locked), email→key, optional | DB-based keys (SHA-256 hashed), identity types (user/team/client), expiry, revocation, request counting |
| **Rate limiting** | None | Token-bucket: 10 RPS default per identity |
| **Quotas** | None | Per-user 24h token quota (100K default) |
| **Usage tracking** | None | Full request ledger: tokens, latency, TTFT, identity, errors, disconnects |
| **GPU pre-check** | None | `nvidia-smi` free-memory check before launch (8GB min, fail-open) |
| **Internal auth** | None — backend trusted directly | SGLang started with `--api-key` via YAML config file (not in process args); health checks verify |
| **Network** | Gateway on login node → compute nodes directly | nginx on head node (TLS/:30000) → proxy on each GPU node (:30001) → SGLang; lua model-affinity routing |
| **Python version** | >=3.10 | >=3.12 |
| **Type checking** | mypy | beartype (runtime, every function) |
| **Tests** | pytest suite (gateway, lifecycle, keystore, config) | None found in installed package; source repo not inspected |
| **Lines of code** | ~6 source files, ~1200 LOC | ~50 source files, ~8000+ LOC |

---

## 1. Architectural Model

### birdbrain: "Always-On Rotating Gateway"

A single FastAPI process runs on the **login node**. It embeds three components per backend pool as asyncio background tasks:

- **Gateway** — OpenAI-compatible proxy. Static routing table (model name + aliases → pool name). On cold start, returns `503 Retry-After` and submits the first Slurm job.
- **Lifecycle Manager** — One per pool. At `wall_time - lead_time_minutes`, submits the next job. Polls `squeue` until RUNNING, waits for `/internal/register` callback (matched by `job_id`), then atomically swaps. On restart, rehydrates from JSON state and re-checks `squeue`.
- **Health Monitor** — One per pool. Polls `/health` every 30s. After 3 consecutive failures, marks unhealthy and triggers lifecycle rotation for that pool only.

The job template (`vllm_job.sh.j2`) runs vLLM in a Singularity container, health-checks, then POSTs to `/internal/register` with `job_id`, `node`, `port`, `backend_name`. A SIGTERM trap is wired (the `cleanup()` currently has a TODO for deregister).

**Key design invariant**: there is always a live backend (or a new one spinning up). Rotation is time-driven, not event-driven.

### Whimbrel: "Lazy-Start Daemon Cluster"

A **daemon** process (Unix socket RPC server) manages SGLang lifecycles. A separate **proxy** (FastAPI uvicorn on `:30001`) handles OpenAI API requests. On each GPU node, nginx on the head node forwards TLS-terminated traffic to the proxy via lua model-affinity routing (crc32 hash of model name → correct GPU node).

Lifecycle is **reactive**: a request for a model not currently running triggers an auto-start via the daemon (if the model is in the `allowed_models.yaml` allow-list). SGLang is started with an internal API key (stored in a YAML config file, not in process args). The daemon tracks `desired_state` and `known_good` configs, restoring desired instances on restart. An idle monitor shuts down instances after `idle_timeout` (default 3600s) of inactivity.

The SLURM job template (`sglang_job.sh.j2`) runs `sglang serve` with a `--config` YAML file (model-path, host, port, api-key). The daemon polls `squeue` for SLURM job state and checks `sinfo` for idle GPU resources before submitting.

**Key design invariant**: resources are only consumed when a model is actually needed. Idle models are reaped.

---

## 2. Job Lifecycle & Rotation

### birdbrain — Proactive rolling rotation

```
T+0h:   Job A submitted, running, gateway routes to A
T+20h:  Lifecycle submits Job B (lead_time=90min before T+24h wall)
T+20h:  Job B starts vLLM, health-checks, registers
T+21h:  Gateway health-checks B, flips to B
T+24h:  Job A killed by Slurm. Zero client impact.
```

- `lead_time_minutes` (default 90) controls when the next job is submitted relative to expiry.
- Registration is callback-based: the Slurm job POSTs to `/internal/register` when vLLM's `/health` returns 200.
- The lifecycle manager waits up to 20 minutes for registration (cold compile cache on shared scratch).
- On gateway restart, each pool's `LifecycleManager._load_state()` reads its JSON state file, then resyncs with `squeue`.
- If a pool's current job is dead, that pool cold-starts independently.
- **SSE chunk normalization**: gateway rewrites vLLM's SSE stream to match OpenAI spec — strips `content:""` from the final finish_reason chunk (vLLM 0.4.x bundles both), removes `usage` from stream chunks (sent only in the final `[DONE]` chunk).
- **Field stripping**: `_UNSUPPORTED = {"store", "stream_options"}` fields are popped from non-streaming requests to avoid vLLM 400 validation errors. `tools`/`tool_choice`/`parallel_tool_calls` are passed through (fleet uses `--enable-auto-tool-choice`).
- **Internal auth header (Reedling template only)**: the Reedling-specific `vllm_job_reedling.sh.j2` sends an `X-Internal-Key` header on `/internal/register` and `/internal/deregister` (via a SIGTERM trap), but the gateway's `_register_backend` endpoint does **not** validate it — the key is sent but not checked. The generic template has only a TODO for deregistration.
- **No request queuing during rotation**: backend swaps are atomic via `asyncio.Lock` in `set_backend()`. If no backend is registered (cold start), the gateway returns `503` with `Retry-After: 60` — no asyncio.Queue or `request_retry_window_seconds` retry window exists in the code (the field appears only in the Reedling TOML, unused by the Pydantic `GatewayConfig` model). **Note**: CLAUDE.md mentions a retry queue, but the implementation does not contain it.
- **5-minute httpx timeout**: `httpx.AsyncClient(timeout=300.0)` accommodates long-generation requests.
- **Hostname matching bug fix**: promotion is matched by `job_id` (not `backend_url`) because squeue's short hostname (`%N`) and the job's `hostname` command (often FQDN) never string-match — matching by URL silently prevented promotion forever.

### Whimbrel — Reactive auto-start + idle timeout

- No pre-started jobs. When a request arrives for a model with no READY instance in the DB:
  1. Proxy checks `allowed_models.yaml` allow-list.
  2. If allowed, proxy calls daemon `run` command via Unix socket.
  3. Daemon: checks if already running in DB (`_check_already_running`), scans `ps` for existing SGLang process, allocates port, launches via `popen` or `sbatch`, waits for health check (600s timeout), persists to DB.
  4. Proxy then re-routes with `route_with_health()`.
- `known_good` flag in `launch_configs` tracks last-known-good ext_args — failed launches don't overwrite it.
- `restore_on_startup()` in daemon: on restart, scans `launch_configs` for `desired_state='running'` AND `known_good=1`, restarts those instances.
- Idle monitor (`daemon/idle_monitor.py`): polls `instances` table for RUNNING entries, checks `active_requests` and `last_request` timestamp against `idle_timeout` (default 3600s). Stops idle instances via `stop_instance()` → `scancel`/`SIGTERM`.
- GPU memory pre-check (`runtime/sglang_launcher.py`): runs `nvidia-smi` before launch, fails with clear error if free memory < 8 GB (fail-open if nvidia-smi unavailable).
- Reserved-arg validation (`runtime/validator.py`): blocks user from passing `--port`, `--host`, `--api-key`, `--config-file` to SGLang (managed internally).
- Single-flight pull (`hf/pull.py`): per-repo + global thread locks prevent duplicate concurrent downloads of the same model.
- `find_running_sglang()`: scans `ps` for existing SGLang processes to reconcile manually-started instances into the daemon's DB state.
- SLURM resource pre-check via `sinfo` before submitting — checks for idle GPU nodes with matching GRES.

### Key difference

| | birdbrain | Whimbrel |
|---|---|---|
| **When jobs start** | Proactively, before current expires | On first request (lazy) |
| **When jobs stop** | Killed by Slurm at wall-time end | Idle timeout (3600s default) or explicit stop |
| **Resource cost** | Always one job per pool running | Zero when idle |
| **Cold-start latency** | Paid during rotation (overlapping) | Paid on first request after idle (503 until warm) |
| **Gap risk** | None if lead_time is adequate | Yes, if idle timeout is too aggressive or resources unavailable |

---

## 3. State Persistence

### birdbrain

- **Per-pool JSON**: `~/.birdbrain/state-{pool-name}.json` with `current_job` and `pending_job` `JobState` dataclasses.
- **Atomic writes**: temp file + `replace()`.
- **Legacy migration**: a pool named `"default"` falls back to reading the old `state.json` if its own file doesn't exist — zero-downtime upgrade path for single-model deployments.
- **On restart**: `_load_state()` + `squeue` re-validation. No schema — just dataclass fields.
- **No concurrent writers**: single gateway process owns all state.

### Whimbrel

- **SQLite database** at `/hpc/db/hf-eval-model-hub/whimbrel-db/whimbrel.db` with **WAL mode** and **busy timeout** (30s).
- **9 migration steps** tracked in `schema_version` table — evolved from initial schema to add daemon registry, API keys, SLURM job ID tracking, node columns, etc.
- **Tables**: `launch_configs`, `instances`, `lifecycle_events`, `requests_ledger`, `api_keys`, `daemon_registry`.
- **`@retry_db_logged`** decorator (tenacity, 5 attempts, exponential backoff) on every DB operation — handles "database is locked" on concurrent access.
- **Daemon registry**: each daemon registers its PID + socket path + node hostname in the DB. CLI discovers the active daemon socket by querying this table (not by convention path lookup).
- **Multi-node**: `node` column on `instances` and `daemon_registry` enables per-node DB state. Head node aggregates via `/v1/cluster/status` (HTTP queries to each GPU node's `:30001`/`/v1/status`).

### Key difference

| | birdbrain | Whimbrel |
|---|---|---|
| **Storage format** | JSON files | SQLite with WAL |
| **Concurrency** | Single writer | Multi-writer (retry-wrapped) |
| **Schema evolution** | None — add field, old files break | 9 migrations, version-tracked |
| **Distributed state** | Single login node owns all state | Per-node DB, head aggregates |
| **Usage ledger** | Not tracked | Full request ledger in DB |

---

## 4. Multi-Model Routing

### birdbrain

- **Static configuration**: `[[backends]]` entries in TOML, each with `name`, `model.name`, `aliases[]`, independent `[backends.slurm]` and `[backends.model]`.
- **Routing table**: built once at startup — canonical name, model name, and every alias all map to the backend pool name. Unknown model → `400 Bad Request`.
- **Multi-pool independence**: each pool has its own `LifecycleManager` + `HealthMonitor` + state file. One pool rotating/cold-starting never blocks another.
- **No auto-start**: if no backend is registered for a pool, returns `503` with `Retry-After: 60`.
- **`/v1/models`** lists canonical model name + aliases for every live pool, with full OpenAI schema (created, owned_by, root, parent, permission).

### Whimbrel

- **Allow-list**: `allowed_models.yaml` at a hardcoded path (`/hpc/db/hf-eval-model-hub/whimbrel-db/allowed_models.yaml`). Only models in this list can be auto-started.
- **Single instance per model**: `repo_id` is the PRIMARY KEY in both `instances` and `launch_configs` — no per-pool concept.
- **Lazy start**: first request for an unstarted allowed model triggers daemon `run`.
- **No aliases or routing table**: clients address models by their HuggingFace repo ID directly. No model name rewriting — the upstream URL is `http://localhost:{port}/v1/{endpoint}` with the original `model` field in the body.
- **`/v1/models`** lists only READY instances from the local DB (no aliases, no full OpenAI schema — just `{"id": repo_id, "object": "model"}`).
- **Multi-node**: nginx uses lua-nginx-module with `crc32_short` of the model name to hash-route to the correct GPU node. If the hashed node doesn't have the model running, `proxy_next_upstream` retries on other nodes (with `proxy_intercept_errors` returning 503).

### Key difference

| | birdbrain | Whimbrel |
|---|---|---|
| **Model addressing** | Canonical name + aliases (routing table) | HuggingFace repo ID only |
| **Auto-start** | No — `503 Retry-After` if not running | Yes — on-demand from allow-list |
| **Instance multiplicity** | One job per pool, always running | One instance per repo_id, lazy start |
| **Model discovery** | Static TOML config | `allowed_models.yaml` allow-list |
| **No-backend behavior** | `503` | `503` with "use ELSA for production" message |
| **Aliases** | Supported | Not supported |

---

## 5. Authentication & Authorization

### birdbrain

- **File-based keystore**: `api_keys.json` mapping `sk_…` → `{email, key, created_at, last_used}`.
- **`fcntl.flock(LOCK_EX)`** for concurrent access — the entire file is read fresh from disk on entry to the lock context, and written back atomically on exit.
- **Email-based identity**: keys are tied to email addresses. `verify_key()` returns the email and updates `last_used`.
- **Optional auth**: if no keys exist, auth is disabled (all requests pass). Simple for research use.
- **No hashing**: keys are stored in plaintext in the JSON file (the file itself is the credential store).
- **No expiry, no revocation granularity beyond key deletion.**
- **Race-condition fix**: the running gateway process calls `verify_key()` on every request (updating `last_used`), which re-reads from disk under the file lock — this prevents the gateway's in-memory snapshot from clobbering keys added via `birdbrain add-key` CLI while the server is running.

### Whimbrel

- **DB-based API keys**: `api_keys` table with `key_hash` (SHA-256), `name`, `identity_type` (user/team/client), `identity_value`, `expires_at`, `revoked`, `request_count`.
- **Key prefix**: `wk_` prefix for easy identification. Plaintext key shown once at creation, only hash stored.
- **Three identity types**: user, team, client — used for rate limiting, quotas, and usage attribution.
- **Expiry & revocation**: keys can expire (timestamp check) or be revoked (soft-delete flag). `delete_api_key` for hard delete.
- **Always-on auth**: proxy validates Bearer token on every `/v1/` request via `_authenticate_request()` → `verify_api_key()`. No "auth disabled" mode.
- **nginx TLS**: head node nginx terminates TLS, proxies to GPU node proxy. Authorization header forwarded.
- **Rate limiting**: token bucket (10 RPS default), keyed by identity_value or IP. Currently in-memory (`RateLimiter` class with dict buckets).
- **Quotas**: 24h rolling window token quota per user (100K default), tracked in `requests_ledger`. Enforcement is noted as requiring middleware configuration — the `limits/` module appears to be partially implemented.

### Key difference

| | birdbrain | Whimbrel |
|---|---|---|
| **Key storage** | JSON file (plaintext) | SQLite (SHA-256 hashed) |
| **Identity model** | Email only | user/team/client |
| **Auth model** | Optional, simple | Mandatory, multi-tenant |
| **Expiry** | No | Yes (per-key) |
| **Revocation** | Hard delete | Soft-delete (revoked flag) + hard delete |
| **Rate limiting** | None | Token bucket (10 RPS) |
| **Quotas** | None | Per-user 24h token quota |
| **TLS** | Not handled (uvicorn only) | nginx TLS termination on head node |
| **Concurrency** | fcntl file locking | SQLite WAL + retry wrapper |

---

## 6. Deployment & Infrastructure

### birdbrain

- **User-level**: `pip install -e ".[dev]"` — no root needed.
- **Single process**: `birdbrain start` runs uvicorn on the login node.
- **Config**: TOML file at `~/.birdbrain/birdbrain.toml` or `./birdbrain.toml`.
- **No infrastructure dependencies**: no nginx, no systemd, no database. Just Python + Slurm.
- **Container**: Singularity SIF pulled and managed by the user.
- **Air-gapped support**: documented in AGENTS.md — pre-pull container and model weights.
- **Logging**: Python `logging` to file + stderr.

### Whimbrel

- **System-installed**: deployed at `/opt/whimbrel`, requires root for:
  - systemd services (`whimbrel-daemon.service`, `whimbrel-proxy.service`)
  - nginx config generation and deployment
  - logrotate config at `/etc/logrotate.d`
  - SQLite DB at `/hpc/db/hf-eval-model-hub/whimbrel-db/` (shared NFS path)
  - SGLang venv at `/opt/whimbrel/.sglang-venv`
- **Interactive installer**: `whimbrel install-prereqs system` — questionary-based wizard that:
  - Discovers GPU nodes via ping
  - Sets up venvs via `uv`
  - Downloads/pulls models
  - Generates systemd units, nginx config, logrotate
  - Configures cluster settings
- **Module system**: `module load whimbrel` available on head node for SLURM job scripts.
- **nginx required**: head node nginx with lua-nginx-module for model-affinity routing, TLS cert paths hardcoded for `reedling4.fda.gov`.
- **Multi-node**: proxy runs on each GPU node; daemon on each GPU node; head node nginx distributes.
- **Logging**: Rich console handler with TTY detection, file handler to `/var/log/whimbrel/`.

### Key difference

| | birdbrain | Whimbrel |
|---|---|---|
| **Privileges required** | User-level (no root) | Root (systemd, nginx, /opt, /var/log) |
| **Process topology** | Single process | Daemon + proxy + nginx + SGLang (per node) |
| **Config management** | TOML file | YAML cluster config + hardcoded constants |
| **Install complexity** | `pip install` + copy TOML | Interactive wizard, venv setup, systemd, nginx |
| **TLS** | Not provided | nginx with hardcoded cert paths |
| **Log management** | Basic file logging | Rich logging + logrotate + manual rotation CLI |
| **Container** | Singularity SIF | SGLang venv (no container) |


### Reedling-specific deployment notes

The two projects are deployed differently on Reedling:

| Aspect | birdbrain (Reedling) | Whimbrel (Reedling) |
|---|---|---|
| **Model** | Poolside/Laguna S & XS 2.1 FP8 (local NFS paths, not HF repo IDs) | Gated models from allow-list (`google/gemma-4-12B-it`, `mistralai/Mistral-7B-Instruct-v0.1`) |
| **vLLM/SGLang version** | vLLM 0.29.0 SIF (pinned — 0.30.0 regressed Laguna config parsing) | SGLang >=0.5.20 in dedicated venv |
| **Install method** | `pip3 install --user -e .` + rsync scripts | `module load whimbrel` + `whimbrel install-prereqs system` |
| **Model path** | Local NFS path (`/nfs/scratch/.../Laguna-S-2.1-FP8`) passed verbatim to `--model` | HF repo ID passed to `sglang serve` |
| **Container** | Singularity SIF with `--writable-tmpfs`, `--bind /nfs/scratch` | No container — SGLang in venv |
| **CUDA graphs** | Graph mode enabled (vLLM 0.29.0 fixes TP-2 H100 NVL crash that forced `--enforce-eager` in 0.25.1) | Not applicable (SGLang graph capture has different topology behavior) |
| **Kernel toggle** | `VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0` via `SINGULARITYENV_` in job template | N/A (SGLang handles differently) |
| **Cache redirect** | `VLLM_CACHE_ROOT`, `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR` to NFS scratch | `SINGULARITYENV_*` not used (no container); caches in venv or env |
| **Wall time** | 48h with 360min lead time (long rotations for stable pool on shared GPUs) | 12h SLURM time limit (idle timeout reaps unused models) |
| **GPUs** | 2×H100 NVL (Laguna-S) or 1×H100 (Laguna-XS) | `gpu:h100:2` GRES per SLURM job |
| **API key storage** | `api_keys.json` (plaintext, fcntl-locked) | SQLite (SHA-256 hashed) |
| **Internal auth** | `X-Internal-Key` header for backend↔gateway callbacks | Internal key in SGLang config YAML, not in process args |
---

## 7. Code Quality & Maintainability

### birdbrain

- **6 source files**, ~1200 LOC: `gateway.py`, `lifecycle.py`, `health.py`, `keystore.py`, `cli.py`, `config.py` + template.
- **Pydantic v2** for config validation with `@model_validator` for legacy migration and backend uniqueness checks.
- **pytest** suite with `respx` for httpx mocking, `tmp_path` for state isolation. Tests cover: job state serialization, state persistence, two-manager isolation, legacy state migration, job submission, squeue querying, auth enforcement, routing by alias, models endpoint schema, backend registration.
- **mypy** type checking with `disallow_untyped_defs = true`.
- **ruff** linting + formatting (line-length 100).
- Comments are thorough — explain *why* (e.g., hostname matching bug fix, 20-minute registration timeout rationale, compile cache on shared scratch).

### Whimbrel

- **~50 source files** across nested packages: `daemon/`, `proxy/`, `runtime/`, `config/`, `limits/`, `usage/`, `auth/`, `db/`, `hf/`, `commands/`, `templates/`, `utils/`, `logging/`, `artifacts.py`.
- **`beartype`** on every function — runtime type checking, not just static.
- **`tenacity`** retry decorator (`retry_db_logged`) on every DB operation.
- **`rich`** for console output, **`questionary`** for interactive prompts.
- **9 DB schema migrations** — indicates significant schema evolution and a team that values upgrade compatibility.
- **No project tests found** — the `find` command for test files only found dependency tests (beartype, fsspec, etc.), not whimbrel's own tests. This is a notable gap for an 8K LOC project with 9 migrations.
- Python 3.12+ required.
- Constants are hardcoded for the Reedling cluster in `constants.py` (default model path `/hpc/db/hf-eval-model-hub/whimbrel-db/whimbrel.db`, GPU nodes `g040`-`g045`, nginx cert paths, etc.) with a YAML override mechanism.

### Key difference

| | birdbrain | Whimbrel |
|---|---|---|
| **Source files** | ~6 | ~50 |
| **Type checking** | mypy (static) | beartype (runtime) + no mypy |
| **Tests** | pytest suite (gateway, lifecycle, keystore, config) | Not present in installed package; source repo not inspected |
| **Error clarity** | Inline comments explain root causes | Helpful error messages ("use ELSA for production") |
| **Schema migrations** | None — JSON files | 9 tracked migrations |
| **Target Python** | >=3.10 | >=3.12 |
| **Dependencies** | 6 runtime + 9 dev | 20 runtime + 9 dev |

---

## 8. Summary of Trade-offs

### Choose birdbrain if you want:

- **Simplicity** — install with `pip`, run on a login node, no root or infrastructure.
- **Zero-downtime guarantee** — proactive rolling rotation means a backend is always warm; the only gap is during first cold start.
- **Full control** — all config in TOML, model names and aliases, vLLM args, container choice.
- **Correctness focus** — the code is dense but well-tested, with clear comments on edge cases (hostname mismatch, graph capture timeouts, compile cache reuse).

### Choose Whimbrel if you want:

- **Resource efficiency** — models only start when used, idle ones are reaped. On shared clusters, this can save substantial GPU time.
- **Enterprise features** — per-user quotas, rate limiting, usage tracking, multi-tenant identity, TLS termination, log rotation, systemd integration.
- **Multi-node** — nginx model-affinity routing across a GPU node pool, with cluster-wide status aggregation.
- **Operational tooling** — interactive installer with node discovery, model allow-list, config management with known-good tracking.

### Notable gaps / risks

| Aspect | birdbrain | Whimbrel |
|---|---|---|
| **Tests** | Good coverage (pytest) | No tests found in installed package; source repo not inspected |
| **No-root deployment** | Supported | No — all paths root-only (`/opt/whimbrel`, `/var/log/whimbrel`, `/hpc/db/...`); popen mode still needs installer + root paths |
| **Proactive availability** | Yes (always-on) | No (cold start on first request) |
| **Idle cost** | Always pays for running jobs | Zero when idle |
| **Single-tenant** | Yes (simple) | Yes (multi-tenant via identity) |
| **Config flexibility** | TOML, user-editable | YAML at fixed paths, requires installer |
| **Backend engine choice** | vLLM only | SGLang only |

### Reedling-specific recommendation

On Reedling HPC specifically:

- **birdbrain** is the better fit if you need **always-on availability** (e.g., agents or services that can't tolerate cold-start latency), **full user-level control** (no admin privileges), or are serving **FP8 models that require vLLM** (e.g., Poolside/Laguna). The 48-hour wall-time + 360-min lead-time config on Reedling means backends are perpetually warm. The codebase ships with a pytest suite covering gateway, lifecycle, keystore, and config.
- **Whimbrel** is the better fit if you need **resource efficiency** (idle models reaped, saving GPU-hours on a shared cluster), **multi-tenant quotas** (per-user token limits, rate limiting), **usage auditing** (full request ledger with token/latency tracking for billing/attribution), or are serving **standard HF models** (not custom FP8 checkpoints) that benefit from SGLang's architecture. It requires root + admin buy-in for the systemd/nginx deployment model.

Key decision factor on Reedling: birdbrain costs **2 GPU-hours per warm pool every hour** (always-on), while Whimbrel costs **0 when idle** but charges cold-start latency (~10-60s for small models via SGLang) on the first request after idle. Whimbrel's test coverage is **unknown** — no tests were present in the installed package and the source repo was not inspected, which is a risk for an 8K LOC project with 9 DB schema migrations.

---

## Appendix: File Structure Comparison

### birdbrain (6 source files)
```
birdbrain/
├── __init__.py
├── gateway.py              # FastAPI app, OpenAI proxy, routing, registration
├── lifecycle.py            # Slurm job submission & rotation (LifecycleManager)
├── health.py               # Backend health monitoring (HealthMonitor)
├── keystore.py             # API key management (KeyStore, fcntl-locked)
├── cli.py                  # Click CLI: start, status, stop, add-key, etc.
├── config.py               # Pydantic config models, TOML loading, legacy migration
└── templates/
    └── vllm_job.sh.j2      # Slurm job template (vLLM in Singularity)
```

### Whimbrel (50+ source files)
```
whimbrel/
├── cli.py                  # Typer CLI entry point, dispatches to subcommands
├── constants.py            # All config as frozen dataclass (RC) — Reedling-specific defaults
├── exceptions.py           # AppError base exception
├── helpers.py              # Rich console helpers, hostname, package info
├── logging_utils.py        # TTY-aware Rich logging, file handler
├── artifacts.py            # systemd unit / nginx / logrotate config generation
├── artifacts.py (cont'd)   # nginx config with lua model-affinity routing
│
├── commands/               # Typer subcommand modules
│   ├── daemon.py           # daemon start/stop/restart/status
│   ├── run.py              # run/forget/ps/stop/restart (sends RPC to daemon)
│   ├── status.py           # daemon status display
│   ├── keys.py             # API key CRUD
│   ├── usage.py            # usage ledger queries
│   ├── limits.py           # rate limit / quota display
│   ├── hf.py               # HF cache ls/pull/rm
│   ├── logs.py             # log tail/clean/rotate/setup
│   └── install_prereqs.py  # interactive installer wizard
│
├── daemon/                 # Unix-socket RPC daemon
│   ├── __init__.py
│   ├── server.py           # start_daemon(), _dispatch(), _handle_run/stop/restart
│   ├── client.py           # call_daemon(), get_active_daemon_socket()
│   ├── status.py           # get_status_summary(), is_daemon_running()
│   └── restore.py          # restore_on_startup() for desired-state instances
│
├── proxy/                  # FastAPI uvicorn app (:30001)
│   ├── __init__.py
│   ├── app.py              # OpenAI-compatible proxy, middleware, /v1/* endpoints
│   ├── health_router.py    # route_with_health(), is_healthy() with latency budget
│   ├── identity.py         # extract_api_key(), authenticate_request()
│   └── errors.py           # map_error() → OpenAI error schema
│
├── runtime/                # SGLang process management
│   ├── __init__.py
│   ├── health.py           # wait_for_health(), check_auth_required()
│   ├── key_registry.py     # in-memory thread-safe (repo_id, pid)→internal_key
│   ├── node_discovery.py   # ping-based reachable node discovery, interactive prompts
│   ├── port_allocator.py   # sequential port allocation in range [31000, 31999]
│   ├── process_manager.py  # reap_process(), stop_instance(), find_running_sglang()
│   ├── sglang_launcher.py  # launch_sglang_popen(), launch_sglang_slurm(),
│   │                       #   _render_slurm_script(), build_sglang_cmd*()
│   └── validator.py        # validate_ext_args() — blocks reserved SGLang args
│
├── config/                 # YAML-based cluster config
│   ├── __init__.py
│   ├── cluster.py          # load/save/generate cluster_config.yaml, validate_config()
│   ├── models.py           # LaunchConfig Pydantic model
│   └── store.py            # DB-backed launch config persistence (known_good tracking)
│
├── limits/                 # Rate limiting & quotas
│   ├── __init__.py
│   ├── rate.py             # RateLimiter (token bucket), rate_limit_middleware
│   └── quota.py            # check_user_quota() (24h rolling window)
│
├── usage/                  # Request usage tracking
│   ├── __init__.py
│   └── ledger.py           # insert_request(), update_usage(), query_usage(),
│                           #   mark_complete(), mark_disconnect(), record_latency()
│
├── auth/                   # API key authentication
│   ├── __init__.py
│   └── keys.py             # create_api_key(), verify_api_key(), list/revoke/delete
│
├── db/                     # SQLite schema & migrations
│   ├── __init__.py
│   ├── schema.py           # TableSchemas (schema v9), 7 tables
│   └── migrations.py       # db_connection(), init_db(), _apply_migrations()
│
├── hf/                     # HuggingFace model management
│   ├── __init__.py
│   ├── cache.py            # list_cached_models()
│   ├── pull.py             # pull_model() — safetensors-only downloads
│   ├── remove.py           # remove_model()
│   └── utils.py            # shared HF utilities
│
├── utils/                  # Shared utilities
│   ├── __init__.py         # retry_db_logged (tenacity SQLite retry decorator)
│   └── subprocess.py       # run_cmd() wrapper
│
├── logging/                # Structured logging
│   ├── __init__.py
│   └── structured.py       # (empty module — placeholder)
│
└── templates/
    ├── sglang_job.sh.j2    # SLURM job script for SGLang (config-file based API key)
    └── allowed_models.yaml # Model allow-list for auto-start
```

### Deployment topology

**birdbrain** (single-node gateway):
```
[Login node]
  birdbrain gateway (FastAPI/uvicorn :8765)
    ├── LifecycleManager (per pool) → sbatch → Slurm → [Compute node]
    ├── HealthMonitor (per pool) → polls /health
    └── KeyStore (file-based)
```

**Whimbrel** (multi-node cluster):
```
[Head node]
  nginx (TLS :30000) → lua model-affinity routing
    ├── → [GPU node g040] whimbrel-proxy (:30001) → SGLang
    ├── → [GPU node g041] whimbrel-proxy (:30001) → SGLang
    ├── → [GPU node g042] whimbrel-proxy (:30001) → SGLang
    ├── → [GPU node g043] whimbrel-proxy (:30001) → SGLang  (single-host: only this node)
    └── → [GPU node g044] whimbrel-proxy (:30001) → SGLang
        → [GPU node g045] whimbrel-proxy (:3001) → SGLang

Each GPU node:
  whimbrel-daemon (Unix socket RPC) ↔ SQLite DB (shared NFS)
    → sbatch/squeue/scancel ↔ SLURM scheduler → SGLang job
```
