# birdbrain

An OpenAI-compatible gateway service that provides stable, continuous inference API access backed by dynamically scheduled LLM jobs on Slurm-managed HPC clusters.

## Overview

**birdbrain** solves a specific problem: on HPC systems with strict wall-time limits and no dedicated service infrastructure, how do you provide a persistent inference endpoint?

The answer: run overlapping rolling vLLM job reservations on the cluster, and have a lightweight gateway on the login node that automatically rotates to a new backend before the current job is killed by the scheduler. Clients never see a gap.

### Why not just use Ollama?

Ollama and vLLM solve a different problem than birdbrain does. Ollama gives you an OpenAI-compatible API and model serving, same as vLLM — but it has no concept of Slurm, job wall-time, or backend rotation, and its multi-GPU support is layer-splitting (like llama.cpp), not true tensor parallelism, so it can't match vLLM's throughput for large models split across multiple GPUs (e.g. `--tensor-parallel-size 2`). It also has no multi-user API key management.

birdbrain's value is the scheduling/rotation glue *around* a serving engine (Slurm job submission, lead-time rotation, health-checked failover, API keystore) — swapping the backend from vLLM to Ollama wouldn't remove any of that code, and would cost you tensor parallelism on multi-GPU jobs.

## Features

- **OpenAI-compatible API** — drop-in replacement for clients using `openai` SDK or `curl`
- **No API changes needed** — works with existing LangChain, Pi.ai, or custom client code
- **Transparent job rotation** — gateway swaps backends without dropping requests
- **Multi-model hosting** — run several independently-rotating model pools (e.g. a large model on multiple GPUs plus small models on single GPUs) behind one endpoint, routed by model name or alias, similar to AWS Bedrock
- **User-level permissions** — no admin privileges required; runs as a regular user
- **Scheduled uptime/downtime** — an optional `[schedule]` section cancels jobs outside configured hours so GPUs go back to other cluster users overnight
- **Simple deployment** — `pip install` and a config file; designed for researchers, not SREs

### Multi-model routing

A gateway instance can manage any number of backend pools, each with its own Slurm sizing, model, and rotation timeline. Configure them as `[[backends]]` entries in the TOML (see `birdbrain.toml`):

```toml
[[backends]]
name = "llama-3.3-70b"
aliases = ["llama", "default"]
[backends.slurm]
gres = "gpu:h100:2"
port = 50000
[backends.model]
name = "meta-llama/Llama-3.3-70B-Instruct"
vllm_extra_args = "--tensor-parallel-size 2"

[[backends]]
name = "small-model-a"
aliases = ["small", "fast"]
[backends.slurm]
gres = "gpu:h100:1"
port = 50001
[backends.model]
name = "..."
```

Clients target a pool by sending either its exact model name or one of its aliases in the `model` field:

```bash
curl ... -d '{"model": "small", "messages": [...]}'
```

Each pool runs its own independent Slurm job lifecycle (submission, rotation, health checks) — one pool rotating or cold-starting never affects another. An unrecognized `model` value returns `400`. `GET /v1/models` lists the canonical model name and every alias for each currently-live pool.

Existing single-model configs (top-level `[slurm]`/`[model]`, no `[[backends]]`) keep working unchanged — they're automatically treated as one backend pool named `"default"`.

### Replicas and co-located models

A pool can run more than one vLLM server inside its single Slurm job:

- `replicas = N` (under `[backends.slurm]`) launches N copies of the pool's model. The job's GPUs are split evenly between them and the gateway round-robins requests across the copies. If one copy dies, the rest keep serving.
- `[[backends.secondary_models]]` adds a different model to the same job, pinned to one of its GPUs and served under its own `backend_name`/aliases. This is meant for small sidecars, e.g. an embedding model next to a chat model, called through `POST /v1/embeddings`. Because a sidecar shares its pool's job, it is rotated with that job, and restarting a dead sidecar means restarting the whole job.

Instances use consecutive ports starting at `slurm.port`, and two pools' port ranges must not overlap. A co-located model shares VRAM with the primary, so the two `--gpu-memory-utilization` values must add up to less than 1.0. See the commented examples in `birdbrain.toml`.

## Quick Start

### Prerequisites

- Python 3.10+
- Access to a Slurm cluster with GPU nodes
- vLLM Singularity container (or venv with vLLM installed)
- Shared filesystem (NFS) accessible from login and compute nodes
- Model weights pre-downloaded to HuggingFace cache (for air-gapped compute nodes)

### Installation

```bash
pip install birdbrain
# or with uv
uv pip install birdbrain
```

### Configuration

Copy the example config and customize for your cluster:

```bash
mkdir -p ~/.birdbrain
cp birdbrain.toml ~/.birdbrain/
# Edit ~/.birdbrain/birdbrain.toml with your partition, wall-time, model, etc.
```

### First-Time Setup (air-gapped clusters)

If running on air-gapped compute nodes, pre-stage container and model weights:

```bash
# On login node (has internet access)
mkdir -p ~/containers
singularity pull ~/containers/vllm-openai.sif docker://vllm/vllm-openai:latest

# Pre-download model weights
pip install --user huggingface-cli
huggingface-cli download meta-llama/Llama-3.1-8B
```

Update config paths to match your environment (see `birdbrain.toml`).

### Running

Generate API key (recommended):

```bash
birdbrain add-key your.email@example.com
# Save the generated key for client configuration
```

Start the gateway:

```bash
birdbrain start
```

This will submit the first vLLM job and enter a holding pattern (`503 Retry-After`) until the backend is ready. Check status in another terminal:

```bash
birdbrain status
```

Once the backend is warm, the gateway accepts requests at `http://localhost:8765` (configurable).

Test it:

```bash
# Get your API key (use email from add-key command)
export OPENAI_API_KEY=$(birdbrain show-key your.email@example.com | grep -o 'sk-[^ ]*')

curl -X POST http://localhost:8765/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

### Command Reference

```bash
birdbrain start                  # Start the gateway (foreground)
birdbrain status                 # Show current backend, job IDs, health
birdbrain stop                   # Graceful shutdown (Ctrl+C in start terminal)

# API key management
birdbrain add-key <email>        # Create API key for user
birdbrain list-keys              # List all API keys
birdbrain show-key [email]       # Show key for email (or all keys)
birdbrain revoke-key <key|email> # Revoke API key
```

## Architecture

Three core components, each instantiated **once per configured backend pool**:

1. **Gateway** — FastAPI app on the login node; routes each request to the backend pool owning the requested model name/alias, and proxies to that pool's active vLLM backend
2. **Lifecycle Manager** — one per backend pool; submits new Slurm jobs before that pool's current job expires, orchestrates rotation independently of other pools
3. **Health Monitor** — one per backend pool; polls that pool's backend health, triggers rotation on failure

See [CLAUDE.md](CLAUDE.md) for detailed architecture and development notes.

## Configuration

See `birdbrain.toml` for all options. Key sections:

- `[gateway]` — host, port, request retry window
- `[[backends]]` — one entry per model pool; each has its own `[backends.slurm]` (partition, GPU request, wall-time, lead-time, port) and `[backends.model]` (model name, vLLM arguments), plus an `aliases` list
- `[paths]` — config file locations, state file, logs directory
- `[schedule]` *(optional)* — daily service window. When present, all jobs are cancelled outside the window and resubmitted when it opens, so GPUs are returned to other cluster users while the gateway is idle. `start` and `stop` are `HH:MM` wall-clock times interpreted in `timezone` (an IANA zone name; default `UTC`). `days` lists the days a window *starts* on (`mon`…`sun`); default is every day. A `stop` that is not after `start` wraps overnight into the next day. Omit the whole section to run around the clock.

Legacy top-level `[slurm]`/`[model]` tables (no `[[backends]]`) are still supported and are migrated automatically into a single `"default"` backend pool.

## Deployment

**Deploying on a new cluster?** Start with **[AGENTS.md](AGENTS.md)** — a
cluster-agnostic deployment guide written for an agent or engineer bringing
birdbrain up on a grid it wasn't built for. It walks the filesystem/bind-mount,
container-version, model-fit, and connectivity gotchas in order, with a
failure→cause quick reference. (Site operators may keep a private,
cluster-specific companion doc with already-solved values for their own grid;
such files are gitignored and not part of this public repo.)

For production, use `nohup` or systemd user unit:

```bash
nohup birdbrain start > ~/.birdbrain/logs/gateway.log 2>&1 &
```

The gateway is designed to run as a persistent process on the login node. On restart, it rehydrates each backend pool's job state from its own `~/.birdbrain/state-<pool-name>.json` and resumes normal operation; a rotation job that was still pending is cancelled and resubmitted. (A pool named `"default"` — the legacy single-model shape — falls back to the older `~/.birdbrain/state.json` if its own state file doesn't exist yet, so upgrading an existing single-model deployment in place doesn't lose track of an already-running job.)

If replacement jobs keep failing (dying at startup, never registering, or failing health checks), the gateway waits longer between them: 60 s after the first failure, doubling up to 30 minutes, until a health check passes. A cluster fault that kills every job therefore costs a couple of jobs per hour rather than one every few minutes.

## Development

See [CLAUDE.md](CLAUDE.md) for setup, testing, and linting commands.

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run type checking and linting
mypy birdbrain/
ruff check birdbrain/
```

## Limitations (v1)

- Aliases are static: fixed in the TOML, changed only by editing config and restarting (no runtime alias re-pointing/blue-green model swaps)
- Single shared API key set across all backend pools (suitable for research teams, not per-model or multi-tenant access control)
- Slurm only (abstraction layer designed for future SGE/PBS support)
- No web UI

## Support

For issues, feature requests, or contributions, see the [GitHub repository](https://github.com/crashfrog/birdbrain).

---

*Named after a drink at a party. You weren't there.*
