# Using birdbrain with VS Code (Continue)

This guide sets up the [Continue](https://marketplace.visualstudio.com/items?itemName=continue.continue)
VS Code extension to use a birdbrain gateway as a code-completion / chat
backend. birdbrain speaks the OpenAI-compatible API, so Continue treats it
like any OpenAI-compatible endpoint.

---

## 1. Prerequisites

1. **VS Code** installed locally.
2. **API key** — you already have one (distributed out of band). It looks like
3. **Gateway endpoint** — the host:port of the birdbrain gateway:
   - On Reedling: `http://Reedling2.fda.gov:8000`
   - Default config (single-user): `http://localhost:8765`
4. **Model pool name or alias** — pick the pool you want. Run this to list
   live pools:

   ```bash
   curl -H "Authorization: Bearer <YOUR-API-KEY>" \
     http://<gateway-host>:<port>/v1/models
   ```

   The response's `id` fields are the exact strings to use as your `model` in
   Continue. They are either canonical names (e.g. `laguna-s-2.1`) or
   short aliases (e.g. `large`, `small`, `fast`). Ask your gateway operator or
   check the config TOML for the full alias list.

---

## 2. Connect to the gateway

The birdbrain gateway runs on the Reedling2 HPC login node and is reachable
directly from your local machine:

```
http://Reedling2.fda.gov:8000
```

Use this URL (with `/v1` appended for the API base) in your Continue
configuration. No SSH tunnel or Remote-SSH connection is needed — the
gateway endpoint is published for client access.

If your local network requires VPN or HPC bastion access to reach
`Reedling2.fda.gov`, ensure that tunnel is active first. The gateway
itself is the only network dependency; model weights and containers stay
on the cluster.

---

## 3. Install and configure Continue

### Install

In VS Code: `Extensions` → search **"Continue"** → install
`continue.continue`.

### Configure the model

Open Command Palette (`Ctrl/Cmd + Shift + P`) → **Continue: Edit Config**.
This opens `~/.continue/config.yaml` (or `config.json`).

#### YAML config (current default)

```yaml
models:
  - name: "Laguna-Large"
    provider: "openai"
    apiBase: "http://Reedling2.fda.gov:8000/v1"
    apiKey: "<your-api-key-here>"
    model: "large"          # or "laguna", "laguna-large", "laguna-s-2.1"
    contextLength: 262144

  - name: "Laguna-Small"
    provider: "openai"
    apiBase: "http://Reedling2.fda.gov:8000/v1"
    apiKey: "<your-api-key-here>"
    model: "fast"           # or "small", "laguna-small", "laguna-xs-2.1"
    contextLength: 262144
```

- `apiBase`: append `/v1` to your gateway address.
- `model`: the alias or canonical name from step 1.
- `contextLength`: match your model's `--max-model-len` (262144 on Reedling).
- `apiKey`: paste the key you received.

#### JSON config (older Continue versions)

```json
{
  "model": "large",
  "provider": "openai",
  "apiKey": "<your-api-key-here>",
  "apiBase": "http://Reedling2.fda.gov:8000/v1",
  "contextLength": 262144,
  "title": "Laguna-Large"
}
```

---

## 4. Set up inline autocomplete (Tab completion)

Continue's tab-autocomplete uses whichever model is configured as the
**autocomplete model**. To point it at birdbrain:

1. Open Command Palette → **Continue: Edit Config**.
2. Add (or set) the `tabAutocompleteModel` field:

```yaml
tabAutocompleteModel:
  name: "Laguna-Small"
  provider: "openai"
  apiBase: "http://Reedling2.fda.gov:8000/v1"
  apiKey: "<your-api-key-here>"
  model: "fast"
```

Or in JSON:

```json
{
  "model": "fast",
  "provider": "openai",
  "apiKey": "<your-api-key-here>",
  "apiBase": "http://Reedling2.fda.gov:8000/v1",
  "tabAutocompleteModel": { ... same block ... }
}
```

### Autocomplete settings

Open **Settings** (`Ctrl/Cm + ,`) → search `continue` to tune:

- **`Continue: Tab Autocomplete Delay`** — ms before autocomplete fires after
  you stop typing (default 500–1000). Lower = snappier, more interrupts.
- **`Continue: Disable Tab Autocomplete`** — uncheck to enable.
- **`Continue: Custom Instructions`** — add team conventions or repo-specific
  context that the model should follow on every completion.

### Using it

- Type code, pause, and a gray suggestion appears.
- `Tab` accepts it; `Ctrl/Cmd + →` cycles through alternatives.
- `Ctrl/Cmd + Shift + L` opens the full Continue sidebar for multi-line / chat
  completion.

---

## 5. Switching models

Once configured, you can switch pools without editing config — use the Continue
sidebar's model dropdown (the `name` field in your config entries). Choose:

- **"Laguna-Large"** for deep reasoning, large-context tasks, or heavy
  multi-file refactors.
- **"Laguna-Small"** for fast inline completions and low-latency iteration.

Both route through the same gateway; the gateway resolves the `model` field to
the correct backend pool automatically.

---

## 6. Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| `401 Unauthorized` | API key missing or wrong — confirm `apiKey` is set and the key is active (`birdbrain list-keys`). |
| `Connection refused` / `ECONNREFUSED` | Gateway not running or VPN/bastion not active. Verify `curl http://Reedling2.fda.gov:8000/health` from your terminal. |
| `400 Bad Request` — "model not found" | The `model` field doesn't match a registered pool alias. Run `/v1/models` to get the exact names. |
| Autocomplete suggestions are slow | The large pool has a longer TTFT. Switch the autocomplete model to the small/fast pool. |
| Gateway returns `503 Retry-After` | Backend is cold-starting or rotating. Wait; the gateway auto-recovers. |

---

## 7. Quick reference (Reedling)

```bash
# 1. Verify gateway is reachable:
curl http://Reedling2.fda.gov:8000/health

# 2. List available model pools:
curl -H "Authorization: Bearer <your-key>" \
  http://Reedling2.fda.gov:8000/v1/models

# 3. Test directly:
curl -X POST http://Reedling2.fda.gov:8000/v1/chat/completions \
  -H "Authorization: Bearer <your-key>" \
  -H "Content-Type: application/json" \
  -d '{"model":"fast","messages":[{"role":"user","content":"def fib(n):"}],"max_tokens":64}'
```

Distribute API keys separately from this guide via your team's secure channel.
