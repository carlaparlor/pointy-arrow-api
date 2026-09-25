# pointy-arrow-api

An OpenAI-compatible local server.

## Quick start

```powershell
pip install -r requirements.txt
python pa_server.py
```

Or with uvicorn:

```powershell
uvicorn pa_server:app --host 127.0.0.1 --port 8001
```

## Models

The current model list is not finalised for the final release. Expect models to not work, be removed, or be added.
Also note that model slugs are self-reported by upstream providers and may not be truthful. We cannot guarantee model performance.

| ID | Context |
| --- | --- |
| `deepseek-v4-flash` | 200k |
| `deepseek-v4-flash-0731` | 1.3M |
| `deepseek-v4-pro` | 1M |
| `deepseek-v4-pro-0813` | 1M |
| `gemini-3.5-flash` | 1M |
| `gemini-3.6-flash` | 200k |
| `glm-5.2` | 1M |
| `gpt-5.6-luna` | 200k |
| `gpt-5.6-sol` | 200k |
| `grok-4.5` | 200k |
| `grok-4.6` | 500k |
| `kimi-k3` | 1M |
| `minimax-m3` | 1M |
| `muse-spark-1.1` | 1M |
| `qwen3.8-2.4t-a95b` | 1M |
| `qwen3.8-27b` | 262k |
| `qwen3.8-flash` | 991k |
| `qwen3.8-max` | 200k |

## Endpoints

- `POST /v1/chat/completions` — chat completions. `stream: true` for SSE. Supports `tools`, `tool_choice`, and `reasoning_effort`.
- `GET  /v1/models` — list models.
- `GET  /v1/models/{id}` — retrieve one model.
- `GET  /health` — liveness plus live credential-pool stats.
- `POST /admin/reload` — reload `credentials.json` from disk without restarting.

`/chat/completions` and `/api/v1/chat/completions` are accepted as aliases of `/v1/chat/completions`.

## Testing

The suite is in `tests/` and runs against **real processes over real sockets** — no
mocks, no monkeypatching, no substituted objects anywhere in the code under test.

```powershell
pip install -r requirements.txt pytest httpx
pytest -v                 # everything except the live tests
pytest -m live -v         # the real Gratisfy endpoints
```

### How it works

`tests/conftest.py` starts a real uvicorn process running `pa_server:app` (exactly
what `python pa_server.py` does) on a real port, and points it at a second real
HTTP server, `tests/reference_upstream.py`. That reference server is a genuine
implementation of the three wire protocols this project speaks:

| Protocol | Endpoint |
| --- | --- |
| Gratisfy website chat (SSE) | `POST /api/chat` |
| Supabase GoTrue auth | `POST /auth/v1/signup`, `POST /auth/v1/token` |
| mail.tm (Hydra/JSON-LD) | `GET /domains`, `POST /accounts`, `POST /token`, `GET /messages` |

Its behaviour is scripted from a JSON scenario file, and it records every request it
receives so the tests can assert on what the router *actually* sent upstream
(payload shape, routing order, retries, credential selection). The production
endpoints are overridable for this purpose:

| Variable | Default |
| --- | --- |
| `PA_CHAT_ENDPOINT` | `https://gratisfy.xyz/api/chat` |
| `PA_SUPABASE_URL` | `https://auth.gratisfy.xyz` |
| `PA_MAILTM_BASE` | `https://api.mail.tm` |
| `PA_CREDENTIALS_PATH` | `./credentials.json` |
| `PA_MODELS_PATH` | `./models.json` |
| `PA_AUTO_HARVEST` / `PA_AUTO_REFRESH` | `1` |

### Live tests

`tests/test_live_upstream.py` drives the real `gratisfy.xyz`, `auth.gratisfy.xyz`
and `api.mail.tm`. Reachability probes run by default; anything that would create a
real account or spend a real request is opt-in:

```powershell
$env:PA_LIVE_HARVEST = "1"      # provision a real throwaway credential
$env:PA_LIVE_TOKEN   = "<access token>"   # chat with a real session
pytest -m live -v
```

`.github/workflows/test.yml` runs both suites on every push, so the live tests
execute on a runner that has unrestricted egress.

### What the suite covers

* **`test_registry.py`** — catalogue integrity, README parity, alias/label
  resolution, slug collisions, route health and cooldowns, live reloads.
* **`test_pure_functions.py`** — message translation, tool-spec rendering,
  plaintext tool-call extraction, error classification.
* **`test_parallel_tool_calls.py`** — parallel tool calls survive extraction.
* **`test_credentials.py`** — token refresh (refresh-token grant, password-grant
  fallback), full signup→verify→login harvest over HTTP, CLI subcommands.
* **`test_http_api.py`** — every documented endpoint, aliases, error shapes,
  `/admin/reload` against files rewritten underneath the running server.
* **`test_chat_e2e.py`** — streaming and non-streaming completions, usage
  normalisation, keepalives, reasoning, native and emulated tool calling, message
  trimming, retry/backoff, credential retirement, route failover, concurrency.
* **`test_robustness.py`** — junk catalogues, junk pools, malformed usage,
  concurrent pool writes.

### Bugs this suite found

1. A `models.json` that was valid JSON but had no `models` key raised
   `TypeError` in `ModelRegistry.reload()` — enough to stop the server from
   starting or to break `POST /admin/reload`.
2. A `credentials.json` row that was not an object raised `AttributeError` in
   `CredentialFile.load()`, taking down `CredentialPool.reload()`.
3. A JSON *list* of parallel tool calls was shredded by `_last_json_object`, so
   emulated tool calling silently dropped every call but the first.
4. Non-numeric `expires_at` / `context_window` values crashed the loaders.
5. Concurrent `CredentialFile.save()` calls could interleave and leave a
   half-written pool file; saves are now atomic (temp file + rename) and
   serialised per path.
6. A malformed upstream `usage` block raised inside `_normalize_usage`, turning a
   bad upstream response into a 500.
