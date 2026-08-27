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
