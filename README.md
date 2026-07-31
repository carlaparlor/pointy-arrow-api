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

| ID | Context |
| --- | --- |
| `claude-haiku-4.5` | 200k |
| `composer-2.5` | 200k |
| `deepseek-v4-flash` | 1M |
| `deepseek-v4-pro` | 1M |
| `gemini-3-flash` | 1M |
| `gemini-3.1-flash-lite` | 1M |
| `gemini-3.1-pro` | 1M |
| `gemini-3.5-flash` | 1M |
| `gemini-3.5-flash-lite` | 1M |
| `gemini-3.6-flash` | 1M |
| `gemma-4-31b` | 0 |
| `glm-5.2` | 1M |
| `gpt-oss-120b` | 131k |
| `grok-4.5` | 200k |
| `kimi-k2.6` | 262k |
| `kimi-k2.7-code` | 262k |
| `kimi-k3` | 1M |
| `laguna-s-2.1` | 262k |
| `ling-3.0-flash` | 262k |
| `minimax-m3` | 0 |
| `nemotron-3-super` | 1M |
| `nemotron-3-ultra` | 1M |
| `qwen-3.8-max` | 0 |
| `step-3.7-flash` | 200k |

## Endpoints

- `POST /v1/chat/completions` — chat completions. `stream: true` for SSE. Supports `tools`, `tool_choice`, and `reasoning_effort`.
- `GET  /v1/models` — list models.
- `GET  /v1/models/{id}` — retrieve one model.
- `GET  /health` — liveness plus live credential-pool stats.
- `POST /admin/reload` — reload `credentials.json` from disk without restarting.

`/chat/completions` and `/api/v1/chat/completions` are accepted as aliases of `/v1/chat/completions`.
