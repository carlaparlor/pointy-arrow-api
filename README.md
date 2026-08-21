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
| `deepseek-v4-flash-0731` | 1M |
| `deepseek-v4-pro` | 1M |
| `deepseek-v4-pro-0813` | 0 |
| `dots3-note-preview` | 512k |
| `gemini-3-flash` | 1M |
| `gemini-3.1-flash-lite` | 1M |
| `gemini-3.1-pro` | 200k |
| `gemini-3.5-flash` | 1M |
| `gemini-3.5-flash-lite` | 1M |
| `gemini-3.6-flash` | 200k |
| `gemma-4-26b` | 0 |
| `glm-5.2` | 1M |
| `gpt-5.4` | 200k |
| `gpt-5.4-mini` | 400k |
| `gpt-5.5` | 200k |
| `gpt-5.6-luna` | 200k |
| `gpt-5.6-sol` | 200k |
| `gpt-5.6-terra` | 200k |
| `gpt-oss-120b` | 0 |
| `gpt-oss-120b-2` | 131k |
| `grok-4.5` | 200k |
| `grok-4.6` | 200k |
| `inkling` | 0 |
| `kimi-k2.6` | 262k |
| `kimi-k2.7-code` | 262k |
| `kimi-k3` | 1M |
| `laguna-s-2.1` | 262k |
| `mimo-v2.5-pro` | 200k |
| `minimax-m2.7` | 0 |
| `minimax-m3` | 200k |
| `muse-glimmer-30b` | 200k |
| `muse-spark-1.2` | 200k |
| `nemotron-3-super` | 1M |
| `nemotron-3-ultra` | 1M |
| `nemotron-3.5-lightning` | 1M |
| `qwen3.8-27b` | 262k |
| `qwen3.8-max` | 200k |
| `step-3.7-flash` | 200k |

## Endpoints

- `POST /v1/chat/completions` — chat completions. `stream: true` for SSE. Supports `tools`, `tool_choice`, and `reasoning_effort`.
- `GET  /v1/models` — list models.
- `GET  /v1/models/{id}` — retrieve one model.
- `GET  /health` — liveness plus live credential-pool stats.
- `POST /admin/reload` — reload `credentials.json` from disk without restarting.

`/chat/completions` and `/api/v1/chat/completions` are accepted as aliases of `/v1/chat/completions`.
