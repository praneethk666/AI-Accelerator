# Observability Setup (Grafana Cloud)

This repo traces every request — agent chat turns, ingestion runs, and query
runs — end to end: every pipeline step, every agent tool call, every LLM/vision
call, and every handled (caught-but-continued) error. Tracing is powered by
OpenTelemetry and exported to **Grafana Cloud Tempo**. It used to run on
Langfuse; that's been fully replaced.


---

## If you already have access to the team's Grafana Cloud stack

You don't need to create anything — just get these four values from whoever
set up the stack (or your team's secrets manager) and skip to
[Local setup](#local-setup):

```
OTEL_EXPORTER_OTLP_ENDPOINT
OTEL_EXPORTER_OTLP_HEADERS
OTEL_EXPORTER_OTLP_PROTOCOL
OTEL_SERVICE_NAME
```

## If you're setting up the stack for the first time

1. Sign up at https://grafana.com/auth/sign-up/create-user (free tier is
   enough). This auto-creates a stack (e.g. `yourteam.grafana.net`).
2. In the stack: **Connections → Add new connection → OpenTelemetry**.
3. That page gives you three things — copy them:
   - **OTLP Endpoint URL** (e.g. `https://otlp-gateway-prod-ap-south-1.grafana.net/otlp`)
   - **Instance ID**
   - A **Generate now** button for an API token — scope it to
     `traces:write` (and `logs:write`/`metrics:write` if you plan to send
     those later too).
4. The same page gives you a ready-made `Authorization: Basic ...` header
   string — copy that directly instead of base64-encoding it yourself.
5. Invite teammates to the stack from **Administration → Users and access**
   so they don't each need their own account/token.

---

## Local setup

### 1. Install dependencies

If you installed langfuse before, then 

```bash
pip uninstall langfuse
```

or

```bash
uv remove langfuse
```

```bash
pip install opentelemetry-api opentelemetry-sdk \
            opentelemetry-exporter-otlp-proto-http \
            opentelemetry-instrumentation-langchain
```
or

```bash
uv add opentelemetry-api opentelemetry-sdk \
            opentelemetry-exporter-otlp-proto-http \
            opentelemetry-instrumentation-langchain
```

(These are already in `requirements.txt` if you pulled latest — this is only
needed if you're on an older checkout or a fresh venv.)


### 2. Add to your `.env`

```dotenv
OTEL_EXPORTER_OTLP_ENDPOINT=https://otlp-gateway-<region>.grafana.net/otlp
OTEL_EXPORTER_OTLP_HEADERS="Authorization=Basic%20<your-base64-token>="
OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
OTEL_SERVICE_NAME=ai-accelerator
```

Notes on formatting, since these have tripped people up before:
- Keep `%20` as-is inside `OTEL_EXPORTER_OTLP_HEADERS` — OpenTelemetry decodes
  it back to a space itself. Don't manually replace it.
- The trailing `=` in the token is base64 padding — normal, don't trim it.
- `OTEL_SERVICE_NAME` is just a label you choose to filter by in Grafana —
  it's unrelated to whatever you named the API token/access policy in
  Grafana's UI.
- If tracing isn't configured (env vars unset), the app runs completely
  normally — tracing silently no-ops, same behavior the old Langfuse setup
  had.

### 3. Bring up the core stack

```bash
docker compose up -d
```

This starts **Postgres** and **Qdrant** only — that's the entire runtime
dependency now. No observability container is needed; traces go straight to
Grafana Cloud over HTTPS.

Optional dev add-on (DB browser, unrelated to tracing):
```bash
docker compose -f docker-compose.yml -f docker-compose.devtools.yml up -d
```

### 4. Run the app as usual

Whatever your normal start command is (`uvicorn ...`, `python -m ...`, etc.)
— nothing about how you run it changes. Just make sure the process actually
loads `.env` (same as it already does for `DEEPSEEK_API_KEY` / `NVIDIA_API_KEY`
— if those work today, the `OTEL_*` vars will too, since it's the same
`os.getenv()` mechanism).

---

## Verifying it's working

1. Send one request — a chat message through the agent, or run an ingest.
2. Go to your Grafana Cloud stack → **Explore** (left nav).
3. Switch the data source dropdown to your **Tempo** source (named something
   like `grafanacloud-<yourstack>-traces`).
4. **Search** tab → filter by `resource.service.name = "ai-accelerator"` (or
   whatever you set `OTEL_SERVICE_NAME` to) → **Run query**.
5. Click into a trace named `agent_chat`, `run_query`, or `ingest_document`.
   You should see the full call tree nested underneath: `execute_task agent`
   → `ChatOpenAI.chat` for LLM calls, `step:categorize` / `step:extract` /
   etc. for pipeline steps, `tool:search_documents` / `tool:list_documents`
   for agent tool dispatches, and `vision:<model>` for vision calls.
6. To confirm error visibility: trigger a deliberate failure (bad file path,
   a tool that errors) and check that span shows a red error marker with
   `error.type` / `error.message` attributes.

Traces can take 5–20 seconds to appear (the exporter batches on a timer) —
give it a moment before assuming something's broken.

---

## Where the tracing code lives

| File | What it does |
|---|---|
| `backend/core/tracing.py` | The core abstraction — `traced_request()` (one root span per request), `traced_tool()` (one child span per step/tool), `record_handled_error()` (marks a caught-and-continued exception as an error on the current span). Also turns on `LangchainInstrumentor` so every LangChain LLM call auto-gets its own span. |
| `backend/core/llm_client.py` | Builds LLM clients. LLM calls are captured automatically via the LangChain instrumentation above — no manual span code needed per call site. |
| `backend/core/vision_client.py` | `_trace()` wraps each vision provider call (`openai`/`google`/`ollama`) in a `traced_tool` span. |
| `backend/agent/executor.py` | Opens the root span for one agent chat turn (`traced_request("agent_chat", ...)`) and a child span per dispatched tool (`traced_tool(f"tool:{name}", ...)`). |
| `backend/pipeline/graph.py` | Opens a child span per pipeline step (`traced_tool(f"step:{tool.name}", ...)`) and calls `record_handled_error` when a step's exception is swallowed instead of raised. |
| `backend/pipeline/ingest.py`, `query.py` | Open the root span for a full ingestion run / query turn respectively, and return `trace_info["trace_id"]` to the caller (API/UI can use this to link "view this request's trace" back to Grafana). |

### If you're adding a new tool or pipeline step

You don't need to touch `tracing.py`. Just:
```python
from backend.core.tracing import traced_tool, record_handled_error

with traced_tool("tool:my_new_tool", input=args) as span:
    try:
        result = do_the_thing(args)
    except Exception as exc:
        result = {"error": str(exc)}
        record_handled_error("my_new_tool_failure", str(exc))
    span["output"] = result
```
That's the whole pattern — it'll automatically nest under whatever root span
(`agent_chat` / `ingest_document` / `run_query`) is active when it runs.

---

## Troubleshooting

- **No traces showing up at all** — print `os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")`
  and `os.getenv("OTEL_EXPORTER_OTLP_HEADERS")` inside the running process to
  confirm `.env` actually reached it (not just your shell).
- **401/403 in app logs from the OTEL exporter** — the `Authorization` header
  got mangled (extra/missing quotes, `%20` decoded twice, wrong token) or the
  token's scope doesn't include `traces:write`. Regenerate the token from
  Grafana Cloud's OpenTelemetry connection page and paste it fresh.
  the app logs.
- **Traces show up but a tool call you expected is missing** — that's usually
  not a tracing bug: it means the agent genuinely didn't call that tool for
  that message (e.g. a greeting, or the model chose not to search). Check the
  trace's `execute_task agent` → what comes right after it; if there's no
  `execute_task tools` span, no tool was dispatched that turn.
- **`ModuleNotFoundError` for `opentelemetry.instrumentation.langchain`** — it's
  an optional dependency; the app still runs fine without it, you just lose
  the per-LLM-call span level (calls still show up nested under their parent
  tool/step span). `pip install opentelemetry-instrumentation-langchain` to
  get it back.

---