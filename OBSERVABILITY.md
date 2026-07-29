# Observability Setup for AI Accelerator

This repo exports three observability signals through OpenTelemetry:

- Traces to Grafana Cloud Tempo
- Metrics to Grafana Cloud Prometheus
- Logs to Grafana Cloud Loki

The current code no longer depends on Langfuse.

The dashboard JSON in this repo, `ai-accelerator-dashboard.json`, expects those
Grafana Cloud data sources and is built around the real metric names emitted by
the app.

## What The App Emits

The tracing layer in `backend/core/tracing.py` creates these metric series:

- `request_calls_total`
- `request_duration_ms`
- `tool_calls_total`
- `tool_call_duration_ms`
- `llm_tokens_total`

It also instruments LangChain so LLM calls can surface in Grafana with the
`gen_ai_*` series used by the dashboard's LLM panels.

## Grafana Cloud Setup

Grafana Cloud uses the term "stack" for the hosted observability workspace.
If your team says "project", they usually mean the same place you will import
the dashboard and configure data sources.

### 1. Create a Grafana Cloud account

1. Sign up at the Grafana Cloud signup page.
2. After signup, Grafana Cloud creates or offers a stack for you.

If you are already on a team stack, you can skip to the next step.

### 2. Create or choose a stack

1. Open the Grafana Cloud portal.
2. In the left menu, select **Add Stack**.
3. Enter a unique stack name.
4. Choose the region for the stack.
5. Wait for the stack to finish provisioning.

You only need one stack for this repo unless your team separates
development/staging/production into different stacks.

### 3. Add the OpenTelemetry connection

1. Open your stack.
2. Go to **Connections**.
3. Select **Add new connection**.
4. Choose **OpenTelemetry**.
5. Follow the connection instructions until Grafana Cloud shows the OTLP
   connection details.
6. Copy the values Grafana Cloud shows you:
   - OTLP endpoint URL
   - Instance ID
   - Authorization header or API token details

If your stack uses an access policy flow instead of showing a ready-made
header, that is fine. In that case, the token you generate from the access
policy is what you place into `OTEL_EXPORTER_OTLP_HEADERS`.

### 4. Generate the token

If you do not already have a token, create it from the stack access policy:

1. Open your Grafana Cloud stack.
2. Go to **Administration**.
3. Select **Users and access**.
4. Open **Cloud access policies**.
5. Click **Create access policy**.
6. Enter a display name for the policy.
7. Set the scopes needed for this app:
   - `traces:write`
   - `logs:write`
   - `metrics:write`
8. Create the policy.
9. Open the policy you just created.
10. Click **Add token**.
11. Enter a token display name.
12. Select **Create**.
13. Copy the token right away.

Grafana Cloud only shows the token once, so save it somewhere secure before
closing the dialog.

For this repo, the application sends data directly to Grafana Cloud, so you do
not need a local collector.

### 5. Make sure the data sources exist

The dashboard uses three Grafana Cloud data sources:

- Prometheus for metrics
- Tempo for traces
- Loki for logs

If they are already provisioned in your stack, keep them.
If not, add them from **Connections** in the stack.

## Import The Dashboard

The dashboard file lives at:

`ai-accelerator-dashboard.json`

To import it into Grafana Cloud:

1. Open your Grafana Cloud stack.
2. Select **Dashboards** from the left menu.
3. Click **New**.
4. Choose **Import dashboard**.
5. Upload `ai-accelerator-dashboard.json`.
6. On the import screen, map the dashboard data sources:
   - Prometheus panels to your stack's Prometheus source
   - Tempo panels to your stack's Tempo source
   - Loki panels to your stack's Loki source
7. Review the dashboard name and folder.
8. Click **Import**.

After import, the dashboard should show panels for:

- Root requests and request latency
- Tool calls and tool latency
- Error rates for requests and tools
- LLM token usage
- LLM call latency
- Recent traces
- Logs
- Session explorer panels

## Local Setup

### 1. Install dependencies

If you previously installed Langfuse, remove it first:

```bash
pip uninstall langfuse
```

or:

```bash
uv remove langfuse
```

Install the OpenTelemetry packages:

```bash
pip install opentelemetry-api opentelemetry-sdk \
            opentelemetry-exporter-otlp-proto-http \
            opentelemetry-instrumentation-langchain
```

or:

```bash
uv add opentelemetry-api opentelemetry-sdk \
       opentelemetry-exporter-otlp-proto-http \
       opentelemetry-instrumentation-langchain
```

If you are on the current checkout, these dependencies are already listed in
`requirements.txt`.

### 2. Add environment variables

Add these values to your `.env`:

```dotenv
OTEL_EXPORTER_OTLP_ENDPOINT=https://otlp-gateway-<region>.grafana.net/otlp
OTEL_EXPORTER_OTLP_HEADERS="Authorization=Basic%20<token-from-your-access-policy>="
OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
OTEL_SERVICE_NAME=ai-accelerator
OTEL_RESOURCE_ATTRIBUTES=service.instance.id=<optional-instance-id>
```

Notes:

- Keep `%20` in `OTEL_EXPORTER_OTLP_HEADERS`.
- Use the token generated from your Grafana Cloud access policy.
- Do not trim the trailing `=` from the token if Grafana Cloud includes it.
- `OTEL_SERVICE_NAME` is the filter label you will use in Tempo and on the
  trace panels.
- `service.instance.id` is optional. Add it only if you want to distinguish
  multiple app instances.
- If tracing is not configured, the app still runs normally and simply skips
  export.


### 3. Run the app

Start the app the same way you normally do.

The observability code reads the same environment loader as the rest of the
application, so if your API keys already load correctly, the `OTEL_*`
variables will load too.

### 4. Start the runtime stack

```bash
docker compose up -d
```

This starts the runtime services the app needs.

No separate observability container is required because traces, metrics, and
logs go directly to Grafana Cloud over HTTPS.

## Verify It Works

1. Send one request, such as a chat message or an ingest.
2. Open your Grafana Cloud stack.
3. Go to **Explore**.
4. Select the Tempo data source.
5. Search for `resource.service.name = "ai-accelerator"` or whatever value you
   set for `OTEL_SERVICE_NAME`.
6. Open a trace named `agent_chat`, `run_query`, or `ingest_document`.
7. Check the dashboard panels for request counts, request latency, tool
   latency, token usage, and logs.

If the traces are slow to appear, wait a few seconds. The exporter batches
data before sending it.

## Where The Observability Code Lives

| File | What it does |
|---|---|
| `backend/core/tracing.py` | Sets up OpenTelemetry tracing, metrics, and logs. It defines the request, tool, and token metrics and adds trace metadata like `service.name` and `session_id`. |
| `backend/core/llm_client.py` | Builds the LLM clients that LangChain instrumentation can observe automatically. |
| `backend/core/vision_client.py` | Wraps each vision provider call in a traced span. |
| `backend/agent/executor.py` | Opens the root span for a chat turn and child spans for dispatched tools. |
| `backend/pipeline/graph.py` | Opens child spans for pipeline steps and records handled errors. |
| `backend/pipeline/ingest.py` and `backend/pipeline/query.py` | Open root spans for ingestion and query runs and return trace IDs to the caller. |
| `backend/core/usage.py` | Sends token usage into `record_llm_usage`, which feeds the token metric and current span attributes. |

### If You Add A New Tool Or Pipeline Step

You usually do not need to touch `backend/core/tracing.py`.

Use the existing tracing helpers:

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

That span will nest under the active root request automatically.

## Troubleshooting

- No traces showing up: confirm `OTEL_EXPORTER_OTLP_ENDPOINT` and
  `OTEL_EXPORTER_OTLP_HEADERS` are present in the running process.
- 401 or 403 from the exporter: the header is wrong or the token scope does
  not include `traces:write`.
- A tool is missing from a trace: the agent may not have called it for that
  request.
- `opentelemetry.instrumentation.langchain` is missing: install
  `opentelemetry-instrumentation-langchain` to restore the extra LLM spans.
