# FC Integration Testing

English | [中文](fc-testing.zh.md)

> **Read when**: you run or write the opt-in `@pytest.mark.fc` integration tests against a
> deployed Function Compute (FC) service — or you hit "FC spun up many instances",
> `403 AccessDenied`, or cold-start timeouts while doing so.
> **Source**: `services/<svc>-server/tests/test_fc*.py`, `framework/src/bioq_service/fc_testing.py`,
> `services/<svc>-server/deploy/fc.yaml`, the live FC console config.
> **Refresh/remove when**: the FC console session-affinity / trigger config changes, or
> `fc_url()` gains an env override for the base URL.

FC tests hit a live deployment and are never part of an offline run. They are marked
`@pytest.mark.fc` and skipped unless `-m fc` / `RUN_FC_TESTS=1` is set (see
[testing.md](testing.md) for the offline layers).

```bash
cd services/<svc>-server
RUN_FC_TESTS=1 uv run --group dev python -m pytest -m fc tests/ -v
```

## URL resolution and the VPC vs. public split

`base_url` fixtures resolve the deployed URL via `fc_url(service_name)` from
`bioq_service.fc_testing`, which reads `services.yaml` (`services.<name>.url`). That entry is
the **VPC HTTP trigger** (`https://fc-...-vpc.fcapp.run`) — it is what the gateway and
everything inside the VPC uses.

The public pendant is the same host without `-vpc` (`https://fc-...cn-hangzhou.fcapp.run`).
Its reachability is controlled by the HTTP trigger's `disableURLInternet` flag (`deploy/fc.yaml`
sets it `true` by default):

| From                 | `-vpc.fcapp.run`                                    | public, `disableURLInternet=true`                                                            | public, `disableURLInternet=false` |
|----------------------|-----------------------------------------------------|---------------------------------------------------------------------------------------------|------------------------------------|
| Inside Aliyun VPC    | works                                               | —                                                                                           | —                                  |
| External machine     | `ConnectTimeout` (DNS → internal `100.x`, no route) | `403 AccessDenied: "access denied due to function internet URL is disabled"`                | works (subject to cold start)      |

**To test from an external machine**: enable the public URL (console, or `disableURLInternet:
false` in `deploy/fc.yaml`), then point the tests at it. `fc_url()` has **no env override**, so
the practical move is to temporarily rewrite the one `services.yaml` line to the public URL,
run the tests, and revert:

```bash
git checkout -- services.yaml   # afterwards; restore the VPC URL (gateway still needs it)
```

## Session affinity — the "many instances" symptom

FC is configured with **HeaderField session affinity** (`deploy/fc.yaml`):

```yaml
sessionAffinity: HEADER_FIELD
affinityHeaderFieldName: bioagent-session-id
sessionConcurrencyPerInstance: 1
```

Every request from a test module must carry the **same** `bioagent-session-id` value so the
submit and all its polls bind to one instance. Without it, FC treats each call as a fresh
session and fans out to many (often brand-new) instances — the "I opened the console and there
are a dozen instances" failure mode.

The framework supports this end-to-end:

- `_SessionAffinityMiddleware` (`framework/src/bioq_service/app.py`) echoes the server-assigned
  session value (`job_id`) into the `bioagent-session-id` response header on any `200` POST that
  returns a `job_id`. It is enabled when `settings.session_header_name` is set
  (env `<PREFIX>_SESSION_HEADER_NAME`, e.g. `PROTEINMPNN_SESSION_HEADER_NAME=bioagent-session-id`).
- The test must send a consistent value on **every** request (submit **and** poll). The repo
  pattern (see alphafold / deeprank-ab / genie3 / … `tests/test_fc.py`) is a module-scoped
  fixture + a client-level header:

```python
SESSION_HEADER = "bioagent-session-id"

@pytest.fixture(scope="module")
def session_headers() -> dict[str, str]:
    return {SESSION_HEADER: f"test-{uuid.uuid4().hex[:12]}"}

@pytest.fixture(scope="module")
def client(base_url: str, session_headers: dict[str, str]) -> httpx.Client:
    # Client-level headers merge into every request, including poll_job's
    # `client.get(..., headers={})`, so submit + poll + smoke + download all carry it.
    with httpx.Client(base_url=base_url, timeout=..., headers=session_headers) as c:
        yield c
```

Alternative: pass `headers=session_headers` on each `client.post(...)` and
`extra_headers=session_headers` to every `poll_job(...)` call. Either is fine; what matters is
consistency across requests. The header name is a historical contract and must stay
`bioagent-session-id` (see [conventions.md](conventions.md)).

## Cold start and 429 retry

GPU functions scale to zero and cold-start in ~12–40 s on the first request. Use generous
timeouts (e.g. `httpx.Timeout(connect=30, read=300, write=600, pool=30)`) so the first
submit/poll doesn't spuriously die.

Account-level GPU quota exhaustion surfaces as `429` on **any** request (including cheap
`/healthz`). `bioq_service.fc_testing` ships two helpers for this:

- `make_retrying_client(base_url, *, timeout=120.0, max_retries=10, backoff_s=20.0)` — an
  `httpx.Client` whose transport retries `429` with linear backoff.
- `poll_job(..., max_transient_errors=..., interval_s=...)` — bump `max_transient_errors` on
  `max_concurrent_jobs=1` services (e.g. `60 × 20s` ≈ 20 min buffer) because FC's quota window
  outlasts the default `10 × 15s`.

## Local prerequisites (why `uv run --group dev` may fail)

- `[dependency-groups] dev = ["pytest>=8.0", "pytest-asyncio>=0.23"]` must exist, otherwise
  `uv run --group dev` fails with "Group `dev` is not defined". Legacy services may still carry a
  stale `[build-system]`+`[tool.setuptools]` block instead of `[tool.uv] package = false` — align
  them with the current convention.
- `bioq_service` must resolve: `[tool.uv.sources] bioq-service-framework = { path = "../../framework", editable = true }`.
- On read-only `~/.cache/uv` sandboxes, set `UV_CACHE_DIR=/tmp/...` (or `UV_NO_CACHE=1`) or uv
  fails before it even resolves.

## Offline fixture gotcha (`helper_scripts` 500)

The submit endpoints build their argv synchronously inside `runner.submit → build_argv`, and
`prepare_inputs()` runs the upstream `helper_scripts/*.py` in that same request. So an offline
`test_app.py` `client` fixture that points `PROTEINMPNN_ROOT` at an empty tmp dir gets `500
"Helper script not found"` instead of `200`. Stub the helper scripts (`parse_multiple_chains.py`
etc.) in the fixture — see `services/proteinmpnn-server/tests/test_app.py`.

## Known gaps / checklist

- Don't trust a test that always `pytest.skip`s. `services/proteinmpnn-server/tests/test_fc.py::
  test_job_uri_cross_reference` skips because plain design emits FASTA, not PDB, so the
  `job://` PDB cross-reference is never exercised — flag such skips rather than assuming
  coverage.
- A passing run should look like: smoke `PASSED`, `422` cases `PASSED`, and at least one
  inference job per endpoint `PASSED` with real `duration_seconds > 0` and non-empty outputs.
- After the run, confirm `git status` is clean (the `services.yaml` override was reverted).