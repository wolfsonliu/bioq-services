# Testing — test skeleton

English | [中文](testing.zh.md)

> ← Back to the [Adding a service cookbook overview](./index.md)

This page covers the four test skeletons: `test_app.py` (offline HTTP unit tests) / `test_cli.py`
(CLI batch) / `test_fc.py` (post-deploy FC regression) / `test_fc_task.py` (FC async task mode
integration). Services called through the gateway should also add a `TestEndToEnd<Svc>` e2e class in
`gateway/tests/test_fc.py` (see the OSS mount config in [deploy.md](./deploy.md)).

### 10. `services/<svc>/tests/test_app.py`

```python
"""Offline tests for <svc>-server (no real algorithm / GPU needed).

`conftest.py` registers the service dir as `server` package, so
`from server.settings import ...` works without pip install.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic_settings import SettingsConfigDict


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("<SVC>_JOBS_BASE_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("<SVC>_ROOT", str(tmp_path / "<svc>"))
    (tmp_path / "<svc>").mkdir(parents=True, exist_ok=True)

    sys.modules.pop("server.app", None)
    server_app = importlib.import_module("server.app")
    return TestClient(server_app.app)


# ----- Healthcheck / manifest -----

def test_health(client):
    health = client.get("/healthz").json()
    assert health["status"] == "ok"
    assert health["service"] == "<svc>"
    assert "version" in health


def test_manifest_service_name(client):
    body = client.get("/api/manifest").json()
    assert body["service"] == "<svc>"


def test_manifest_lists_endpoints(client):
    body = client.get("/api/manifest").json()
    paths = {e["path"] for e in body["endpoints"]}
    assert "/api/generate" in paths


def test_manifest_extras_has_tool_outputs(client):
    extras = client.get("/api/manifest").json()["service_specific"]
    assert "generate" in extras["tool_outputs"]


# ----- Settings -----

def test_settings_defaults():
    from server.settings import <Svc>Settings

    class _Off(<Svc>Settings):
        model_config = SettingsConfigDict(env_prefix="<SVC>_TEST_", env_file=None, extra="ignore")

    s = _Off()
    assert s.jobs_base_dir == Path("/data/<svc>_jobs")
    assert s.root == Path("/opt/<svc>")
    # weight externalization: default must point at the NAS mount point
    assert s.weights_dir == Path("/data/models/<svc>")


# ----- Endpoint smoke (no real pipeline) -----

def test_endpoint_returns_job(client):
    resp = client.post("/api/generate", data={"n_samples": "2"})
    assert resp.status_code == 200
    body = resp.json()
    assert "job_id" in body
    assert body["input_params"] is not None
```

> **Test points**: (1) the `client` fixture uses `monkeypatch.setenv` + `importlib.import_module` to
> re-import the app, ensuring settings use the test directories. (2) Each endpoint has at least one
> smoke test verifying the submit returns a `job_id`. (3) Manifest tests cover the service name, the
> endpoint list, and extras. (4) Settings tests verify defaults and env override.

### 11. `services/<svc>/tests/test_cli.py`

Offline tests for the CLI batch mode. Three layers: (1) endpoint registration correctness; (2)
`build_argv` callbacks producing the right command line; (3) end-to-end `create_cli` flow (mocking
SubprocessRunner).

```python
"""CLI batch-mode tests for <svc>-server.

Tests endpoint registration, build_argv callbacks, and end-to-end create_cli.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic_settings import SettingsConfigDict

from bioq_service.cli import CLIEndpoint, create_cli

from server.adapter import <Svc>Adapter
from server.models import GenerateRequest
from server.settings import <Svc>Settings
from server.tools import generate_argv


class _Off(<Svc>Settings):
    """Settings subclass that ignores .env and real env vars."""
    model_config = SettingsConfigDict(
        env_prefix="<SVC>_TEST_", env_file=None, extra="ignore",
    )


def _generate_build(req, inputs, job_dir, settings):
    return generate_argv(
        req, job_dir=job_dir, input_pdb=inputs["input_pdb"], settings=settings,
    )


ENDPOINTS = {
    "generate": CLIEndpoint(
        name="generate",
        help="Run generation on an input PDB",
        request_model=GenerateRequest,
        build_argv=_generate_build,
        inputs={"input_pdb": ("Input PDB file", True)},
    ),
}


# ---- Endpoint registration ----


def test_endpoint_keys():
    assert set(ENDPOINTS.keys()) == {"generate"}


def test_generate_endpoint_fields():
    ep = ENDPOINTS["generate"]
    assert ep.request_model is GenerateRequest
    assert ep.inputs["input_pdb"][1] is True  # required


# ---- Build_argv callbacks ----


def test_generate_build_argv(tmp_path):
    s = _Off()
    job_dir = tmp_path / "j"
    job_dir.mkdir()
    pdb = tmp_path / "input.pdb"
    pdb.write_text("ATOM")

    argv = _generate_build(
        GenerateRequest(n_samples=3),
        {"input_pdb": pdb},
        job_dir,
        s,
    )
    # Assert argv structure matches what tools.py generates.
    # Concrete assertions depend on generate_argv implementation.
    assert len(argv) > 0


# ---- End-to-end create_cli ----


def test_cli_generate_success(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs")
    adapter = <Svc>Adapter(settings=s)

    pdb = tmp_path / "input.pdb"
    pdb.write_text("ATOM")
    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "generate",
        "--input-pdb", str(pdb),
        "--output-dir", str(output_dir),
    ]):
        with patch("bioq_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 0
            with patch.object(adapter, "detect_outputs", return_value=True):
                with pytest.raises(SystemExit) as exc_info:
                    create_cli(adapter, s, ENDPOINTS, version="0.0.1")
                assert exc_info.value.code == 0


def test_cli_generate_json_output(tmp_path, capsys):
    s = _Off(jobs_base_dir=tmp_path / "jobs")
    adapter = <Svc>Adapter(settings=s)

    pdb = tmp_path / "input.pdb"
    pdb.write_text("ATOM")
    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "generate",
        "--input-pdb", str(pdb),
        "--json", "--output-dir", str(output_dir),
    ]):
        with patch("bioq_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 0
            with patch.object(adapter, "detect_outputs", return_value=True):
                with pytest.raises(SystemExit) as exc_info:
                    create_cli(adapter, s, ENDPOINTS, version="0.0.1")
                assert exc_info.value.code == 0

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "completed"
    assert result["return_code"] == 0


def test_cli_no_subcommand_exits_2(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs")
    adapter = <Svc>Adapter(settings=s)

    with patch.object(sys, "argv", ["prog"]):
        with pytest.raises(SystemExit, match="2"):
            create_cli(adapter, s, ENDPOINTS)
```

> **Test points**:
> - **`_Off` settings** — use a separate `env_prefix` + `env_file=None` to isolate env vars, so the
>   dev machine's `.env` or real env vars don't interfere with the tests
> - **Endpoint registration** — verify the `ENDPOINTS` dict's key set and each endpoint's
>   `request_model` and `inputs` declarations
> - **Build_argv** — call the `_xxx_build()` callback directly and assert the generated argv contains
>   the expected script path and key args; use fixture files from `tests/data/` for the
>   "with data file" variant
> - **End-to-end create_cli** — mock `SubprocessRunner` and `adapter.detect_outputs`, inject CLI args
>   via `patch.object(sys, "argv", [...])`, assert `SystemExit.code`. Cover all four scenarios:
>   success / json output / failure / no-subcommand
> - **Complex params** — for fields like `dict` / `list[Model]` that can't be expressed as an argparse
>   flag, pass `--params-json` in the end-to-end tests (see rfdiffusion2-server's `contig_atoms`,
>   boltz-server's `sequences`)

### 12. `services/<svc>/tests/test_fc.py` + `tests/data/`

Post-deploy FC URL regression tests. Marker = `fc`, skipped by default; enable explicitly with
`pytest -m fc` or `RUN_FC_TESTS=1`. The URL is read from [services.yaml](../../services.yaml) — write
the new entry into that file once deployment is done.

**Test data placement principle (default self-contained)**: any fixture (PDB / JSON / FASTA /
prm7+rst7 / etc.) needed by any test script (`test_fc.py` / `test_fc_task.py` / `test_cli.py` / …)
**must, as much as possible**, be copied into `tests/data/` under that test's directory and committed
with it — so the suite is self-contained and runs on a fresh clone. **Never** reference
`opensource/*` (gitignored, missing on a fresh clone) or other services' directories. Locate fixtures
in tests via `DATA_DIR = Path(__file__).resolve().parent / "data"`, not a path relative to
`opensource`.

**The only exception — files too large for git**: only skip committing when a single fixture reaches
multiple megabytes (the repo's largest tracked fixture is currently ~700 KB and there is no git-LFS).
In that case:

- keep `tests/data/` as the fixture's **default lookup location**, but add that directory (or specific
  files) to `.gitignore`;
- the test overrides the default path via an env var (e.g. `<SVC>_TEST_STRUCTURE` / `_PARAMETERS`) and
  uses `pytest.mark.skipif` to skip rather than error when the fixture is missing;
- document in the service README + test docstring where to fetch it from / how to stage it into
  `tests/data/`.

> Criterion: **if it can fit in git, commit it** (a single MD/structure fixture of a few hundred KB to
> a few MB is usually acceptable, especially when there is no smaller legitimate alternative — e.g.
> OpenBPMD's ~10 MB solvated system is still committed to stay self-contained). Only take the
> exception path when the file clearly slows down clones (tens of MB+) or is inherently binary
> weights/datasets. When in doubt, prefer self-contained.

```python
"""End-to-end tests against the deployed <Svc> Function Compute service.

Marked `@pytest.mark.fc`, skipped by default. Run with:

    pytest -m fc services/<svc>-server/tests/test_fc.py

Test fixtures ship in `tests/data/`, so the suite is self-contained — no
dependency on `opensource/`.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from bioq_service.fc_testing import fc_url, poll_job

DATA_DIR = Path(__file__).resolve().parent / "data"
TEST_PDB = DATA_DIR / "<example>.pdb"

pytestmark = pytest.mark.fc


@pytest.fixture(scope="module")
def base_url() -> str:
    return fc_url("<svc>-server", start=Path(__file__))


@pytest.fixture(scope="module")
def client(base_url: str) -> httpx.Client:
    with httpx.Client(base_url=base_url, timeout=httpx.Timeout(120.0)) as c:
        yield c


# ----- Smoke -----

def test_healthz(client: httpx.Client) -> None:
    r = client.get("/healthz")
    r.raise_for_status()
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "<svc>"
    assert "version" in body


def test_manifest_lists_endpoints(client: httpx.Client) -> None:
    r = client.get("/api/manifest")
    r.raise_for_status()
    paths = {e["path"] for e in r.json()["endpoints"]}
    assert paths == {"/api/<endpoint>", ...}  # each service lists its own


def test_openapi_served(client: httpx.Client) -> None:
    client.get("/openapi.json").raise_for_status()


def test_unknown_job_returns_404(client: httpx.Client) -> None:
    assert client.get("/api/jobs/missing-job-id").status_code == 404


# ----- Inference: at least 1 minimal job per endpoint -----

def test_<endpoint>_minimal_job(client: httpx.Client, base_url: str) -> None:
    with open(TEST_PDB, "rb") as fh:
        r = client.post(
            "/api/<endpoint>",
            files={"pdb": (TEST_PDB.name, fh, "chemical/x-pdb")},
            data={"name": "fc_smoke", "<smallest-param>": "..."},
        )
    r.raise_for_status()
    final = poll_job(client, base_url, r.json()["job_id"])
    assert final["status"] == "completed", final
    files = client.get(f"/api/jobs/{final['job_id']}/files").json()["files"]
    assert files
```

`conftest.py` must do two things: (1) register `services/<svc>/` as the `server` package — so
`from server.app import app` works in tests (the Dockerfile COPYs it to `/opt/.../server/`, but local
dev has no such directory remapping); (2) register the `fc` marker.

```python
"""Register `server` package alias + `fc` marker."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parent.parent

if "server" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "server",
        SERVICE_DIR / "__init__.py",
        submodule_search_locations=[str(SERVICE_DIR)],
    )
    if spec is not None and spec.loader is not None:
        module = importlib.util.module_from_spec(spec)
        sys.modules["server"] = module
        spec.loader.exec_module(module)

from bioq_service.fc_testing import (  # noqa: E402
    register_fc_marker,
    skip_fc_tests_unless_enabled,
)


def pytest_configure(config):
    register_fc_marker(config)


def pytest_collection_modifyitems(config, items):
    skip_fc_tests_unless_enabled(config, items)
```

> **Why `importlib.util`?** Offline tests run directly via `uv run pytest services/<svc>/tests/` —
> the `server` package isn't on the Python search path. This code registers `services/<svc>/` as the
> `server` module, after which `from server.settings import ...` resolves normally. Most existing
> services already use this pattern.

For general offline / FC test conventions see [testing.md](../topics/testing.md).

### 12b. `services/<svc>/tests/test_fc_task.py`

FC **async task mode** integration tests. Goal: verify the `/api/tasks/<name>` endpoint end-to-end
under `X-Fc-Invocation-Type: Async` — submit returns 202 immediately, the job eventually completes,
the lifecycle endpoints are readable, and platform-layer dedup works. Also `@pytest.mark.fc` (reuses
Section 12's `conftest.py`; no extra registration needed); skipped by default, run with:

```bash
RUN_FC_TESTS=1 \
uv run python -m pytest -m fc services/<svc>-server/tests/test_fc_task.py -v
```

Why a separate file instead of folding into `test_fc.py`?
- The two calling modes (sync submit/poll vs. async task) differ enough in headers / assertions /
  lifecycle semantics that squeezing them together makes fixtures collide; separate files are clearer.
- Async task mode is the **recommended** entry for long-running GPU services (no HTTP-gateway 30s
  recycle + platform-layer dedup + queueing instead of 429). Sync is kept for legacy compatibility —
  it's the first thing to switch when the pipeline migrates to K8s or another platform. Get async
  working first; sync is the fallback.
- In task mode the FC event payload has a **128 KiB cap** (`EntityTooLarge` 400 otherwise); large
  PDBs / large zips must use URI fallback, which needs its own fixture organization.

#### Skeleton

```python
"""FC async task mode tests for <svc>-server (opt-in).

Run with::

    RUN_FC_TESTS=1 \\
    uv run python -m pytest -m fc services/<svc>-server/tests/test_fc_task.py -v

Validates the /api/tasks/<name> endpoints end-to-end under FC async task mode
(``X-Fc-Invocation-Type: Async``).  Async task mode pins the FC instance
for the whole job (no 30 s HTTP-gateway recycle risk) and dedups by
``X-Fc-Async-Task-Id`` at the platform layer.
"""

from __future__ import annotations

import io
import time
import uuid
import zipfile
from pathlib import Path

import httpx
import pytest

from bioq_service.fc_testing import fc_url, poll_job

SERVICE = "<svc>-server"

# how long each endpoint needs; covers the service's slowest inference.
POLL_TIMEOUT_S = 1800
POLL_INTERVAL_S = 20
TIMEOUT = httpx.Timeout(connect=30, read=300, write=600, pool=30)


@pytest.fixture(scope="module")
def base_url() -> str:
    return fc_url(SERVICE, start=Path(__file__))


@pytest.fixture(scope="module")
def client(base_url: str):
    with httpx.Client(base_url=base_url, timeout=TIMEOUT) as c:
        yield c


# one task_id per endpoint, shared by all assertions in the same module.
@pytest.fixture(scope="module")
def <endpoint>_task_id() -> str:
    return f"fc-async-<endpoint>-{int(time.time())}-{uuid.uuid4().hex[:6]}"


# ---- key headers ----
def _async_headers(task_id: str) -> dict[str, str]:
    return {
        "X-Fc-Invocation-Type": "Async",
        "X-Bioagent-Job-Id": task_id,
        "X-Fc-Async-Task-Id": task_id,
    }


# ---- 429-jitter fallback GET (sync test_fc.py should have one too) ----
def _get_with_retry(client, path, *, max_attempts=20, backoff_s=30):
    """Poll and retry on FC gateway 429. Especially needed for max_concurrent_jobs=1 services."""
    last = None
    for _ in range(max_attempts):
        last = client.get(path)
        if last.status_code != 429:
            return last
        time.sleep(backoff_s)
    return last


def _poll_to_completion(client, task_id: str) -> dict:
    # framework default max_transient_errors=10 × interval=15s = 150s, not enough
    # to ride out FC's 4-7 min 429 window. Here 60 × 20s = 20 min buffer.
    final = poll_job(
        client, "", task_id,
        timeout_s=POLL_TIMEOUT_S, interval_s=POLL_INTERVAL_S,
        max_transient_errors=60,
    )
    assert final["status"] == "completed", f"task did not complete: {final}"
    return final


# ---- one submit_response / task fixture pair per endpoint ----
@pytest.fixture(scope="module")
def <endpoint>_submit_response(client, <endpoint>_task_id):
    return client.post(
        "/api/tasks/<endpoint>",
        data={"<param>": "<smallest-valid>"},
        # if the upload exceeds 128 KiB, switch to URI fallback (see "Large files" below)
        headers=_async_headers(<endpoint>_task_id),
    )


@pytest.fixture(scope="module")
def <endpoint>_task(client, <endpoint>_task_id, <endpoint>_submit_response) -> dict:
    assert <endpoint>_submit_response.status_code == 202, (
        f"async <endpoint> submit returned {<endpoint>_submit_response.status_code}: "
        f"{<endpoint>_submit_response.text!r}"
    )
    return _poll_to_completion(client, <endpoint>_task_id)


# ===================================================================
# Section 1: submit semantics + OpenAPI
# ===================================================================


@pytest.mark.fc
class TestAsyncSubmit:
    def test_<endpoint>_returns_202(self, <endpoint>_submit_response):
        assert <endpoint>_submit_response.status_code == 202

    def test_task_endpoints_registered_in_openapi(self, client):
        r = _get_with_retry(client, "/openapi.json")
        assert r.status_code == 200
        expected = {"/api/tasks/<endpoint>", ...}
        missing = expected - set(r.json()["paths"])
        assert not missing, (
            f"task endpoints missing: {missing}; "
            f"settings.task_endpoints_enabled may be False"
        )


# ===================================================================
# Section 2: completion + output per endpoint
# ===================================================================


def _assert_completed_with_output(task, task_id, client, expected_name,
                                  *, min_duration_s=3.0):
    """job completed + has the expected_name artifact. min_duration_s is a fallback
    guard that the subprocess actually ran."""
    assert task["status"] == "completed"
    assert task["job_id"] == task_id
    d = task.get("duration_seconds")
    assert d is not None and d > min_duration_s, (
        f"duration {d}s too short (min {min_duration_s}s) — subprocess may not have actually run"
    )
    assert task.get("output_count", 0) > 0
    r = _get_with_retry(client, f"/api/jobs/{task_id}/files")
    assert r.status_code == 200
    assert any(expected_name in f for f in r.json()["files"])


@pytest.mark.fc
class TestAsync<Endpoint>:
    def test_completed(self, <endpoint>_task, <endpoint>_task_id, client):
        _assert_completed_with_output(
            <endpoint>_task, <endpoint>_task_id, client, "<expected-out>",
            min_duration_s=60,  # pass a smaller number for fast endpoints
        )

    def test_input_params_echoed(self, <endpoint>_task):
        params = <endpoint>_task.get("input_params") or {}
        assert params.get("<param>") == <expected-value>

    def test_output_downloadable(self, client, <endpoint>_task_id, <endpoint>_task):
        files = _get_with_retry(
            client, f"/api/jobs/{<endpoint>_task_id}/files"
        ).json()["files"]
        f = next(x for x in files if "<expected-out>" in x)
        r = _get_with_retry(client, f"/api/jobs/{<endpoint>_task_id}/file/{f}")
        assert r.status_code == 200
        assert len(r.content) > 100


# ===================================================================
# Section 3: lifecycle (use the cheapest endpoint fixture)
# ===================================================================


@pytest.mark.fc
class TestJobLifecycle:
    def test_status_endpoint(self, client, <endpoint>_task_id, <endpoint>_task):
        body = _get_with_retry(client, f"/api/jobs/{<endpoint>_task_id}").json()
        assert body["status"] == "completed"

    def test_log_endpoint(self, client, <endpoint>_task_id, <endpoint>_task):
        r = _get_with_retry(client, f"/api/jobs/{<endpoint>_task_id}/log")
        assert r.status_code == 200
        assert len((r.json().get("log") or r.json().get("text") or "")) > 0

    def test_download_zip(self, client, <endpoint>_task_id, <endpoint>_task):
        r = _get_with_retry(client, f"/api/jobs/{<endpoint>_task_id}/download")
        assert r.status_code == 200
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        assert any("<expected-out>" in n for n in zf.namelist())


# ===================================================================
# Section 4: platform-layer dedup — resubmit with the same task_id
# ===================================================================


@pytest.mark.fc
class TestAsyncDuplicateDedup:
    """Resubmit with the same X-Fc-Async-Task-Id — FC should return 409 (platform
    layer) or 202 (after forwarding, the framework's execute_task layer returns the
    existing JobInfo); either way, it must not re-run."""

    def test_duplicate_does_not_rerun(self, client, <endpoint>_task_id, <endpoint>_task):
        first_created = <endpoint>_task["created_at"]
        first_completed = <endpoint>_task["completed_at"]

        r2 = client.post(
            "/api/tasks/<endpoint>",
            data={"<param>": "<different-value>"},  # deliberately different to prove it takes no effect
            headers=_async_headers(<endpoint>_task_id),
        )
        assert r2.status_code in (202, 409), (
            f"expected 409 or 202; got {r2.status_code} {r2.text!r}"
        )
        if r2.status_code == 202:
            time.sleep(15)

        re_query = _get_with_retry(client, f"/api/jobs/{<endpoint>_task_id}").json()
        assert re_query["status"] == "completed"
        assert re_query["created_at"] == first_created, "created_at was reset"
        assert re_query["completed_at"] == first_completed, "task was re-run"
```

#### Large files / large zips: the 128 KiB event payload cap

The FC async gateway caps the event body at **128 KiB** (otherwise 400 `EntityTooLarge`). When
estimating payload size, account for form fields + file base64 + multipart boundary; files over
~100 KiB should use the URI fallback.

**Server-side prep**: add a URI variant for large-file fields and dispatch via
`resolve_input(upload, uri, ...)`. See the `target_uri` / `framework_uri` params in
`services/rfantibody-server/app.py` and `services/rfantibody-server/uris.py::resolve_input`.

**Test-side sync-bootstrap pattern**: run one sync POST to land the large file on NAS; the async tests
reference it via a `file://<jobs_base>/<bootstrap_id>/input/<file>` URI. See the `staged_pdb_uris`
fixture and the `RFANTIBODY_TEST_*_NAS_PATH` env override entry in
[services/rfantibody-server/tests/test_fc_task.py](../../services/rfantibody-server/tests/test_fc_task.py).

**Pipeline chaining**: later steps reference a previous step's artifacts via the
`job://<prev_job_id>/<file>` URI, no need to download-then-reupload; see the `TestAsyncProteinMPNN` /
`TestAsyncRF2` fixtures.

#### 429 handling for `max_concurrent_jobs=1` services

If the FC console's `max_concurrent_request` is tight (the default for most GPU services), GET
`/api/jobs/<id>` gets blocked by platform-layer throttling with 429, and the framework's `poll_job`
only tolerates 10 consecutive errors by default (`max_transient_errors=10` × `interval_s=15` =
2.5 min, not enough for FC's 4-7 min 429 window). The tests must explicitly override it:

```python
poll_job(client, "", task_id,
         timeout_s=1800, interval_s=20,
         max_transient_errors=60)  # 60 × 20s = 20 min buffer
```

Likewise, every non-`poll_job` GET / DELETE should go through a `_get_with_retry`-style wrapper — GET
endpoints will inevitably jitter under high concurrency.

#### Reference implementations

- [genie3-server/tests/test_fc_task.py](../../services/genie3-server/tests/test_fc_task.py) — the
  no-file-upload scenario (unconditional) + small zip upload (motif/binder < 30 KB) + custom YAML.
- [rfantibody-server/tests/test_fc_task.py](../../services/rfantibody-server/tests/test_fc_task.py) — large
  PDB → sync bootstrap → `file://` URI, pipeline chained `job://` URI.