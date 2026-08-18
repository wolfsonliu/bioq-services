# Adding a Service (cookbook)

English | [中文](index.zh.md)

A new service = one Docker image + a set of HTTP endpoints + a CLI batch entry point.
It is deployed on FC and called by agents / pipelines over HTTP; on a Slurm cluster it runs as an
sbatch task via `python -m server`.
**You must build it on top of [framework](../../framework/)** — do not re-implement the shared
HTTP / job lifecycle / error handling / persistence / manifest / CLI / upload-download layers.

This page is the single authoritative workflow document for adding a service; if it conflicts with a
sub-page or `services/<svc>-server/README.md`, this cookbook wins.

## About this guide

This guide is the authoritative workflow document for adding a service **in this repository
(`bioq-services`)**: from skeleton → Dockerfile → testing → deployment. All `services/<svc>/` paths
below are trees under **the repository root**.

**Naming conventions:** the framework import name is **`bioq_service`**, the distribution name is
**`bioq-service-framework`**, and the console script is `bioq-service-mcp-stdio`. The HTTP headers
(`X-Bioagent-*`), the FC session header `bioagent-session-id`, etc. **stay unchanged** (a historical
contract — do not touch them).

**The design document (§0, mandatory before starting)** is an artifact independent of code. This
guide specifies only the **sections it must contain** (see §0); it lives uniformly in
[`docs/specs/`](../specs/) (alongside the existing `*-design.md` files).

## Sub-page navigation

This cookbook is split into sub-pages by topic; this page keeps the overview (design doc first +
required file list + verification / submission checklist):

| Sub-page | Contents |
|------|------|
| [skeleton](./skeleton.md) | 5-minute echo skeleton: `__init__` / `settings` / `models` / `adapter` / `app.py` (incl. task endpoint) / `__main__` / `pyproject` / `VERSION` / `README` |
| [dockerfile](./dockerfile.md) | Dockerfile: `vendor.sh` / `fetch_weights.sh` / uv + conda skeletons / wrapper vs patch decision |
| [conda-pitfalls](./conda-pitfalls.md) | Common pitfalls when wrapping conda-based upstream (LANG / yaml override / dead-import stub / uv git+ etc.) |
| [testing](./testing.md) | `test_app` / `test_cli` / `test_fc` / `test_fc_task` test skeletons |
| [deploy](./deploy.md) | Deploying to FC + console configuration (async task mode + OSS mount) + concurrency testing |

## 0. Write the design doc first (**mandatory before starting**)

Before any code, produce a `YYYY-MM-DD-<svc>-server-design.md` design document (archived in
[`docs/specs/`](../specs/)). This is not ceremony — this guide is "HOW to stand up the skeleton",
the design doc is "WHY it is designed this way + the boundary with the project's other services".
The two are complementary and both are required.

**Design doc writing conventions**: an H1 title + a metadata line (date / status / applies-to /
related) + the required sections below; after writing it, add a link line in `docs/specs/`.

The **required sections** for a new-service design doc (clone a recent sample and edit):

| Section | Key contents |
|---|---|
| Overview | The upstream model/algorithm's positioning, what role it plays in this project's pipeline, and its boundary with existing services |
| Design goals | 3-6 principles to land (single/multi endpoint, wrapper vs patch, weight externalization strategy, conda vs uv, CLI dual-mode alignment, etc.) |
| Endpoint topology | One sentence per endpoint; **explicitly list what v0.0.1 will NOT do** (to avoid later scope creep) |
| Request schema | pydantic field table: field/type/default/constraint/description; file-upload fields listed separately (not inside model) |
| Output | The `<jobs_base_dir>/<job_id>/` tree + the semantics of each file (agents need to know this when consuming) |
| Implementation notes | wrapper/patch decision (use the table below) + data-layout shim (if needed) + argv construction + `detect_outputs()` criteria + conda env pin + `/healthz/detail` probe items |
| Configuration | `env_prefix` + `Settings` field table |
| Deployment target | FC instance spec / timeout / memory / NAS mount / whether async task mode is enabled |
| Testing strategy | four tables for offline HTTP / offline CLI / FC sync / FC async + fixture sources |
| Risks / limitations | build-time gotchas, dependency-pin sensitivity, GPU-generation compatibility, functional overlap with other services |
| Sources | paper citations, upstream URL + pinned SHA, fixture provenance, other service designs referenced |

**Sample docs** (pick one whose structure is closest and clone it):

| Scenario | Reference |
|---|---|
| pytorch + PyG + conda skeleton (GPU diffusion/equivariant networks) | diffusion-hopping-server design |
| conda + ESM/AF cache + monkey-patch upstream | deeprank-ab-server design |
| uv venv + single/multi-variant endpoint | proteinmpnn-server design |
| multi endpoint + config-YAML driven | genie3-server / drughive-server design |
| CPU service + large external resource (DB) | chembounce-server design |

After writing the design doc and syncing `docs/specs/`, return to this guide and start implementing
from the "required file list" below.

## Required file list

```
services/<svc>/
├── __init__.py              # package marker, usually empty
├── app.py                   # create_app + service-specific endpoints + /healthz/detail weight probe (if NAS-backed)
├── __main__.py              # CLI batch entry point (python -m server <endpoint>)
├── adapter.py               # JobAdapter subclass (name + manifest_extras + endpoint_examples)
├── settings.py              # ServiceSettings subclass (env_prefix=<SVC>_); weights_dir defaults to /data/models/<svc>/
├── models.py                # request pydantic models
├── Dockerfile               # COPY framework + services/<svc>/upstream/ + algorithm stack
├── pyproject.toml           # package deps (optional — only for the uv venv skeleton + pip install -e .)
├── README.md                # endpoint / config / deploy notes + Weights section (NAS / SIF --bind)
├── VERSION                  # image tag, read by the Makefile (e.g. "v0.0.1")
├── scripts/
│   ├── vendor.sh            # required: clone upstream source into upstream/ at a pinned SHA (see below)
│   └── fetch_weights.sh     # optional: download model weights into weights/ or directly to NAS (supports WEIGHTS_DST env override)
├── upstream/                # gitignored: vendor.sh artifacts; the Docker build COPYs upstream source from here
├── weights/                 # gitignored: fetch_weights.sh artifacts, staging only; uploaded to NAS for production
└── tests/                   # pytest tests
    ├── __init__.py
    ├── conftest.py          # server module registration (importlib.util) + fc marker
    ├── test_app.py          # offline TestClient unit tests (health / manifest / one endpoint)
    ├── test_cli.py          # CLI batch unit tests (endpoint registration / argv builder / create_cli)
    ├── test_fc.py           # FC integration tests for the sync submit/poll path (marker=fc, skipped by default)
    ├── test_fc_task.py      # /api/tasks/<name> async task mode integration tests (marker=fc, skipped by default)
    └── data/                # PDB / JSON / etc. fixtures for test_fc / test_fc_task / test_cli
```

Service-specific (add as needed, not required):

| File | Purpose | Example |
|---|---|---|
| `tools.py` / `configs.py` | argv builder / config-YAML constructor | [rfantibody-server/tools.py](../../services/rfantibody-server/tools.py), [genie3-server/configs.py](../../services/genie3-server/configs.py) |
| `datasets.py` | dataset zip extraction + path rewriting | [genie3-server/datasets.py](../../services/genie3-server/datasets.py) |
| `uris.py` | URI scheme parsing (job:// / oss:// / file:// / http(s):// etc.) — **now provided uniformly by the framework** | [framework uris.py](../../framework/src/bioq_service/uris.py) |
| `patches/` | upstream source patches + apply rules (applied in the Dockerfile via `patch -p1` after vendor.sh) | [genie3-server/patches/](../../services/genie3-server/patches/) |
| `scripts/vendor.sh` | **required**: clone upstream source into `upstream/` at a pinned SHA + retries + verification. The Dockerfile COPYs from `upstream/` at build time — no in-image `git clone` | [proteinmpnn-server/scripts/vendor.sh](../../services/proteinmpnn-server/scripts/vendor.sh) (single upstream), [promera-server/scripts/vendor.sh](../../services/promera-server/scripts/vendor.sh) (multi upstream) |
| `scripts/fetch_weights.sh` | pre-download model weights into `weights/`, **supports `WEIGHTS_DST=` to download straight to NAS**. The Docker image no longer COPYs weights | [boltzgen-server/scripts/fetch_weights.sh](../../services/boltzgen-server/scripts/fetch_weights.sh), [immunebuilder-server/scripts/fetch_weights.sh](../../services/immunebuilder-server/scripts/fetch_weights.sh) |
| `upstream/` | gitignored vendored upstream source dir (vendor.sh artifacts); the Docker build COPYs from here | proteinmpnn-server/upstream/ |
| `weights/` | gitignored local weight staging dir; rsync'd to NAS for production | boltz-server/weights/ |

## Verification checklist

Run through this before submitting a new service:

```bash
# 1. lint
uvx ruff check services/<svc>/

# 2. unit tests — HTTP (test_app) + CLI (test_cli)
uv run python -m pytest services/<svc>/tests/test_app.py -v
uv run python -m pytest services/<svc>/tests/test_cli.py -v

# 3. import smoke (catch typos in adapter / settings)
python -c "from server.app import app; print(app.title)"

# 3.5. vendor upstream source (required, run before the Docker build)
./services/<svc>/scripts/vendor.sh
# verify the artifacts
ls services/<svc>/upstream/ | head

# 3.6. (if any) pre-download weights — only needed for local SIF tests; upload to NAS directly for FC deploys
# ./services/<svc>/scripts/fetch_weights.sh

# 4. local docker build (verify the Dockerfile didn't miss a COPY)
make build-<svc>

# 5. manifest payload sanity (confirm every endpoint_examples entry is filled in)
docker run --rm -p 9000:9000 <svc>-server &
curl http://localhost:9000/api/manifest | jq .endpoints
kill %1

# 5.5. CLI smoke — verify __main__.py can parse args (no need to actually run the algorithm)
docker run --rm <svc>-server .venv/bin/python -m server --help
docker run --rm <svc>-server .venv/bin/python -m server generate --help

# 5.7. task endpoint route sanity (verify task endpoints are registered)
docker run --rm -p 9000:9000 <svc>-server &
sleep 5
curl http://localhost:9000/openapi.json | jq '.paths | keys | .[]' | grep "/api/tasks/"
# you should see /api/tasks/<name>... mapped to each /api/<name>...
kill %1

# 6. after deploying to FC, run the FC tests (smoke first, then inference)
pytest -m fc -k "not minimal_job" services/<svc>/tests/test_fc.py
pytest -m fc services/<svc>/tests/test_fc.py

# 7. after deploying + enabling async task mode in the console, run the task tests (recommended: the primary entry for modern GPU services)
pytest -m fc services/<svc>/tests/test_fc_task.py
```

## Submission checklist (check off in the PR description)

- [ ] **Design doc in place**: `YYYY-MM-DD-<svc>-server-design.md` contains the required sections from §0, archived in [docs/specs/](../specs/)
- [ ] Required files (including VERSION + `__main__.py` + scripts/vendor.sh + tests/{conftest,test_app,test_cli,test_fc,test_fc_task,data} + README; pyproject.toml as needed)
- [ ] `scripts/vendor.sh` exists and runs: pinned upstream SHA, 5 retries, SHA verification, rsync into `upstream/`
- [ ] `services/<svc>/upstream/` is added to the repo-root `.gitignore`
- [ ] **No `git clone` in the Dockerfile, no `COPY opensource/`**: upstream is always COPYed from `services/<svc>/upstream/`; apt does not install git (unless there is another genuine use)
- [ ] **No `COPY services/<svc>/weights/` in the Dockerfile**: weights default to NAS; if you must bake small weights (< 100 MB), state the reason in a comment
- [ ] `settings.py`'s `weights_dir` default points at `/data/models/<svc>/...`
- [ ] `app.py` implements a custom `/healthz/detail` (with `_strip_route` + `weights_loaded` / `weights_missing` fields); when weights are missing it returns HTTP 200 + `weights_loaded=false` and does not raise at import time
- [ ] `services/<svc>/README.md` has a `## Weights` section: NAS layout tree / pre-stage command / FC verify / SIF `--bind` example
- [ ] `scripts/fetch_weights.sh` supports the `WEIGHTS_DST=` env override to download straight to NAS (if the service needs local weights)
- [ ] Weights are uploaded to NAS `/data/models/<svc>/` (prerequisite before FC deploy)
- [ ] `__main__.py` registers a `CLIEndpoint` descriptor for every endpoint
- [ ] `JobAdapter.manifest_extras()` has at least `tool_outputs` + `input_uri_schemes`
- [ ] `JobAdapter.endpoint_examples()` has at least 1 curl example per endpoint
- [ ] Endpoints receive form params via `Depends(model_form_depends(Model))` (see app.py notes)
- [ ] Every submit/poll endpoint has a paired task endpoint (`/api/tasks/<name>`), via `register_task_endpoint` (no upload) or `execute_task` (with upload)
- [ ] Task endpoint registration sits inside the `if settings.task_endpoints_enabled:` guard (for custom endpoints)
- [ ] After FC deploy, enable "async task mode" in the console + clear the keepalive URL + confirm the NAS mount `/data/models/<svc>` is readable
- [ ] **Services called through the gateway**: in the FC console, mount the data-plane OSS bucket `bioagent-inputs` to `/mnt/oss` (RW); the Dockerfile runtime has `ENV <SVC>_OSS_OUTPUT_MOUNT=/mnt/oss`; the framework is installed via `COPY framework` (not bind-mount, otherwise the output-sink fix doesn't make it into the image); for file-input services its `uris.py` must support bare `/` absolute paths (the gateway rewrites `oss://` to `/mnt/oss/...` and downstream reads directly via `shutil.copy2`)
- [ ] After FC deploy, `curl /healthz/detail` confirms `weights_loaded: true`
- [ ] The Dockerfile uses the uv venv skeleton or the conda/micromamba multi-stage build
- [ ] **conda services**: the Dockerfile runtime stage has `ENV LANG=C.UTF-8` + `ENV LC_ALL=C.UTF-8` (disables the upstream `open()` ASCII locale trap, see [conda-pitfalls.md](./conda-pitfalls.md))
- [ ] **conda services**: the bundled upstream config yaml has had all wrapper-controlled keys stripped (`data_file`, `data_path`, `num_samples`, `seed`, `mode`, `checkpoint_path`, etc.); the wrapper adds a defensive assertion like `assert upstream_args.data_file == csv_path`
- [ ] **conda services**: the Dockerfile has a filesystem sanity check (`[ -f "$f" ] || exit 1` for each critical upstream .py) + a full upstream module-chain import smoke (with wandb/matplotlib stub injection, if needed)
- [ ] **conda services**: in the wrapper, all top-level dead-code imports (wandb / matplotlib / etc.) are stubbed via `sys.modules.setdefault(name, ModuleType(name))` — **don't install the real packages**
- [ ] If you changed `scripts/vendor.sh` exclude rules: **`rm -rf services/<svc>/upstream/` first, then re-run vendor.sh**, to avoid the Docker COPY reusing stale vendor artifacts
- [ ] If present, `pyproject.toml` depends on `bioq-service-framework` (some services don't need a pyproject.toml — server code is injected via COPY rather than pip install)
- [ ] `settings.py` has no `os.getenv` calls (everything goes through pydantic-settings)
- [ ] `uvx ruff check` passes
- [ ] `pytest tests/test_app.py tests/test_cli.py` passes (offline)
- [ ] After deploy, `pytest -m fc services/<svc>/tests/test_fc.py` passes (at least 1 inference job ran per endpoint)
- [ ] After enabling async task mode in the console, `pytest -m fc services/<svc>/tests/test_fc_task.py` passes (covering `/api/tasks/<name>` submit=202 / completion / lifecycle / platform-layer dedup)
- [ ] [services.yaml](../../services.yaml) gains an entry `<svc>-server:` + `url: https://...` (optional `tier` / `region` / `function` / `gpu`; **for file-input services add `oss_mount: true`** so the gateway rewrites `oss://` inputs to `/mnt/oss/...` for downstream credential-free direct read)
- [ ] **Services called through the gateway**: add a `TestEndToEnd<Svc>` e2e class in [gateway/tests/test_fc.py](../../gateway/tests/test_fc.py) (follow `TestEndToEndProteinMPNN` for presign-file input / `TestEndToEndMMseqs2` for inline params), asserting download is 302→OSS + `results.zip` contains the expected artifacts. **Nested endpoints (e.g. `generate/motif`) require gateway ≥ v0.0.2** (`{endpoint:path}` routing)

## Related

- [framework/](../../framework/) —— the framework package source + unit tests as reference
- Framework API / conventions: [framework-api.md](../topics/framework-api.md) · [conventions.md](../topics/conventions.md)
- Build / verify: [build-deploy.md](../topics/build-deploy.md) · [testing.md](../topics/testing.md)
- Called through the gateway: [gateway.md](../topics/gateway.md) · [service-anatomy.md](../topics/service-anatomy.md)
- Design doc archive: [docs/specs/](../specs/)
- Reference implementations:
  - [proteinmpnn-server](../../services/proteinmpnn-server/) —— **the standard vendor.sh example**: single upstream + cherry-pick + the simplest test-passing skeleton
  - [boltzgen-server](../../services/boltzgen-server/) —— vendor.sh + externalized weights + healthz probe + the large-image (~15 GB → ~2.5 GB) comparison
  - [promera-server](../../services/promera-server/) —— vendor.sh **multi-upstream** (tinyprot + promera + LigandMPNN) + symlink-mode weights
  - [genie3-server](../../services/genie3-server/) —— vendor.sh + patches applied + symlink-mode weights
  - [rfantibody-server](../../services/rfantibody-server/) / [boltz-server](../../services/boltz-server/) —— uv venv skeleton, weights already externalized
  - [deeprank-ab-server](../../services/deeprank-ab-server/) —— conda/micromamba multi-stage skeleton, weights already externalized