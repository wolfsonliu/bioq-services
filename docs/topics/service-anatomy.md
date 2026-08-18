# Service Anatomy

English | [中文](service-anatomy.zh.md)

> **Read when**: you create or modify files inside a `services/<svc>-server/`.
> **Source**: condensed from real services (see reference table) and the checklist in [`../adding-a-new-service/index.zh.md`](../adding-a-new-service/index.zh.md).
> **Refresh/remove when**: the required-file contract changes (e.g. a new mandatory file is added to the checklist).

## Required files

```
services/<svc>-server/
├── __init__.py          # package marker, usually empty
├── app.py               # create_app + service endpoints + /healthz/detail + task endpoints
├── __main__.py          # CLI batch entry (python -m server <endpoint>)
├── adapter.py           # JobAdapter subclass (name + detect_outputs + manifest_extras + endpoint_examples)
├── settings.py          # ServiceSettings subclass (env_prefix=<SVC>_); weights_dir defaults to /data/models/<svc>/
├── models.py            # request pydantic models
├── tools.py             # argv builders (optional, by complexity)
├── Dockerfile           # COPY framework + services/<svc>/upstream/ + algorithm stack
├── pyproject.toml       # offline test/dev deps (only the uv-venv skeleton needs it; some services omit it)
├── README.md            # endpoints / config / deploy notes + a «Weights» section
├── VERSION              # image tag read by the Makefile (e.g. "v0.0.1")
├── scripts/
│   ├── vendor.sh        # required: clone upstream into upstream/ at a pinned SHA
│   └── fetch_weights.sh # optional: pre-download weights to weights/ or straight to NAS (WEIGHTS_DST override)
└── tests/
    ├── __init__.py / conftest.py   # server module registration (importlib) + fc marker
    ├── test_app.py      # offline TestClient unit tests
    ├── test_cli.py      # CLI batch unit tests
    ├── test_fc.py       # FC sync integration tests (@pytest.mark.fc, skipped by default)
    ├── test_fc_task.py  # FC async task-mode tests (skipped by default)
    └── data/            # test fixtures (small PDB/JSON, etc.)
```

`upstream/` (vendor.sh output) and `weights/` (fetch_weights.sh output) are git-ignored — never
committed.

## Fastest start

Copy an existing structurally-similar service in full, rename it, then edit file by file.

| Scenario | Reference |
|---|---|
| uv venv + sequence design + external weights; canonical single-upstream `vendor.sh` | `services/proteinmpnn-server/` |
| CPU-only uv-venv slim | `services/dockq-server/`, `services/diamond-server/` |
| conda/micromamba multi-stage | `services/deeprank-ab-server/`, `services/pocketxmol-server/` |
| `manifest_extras` + `endpoint_examples` full examples | `services/rfantibody-server/`, `services/genie3-server/` |
| multi-endpoint + config-YAML driven | `services/genie3-server/`, `services/drughive-server/` |
| `vendor.sh` multi-upstream / symlinked weights | `services/promera-server/` |
| `fetch_weights.sh` + large-image slimming | `services/boltzgen-server/` |

## Notes

- Optional per-service files (`configs.py`, `datasets.py`, `patches/`, extra scripts) are documented
  in [`../adding-a-new-service/index.zh.md`](../adding-a-new-service/index.zh.md).
- For the job-lifecycle model behind `adapter.py` / `app.py`, see [framework-api.md](./framework-api.md) and [mental-model.md](./mental-model.md).
