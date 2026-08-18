# Testing

English | [中文](testing.zh.md)

> **Read when**: you run or write tests at any layer (service / framework / gateway).
> **Source**: the per-service `pyproject.toml` `[dependency-groups]`, `framework/pyproject.toml` extras, and `gateway/pyproject.toml`.
> **Refresh/remove when**: a test command changes (watch the gateway command — it needs `--with pytest` since gateway has no dev group).

Tests are isolated per component. **Read a service's `README.md` + its tests before changing it.**

```bash
# one service's offline unit tests (run inside the service directory)
cd services/<svc>-server
uv run --group dev python -m pytest tests/ -q

# FC integration tests (require a deployed service; skipped by default)
RUN_FC_TESTS=1 uv run --group dev python -m pytest -m fc tests/ -v
# hitting the live deployment? read fc-testing.md first (session affinity /
# VPC vs public URL / cold start / 429).

# framework itself
cd framework && uv run --extra dev python -m pytest tests/ -q

# gateway (no dev group → inject pytest)
cd gateway && uv run --with pytest --with pytest-asyncio python -m pytest tests/ -v

# lint one service
uvx ruff check services/<svc>/

# gateway functional test against the local kind deploy
make local-test
```

## Test file layout (per service)

| File | Covers |
|---|---|
| `test_app.py` | offline TestClient unit tests (health / manifest / one endpoint) |
| `test_cli.py` | CLI registration / argv builder / `create_cli` |
| `test_fc.py` | FC sync submit/poll integration (`@pytest.mark.fc`) |
| `test_fc_task.py` | `/api/tasks/<name>` async-task integration (`@pytest.mark.fc`) |

`conftest.py` registers the `server` module via importlib and defines the `fc` marker.

## Gotchas

- A few services' tests read the vendored `upstream/` (git-ignored). Run that service's
  `scripts/vendor.sh` first, or those tests fail on missing files.
- FC tests (`@pytest.mark.fc`) deliberately skip by default; `RUN_FC_TESTS=1` (or `-m fc`) enables
  them against a live deployment — before that, read [fc-testing.md](fc-testing.md) (session
  affinity header `bioagent-session-id`, VPC vs public URL, cold start, 429).
- Offline service tests mock the subprocess; they need only the lightweight `[dependency-groups] dev`
  deps, not the heavy runtime stack.
- `make local-test` targets the local deployment (`http://127.0.0.1:9000`); it is a gateway test,
  not a service test.
