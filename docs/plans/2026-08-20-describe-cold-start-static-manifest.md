# describe 冷启动与静态自描述契约 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `bioq describe <svc>` 冷启动免疫——命中静态契约时零下游调用；未命中时也在 ≤8s 内返回可行动的 warming 哨兵，绝不 60s 裸超时。

**Architecture:** 分两阶段。阶段一只改 gateway：把 `Discovery` 从「串行两次 60s 抓取 + 失败塌缩为 `{}`」重写为「manifest-first 短接 + 结构化失败分类 + 负缓存 + 单飞」，并把超时接入 settings。阶段二把每个服务的 manifest/openapi 在 release 期物化为 `manifests/<svc>.(manifest|openapi).json` 随网关下发，`describe_service` 命中即返回（source=registry），未命中回退阶段一的 live 路径。

**Tech Stack:** Python 3.11 · httpx · FastAPI/Starlette · pydantic-settings · `threading`(stdlib) · `uv`(orchestration) · Docker · GNU Make。

**设计文档:** `docs/specs/2026-08-20-describe-cold-start-static-manifest-design.md`（先读它了解背景、根因与方案对比；本计划是该设计的可执行拆分，并有一处简化：物化脚本做成 repo 根 `scripts/gen_manifests.py`，复用既有 `build_manifest()`，零 framework 改动、零 per-service 文件改动）。

---

## 文件结构

- **Modify** `gateway/settings.py` — 新增 4 个 `discovery_*` 配置项（阶段一 Task 1）。
- **Modify** `gateway/discover.py` — 重写 `Discovery`：结构化 `_get_json`、失败分类、manifest-first 短接、负缓存、单飞（阶段一 Task 2）。
- **Modify** `gateway/app.py` — 从 settings 构造 `Discovery`（Task 3）；`describe_service` 静态优先（Task 6）。
- **Modify** `gateway/registry.py` — `ServiceRegistry.manifest(svc)` / `openapi(svc)` 按约定读 `manifests/`（Task 5）。
- **Modify** `gateway/Dockerfile` — `COPY manifests/ /opt/gateway/manifests/`（Task 7）。
- **Modify** `Makefile` — `gen-manifests` / `check-manifests` 目标（Task 7）。
- **Create** `scripts/gen_manifests.py` — 物化 orchestrator + per-service dump leaf（Task 4）。
- **Create** `manifests/<svc>.manifest.json` + `manifests/<svc>.openapi.json` — 提交一个服务的样例（Task 8）。
- **Test** `gateway/tests/test_settings.py`(新) · `gateway/tests/test_discover.py` · `gateway/tests/test_registry.py` · `gateway/tests/test_app.py`。

交互契约（阶段一/二共同输出，字段向后兼容，绝无 `response_model`）：

```json
{"service": "<svc>", "manifest": {...}, "openapi": {...},
 "status": "ok|warming|partial|no_manifest|error",
 "source": "registry|live",
 "detail": "非 ok 时的单行提示"}
```

---

## 阶段一：加固实时发现（护栏）

### Task 0: 基线验证（先确认当前 GREEN）

- [ ] **Step 1: 跑 gateway 测试确认基线通过**

Run:
```bash
cd gateway && uv run --with pytest --with pytest-asyncio python -m pytest tests/ -q
```
Expected: `all passed`（记录当前通过数量，后续每步不得回退）。

- [ ] **Step 2: 确认 `gateway/discover.py` 现状**（供对比）
    `gateway/discover.py` 现为 44 行：`Discovery(ttl_sec, timeout_sec)`，`describe()` 串行两次 `_get_json`，`_get_json` 把所有异常返回 `{}`，只在 `manifest and openapi` 都真时缓存。

### Task 1: 新增 discovery 配置项

**Files:**
- Modify: `gateway/settings.py:72-73`（`dispatch_timeout_sec` 之后）
- Create: `gateway/tests/test_settings.py`

- [ ] **Step 1: 写失败测试**

Create `gateway/tests/test_settings.py`:
```python
from __future__ import annotations

from server.settings import GatewaySettings


def test_discovery_timeout_defaults():
    s = GatewaySettings()
    assert s.discovery_ttl_sec == 300.0
    assert s.discovery_negative_ttl_sec == 15.0
    assert s.discovery_read_timeout_sec == 8.0
    assert s.discovery_connect_timeout_sec == 5.0
```

- [ ] **Step 2: 运行确认失败**

Run:
```bash
cd gateway && uv run --with pytest --with pytest-asyncio python -m pytest tests/test_settings.py::test_discovery_timeout_defaults -v
```
Expected: FAIL — `AttributeError: 'GatewaySettings' object has no attribute 'discovery_ttl_sec'`。

- [ ] **Step 3: 实现 settings 字段**

In `gateway/settings.py`, after the `dispatch_timeout_sec` line:
```python
    # downstream HTTP dispatch
    dispatch_timeout_sec: float = Field(default=60.0, ge=5)

    # describe/self-description discovery (see
    # docs/specs/2026-08-20-describe-cold-start-static-manifest-design.md).
    # read timeout is the cold-start binding item: a scaled-to-zero FC holds the
    # request until the instance boots, so read must be short to fail fast.
    discovery_ttl_sec: float = Field(default=300.0, ge=0)
    discovery_negative_ttl_sec: float = Field(default=15.0, ge=0)
    discovery_read_timeout_sec: float = Field(default=8.0, ge=1)
    discovery_connect_timeout_sec: float = Field(default=5.0, ge=1)
```

- [ ] **Step 4: 运行确认通过**

Run 同上；Expected: `1 passed`。

- [ ] **Step 5: Commit**

```bash
git add gateway/settings.py gateway/tests/test_settings.py
git commit -m "feat(gateway): add discovery timeout/ttl settings"
```

### Task 2: 重写 `Discovery`（分类 + 短接 + 负缓存 + 单飞）

**Files:**
- Modify: `gateway/discover.py`（整体重写）
- Modify: `gateway/tests/test_discover.py`（改 3 个、新增 5 个）

- [ ] **Step 1: 改写/新增失败测试**

把 `gateway/tests/test_discover.py` 整体替换为下面内容（保留的 3 个测试语义不变、断言不变；`test_describe_errors_degrade` 增加 `status` 断言；两个 `_not_cache_*` 改为新语义；新增分类/短接/负缓存/单飞测试）：

```python
from __future__ import annotations

import threading
import time

import httpx

from server.discover import Discovery


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_describe_merges_manifest_and_openapi():
    def handler(request):
        if request.url.path == "/api/manifest":
            return httpx.Response(200, json={"service": "openbpmd", "endpoints": ["score"]})
        if request.url.path == "/openapi.json":
            return httpx.Response(200, json={"paths": {"/api/score": {}}})
        return httpx.Response(404)

    disc = Discovery(client=_client(handler), ttl_sec=60)
    info = disc.describe("openbpmd-server", "https://svc.local")
    assert info["manifest"]["service"] == "openbpmd"
    assert "/api/score" in info["openapi"]["paths"]
    assert info["status"] == "ok"
    assert info["source"] == "live"


def test_describe_cached():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if request.url.path == "/api/manifest":
            return httpx.Response(200, json={"service": "x"})
        return httpx.Response(200, json={"paths": {}})

    disc = Discovery(client=_client(handler), ttl_sec=60)
    disc.describe("s", "https://svc.local")
    disc.describe("s", "https://svc.local")
    assert calls["n"] == 2  # one manifest + one openapi; second describe from cache


def test_describe_ttl_expiry_refetches():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if request.url.path == "/api/manifest":
            return httpx.Response(200, json={"service": "x"})
        return httpx.Response(200, json={"paths": {}})

    disc = Discovery(client=_client(handler), ttl_sec=0)
    disc.describe("s", "https://svc.local")
    disc.describe("s", "https://svc.local")
    assert calls["n"] == 4  # ttl=0 => always expired, refetch manifest + openapi


def test_describe_errors_degrade():
    disc = Discovery(client=_client(lambda req: httpx.Response(500)), ttl_sec=60)
    info = disc.describe("s", "https://svc.local")
    assert info["service"] == "s"
    assert info["manifest"] == {}
    assert info["openapi"] == {}
    assert info["status"] == "error"


def test_timeout_is_warming_and_short_circuits():
    state = {"openapi": 0}

    def handler(request):
        if request.url.path == "/api/manifest":
            raise httpx.ReadTimeout("cold")
        if request.url.path == "/openapi.json":
            state["openapi"] += 1
            return httpx.Response(200, json={"paths": {}})
        return httpx.Response(404)

    disc = Discovery(client=_client(handler), ttl_sec=60)
    info = disc.describe("s", "https://svc.local")
    assert info["status"] == "warming"
    assert info["manifest"] == {} and info["openapi"] == {}
    assert "detail" in info
    assert state["openapi"] == 0  # short-circuited: no openapi fetch after manifest fail


def test_404_is_no_manifest():
    disc = Discovery(client=_client(lambda req: httpx.Response(404)), ttl_sec=60)
    info = disc.describe("s", "https://svc.local")
    assert info["status"] == "no_manifest"
    assert "detail" in info


def test_negative_cache_avoids_refetch():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(500)

    disc = Discovery(client=_client(handler), ttl_sec=60, negative_ttl_sec=60)
    disc.describe("s", "https://svc.local")
    disc.describe("s", "https://svc.local")
    assert calls["n"] == 1  # second describe served from negative cache


def test_describe_does_not_cache_partial_failure():
    state = {"openapi_fail": True}

    def handler(request):
        if request.url.path == "/api/manifest":
            return httpx.Response(200, json={"service": "x"})
        if request.url.path == "/openapi.json":
            return httpx.Response(500) if state["openapi_fail"] \
                else httpx.Response(200, json={"paths": {"/api/s": {}}})
        return httpx.Response(404)

    disc = Discovery(client=_client(handler), ttl_sec=300)
    first = disc.describe("s", "https://svc.local")
    assert first["status"] == "partial"
    assert first["manifest"] == {"service": "x"} and first["openapi"] == {}
    state["openapi_fail"] = False
    second = disc.describe("s", "https://svc.local")  # partial was NOT cached
    assert second["status"] == "ok"
    assert second["openapi"] == {"paths": {"/api/s": {}}}


def test_describe_does_not_cache_total_failure():
    state = {"fail": True}

    def handler(request):
        if state["fail"]:
            return httpx.Response(500)
        return httpx.Response(200, json={"service": "x"} if request.url.path == "/api/manifest" else {"paths": {}})

    disc = Discovery(client=_client(handler), ttl_sec=300, negative_ttl_sec=0)
    first = disc.describe("s", "https://svc.local")
    assert first["status"] == "error"
    state["fail"] = False
    second = disc.describe("s", "https://svc.local")  # negative ttl 0 => refetch
    assert second["manifest"] == {"service": "x"}


def test_single_flight_coalesces_concurrent():
    state = {"manifest": 0, "openapi": 0}
    guard = threading.Lock()
    gate = threading.Event()

    def handler(request):
        if request.url.path == "/api/manifest":
            with guard:
                state["manifest"] += 1
            gate.wait(timeout=2.0)
            return httpx.Response(200, json={"service": "x"})
        if request.url.path == "/openapi.json":
            with guard:
                state["openapi"] += 1
            return httpx.Response(200, json={"paths": {}})
        return httpx.Response(404)

    disc = Discovery(client=_client(handler), ttl_sec=60)
    results = []

    def worker():
        results.append(disc.describe("s", "https://svc.local"))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    time.sleep(0.05)
    gate.set()
    for t in threads:
        t.join()

    assert state["manifest"] == 1
    assert state["openapi"] == 1
    assert all(r["status"] == "ok" for r in results)
```

- [ ] **Step 2: 运行确认失败**

Run:
```bash
cd gateway && uv run --with pytest --with pytest-asyncio python -m pytest tests/test_discover.py -v
```
Expected: FAIL——多用例报 `TypeError`（`Discovery` 尚无 `negative_ttl_sec` 参数）/ `KeyError`（尚无 `status` 字段）/ 单飞用例计数不为 1。

- [ ] **Step 3: 重写 `gateway/discover.py`**

整体替换 `gateway/discover.py`:
```python
"""Fetch + cache downstream /api/manifest + /openapi.json for `describe`.

Phase-1 hardening: bounded split timeouts, manifest-first short-circuit,
structured failure taxonomy, short-TTL negative caching, and per-service
single-flight coalescing. See
docs/specs/2026-08-20-describe-cold-start-static-manifest-design.md.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Literal

import httpx

FetchOutcome = Literal["ok", "warming", "no_manifest", "error"]


class Discovery:
    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        ttl_sec: float = 300.0,
        negative_ttl_sec: float = 15.0,
        connect_timeout_sec: float = 5.0,
        read_timeout_sec: float = 8.0,
    ) -> None:
        if client is None:
            client = httpx.Client(
                timeout=httpx.Timeout(connect=connect_timeout_sec, read=read_timeout_sec)
            )
        self._client = client
        self._ttl = ttl_sec
        self._negative_ttl = negative_ttl_sec
        self._cache: dict[str, tuple[dict[str, Any], float]] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def describe(self, svc: str, base_url: str) -> dict[str, Any]:
        if info := self._cached(svc):
            return info
        with self._lock_for(svc):
            if info := self._cached(svc):
                return info
            info = self._fetch(svc, base_url)
            ttl = self._cache_ttl(info["status"])
            if ttl > 0:
                self._cache[svc] = (info, time.time() + ttl)
            return info

    # ---- internal ----

    def _cached(self, svc: str) -> dict[str, Any] | None:
        entry = self._cache.get(svc)
        if entry is None:
            return None
        info, exp = entry
        return info if exp > time.time() else None

    def _lock_for(self, svc: str) -> threading.Lock:
        with self._locks_guard:
            lock = self._locks.get(svc)
            if lock is None:
                lock = threading.Lock()
                self._locks[svc] = lock
            return lock

    def _cache_ttl(self, status: str) -> float:
        if status == "ok":
            return self._ttl
        if status in ("warming", "error"):
            return self._negative_ttl
        return 0.0  # partial / no_manifest: never cached

    def _fetch(self, svc: str, base_url: str) -> dict[str, Any]:
        m_status, manifest = self._get_json(f"{base_url}/api/manifest")
        if m_status != "ok":
            # Short-circuit: never hit /openapi.json once the manifest already
            # failed (cold start / missing framework / hard error).
            return self._info(svc, {}, {}, m_status)
        o_status, openapi = self._get_json(f"{base_url}/openapi.json")
        status = "ok" if (openapi and o_status == "ok") else "partial"
        return self._info(svc, manifest, openapi, status)

    def _info(self, svc: str, manifest: dict, openapi: dict, status: str) -> dict[str, Any]:
        info = {
            "service": svc,
            "manifest": manifest,
            "openapi": openapi,
            "status": status,
            "source": "live",
        }
        if status != "ok":
            info["detail"] = _DETAIL[status]
        return info

    def _get_json(self, url: str) -> tuple[FetchOutcome, dict[str, Any]]:
        try:
            r = self._client.get(url)
            if r.status_code == 404:
                return "no_manifest", {}
            if r.status_code in (502, 504):
                return "warming", {}
            r.raise_for_status()
            return "ok", r.json()
        except httpx.TimeoutException:
            return "warming", {}
        except httpx.NetworkError:
            return "warming", {}
        except httpx.HTTPStatusError:
            return "error", {}
        except ValueError:  # 200-but-not-JSON
            return "error", {}
        except Exception:  # noqa: BLE001 — describe degrades gracefully
            return "error", {}


_DETAIL = {
    "warming": "downstream cold-start timed out; retry in ~15s",
    "no_manifest": "service has no /api/manifest (framework self-description not adopted)",
    "partial": "manifest ok; openapi unavailable (CLI/human path unaffected)",
    "error": "downstream returned an unexpected error",
}
```

- [ ] **Step 4: 运行确认通过**

Run:
```bash
cd gateway && uv run --with pytest --with pytest-asyncio python -m pytest tests/test_discover.py -v
```
Expected: `10 passed`。

- [ ] **Step 5: Commit**

```bash
git add gateway/discover.py gateway/tests/test_discover.py
git commit -m "feat(gateway): harden discovery with taxonomy, short-circuit, negative cache, single-flight"
```

### Task 3: 从 settings 构造 `Discovery`

**Files:**
- Modify: `gateway/app.py:59`

- [ ] **Step 1: 接线**

把 `gateway/app.py:59` 的
```python
app.state.discover = Discovery(ttl_sec=300)
```
改为：
```python
app.state.discover = Discovery(
    ttl_sec=settings.discovery_ttl_sec,
    negative_ttl_sec=settings.discovery_negative_ttl_sec,
    connect_timeout_sec=settings.discovery_connect_timeout_sec,
    read_timeout_sec=settings.discovery_read_timeout_sec,
)
```

- [ ] **Step 2: 回归网关测试**

Run:
```bash
cd gateway && uv run --with pytest --with pytest-asyncio python -m pytest tests/ -q
```
Expected: `all passed`（与 Task 0 一致或更多——`test_settings.py` 新增 1 个）。

- [ ] **Step 5: Commit**

```bash
git add gateway/app.py
git commit -m "feat(gateway): wire discovery timeouts from settings"
```

---

## 阶段二：静态契约根修

### Task 4: `scripts/gen_manifests.py`（物化 + 校验）

**Files:**
- Create: `scripts/gen_manifests.py`

- [ ] **Step 1: 写脚本**（完整内容，两模式：orchestrator + dump-one leaf）

Create `scripts/gen_manifests.py`:
```python
#!/usr/bin/env python3
"""Materialize static `describe` contracts (see
docs/specs/2026-08-20-describe-cold-start-static-manifest-design.md).

Modes:
  generate   `python scripts/gen_manifests.py`            all services in services.yaml
  check      `python scripts/gen_manifests.py --check`     regenerate to tmp + diff
  dump-one   internal leaf, called per service in its own venv

`generate`/`check` run the orchestrator in the *gateway* venv (which has
bioq-service-framework + pyyaml); for each service it re-enters that service's
own venv via `uv run --project services/<svc>-server`. `dump-one` registers the
service dir as the `server` package (same trick as the service tests' conftest)
and dumps build_manifest() + app.openapi().
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = Path(__file__).resolve()


def register_server_package(service_dir: Path) -> None:
    if "server" in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(
        "server",
        service_dir / "__init__.py",
        submodule_search_locations=[str(service_dir)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot build `server` package spec from {service_dir}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["server"] = module
    spec.loader.exec_module(module)


def dump_one(svc: str, service_dir: Path, out_dir: Path) -> None:
    register_server_package(service_dir)
    from bioq_service.manifest import build_manifest
    from server.app import adapter, app, settings

    manifest = build_manifest(app, adapter, settings)
    openapi = app.openapi()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{svc}.manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / f"{svc}.openapi.json").write_text(
        json.dumps(openapi, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def service_names() -> list[str]:
    from bioq_service.service_registry import load_services
    return sorted(load_services(REPO_ROOT / "services.yaml"))


def service_dir(svc: str) -> Path:
    return REPO_ROOT / "services" / svc


def render_one(svc: str, out_dir: Path) -> None:
    d = service_dir(svc)
    if not (d / "Dockerfile").is_file():
        raise SystemExit(f"missing service dir {d}")
    subprocess.run(
        ["uv", "run", "--project", str(d), "python", str(SCRIPT),
         "--dump-one", "--svc", svc, "--service-dir", str(d), "--out-dir", str(out_dir)],
        check=True,
    )


def generate() -> None:
    out_dir = REPO_ROOT / "manifests"
    for svc in service_names():
        print(f"gen-manifests: {svc}")
        render_one(svc, out_dir)


def check() -> int:
    rc = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for svc in service_names():
            committed_m = REPO_ROOT / "manifests" / f"{svc}.manifest.json"
            committed_o = REPO_ROOT / "manifests" / f"{svc}.openapi.json"
            if not committed_m.is_file() or not committed_o.is_file():
                print(f"check-manifests: MISSING manifest for {svc}", file=sys.stderr)
                rc = 1
                continue
            render_one(svc, tmp_dir)
            for kind, committed in (("manifest", committed_m), ("openapi", committed_o)):
                fresh = tmp_dir / f"{svc}.{kind}.json"
                if committed.read_text(encoding="utf-8") != fresh.read_text(encoding="utf-8"):
                    print(f"check-manifests: STALE {committed} (re-run make gen-manifests)",
                          file=sys.stderr)
                    rc = 1
    return rc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--dump-one", action="store_true")
    parser.add_argument("--svc")
    parser.add_argument("--service-dir")
    parser.add_argument("--out-dir")
    args = parser.parse_args()

    if args.dump_one:
        dump_one(args.svc, Path(args.service_dir).resolve(), Path(args.out_dir))
        return 0
    if args.check:
        return check()
    generate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: 单服务冒烟（先于写 registry/测试，验证脚本本身可用）**

Run:
```bash
cd services/dockq-server && uv run python ../../scripts/gen_manifests.py \
  --dump-one --svc dockq-server --service-dir . --out-dir ../../manifests
```
Expected: 生成 `manifests/dockq-server.manifest.json` + `manifests/dockq-server.openapi.json`。
    （dockq-server 是 CPU wrapper，`app.py` 模块级不 import 上游库；若某服务模块级 import 重型库导致 import 失败，换任一同样干净的 CPU 服务，如 `plip-server` / `lightdock-server`，`--svc`/`--service-dir` 相应替换。）

- [ ] **Step 3: 校验 JSON 形状**

Run（在仓库根目录）:
```bash
python3 - <<'PY'
import json
m = json.load(open("manifests/dockq-server.manifest.json"))
o = json.load(open("manifests/dockq-server.openapi.json"))
assert m["service"], "manifest.service empty"
assert m["endpoints"], "manifest.endpoints empty"
assert o["paths"], "openapi.paths empty"
print("OK: manifest.service=%r endpoints=%d openapi.paths=%d"
      % (m["service"], len(m["endpoints"]), len(o["paths"])))
PY
```
Expected: `OK: manifest.service=... endpoints=... openapi.paths=...`（结尾非 0 端点、非 0 paths）。

- [ ] **Step 4: Commit**

```bash
git add scripts/gen_manifests.py manifests/dockq-server.manifest.json manifests/dockq-server.openapi.json
git commit -m "feat: add manifest materialization script and dockq sample manifests"
```

### Task 5: `ServiceRegistry.manifest()` / `openapi()`

**Files:**
- Modify: `gateway/registry.py`
- Modify: `gateway/tests/test_registry.py`

- [ ] **Step 1: 写失败测试**

在 `gateway/tests/test_registry.py` 末尾追加：
```python
def _manifest_tree(tmp_path):
    p = _yaml(tmp_path)
    mdir = tmp_path / "manifests"
    mdir.mkdir()
    (mdir / "openbpmd-server.manifest.json").write_text(
        '{"service": "openbpmd", "endpoints": [{"path": "/api/score"}]}',
        encoding="utf-8",
    )
    (mdir / "openbpmd-server.openapi.json").write_text(
        '{"paths": {"/api/score": {}}}', encoding="utf-8",
    )
    return p


def test_manifest_reads_manifests_dir(tmp_path):
    reg = ServiceRegistry(_manifest_tree(tmp_path))
    assert reg.manifest("openbpmd-server")["service"] == "openbpmd"
    assert reg.openapi("openbpmd-server")["paths"] == {"/api/score": {}}


def test_manifest_missing_returns_none(tmp_path):
    reg = ServiceRegistry(_yaml(tmp_path))  # no manifests/ dir
    assert reg.manifest("openbpmd-server") is None
    assert reg.openapi("openbpmd-server") is None
```

- [ ] **Step 2: 运行确认失败**

Run:
```bash
cd gateway && uv run --with pytest --with pytest-asyncio python -m pytest tests/test_registry.py -v
```
Expected: FAIL — `AttributeError: 'ServiceRegistry' object has no attribute 'manifest'`。

- [ ] **Step 3: 实现**

把 `gateway/registry.py` 改为（在 `base_url` 之前新增 `manifest`/`openapi`，并加 `import json`）：
```python
"""Downstream service registry: load services.yaml (svc -> record).

Thin wrapper over the framework's `load_services` (which parses the YAML
registry into `ServiceRecord`s). Kept as a small class so the app can hold it
on `app.state` and reload it if needed. Also serves the static describe
contracts committed under manifests/ (see
docs/specs/2026-08-20-describe-cold-start-static-manifest-design.md).
"""

from __future__ import annotations

import json
from pathlib import Path

from bioq_service.service_registry import ServiceRecord, load_services


class ServiceRegistry:
    def __init__(self, registry_path: Path) -> None:
        self._path = Path(registry_path)
        self._manifests_dir = self._path.parent / "manifests"
        self._manifest_cache: dict[str, dict | None] = {}
        self._openapi_cache: dict[str, dict | None] = {}
        self._services: dict[str, ServiceRecord] = {}
        self.reload()

    def reload(self) -> None:
        self._services = load_services(self._path)

    def list(self) -> list[str]:
        return sorted(self._services)

    def record(self, svc: str) -> ServiceRecord:
        if svc not in self._services:
            raise KeyError(svc)
        return self._services[svc]

    def base_url(self, svc: str) -> str:
        return self.record(svc).url

    def manifest(self, svc: str) -> dict | None:
        if svc not in self._manifest_cache:
            self._manifest_cache[svc] = self._load_json(
                self._manifests_dir / f"{svc}.manifest.json"
            )
        return self._manifest_cache[svc]

    def openapi(self, svc: str) -> dict | None:
        if svc not in self._openapi_cache:
            self._openapi_cache[svc] = self._load_json(
                self._manifests_dir / f"{svc}.openapi.json"
            )
        return self._openapi_cache[svc]

    def _load_json(self, path: Path) -> dict | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
```

- [ ] **Step 4: 运行确认通过**

Run 同上；Expected: `4 passed`（原 2 个 + 新 2 个）。

- [ ] **Step 5: Commit**

```bash
git add gateway/registry.py gateway/tests/test_registry.py
git commit -m "feat(gateway): serve static describe contracts from manifests/"
```

### Task 6: `describe_service` 静态优先

**Files:**
- Modify: `gateway/app.py:86-98`
- Modify: `gateway/tests/test_app.py`

- [ ] **Step 1: 写失败测试**

在 `gateway/tests/test_app.py` 末尾追加（复用现有 `client` fixture，其 `GATEWAY_REGISTRY_PATH` 指向 `tmp_path/services.yaml`，故 manifests 目录为 `tmp_path/manifests`）：
```python
def test_describe_serves_static_manifest(client, tmp_path):
    import server.app as appmod
    from bioq_service.service_registry import ServiceRecord
    appmod.app.state.registry._services = {
        "openbpmd-server": ServiceRecord(url="https://svc.local")
    }
    mdir = tmp_path / "manifests"
    mdir.mkdir()
    (mdir / "openbpmd-server.manifest.json").write_text(
        '{"service": "openbpmd", "endpoints": [{"path": "/api/score"}]}',
        encoding="utf-8",
    )
    (mdir / "openbpmd-server.openapi.json").write_text(
        '{"paths": {"/api/score": {}}}', encoding="utf-8",
    )

    r = client.get("/v1/services/openbpmd-server", headers={"x-test-account": "alice"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["source"] == "registry"
    assert body["status"] == "ok"
    assert body["manifest"]["service"] == "openbpmd"
    assert body["openapi"] == {"paths": {"/api/score": {}}}


def test_describe_falls_back_to_live_when_no_static(client):
    import server.app as appmod
    from bioq_service.service_registry import ServiceRecord
    appmod.app.state.registry._services = {
        "openbpmd-server": ServiceRecord(url="https://svc.local")
    }

    class _Disp:
        def describe_base_url(self, rec):
            return rec.url

    class _Disc:
        def describe(self, svc, base):
            return {"service": svc, "manifest": {}, "openapi": {},
                    "status": "warming", "source": "live"}

    appmod.app.state.dispatch = _Disp()
    appmod.app.state.discover = _Disc()

    r = client.get("/v1/services/openbpmd-server", headers={"x-test-account": "alice"})
    assert r.status_code == 200
    assert r.json()["source"] == "live"
    assert r.json()["status"] == "warming"
```

- [ ] **Step 2: 运行确认失败**

Run:
```bash
cd gateway && uv run --with pytest --with pytest-asyncio python -m pytest tests/test_app.py -q
```
Expected: FAIL（`test_describe_serves_static_manifest` 当前走 live 路径，`source` 不是 registry）。

- [ ] **Step 3: 实现静态优先**

把 `gateway/app.py` 的 `describe_service`（现 86-98 行）改为：
```python
@app.get("/v1/services/{svc}")
def describe_service(svc: str, request: Request,
                     _: AuthIdentity = Depends(require_auth)) -> dict:
    reg = request.app.state.registry
    try:
        rec = reg.record(svc)
    except KeyError:
        raise HTTPException(404, f"unknown service {svc!r}")
    # Static contract first: zero downstream calls, cold-start immune. Fall back
    # to live discovery only for services without committed manifests yet.
    manifest = reg.manifest(svc)
    if manifest is not None:
        return {
            "service": svc,
            "manifest": manifest,
            "openapi": reg.openapi(svc) or {},
            "status": "ok",
            "source": "registry",
        }
    base = request.app.state.dispatch.describe_base_url(rec)
    return request.app.state.discover.describe(svc, base)
```

- [ ] **Step 4: 运行确认通过**

Run 同上；Expected: 全部通过（新增 2 个）。

- [ ] **Step 5: Commit**

```bash
git add gateway/app.py gateway/tests/test_app.py
git commit -m "feat(gateway): serve static manifests first in describe"
```

### Task 7: Dockerfile COPY + Makefile 目标

**Files:**
- Modify: `gateway/Dockerfile:38`（`COPY services.yaml` 之后）
- Modify: `Makefile`

- [ ] **Step 1: Dockerfile**

在 `gateway/Dockerfile` 的 `COPY services.yaml /opt/gateway/services.yaml` 之后新增一行：
```dockerfile
# static describe contracts (see docs/specs/2026-08-20-describe-cold-start-static-manifest-design.md)
COPY manifests/ /opt/gateway/manifests/
```

- [ ] **Step 2: Makefile 增补 `.PHONY`**

把 Makefile 的 `.PHONY:` 行追加 `gen-manifests check-manifests`：
```make
.PHONY: help build push clean list version login-harbor bump sif \
	local-up local-down local-purge local-status local-logs local-test \
	local-info local-forward local-user local-users local-svc local-svcs \
	gen-config check-config gen-manifests check-manifests
```

- [ ] **Step 3: Makefile 新增两个目标**

在 `gen-config` / `check-config` 目标附近新增：
```make
# Static describe contracts — materialize each service's manifest+openapi into
# manifests/ (committed) or verify committed copies are current (CI gate).
# Orchestration runs under the gateway venv (needs pyyaml + framework).
gen-manifests:
	cd gateway && uv run python ../scripts/gen_manifests.py

check-manifests:
	cd gateway && uv run python ../scripts/gen_manifests.py --check
```

- [ ] **Step 4: 验证 `make check-manifests` 通过**

Run:
```bash
make check-manifests
```
Expected: exit 0（此时仅 dockq-server 有已提交 manifest，脚本遍历 services.yaml 会对其余服务报 `MISSING` 并 exit 1）。

    > 说明：全量 `make gen-manifests` 需要每个服务各自的 venv（含 GPU 依赖），是一次 release/CI 动作，不在本计划内跑完全量；本步只验证脚本路径与 Makefile 接线正确。此时 `check-manifests` 预期 exit 1 属**已知的待补全状态**，不是实现缺陷——全量物化见 Task 8 之后的「交底」。

- [ ] **Step 5: Commit**

```bash
git add gateway/Dockerfile Makefile
git commit -m "feat(gateway): ship manifests/ in image and add gen/check-manifests targets"
```

### Task 8: 冒烟复核与全量物化交底

- [ ] **Step 1: 全量回归 gateway 测试**

Run:
```bash
cd gateway && uv run --with pytest --with pytest-asyncio python -m pytest tests/ -q
```
Expected: `all passed`（含新加的 settings/discover/registry/app 测试）。

- [ ] **Step 2: 对照设计文档自检**

逐条核对 `docs/specs/2026-08-20-describe-cold-start-static-manifest-design.md` §成功标准：
    阶段一的三条（≤8s 哨兵 / 单飞 / `--output json` openapi 降级）由 `test_discover.py` 覆盖；
    阶段二的「registry 命中」由 `test_app.py::test_describe_serves_static_manifest` 覆盖。
    注明：`make check-manifests` 目前非 exit 0，因全量 manifest 尚未物化（见下）。

- [ ] **Step 3: Commit（如 Task 8 前有遗漏改动）**

```bash
git status --short   # 确认仅剩本计划预期改动 + dockq manifests
```

---

## 范围外 / 交底（不在本仓库执行）

- **`bioq` CLI 仓库**：client 超时拉长到 > 网关最坏读超时（如 30s）；读 `status`/`detail` 时打印 `"服务冷启动中，约 Ns 后重试"` 替代裸 `ReadTimeout`。现有 `--wait`/`--timeout` 已可复用。
- **esmfold2-server 框架自描述落地**：独立任务；`make check-manifests` 会在全量物化时以 `MISSING` 暴露它。
- **全量物化**：在各服务 release 流水线（依赖齐全）跑 `make gen-manifests`，随后 `make check-manifests` 作为 CI 门禁；`make bump-<svc>` 后必须重跑，否则 CI 报 `STALE`。

## 验证清单（完成判定）

- [ ] `gateway` 测试全绿（`pytest tests/ -q`）。
- [ ] `make check-manifests` 在**全量物化后** exit 0（本计划内仅 dockq 样例，其余为已知 MISSING）。
- [ ] 阶段一行为：`test_discover.py` 覆盖 warming/partial/no_manifest/error 四类 + 负缓存 + 单飞。
- [ ] 阶段二行为：`test_app.py::test_describe_serves_static_manifest` 证明 registry 命中零下游调用。
- [ ] `gateway/Dockerfile` 已 `COPY manifests/`，`manifests/` 非空（含 dockq 样例），`docker build -f gateway/Dockerfile .` 不因空目录失败。