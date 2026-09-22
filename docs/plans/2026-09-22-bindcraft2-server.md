# bindcraft2-server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 BindCraft2 v1.0.1 包装成 `services/bindcraft2-server/`——一个双模式（FastAPI HTTP + `python -m server` CLI 批处理）GPU 服务，暴露 `design` / `rank` / `filter` 三组 endpoint。

**Architecture:** 纯 argv 包装，零 upstream patch。服务把 pydantic 请求编译成上游 campaign JSON，交给 `python -m bindcraft.cli <sub>` 执行；框架的 `JobRunner` 负责 job 目录、子进程、日志、状态。权重（AF2 5.3 GB）外置 NAS，通过 `BINDCRAFT_AF2_PARAMS` 注入；ProteinMPNN 三变体随包。三组 endpoint 各自有 `/api/tasks/*` 孪生。

**Tech Stack:** Python 3.12 / JAX ≥0.11 <0.12（cuda13 extra）/ FastAPI / pydantic v2 / pydantic-settings / uv / Docker（ubuntu:24.04）/ pytest。

**设计文档:** `docs/specs/2026-09-22-bindcraft2-server-design.md`——先读它了解许可证红线、Endpoint 拓扑与显存/时长约束。

**上游 pin:** `PacesaLab/BindCraft2` @ `5342aefa18dedad653f7a5f6dbee1e566ca24d8f`（v1.0.1）

## 读之前必知：两条输入路径的契约不同（踩过一次的坑）

`binder_lengths` / `on` / `where` 是复杂类型，HTTP 与 CLI 两条路径接受的**写法不一样**：

| 路径 | 机制 | 合法写法 | 非法写法 |
|---|---|---|---|
| HTTP 表单 | `forms.py` 在模型校验**之前**对复杂字段 `json.loads` | `-F 'binder_lengths=[80,80]'`、`-F 'on=["i_pTM"]'` | `-F binder_lengths=80,80` → **422 json_invalid** |
| CLI | `cli._add_model_args` 对除 bool/int/float 外一律 `type=str`，从不做 JSON 解析，原样交给 `model_validate` | `--binder-lengths 80,80`、`--on i_pTM,i_pDAE`；**JSON 写法也照收**（`--binder-lengths '[80,80]'` 一样成立，因为模型解码器两种都认） | 无——CLI 上两种写法都合法。区别只在 HTTP 侧**只**认 JSON |

模型里的 `field_validator(mode="before")` 同时解码两种写法，**但 HTTP 上逗号分支永远不会被用到**
（非法 JSON 在进模型前就已 422）。因此：**凡是 HTTP 示例/测试（`endpoint_examples()`、README、
`test_app.py`、`test_fc*.py`）里的复杂字段都必须写成 JSON 字符串**；逗号写法只出现在 CLI 示例与
CLI 测试里。另注：`--on` 不支持重复传参（后者覆盖前者），CLI 上多指标也用一个逗号串。

### 顺带记下的框架级限制（不修，属既有行为）

`framework/src/bioq_service/cli.py::_add_model_args` 从不给"模型里必填、但 argparse
层没标 required"的字段设 `required=True`。因此 `python -m server rank`（漏 `--campaign-uri`）
不会打 usage，而是抛 pydantic `ValidationError` 的整段 traceback、退出码 1。所有服务同样
受影响，与本计划无关；本服务只能对"跨字段条件必填"（`--target` / `--target-name` 二选一）
在 build 回调里自查（见 Task 7）。

---

## 与设计文档的三处偏差（Task 13 回写）

计划阶段读完 framework 真实代码后，有三处需要修正设计文档。**Task 13 负责回写**，实现时以下表为准：

| # | 设计文档写的 | 实际做法 | 原因 |
|---|---|---|---|
| A1 | `binder_lengths` / `on` / `where` 以 **JSON 字符串**表单字段传入 | **`mode="before"` 解码器同时接受两种写法，但两条路上的合法写法不同**：HTTP **只能**用 JSON（`[80,80]`、`["i_pTM"]`），CLI **只能**用逗号分隔（`--on i_pTM --on` 不支持，`80,80` 有效而 `[80,80]` 无效） | `model_form_depends` 在模型校验**之前**就对复杂字段做 `json.loads`，非法 JSON 直接 422，所以模型里的逗号分支永远不会被 HTTP 路径用到；CLI 路径（`cli._add_model_args`）对除 bool/int/float 外的字段一律用 `type=str`，绝不会 `json.loads`。**推论：所有 HTTP 示例（`endpoint_examples()` / README / 测试）里的复杂字段都必须写成 JSON 字符串**，写逗号形式会 422 |
| A2 | `target_name` / `target` / `target_uri` 用 `model_validator(mode="after")` 交叉校验 | 用 `models.validate_target_selection()` 普通函数，在路由层调用 | 上传是路由级 `File(...)` / `Form(...)`，不是 model 字段，`model_validator` 看不到它们 |
| A3 | FC 测试用 `tests/data/mini_target.pdb` 跑 design smoke | FC smoke 改用**shipped target `hPDL1`**；`mini_target.pdb` 只作离线上传路径的 fixture | 手写 PDB 有被上游 preflight 判为畸形结构的风险，会让 FC 测试以误导性方式失败。shipped target 保证可解析 |

---

## 计划期修订（动手前核验仓库约定后加的三项）

首版计划漏了三条仓库既有约定，已改入 Task 1；记在这里以免后续步骤又按旧写法写。

| # | 首版计划写的 | 实际做法 | 原因 |
|---|---|---|---|
| B1 | 只有 `settings.py`，测试命令用 `uv run --with pytest --with fastapi ...` 现搭临时环境 | 建 `pyproject.toml`（`dependencies` / `[dependency-groups] dev` / `[tool.ruff]` / `[tool.uv] package = false`），所有测试命令统一为 `uv run --group dev python -m pytest ...` | 仓库里 **39/39** 服务每个都是独立 uv 项目；AGENTS.md 的离线测试入口就是 `uv run --group dev`。`pyproject.toml` 进版本控制，`uv.lock` 被 gitignore |
| B2 | `tests/conftest.py` 在 Task 6 才建（当时计划 Task 2–5 用 `PYTHONPATH=..` 绕） | conftest 提前到 Task 1，只含 `server` 别名 + fc marker；Task 6 改为**追加**离线 fixture。Task 2–5 跑 `uv run --group dev python -m pytest tests/test_x.py -q` | 服务目录名 `bindcraft2-server` 含连字符，不是合法 Python 标识符；`PYTHONPATH=..` 会指向 `services/`，`import server.models` 必然失败。别名只能由 conftest 提供，且必须早于第一个 `from server.x import y` |
| B3 | `pyproject.toml` 只有 `[tool.ruff] line-length/target-version` | 补 `[tool.ruff.lint]` 并**钉住规则集**：`select = ["E4","E7","E9","F"]` + `extend-select = ["E402"]` | 不钉住就没有可复现的 lint：`uv.lock` 被 gitignore → ruff 版本随每次 `uv sync` 漂移 → ruff 0.16.8 默认启用 **413** 条规则，把 `Optional[X]`（UP045）、FastAPI 的 `Depends()/File()/Form()` 参数默认值（B008）等既有写法一并报错。实测 plan-exact 的 `models.py` 在钉住的规则集下 `All checks passed!`、在 0.16 默认集下报 16 个错。钉住后本计划所有代码无需为 lint 改写 |

---

## 文件结构

```
services/bindcraft2-server/
├── __init__.py                 # 空包标记
├── VERSION                     # v0.0.1（Makefile 读它做镜像 tag）
├── pyproject.toml              # 独立 uv 项目（39 个服务共用的骨架）；uv.lock 被 gitignore
├── settings.py                 # Bindcraft2Settings（env_prefix=BINDCRAFT2_）
├── models.py                   # DesignRequest / RankRequest / FilterRequest + validate_target_selection
├── campaigns.py                # campaign *目录* URI 零拷贝解析（不复用 uris.resolve_uri）
├── tools.py                    # CAMPAIGN_MODELS / build_campaign_json / design|rank|filter_argv / 权重探针
├── adapter.py                  # Bindcraft2Adapter（detect_outputs / subprocess_env / manifest_extras / examples）
├── app.py                      # create_app + 6 endpoints + 自定义 /healthz/detail（GPU 子进程探针）
├── __main__.py                 # CLI 批处理入口（3 个 CLIEndpoint）
├── Dockerfile                  # ubuntu:24.04 + uv venv + jax cuda13 + editable upstream
├── README.md                   # endpoint / 配置 / 权重 / 许可证红线
├── deploy/fc.yaml              # FC 部署描述
├── scripts/vendor.sh           # clone upstream @ pinned SHA → upstream/
├── scripts/fetch_weights.sh    # AF2 7 个 npz → NAS（支持 WEIGHTS_DST）
├── upstream/                   # gitignored：vendor.sh 产物
└── tests/
    ├── __init__.py
    ├── conftest.py             # server 包别名 + fc marker（Task 1）；离线 fixture（Task 6）
    ├── test_settings.py
    ├── test_models.py
    ├── test_campaigns.py
    ├── test_tools.py
    ├── test_adapter.py
    ├── test_app.py             # 离线 HTTP（用 fake_bindcraft stub）
    ├── test_cli.py             # 离线 CLI
    ├── test_fc.py              # @pytest.mark.fc
    ├── test_fc_task.py         # @pytest.mark.fc
    └── data/
        └── fake_bindcraft.sh   # 离线 stub：模拟 design/rank/filter 的产物与 GPU 探针
```

同时修改（仓库根）：

- `.gitignore` — `upstream/` 已被 `services/*-server/upstream/` 覆盖，**无需改动**（Task 1 只做确认）。
- `services.yaml` — 新增 `bindcraft2-server` 条目（Task 11）。
- `THIRD_PARTY_NOTICES` — 新增 BC2 行（Task 11）。

---

### Task 1: 骨架、uv 项目与 settings

**Files:**
- Create: `services/bindcraft2-server/__init__.py`
- Create: `services/bindcraft2-server/VERSION`
- Create: `services/bindcraft2-server/pyproject.toml`
- Create: `services/bindcraft2-server/settings.py`
- Create: `services/bindcraft2-server/tests/__init__.py`
- Create: `services/bindcraft2-server/tests/conftest.py`
- Test: `services/bindcraft2-server/tests/test_settings.py`

**为什么必须建 `pyproject.toml`：** 仓库里 39 个服务每个都有自己的 uv 项目
（`pyproject.toml` 进版本控制，`uv.lock` 被 gitignore）。AGENTS.md 的离线测试入口是
`cd services/<svc>-server && uv run --group dev python -m pytest tests/ -q`——没有
`pyproject.toml` 这一步跑不了。本计划里所有测试命令都走这个入口。

**为什么 `tests/conftest.py` 在 Task 1 就要有：** 服务目录名 `bindcraft2-server` 含连字符，
不是合法 Python 标识符，测试无法直接 `import bindcraft2_server`。镜像里是
`/opt/bindcraft2/server/` + `server.app:app`；本地 pytest 靠 conftest 把该目录按 `server`
这个别名挂进 `sys.modules` 来对齐。Task 2 起每个测试文件都 `from server.models import ...`，
所以别名必须先存在。

- [ ] **Step 1: 建目录与空包标记**

```bash
mkdir -p services/bindcraft2-server/{tests/data,scripts,deploy}
touch services/bindcraft2-server/__init__.py services/bindcraft2-server/tests/__init__.py
printf 'v0.0.1\n' > services/bindcraft2-server/VERSION
```

- [ ] **Step 2: 写 pyproject.toml**

以 `services/rfantibody-server/pyproject.toml` 为模板（39 个服务共用这套骨架）：

```toml
[project]
name = "bindcraft2-server"
version = "0.0.1"
description = "HTTP service wrapping BindCraft2 protein binder design campaigns for FC GPU deployment"
requires-python = ">=3.10"
dependencies = [
    "bioq-service-framework[mcp]",
    "httpx>=0.27",
    "alibabacloud-oss-v2>=0.4",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "ruff>=0.4",
]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
# 显式钉住规则集。原因：uv.lock 被 gitignore，ruff 版本随每次 `uv sync` 漂移，
# 而 ruff 0.16 的默认集比仓库其它 38 个服务写代码时的默认集宽得多（实测启用
# 413 条规则）。不钉住的话，同一份代码在不同机器上 lint 结果不同，并且会要求把
# 既有写法一并改掉（`Optional[X]`、FastAPI 的 `Depends()` / `File()` 作为参数
# 默认值等）。钉在 pycodestyle + Pyflakes 这套长期默认上：结果可复现、与兄弟
# 服务风格一致。注意 N999（连字符目录名）、I001（`server` 别名让 ruff 误判
# `server.*` 为第三方包）、B008（FastAPI 参数默认值里的 Depends/File/Form）都
# 不在这个子集内，所以无需 ignore —— 仓库既有先例 services/seqkit-server 用
# per-file-ignores 处理 B008，是因为它没钉 select。
select = ["E4", "E7", "E9", "F"]
# E402 不在上面的子集里但确实适用：conftest 是"先注册 server 别名、后 import"。
# 显式启用，让既有的 `# noqa: E402` 真正起作用（而不是被 RUF100 判成无用 noqa）。
extend-select = ["E402"]

[tool.uv.sources]
bioq-service-framework = { path = "../../framework", editable = true }

[tool.uv]
package = false
```

**这条 `select` 是硬要求，不是风格偏好。** 本计划里所有「`ruff check` 全绿」的验收
（Task 7 Step 5、Task 13 Step 6）都以它为前提。**不要为了让更宽的规则集通过而去改动
计划给出的代码**——`Optional[X]`、`Depends(...)` / `File(...)` / `Form(...)` 作参数
默认值等写法在本计划里是刻意保留的，实测：plan-exact 的 `models.py` 在钉住的规则集下
`All checks passed!`，在 ruff 0.16 默认集下报 16 个错（全是 UP045 之类的风格改写）。

`package = false` 是关键：服务代码是扁平目录、靠 `PYTHONPATH` + `server` 别名导入，
不作为 wheel 打包。

- [ ] **Step 3: 写 tests/conftest.py（本步只含 server 别名 + fc marker）**

```python
"""测试装置：把服务目录按镜像里的方式挂成 `server` 包。

Dockerfile 把 `services/bindcraft2-server/` 拷到 `/opt/bindcraft2/server/` 并以
`server.app:app` 导入；本地 pytest 用同样的别名，保证 import 路径与生产一致。
"""

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


# `fc` marker：opt-in 的 FC 线上测试。
from bioq_service.fc_testing import (  # noqa: E402
    register_fc_marker,
    skip_fc_tests_unless_enabled,
)


def pytest_configure(config):
    register_fc_marker(config)


def pytest_collection_modifyitems(config, items):
    skip_fc_tests_unless_enabled(config, items)
```

（Task 6 会往这个文件追加 `OfflineSettings` 与 `offline_settings` fixture。）

- [ ] **Step 4: 写 settings.py**

```python
"""bindcraft2-server 的运行时配置。

所有环境变量驱动的配置集中在这里；其余代码读 `settings.foo`，不调用
`os.getenv`。默认值对齐 Docker 镜像布局（`/opt/bindcraft`、`/opt/venv`）。
"""

from __future__ import annotations

from pathlib import Path

from bioq_service import ServiceSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict


class Bindcraft2Settings(ServiceSettings):
    model_config = SettingsConfigDict(
        env_prefix="BINDCRAFT2_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # 覆盖框架默认（/data/jobs）。
    jobs_base_dir: Path = Field(default=Path("/data/bindcraft2_jobs"))

    # 上游源码树。editable install（pip install -e）记录了它的路径，运行时必须存在。
    root: Path = Field(default=Path("/opt/bindcraft"))

    # 解释器。离线测试指向 tests/data/fake_bindcraft.sh。
    python: str = Field(default="/opt/venv/bin/python")

    # 上游 CLI 模块。置空则只执行 `python <args>`（离线 stub 用；上游改名时也可用）。
    module: str = Field(default="bindcraft.cli")

    # AlphaFold 参数目录：其下 params/params_<model>.npz 或 params_<model>.npz。
    # 必须显式设置——否则上游会去下载 5.3 GB。见 fetch_weights.sh / 设计文档 P1。
    alphafold_params_dir: Path = Field(default=Path("/data/models/bindcraft2/alphafold"))

    # 上游源码中的 ProteinMPNN 权重根（随包，26 MB/变体）。
    shipped_weights_dir: Path = Field(default=Path("/opt/bindcraft"))

    # JAX 编译缓存。镜像内没有 nvidia-smi，不设它上游会把图缓存丢进 /tmp，
    # 每次冷启动重付编译成本。
    compile_cache_dir: Path = Field(default=Path("/data/models/bindcraft2/xla_cache"))

    # BC2 会吃掉所有可见卡并按显存 pack worker；FC 一实例一卡，必须独占。
    max_concurrent_jobs: int = Field(default=1, ge=1, le=4)
    workers_per_gpu: int = Field(default=1, ge=1, le=8)
    max_workers_per_gpu: int = Field(default=1, ge=1, le=8)

    # max_trajectories 未由调用方指定时的兜底（上游对尝试次数无上限）。
    default_max_trajectories: int = Field(default=500, ge=1)

    # /healthz/detail 的 GPU 探针缓存秒数。0 = 每次探测。
    gpu_probe_ttl_seconds: int = Field(default=300, ge=0)
```

- [ ] **Step 5: 写 tests/test_settings.py**

```python
"""settings 的默认值与 env 覆盖行为。"""

from __future__ import annotations

from pathlib import Path

import pytest

from server.settings import Bindcraft2Settings


def test_defaults_match_container_layout():
    s = Bindcraft2Settings(_env_file=None)
    assert s.jobs_base_dir == Path("/data/bindcraft2_jobs")
    assert s.root == Path("/opt/bindcraft")
    assert s.shipped_weights_dir == Path("/opt/bindcraft")
    assert s.python == "/opt/venv/bin/python"
    assert s.module == "bindcraft.cli"
    assert s.alphafold_params_dir == Path("/data/models/bindcraft2/alphafold")
    assert s.compile_cache_dir == Path("/data/models/bindcraft2/xla_cache")
    assert s.max_concurrent_jobs == 1
    assert s.workers_per_gpu == 1
    assert s.max_workers_per_gpu == 1
    assert s.default_max_trajectories == 500
    assert s.gpu_probe_ttl_seconds == 300


def test_env_prefix_is_bindcraft2(monkeypatch):
    monkeypatch.setenv("BINDCRAFT2_MAX_CONCURRENT_JOBS", "3")
    monkeypatch.setenv("BINDCRAFT2_DEFAULT_MAX_TRAJECTORIES", "7")
    s = Bindcraft2Settings(_env_file=None)
    assert s.max_concurrent_jobs == 3
    assert s.default_max_trajectories == 7


def test_unknown_env_vars_are_ignored(monkeypatch):
    monkeypatch.setenv("BINDCRAFT2_NOT_A_FIELD", "1")
    assert Bindcraft2Settings(_env_file=None).module == "bindcraft.cli"


def test_bounds_are_enforced():
    with pytest.raises(ValueError):
        Bindcraft2Settings(_env_file=None, workers_per_gpu=0)
    with pytest.raises(ValueError):
        Bindcraft2Settings(_env_file=None, max_concurrent_jobs=99)
```

- [ ] **Step 6: 建环境并跑测试**

```bash
cd services/bindcraft2-server && uv sync --group dev
cd services/bindcraft2-server && uv run --group dev python -m pytest tests/ -q
```

Expected：`4 passed`。`uv sync` 会生成 `uv.lock` 与 `.venv`——两者都被 .gitignore 覆盖
（`uv.lock` 在 "per-service test-env lockfiles" 一节里），不进版本控制。

- [ ] **Step 7: 确认 .gitignore 已覆盖 upstream/**

```bash
grep -n 'services/\*-server/upstream/' .gitignore
```

Expected：`2:services/*-server/upstream/`。已覆盖，无需改动。

- [ ] **Step 8: Commit**

```bash
git add services/bindcraft2-server/__init__.py services/bindcraft2-server/VERSION \
        services/bindcraft2-server/pyproject.toml services/bindcraft2-server/settings.py \
        services/bindcraft2-server/tests/__init__.py services/bindcraft2-server/tests/conftest.py \
        services/bindcraft2-server/tests/test_settings.py
git commit -m "feat(bindcraft2-server): add settings skeleton and uv project"
```

---

### Task 2: models.py（请求模型 + 校验）

**Files:**
- Create: `services/bindcraft2-server/models.py`
- Test: `services/bindcraft2-server/tests/test_models.py`

- [ ] **Step 1: 写失败测试**

```python
"""DesignRequest / RankRequest / FilterRequest 的校验行为。"""

from __future__ import annotations

import pytest

from server.models import (
    DesignRequest,
    FilterRequest,
    RankRequest,
    RankTable,
    validate_target_selection,
)


def test_defaults_are_upstream_defaults():
    req = DesignRequest()
    assert req.modality == "binder"
    assert req.number_of_final_designs == 10
    assert req.max_trajectories is None
    assert req.properties == []


def test_properties_lists_only_enabled_flags():
    req = DesignRequest(humanize=True, bigbang=True)
    assert req.properties == ["humanize", "bigbang"]


def test_binder_lengths_accepts_list():
    assert DesignRequest(binder_lengths=[60, 100]).binder_lengths == [60, 100]


def test_binder_lengths_accepts_comma_string():
    # CLI 路径把 list 字段当 type=str 传入。
    assert DesignRequest(binder_lengths="60,100").binder_lengths == [60, 100]


def test_binder_lengths_accepts_json_string():
    # HTTP multipart 路径经 model_form_depends 的 json.loads 后应同时兼容。
    assert DesignRequest(binder_lengths="[60,100]").binder_lengths == [60, 100]


def test_binder_lengths_rejects_descending():
    with pytest.raises(ValueError, match="ascending"):
        DesignRequest(binder_lengths=[100, 60])


def test_binder_lengths_rejects_three_entries():
    with pytest.raises(ValueError, match="1 or 2"):
        DesignRequest(binder_lengths=[60, 70, 80])


def test_binder_lengths_rejects_non_positive():
    # 覆盖 `any(value < 1 ...)` 分支——少了这个测试，删掉该分支所有测试仍全绿
    # （Task 2 实现者用变异测试证明了这个缺口）。
    with pytest.raises(ValueError, match="must be >= 1"):
        DesignRequest(binder_lengths=[0])
    with pytest.raises(ValueError, match="must be >= 1"):
        DesignRequest(binder_lengths=[-5, 80])


def test_binder_lengths_rejects_empty_list():
    # 钉住 `is not None` 语义：改成真值判断会让 [] 被放行。
    with pytest.raises(ValueError, match="1 or 2"):
        DesignRequest(binder_lengths=[])


def test_binder_lengths_rejects_float_and_bool():
    # 钉住严格整数解析：`int(80.5)` 会静默截断成 80，必须报错而不是改值。
    with pytest.raises(ValueError, match="must be integers"):
        DesignRequest(binder_lengths=[80.5, 90.5])
    with pytest.raises(ValueError, match="must be integers"):
        DesignRequest(binder_lengths="[80.5,90.5]")
    with pytest.raises(ValueError, match="must be integers"):
        DesignRequest(binder_lengths=[True, True])


def test_max_trajectories_must_be_positive():
    with pytest.raises(ValueError):
        DesignRequest(max_trajectories=0)


def test_max_trajectories_is_bounded():
    # 上界兜住共享卡上的 GPU 成本：没有它 10**30 会被原样写进 campaign。
    assert DesignRequest(max_trajectories=100_000).max_trajectories == 100_000
    with pytest.raises(ValueError):
        DesignRequest(max_trajectories=100_001)


def test_top_must_be_positive():
    with pytest.raises(ValueError):
        RankRequest(campaign_uri="job://abc", top=0)
    with pytest.raises(ValueError):
        FilterRequest(campaign_uri="job://abc", top=0)


def test_binder_lengths_rejects_non_integer():
    with pytest.raises(ValueError, match="integers"):
        DesignRequest(binder_lengths="sixty,80")


def test_number_of_final_designs_bounds():
    with pytest.raises(ValueError):
        DesignRequest(number_of_final_designs=0)
    with pytest.raises(ValueError):
        DesignRequest(number_of_final_designs=1001)


def test_campaign_name_pattern():
    assert DesignRequest(campaign_name="pdl1_v2").campaign_name == "pdl1_v2"
    with pytest.raises(ValueError):
        DesignRequest(campaign_name="has spaces")


def test_rank_request_defaults():
    req = RankRequest(campaign_uri="job://abc")
    assert req.on == ["i_pDAE"]
    assert req.table is RankTable.accepted
    assert req.lowest_first is None


def test_rank_request_parses_comma_string_metrics():
    assert RankRequest(campaign_uri="job://abc", on="i_pTM,i_pDAE").on == ["i_pTM", "i_pDAE"]


def test_rank_request_rejects_empty_metrics():
    with pytest.raises(ValueError, match="at least one metric"):
        RankRequest(campaign_uri="job://abc", on=[])


def test_filter_request_parses_comma_string_where():
    req = FilterRequest(campaign_uri="job://abc", where="i_pAE=0.45,Interface_Residues>=7")
    assert req.where == ["i_pAE=0.45", "Interface_Residues>=7"]


def test_filter_request_defaults_to_candidates_table():
    assert FilterRequest(campaign_uri="job://abc").table is RankTable.candidates


def test_validate_target_selection_accepts_exactly_one():
    validate_target_selection("hPDL1", False, None)
    validate_target_selection(None, True, None)
    validate_target_selection(None, False, "job://abc/target.pdb")


def test_validate_target_selection_rejects_none():
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        validate_target_selection(None, False, None)
    assert exc.value.status_code == 422


def test_validate_target_selection_rejects_two():
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        validate_target_selection("hPDL1", True, None)
    assert exc.value.status_code == 422
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cd services/bindcraft2-server && uv run --group dev python -m pytest tests/test_models.py -q
```

Expected：collection error — `ModuleNotFoundError: No module named 'server.models'`
（`models.py` 还没写；`server` 别名由 Task 1 的 conftest 提供，所以报的是子模块而不是包）。

- [ ] **Step 3: 写 models.py**

```python
"""各 endpoint 的 pydantic 请求模型。

文件输入（target 上传 / target_uri）走路由层 `File(...)` / `Form(...)`，不在
model 上——因此"三选一"校验是 `validate_target_selection()` 这个普通函数，
在路由里调用，而不是 `model_validator`。
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Optional

from fastapi import HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator

__all__ = [
    "PROPERTY_FIELDS",
    "DesignRequest",
    "FilterRequest",
    "RankRequest",
    "RankTable",
    "validate_target_selection",
]

# 长度解析失败的统一报错文案（两处 raise 共用，测试按它匹配）。
_BINDER_LENGTHS_MESSAGE = (
    "binder_lengths must be integers, e.g. '80,80' or '[60,100]'"
)

# 与上游 campaign JSON 顶层 key 一一对应的 9 个可选属性。
PROPERTY_FIELDS: tuple[str, ...] = (
    "forced_targeting",
    "humanize",
    "protease_stable",
    "disulfide_staple",
    "mixed_topology",
    "termini_together",
    "termini_accessible",
    "initial_guess",
    "bigbang",
)


class RankTable(str, Enum):
    """可选的源表（上游 `rank --table` / `filter --table`）。"""

    accepted = "accepted"
    candidates = "candidates"
    trajectories = "trajectories"


def _decode_list(value: Any) -> Any:
    """把 JSON 字符串或逗号分隔字符串解码成 list。

    HTTP 路径：`model_form_depends` 对复杂字段做 `json.loads`，拿到 list。
    CLI 路径：`cli._add_model_args` 对非 scalar 字段用 `type=str`，拿到原始字符串。
    两条路都要能走通，所以这里同时接受两种写法。
    """
    if value is None or isinstance(value, list):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            return json.loads(text)
        return [part.strip() for part in text.split(",") if part.strip()]
    return value


def validate_target_selection(
    target_name: Optional[str],
    has_upload: bool,
    target_uri: Optional[str],
) -> None:
    """`target_name` / `target` 上传 / `target_uri` 恰好提供一个。"""
    provided = [bool(target_name), bool(has_upload), bool(target_uri)]
    if sum(provided) != 1:
        raise HTTPException(
            status_code=422,
            detail=(
                "Exactly one of `target_name`, `target` (file upload) or `target_uri` "
                "is required."
            ),
        )


class DesignRequest(BaseModel):
    """`POST /api/design` 的参数；目标结构走路由层上传/URI。"""

    target_name: Optional[str] = Field(
        default=None,
        description="上游 shipped target 名（bindcraft design --list-targets）。",
    )
    target_chains: Optional[str] = Field(
        default=None, description="目标链，如 'A' 或 'A,B'。", examples=["A"]
    )
    hotspots: Optional[str] = Field(
        default=None,
        description="结合位点残基，编号取自输入结构。",
        examples=["54,56,66-70"],
    )
    coldspots: Optional[str] = Field(
        default=None, description="要求保持自由的区域。", examples=["90-95"]
    )
    modality: str = Field(
        default="binder",
        description="上游 modality 名，允许逗号组合（binder,VHH,ARP,scFv,Fab,...）。",
    )
    binder_lengths: Optional[list[int]] = Field(
        default=None,
        description="[80,80] 定长；[60,100] 范围。scaffold 模态不需要。",
    )
    number_of_final_designs: int = Field(
        default=10, ge=1, le=1000, description="收够 N 个 accepted 即停。"
    )
    max_trajectories: Optional[int] = Field(
        default=None,
        ge=1,
        le=100_000,
        # 上界用来兜住共享卡上的 GPU 成本：不给上限时 10**30 这类值会被原样写进
        # campaign，等于一次无界 GPU 运行（服务端默认 500，10 万已经很宽松）。
        description=(
            "尝试次数硬上限；未给则用服务端 default_max_trajectories。上限 100000。"
        ),
    )

    forced_targeting: bool = False
    humanize: bool = False
    protease_stable: bool = False
    disulfide_staple: bool = False
    mixed_topology: bool = False
    termini_together: bool = False
    termini_accessible: bool = False
    initial_guess: bool = False
    bigbang: bool = False

    core: Optional[str] = Field(
        default=None, description="上游 core profile，如 'benchmark'。"
    )
    campaign_name: Optional[str] = Field(
        default=None,
        pattern=r"^[A-Za-z0-9_-]{1,64}$",
        description="结果命名与元数据标签；未给则用 job_id。",
    )

    @field_validator("binder_lengths", mode="before")
    @classmethod
    def _decode_binder_lengths(cls, value: Any) -> Any:
        decoded = _decode_list(value)
        if decoded is None:
            return None
        lengths: list[int] = []
        for item in decoded:
            # bool 是 int 的子类，必须先拦掉；float 也必须拦——`int(80.5)` 会静默
            # 截断成 80，值变了却不报错，是最坏的一类错误。
            if isinstance(item, bool) or not isinstance(item, (int, str)):
                raise ValueError(_BINDER_LENGTHS_MESSAGE)
            text = str(item).strip()
            if not text.lstrip("+-").isdigit():
                raise ValueError(_BINDER_LENGTHS_MESSAGE)
            lengths.append(int(text))
        return lengths

    @model_validator(mode="after")
    def _check_binder_lengths(self) -> "DesignRequest":
        if self.binder_lengths is not None:
            if len(self.binder_lengths) not in (1, 2):
                raise ValueError("binder_lengths must have 1 or 2 entries")
            if any(value < 1 for value in self.binder_lengths):
                raise ValueError("binder_lengths entries must be >= 1")
            if (
                len(self.binder_lengths) == 2
                and self.binder_lengths[0] > self.binder_lengths[1]
            ):
                raise ValueError("binder_lengths must be ascending: [min, max]")
        return self

    @property
    def properties(self) -> list[str]:
        """已开启的属性名，顺序与 PROPERTY_FIELDS 一致。"""
        return [name for name in PROPERTY_FIELDS if getattr(self, name)]


class RankRequest(BaseModel):
    """`POST /api/rank` 的参数。"""

    campaign_uri: str = Field(
        description="源 campaign 目录：job://<id>、job://<id>/output、file:///abs 或裸绝对路径。"
    )
    on: list[str] = Field(default=["i_pDAE"], description="排序指标；多个用于 tie-break。")
    lowest_first: Optional[bool] = Field(
        default=None, description="为 None 时用上游自动方向判定。"
    )
    table: RankTable = RankTable.accepted
    top: Optional[int] = Field(default=None, ge=1, description="控制台显示行数上限。")

    @field_validator("on", mode="before")
    @classmethod
    def _decode_on(cls, value: Any) -> Any:
        return _decode_list(value)

    @model_validator(mode="after")
    def _check_on(self) -> "RankRequest":
        if not self.on:
            raise ValueError("on must contain at least one metric")
        return self


class FilterRequest(BaseModel):
    """`POST /api/filter` 的参数。"""

    campaign_uri: str = Field(
        description="源 campaign 目录：job://<id>、job://<id>/output、file:///abs 或裸绝对路径。"
    )
    where: Optional[list[str]] = Field(
        default=None,
        description="阈值表达式；空表示用 campaign 自身阈值重放。",
        examples=["i_pAE=0.45", "Interface_Residues>=7"],
    )
    table: RankTable = RankTable.candidates
    top: Optional[int] = Field(default=None, ge=1, description="控制台显示行数上限。")

    @field_validator("where", mode="before")
    @classmethod
    def _decode_where(cls, value: Any) -> Any:
        return _decode_list(value)
```

- [ ] **Step 4: 跑测试确认通过**

```bash
cd services/bindcraft2-server && uv run --group dev python -m pytest tests/test_models.py -q
```

Expected：`24 passed`。

- [ ] **Step 5: Commit**

```bash
git add services/bindcraft2-server/models.py services/bindcraft2-server/tests/test_models.py
git commit -m "feat(bindcraft2-server): add request models with offline-testable validation"
```

---

### Task 3: campaigns.py（campaign 目录零拷贝解析）

**Files:**
- Create: `services/bindcraft2-server/campaigns.py`
- Test: `services/bindcraft2-server/tests/test_campaigns.py`

- [ ] **Step 1: 写失败测试**

```python
"""campaign 目录 URI 解析：零拷贝 + 目录穿越防护。"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException
from pydantic_settings import SettingsConfigDict

from server.campaigns import resolve_campaign_dir
from server.settings import Bindcraft2Settings


class _Off(Bindcraft2Settings):
    model_config = SettingsConfigDict(
        env_prefix="BINDCRAFT2_TEST_", env_file=None, extra="ignore", case_sensitive=False
    )


@pytest.fixture
def settings(tmp_path: Path) -> _Off:
    return _Off(jobs_base_dir=tmp_path / "jobs")


def _make_campaign(settings, job_id: str) -> Path:
    out = settings.jobs_base_dir / job_id / "output"
    (out / "3_Ranked").mkdir(parents=True)
    (out / "3_Ranked" / "!_Ranked.csv").write_text("design,i_pDAE\n", encoding="utf-8")
    return out


def test_job_uri_resolves_to_output_dir(settings):
    out = _make_campaign(settings, "abc123")
    assert resolve_campaign_dir("job://abc123", settings) == out


def test_job_uri_with_redundant_output_suffix(settings):
    out = _make_campaign(settings, "abc123")
    assert resolve_campaign_dir("job://abc123/output", settings) == out


def test_job_uri_with_subdirectory(settings):
    out = _make_campaign(settings, "abc123")
    sub = out / "3_Ranked"
    assert resolve_campaign_dir("job://abc123/3_Ranked", settings) == sub


def test_job_uri_is_not_copied(settings):
    """零拷贝：返回的必须是 NAS 上的原路径，不是副本。"""
    out = _make_campaign(settings, "abc123")
    resolved = resolve_campaign_dir("job://abc123", settings)
    assert resolved == out
    assert resolved.is_dir()


def test_job_uri_accepts_redundant_output_suffix_with_subdir(settings):
    out = _make_campaign(settings, "abc123")
    assert resolve_campaign_dir("job://abc123/output/3_Ranked", settings) == out / "3_Ranked"


def test_job_uri_rejects_dotdot_job_id(settings):
    """`job://..` 会让 root 落到 <jobs_base_dir>/../output——必须在拼接前拒掉。"""
    _make_campaign(settings, "abc123")
    for uri in ("job://..", "job://../secret", "job://."):
        with pytest.raises(HTTPException) as exc:
            resolve_campaign_dir(uri, settings)
        assert exc.value.status_code == 422, uri


def test_job_uri_does_not_strip_output_lookalike(settings):
    """只认整段 `output`；`output2` 不能被当作冗余前缀剥掉。"""
    _make_campaign(settings, "abc123")
    with pytest.raises(HTTPException) as exc:
        resolve_campaign_dir("job://abc123/output2", settings)
    assert exc.value.status_code == 404


def test_job_uri_rejects_symlinked_output_dir(settings):
    """`output/` 是指向树外的符号链接时必须 422。

    这条钉住的是 root 相对 jobs_base_dir 的包含性检查——少了它，`.resolve()`
    会跟随符号链接把 root 定到树外，`job://abc123` 就会直接返回外部目录。
    """
    outside = settings.jobs_base_dir.parent / "outside"
    outside.mkdir()
    job = settings.jobs_base_dir / "abc123"
    job.mkdir(parents=True)
    (job / "output").symlink_to(outside, target_is_directory=True)
    with pytest.raises(HTTPException) as exc:
        resolve_campaign_dir("job://abc123", settings)
    assert exc.value.status_code == 422


def test_job_uri_rejects_traversal(settings):
    _make_campaign(settings, "abc123")
    with pytest.raises(HTTPException) as exc:
        resolve_campaign_dir("job://abc123/../../../etc", settings)
    assert exc.value.status_code == 422


def test_absolute_path_passthrough(settings, tmp_path: Path):
    out = _make_campaign(settings, "abc123")
    assert resolve_campaign_dir(str(out), settings) == out


def test_file_uri_passthrough(settings):
    out = _make_campaign(settings, "abc123")
    assert resolve_campaign_dir(f"file://{out}", settings) == out


def test_missing_directory_is_404(settings):
    with pytest.raises(HTTPException) as exc:
        resolve_campaign_dir("job://nope", settings)
    assert exc.value.status_code == 404


def test_unsupported_scheme_is_422(settings):
    with pytest.raises(HTTPException) as exc:
        resolve_campaign_dir("oss://bucket/key", settings)
    assert exc.value.status_code == 422


def test_empty_uri_is_422(settings):
    with pytest.raises(HTTPException) as exc:
        resolve_campaign_dir("   ", settings)
    assert exc.value.status_code == 422
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cd services/bindcraft2-server && uv run --group dev python -m pytest tests/test_campaigns.py -q
```

Expected：`ModuleNotFoundError: No module named 'server.campaigns'`。

- [ ] **Step 3: 写 campaigns.py**

```python
"""把 campaign *目录* URI 解析成本地路径——零拷贝。

`bioq_service.uris.resolve_uri` 是文件级的，且 `shutil.copy2` 源文件。campaign
目录可达数百 MB（`3_Ranked/*.cif`），拷贝不可接受。本模块直接返回 NAS 上的原
路径；调用方依赖 NAS 是共享挂载。

安全性：`job://` 分支只允许落在该 job 的 `output/` 目录内，且 `job_id` 必须是
单个普通路径段（拒 `.` / `..`）。`file://` 与裸绝对路径分支**故意不做白名单**：
网关会把 `oss://` 输入重写成 `/mnt/oss/...` 的裸绝对路径，且本服务按设计仅限
组织内部使用（见设计文档的许可证边界），调用方已被信任。
"""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException

from .settings import Bindcraft2Settings

__all__ = ["OUTPUT_DIRNAME", "resolve_campaign_dir"]

# job 目录下存放上游产物的子目录名；`job://<id>` 与 `job://<id>/output` 同义。
OUTPUT_DIRNAME = "output"


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _normalize_subpath(sub: str) -> str:
    """去掉冗余的前导 `output` 段。

    `job://<id>`、`job://<id>/output`、`job://<id>/output/3_Ranked` 与
    `job://<id>/3_Ranked` 指向同一处。只认整段 `output`——`output2` 不受影响。
    """
    if sub == OUTPUT_DIRNAME:
        return ""
    prefix = OUTPUT_DIRNAME + "/"
    if sub.startswith(prefix):
        return sub[len(prefix) :]
    return sub


def resolve_campaign_dir(campaign_uri: str, settings: Bindcraft2Settings) -> Path:
    """`job://<id>[/<sub>]` / `file://<abs>` / 裸绝对路径 → 已存在的目录路径。

    不支持 `oss://` 与 `http(s)://`：它们是单对象语义，且网关把 `oss://` 输入
    重写成 `/mnt/oss/...` 的裸绝对路径，所以裸路径分支已覆盖网关场景。
    """
    uri = (campaign_uri or "").strip()
    if not uri:
        raise HTTPException(status_code=422, detail="campaign_uri is required.")

    if uri.startswith("job://"):
        body = uri[len("job://") :]
        job_id, _, sub = body.partition("/")
        # job_id 必须是单个普通路径段。在拼接之前就拒掉 `.` / `..`：否则
        # `job://..` 会让 root 落到 `<jobs_base_dir>/../output`，逃出 jobs 树。
        if not job_id or job_id in {".", ".."}:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid job URI; expected job://<job_id>[/<subdir>]: {uri}",
            )
        jobs_root = settings.jobs_base_dir.resolve()
        root = (jobs_root / job_id / OUTPUT_DIRNAME).resolve()
        # 双保险：即便 job_id 校验被绕过，也不允许 root 逃出 jobs_base_dir。
        if not _is_within(root, jobs_root):
            raise HTTPException(
                status_code=422,
                detail=f"job:// path escapes the jobs base directory: {uri}",
            )
        sub = _normalize_subpath(sub)
        path = (root / sub).resolve() if sub else root
        if not _is_within(path, root):
            raise HTTPException(
                status_code=422,
                detail=f"job:// path escapes the job output directory: {uri}",
            )
    elif uri.startswith("file://") or uri.startswith("/"):
        path = Path(uri[len("file://") :] if uri.startswith("file://") else uri)
    else:
        raise HTTPException(
            status_code=422,
            detail=(
                "Unsupported campaign URI scheme; expected job://, file:// or an "
                f"absolute path: {uri}"
            ),
        )

    if not path.is_dir():
        raise HTTPException(
            status_code=404, detail=f"Campaign directory not found: {path}"
        )
    return path
```

- [ ] **Step 4: 跑测试确认通过**

```bash
cd services/bindcraft2-server && uv run --group dev python -m pytest tests/test_campaigns.py -q
```

Expected：`14 passed`。

- [ ] **Step 5: Commit**

```bash
git add services/bindcraft2-server/campaigns.py services/bindcraft2-server/tests/test_campaigns.py
git commit -m "feat(bindcraft2-server): add zero-copy campaign directory resolver"
```

---

### Task 4: tools.py（campaign 拼装 + argv + 权重探针）

**Files:**
- Create: `services/bindcraft2-server/tools.py`
- Test: `services/bindcraft2-server/tests/test_tools.py`

- [ ] **Step 1: 写失败测试**

```python
"""campaign JSON 拼装、三组 argv、权重探针。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic_settings import SettingsConfigDict

from server.models import DesignRequest, FilterRequest, RankRequest
from server.settings import Bindcraft2Settings
from server.tools import (
    CAMPAIGN_MODELS,
    build_campaign_json,
    design_argv,
    filter_argv,
    missing_alphafold_params,
    missing_proteinmpnn_weights,
    prepare_design,
    rank_argv,
    write_campaign_file,
)


class _Off(Bindcraft2Settings):
    model_config = SettingsConfigDict(
        env_prefix="BINDCRAFT2_TEST_", env_file=None, extra="ignore", case_sensitive=False
    )


@pytest.fixture
def settings(tmp_path: Path) -> _Off:
    return _Off(
        jobs_base_dir=tmp_path / "jobs",
        root=tmp_path / "root",
        python="/usr/bin/python3",
        alphafold_params_dir=tmp_path / "af",
    )


@pytest.fixture
def job_dir(tmp_path: Path) -> Path:
    d = tmp_path / "jobs" / "j1"
    (d / "output").mkdir(parents=True)
    return d


def test_campaign_models_match_upstream_order():
    assert CAMPAIGN_MODELS == (
        "model_1_multimer_v3",
        "model_2_multimer_v3",
        "model_3_multimer_v3",
        "model_4_multimer_v3",
        "model_5_multimer_v3",
        "model_1_ptm",
        "model_2_ptm",
    )


def test_shipped_target_campaign(job_dir):
    campaign = build_campaign_json(
        DesignRequest(target_name="hPDL1", number_of_final_designs=5),
        job_dir=job_dir,
        target_path=None,
        max_trajectories=50,
    )
    assert campaign["target"] == "hPDL1"
    assert "targets" not in campaign
    assert campaign["number_of_final_designs"] == 5
    assert campaign["max_trajectories"] == 50
    assert campaign["modality"] == "binder"
    assert campaign["project_folder"] == str((job_dir / "output").resolve())


def test_uploaded_target_campaign_uses_service_generated_name(job_dir):
    target = job_dir / "input" / "target.pdb"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("ATOM\n", encoding="utf-8")

    campaign = build_campaign_json(
        DesignRequest(target_chains="A", hotspots="54,56", coldspots="90-95"),
        job_dir=job_dir,
        target_path=target,
        max_trajectories=10,
    )
    entry = campaign["targets"][0]
    assert entry["name"] == "target"  # 不取用户文件名
    assert entry["target_path"] == str(target.resolve())
    assert entry["chains"] == "A"
    assert entry["hotspots"] == "54,56"
    assert entry["coldspots"] == "90-95"


def test_campaign_omits_unset_optionals(job_dir):
    campaign = build_campaign_json(
        DesignRequest(target_name="hPDL1"), job_dir=job_dir, target_path=None, max_trajectories=1
    )
    assert "binder_lengths" not in campaign
    assert "core" not in campaign
    assert "humanize" not in campaign
    assert campaign["campaign_name"] == "j1"  # 未给则用 job_id


def test_campaign_writes_enabled_properties(job_dir):
    campaign = build_campaign_json(
        DesignRequest(target_name="hPDL1", humanize=True, bigbang=True, binder_lengths=[60, 60]),
        job_dir=job_dir,
        target_path=None,
        max_trajectories=1,
    )
    assert campaign["humanize"] is True
    assert campaign["bigbang"] is True
    assert "protease_stable" not in campaign
    assert campaign["binder_lengths"] == [60, 60]


def test_write_campaign_file_roundtrips(job_dir):
    path = write_campaign_file({"target": "hPDL1"}, job_dir)
    assert path == job_dir / "input" / "campaign.json"
    assert json.loads(path.read_text(encoding="utf-8")) == {"target": "hPDL1"}


def test_design_argv(settings, job_dir):
    argv = design_argv(job_dir / "input" / "campaign.json", job_dir, settings)
    assert argv == [
        "/usr/bin/python3",
        "-m",
        "bindcraft.cli",
        "design",
        str((job_dir / "input" / "campaign.json").resolve()),
    ]


def test_runner_prefix_omits_module_when_blank(tmp_path, job_dir):
    s = _Off(python="/bin/true", module="")
    argv = design_argv(job_dir / "input" / "campaign.json", job_dir, s)
    assert argv[0] == "/bin/true"
    assert "-m" not in argv


def test_prepare_design_returns_written_campaign(settings, job_dir):
    path = prepare_design(
        DesignRequest(target_name="hPDL1"), job_dir=job_dir, target_path=None, settings=settings
    )
    campaign = json.loads(path.read_text(encoding="utf-8"))
    assert campaign["target"] == "hPDL1"
    # 未指定 max_trajectories 时用 settings 兜底
    assert campaign["max_trajectories"] == settings.default_max_trajectories


def test_rank_argv_maps_every_field(settings, job_dir, tmp_path):
    campaign = tmp_path / "src"
    campaign.mkdir()
    argv = rank_argv(
        RankRequest(campaign_uri=str(campaign), on=["i_pTM", "i_pDAE"], lowest_first=True, top=20),
        campaign,
        job_dir,
        settings,
    )
    joined = " ".join(argv)
    assert "rank" in argv
    assert "--on i_pTM --on i_pDAE" in joined
    assert "--table accepted" in joined
    assert "--lowest-first" in joined
    assert "--top 20" in joined
    assert argv[argv.index("--output") + 1] == str((job_dir / "output" / "ranked_by_i_pTM.csv").resolve())


def test_rank_argv_omits_optional_flags(settings, job_dir, tmp_path):
    campaign = tmp_path / "src"
    campaign.mkdir()
    argv = rank_argv(RankRequest(campaign_uri=str(campaign)), campaign, job_dir, settings)
    assert "--lowest-first" not in argv
    assert "--highest-first" not in argv
    assert "--top" not in argv


def test_rank_argv_uses_highest_first_when_false(settings, job_dir, tmp_path):
    campaign = tmp_path / "src"
    campaign.mkdir()
    argv = rank_argv(
        RankRequest(campaign_uri=str(campaign), lowest_first=False), campaign, job_dir, settings
    )
    assert "--highest-first" in argv


def test_filter_argv_maps_every_field(settings, job_dir, tmp_path):
    campaign = tmp_path / "src"
    campaign.mkdir()
    argv = filter_argv(
        FilterRequest(campaign_uri=str(campaign), where=["i_pAE=0.45", "Interface_Residues>=7"], top=5),
        campaign,
        job_dir,
        settings,
    )
    joined = " ".join(argv)
    assert "filter" in argv
    assert "--where i_pAE=0.45 --where Interface_Residues>=7" in joined
    assert "--table candidates" in joined
    assert "--top 5" in joined
    assert argv[argv.index("--output") + 1] == str((job_dir / "output" / "filtered.csv").resolve())


def test_filter_argv_without_where(settings, job_dir, tmp_path):
    campaign = tmp_path / "src"
    campaign.mkdir()
    argv = filter_argv(FilterRequest(campaign_uri=str(campaign)), campaign, job_dir, settings)
    assert "--where" not in argv


def test_missing_alphafold_params_reports_all_seven(tmp_path):
    missing = missing_alphafold_params(tmp_path / "af")
    assert len(missing) == 7
    assert "params_model_1_multimer_v3.npz" in missing


@pytest.mark.parametrize(
    "layout",
    [
        "params/params_{model}.npz",
        "params_{model}.npz",
        "params/{model}.npz",
        "{model}.npz",
    ],
)
def test_missing_alphafold_params_accepts_every_layout(tmp_path, layout):
    """四种候选布局都要认——Task 9/12 才会确定 NAS 上真实是哪一种。"""
    af = tmp_path / "af"
    for name in CAMPAIGN_MODELS:
        path = af / layout.format(model=name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * (101 << 20))
    assert missing_alphafold_params(af) == []


def test_missing_alphafold_params_flags_truncated(tmp_path):
    af = tmp_path / "af" / "params"
    af.mkdir(parents=True)
    for name in CAMPAIGN_MODELS:
        (af / f"params_{name}.npz").write_bytes(b"x" * (101 << 20))
    (af / "params_model_1_multimer_v3.npz").write_bytes(b"x")  # 中断的下载
    assert missing_alphafold_params(tmp_path / "af") == ["params_model_1_multimer_v3.npz"]


def test_missing_proteinmpnn_weights(tmp_path):
    root = tmp_path / "root"
    for variant in ("neutral", "negative", "positive"):
        d = root / "bindcraft" / "weights" / "proteinmpnn" / f"weights_{variant}"
        d.mkdir(parents=True)
        (d / "v_48_020.npz").write_bytes(b"x" * (2 << 20))
    assert missing_proteinmpnn_weights(root) == []

    (root / "bindcraft" / "weights" / "proteinmpnn" / "weights_positive" / "v_48_020.npz").unlink()
    assert missing_proteinmpnn_weights(root) == ["weights_positive/v_48_020.npz"]


def test_missing_proteinmpnn_weights_flags_truncated(tmp_path):
    """半截下载的 checkpoint 必须算缺失，否则要跑到一半才炸。"""
    root = tmp_path / "root"
    for variant in ("neutral", "negative", "positive"):
        d = root / "bindcraft" / "weights" / "proteinmpnn" / f"weights_{variant}"
        d.mkdir(parents=True)
        (d / "v_48_020.npz").write_bytes(b"x" * (2 << 20))
    d = root / "bindcraft" / "weights" / "proteinmpnn" / "weights_negative"
    (d / "v_48_020.npz").write_bytes(b"x")
    assert missing_proteinmpnn_weights(root) == ["weights_negative/v_48_020.npz"]
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cd services/bindcraft2-server && uv run --group dev python -m pytest tests/test_tools.py -q
```

Expected：`ModuleNotFoundError: No module named 'server.tools'`。

- [ ] **Step 3: 写 tools.py**

```python
"""campaign 文件拼装与 argv 构造。

本模块只构造 argv，从不 spawn 进程——子进程生命周期归框架
（`bioq_service.runner.SubprocessRunner`）。
"""

from __future__ import annotations

import json
from pathlib import Path

from .models import DesignRequest, FilterRequest, RankRequest
from .settings import Bindcraft2Settings

__all__ = [
    "CAMPAIGN_MODELS",
    "DESIGN_RANKED_CSV",
    "DESIGN_SUMMARY_CSV",
    "FILTER_OUTPUT",
    "MPNN_VARIANTS",
    "build_campaign_json",
    "design_argv",
    "filter_argv",
    "missing_alphafold_params",
    "missing_proteinmpnn_weights",
    "prepare_design",
    "rank_argv",
    "write_campaign_file",
]

# 上游 bindcraft/model_weights.py: CAMPAIGN_MODELS —— 5 个 multimer + 2 个 monomer。
CAMPAIGN_MODELS: tuple[str, ...] = tuple(
    f"model_{index}_multimer_v3" for index in range(1, 6)
) + ("model_1_ptm", "model_2_ptm")

MPNN_VARIANTS: tuple[str, ...] = ("neutral", "negative", "positive")
MPNN_CHECKPOINT = "v_48_020.npz"

# 上游输出约定（docs/outputs.md）。
DESIGN_RANKED_CSV = "3_Ranked/!_Ranked.csv"
DESIGN_SUMMARY_CSV = "summary.csv"
TRAJECTORIES_CSV = "1_Trajectories/!_Trajectories.csv"
CAMPAIGN_METADATA_JSON = "campaign_metadata.json"
RANK_OUTPUT_TEMPLATE = "ranked_by_{metric}.csv"
FILTER_OUTPUT = "filtered.csv"


def runner_prefix(settings: Bindcraft2Settings) -> list[str]:
    """`[python, -m, module]`；module 为空时退化成 `[python]`（离线 stub / 上游改名）。"""
    if settings.module:
        return [settings.python, "-m", settings.module]
    return [settings.python]


def alphafold_parameter_path(parameters_dir: Path, model_name: str) -> Path | None:
    """按上游 `alphafold_parameter_file` 的候选顺序查找，返回第一个存在的。"""
    for candidate in (
        parameters_dir / "params" / f"params_{model_name}.npz",
        parameters_dir / f"params_{model_name}.npz",
        parameters_dir / "params" / f"{model_name}.npz",
        parameters_dir / f"{model_name}.npz",
    ):
        if candidate.is_file():
            return candidate
    return None


def missing_alphafold_params(
    parameters_dir: Path, *, floor_bytes: int = 100 << 20
) -> list[str]:
    """返回缺失或体积不足（下载中断）的检查点名。

    上游对 AlphaFold 检查点用 100 MB 下限判"未完成"（
    `ALPHAFOLD_CHECKPOINT_FLOOR = 100 << 20`）。
    """
    missing: list[str] = []
    for name in CAMPAIGN_MODELS:
        path = alphafold_parameter_path(parameters_dir, name)
        if path is None or path.stat().st_size < floor_bytes:
            missing.append(f"params_{name}.npz")
    return missing


def missing_proteinmpnn_weights(root: Path, *, floor_bytes: int = 1 << 20) -> list[str]:
    """ProteinMPNN 三变体随包（~6.6 MB each，上游下限 1 MB）。"""
    missing: list[str] = []
    for variant in MPNN_VARIANTS:
        path = (
            root
            / "bindcraft"
            / "weights"
            / "proteinmpnn"
            / f"weights_{variant}"
            / MPNN_CHECKPOINT
        )
        if not path.is_file() or path.stat().st_size < floor_bytes:
            missing.append(f"weights_{variant}/{MPNN_CHECKPOINT}")
    return missing


def build_campaign_json(
    req: DesignRequest,
    *,
    job_dir: Path,
    target_path: Path | None,
    max_trajectories: int,
) -> dict:
    """把请求编译成上游 campaign JSON。

    上游 campaign JSON 的 key 与请求字段一一对应；服务额外写死
    `project_folder`（指向本 job 的 output/），并注入 `max_trajectories`。
    """
    campaign: dict = {
        "campaign_name": req.campaign_name or job_dir.name,
        "modality": req.modality,
        "number_of_final_designs": req.number_of_final_designs,
        "max_trajectories": max_trajectories,
        "project_folder": str((job_dir / "output").resolve()),
    }

    if req.target_name:
        campaign["target"] = req.target_name
    else:
        if target_path is None:
            raise ValueError("target_path is required when target_name is unset")
        entry: dict = {
            # 服务生成的名字：用户文件名可能含空格 / 非 ASCII / 长度不定，
            # 而上游用它拼结果文件名与 hash。
            "name": "target",
            "target_path": str(target_path.resolve()),
        }
        if req.target_chains:
            entry["chains"] = req.target_chains
        if req.hotspots:
            entry["hotspots"] = req.hotspots
        if req.coldspots:
            entry["coldspots"] = req.coldspots
        campaign["targets"] = [entry]

    if req.binder_lengths:
        campaign["binder_lengths"] = list(req.binder_lengths)
    if req.core:
        campaign["core"] = req.core
    for name in req.properties:
        campaign[name] = True

    return campaign


def write_campaign_file(campaign: dict, job_dir: Path) -> Path:
    """把 campaign 写到 `<job_dir>/input/campaign.json` 并返回该路径。"""
    path = job_dir / "input" / "campaign.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(campaign, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def prepare_design(
    req: DesignRequest,
    *,
    job_dir: Path,
    target_path: Path | None,
    settings: Bindcraft2Settings,
) -> Path:
    """HTTP 与 CLI 共用的 design 前置：拼装 + 落盘 campaign 文件。"""
    max_trajectories = req.max_trajectories or settings.default_max_trajectories
    campaign = build_campaign_json(
        req, job_dir=job_dir, target_path=target_path, max_trajectories=max_trajectories
    )
    return write_campaign_file(campaign, job_dir)


def design_argv(
    campaign_file: Path, job_dir: Path, settings: Bindcraft2Settings
) -> list[str]:
    """`python -m bindcraft.cli design <campaign.json>`。"""
    return runner_prefix(settings) + ["design", str(campaign_file.resolve())]


def rank_argv(
    req: RankRequest,
    campaign_dir: Path,
    job_dir: Path,
    settings: Bindcraft2Settings,
) -> list[str]:
    """`... rank <campaign_dir> --on M [--on M2] --table T --output <job>/output/ranked_by_<M>.csv`。"""
    output = job_dir / "output" / RANK_OUTPUT_TEMPLATE.format(metric=req.on[0])
    cmd = runner_prefix(settings) + ["rank", str(campaign_dir)]
    for metric in req.on:
        cmd += ["--on", metric]
    cmd += ["--table", req.table.value, "--output", str(output.resolve())]
    if req.lowest_first is not None:
        cmd.append("--lowest-first" if req.lowest_first else "--highest-first")
    if req.top is not None:
        cmd += ["--top", str(req.top)]
    return cmd


def filter_argv(
    req: FilterRequest,
    campaign_dir: Path,
    job_dir: Path,
    settings: Bindcraft2Settings,
) -> list[str]:
    """`... filter <campaign_dir> [--where EXPR]... --table T --output <job>/output/filtered.csv`。"""
    output = job_dir / "output" / FILTER_OUTPUT
    cmd = runner_prefix(settings) + ["filter", str(campaign_dir)]
    for expression in req.where or []:
        cmd += ["--where", expression]
    cmd += ["--table", req.table.value, "--output", str(output.resolve())]
    if req.top is not None:
        cmd += ["--top", str(req.top)]
    return cmd
```

- [ ] **Step 4: 跑测试确认通过**

```bash
cd services/bindcraft2-server && uv run --group dev python -m pytest tests/test_tools.py -q
```

Expected：`22 passed`。

- [ ] **Step 5: Commit**

```bash
git add services/bindcraft2-server/tools.py services/bindcraft2-server/tests/test_tools.py
git commit -m "feat(bindcraft2-server): add campaign builder, argv builders and weight probes"
```

---

### Task 5: adapter.py

**Files:**
- Create: `services/bindcraft2-server/adapter.py`
- Test: `services/bindcraft2-server/tests/test_adapter.py`

- [ ] **Step 1: 写失败测试**

```python
"""Bindcraft2Adapter：输出判定、子进程环境、manifest extras。"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic_settings import SettingsConfigDict

from server.adapter import Bindcraft2Adapter
from server.settings import Bindcraft2Settings


class _Off(Bindcraft2Settings):
    model_config = SettingsConfigDict(
        env_prefix="BINDCRAFT2_TEST_", env_file=None, extra="ignore", case_sensitive=False
    )


@pytest.fixture
def adapter(tmp_path: Path) -> Bindcraft2Adapter:
    s = _Off(
        jobs_base_dir=tmp_path / "jobs",
        root=tmp_path / "root",
        alphafold_params_dir=tmp_path / "af",
        compile_cache_dir=tmp_path / "cache",
    )
    return Bindcraft2Adapter(settings=s)


def _job(tmp_path: Path) -> Path:
    d = tmp_path / "jobs" / "j1"
    (d / "output").mkdir(parents=True)
    return d


def test_name(adapter):
    assert adapter.name == "bindcraft2"


def test_subprocess_cwd_is_upstream_root(adapter, tmp_path):
    assert adapter.subprocess_cwd() == tmp_path / "root"


def test_subprocess_env_sets_required_vars(adapter, tmp_path):
    env = adapter.subprocess_env()
    # 必须显式设置，否则上游会尝试下载 5.3 GB 参数。
    assert env["BINDCRAFT_AF2_PARAMS"] == str(tmp_path / "af")
    assert env["JAX_COMPILATION_CACHE_DIR"] == str(tmp_path / "cache")
    assert env["BINDCRAFT_WORKERS_PER_GPU"] == "1"
    assert env["BINDCRAFT_MAX_WORKERS_PER_GPU"] == "1"


def test_subprocess_env_creates_compile_cache_dir(adapter, tmp_path):
    adapter.subprocess_env()
    assert (tmp_path / "cache").is_dir()


def test_subprocess_env_survives_unwritable_cache_dir(adapter, tmp_path):
    (tmp_path / "af").mkdir(exist_ok=True)
    # 把 cache 路径指到一个已存在的普通文件 → mkdir 必然失败，但不能抛。
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    adapter.settings.compile_cache_dir = blocker / "cache"
    env = adapter.subprocess_env()
    assert env["JAX_COMPILATION_CACHE_DIR"] == str(blocker / "cache")


def test_detect_outputs_false_when_empty(adapter, tmp_path):
    assert adapter.detect_outputs(_job(tmp_path)) is False


def test_detect_outputs_true_on_ranked_csv(adapter, tmp_path):
    d = _job(tmp_path)
    p = d / "output" / "3_Ranked" / "!_Ranked.csv"
    p.parent.mkdir(parents=True)
    p.write_text("design,i_pDAE\n", encoding="utf-8")
    assert adapter.detect_outputs(d) is True


def test_detect_outputs_true_on_summary_only(adapter, tmp_path):
    """accepted=0 的合法 campaign 只写 summary.csv——不能被判为失败。"""
    d = _job(tmp_path)
    (d / "output" / "summary.csv").write_text("scope,metric\ncampaign,final\n", encoding="utf-8")
    assert adapter.detect_outputs(d) is True


def test_detect_outputs_true_on_rank_output(adapter, tmp_path):
    d = _job(tmp_path)
    (d / "output" / "ranked_by_i_pTM.csv").write_text("design,rank\n", encoding="utf-8")
    assert adapter.detect_outputs(d) is True


def test_detect_outputs_true_on_filter_output(adapter, tmp_path):
    d = _job(tmp_path)
    (d / "output" / "filtered.csv").write_text("design,outcome\n", encoding="utf-8")
    assert adapter.detect_outputs(d) is True


def test_detect_outputs_ignores_empty_files(adapter, tmp_path):
    d = _job(tmp_path)
    (d / "output" / "summary.csv").write_text("", encoding="utf-8")
    assert adapter.detect_outputs(d) is False


def test_infer_job_from_dir_reports_accepted_count(adapter, tmp_path):
    d = _job(tmp_path)
    p = d / "output" / "3_Ranked" / "!_Ranked.csv"
    p.parent.mkdir(parents=True)
    p.write_text("design,i_pDAE\na,0.7\nb,0.6\n", encoding="utf-8")
    info = adapter.infer_job_from_dir(d)
    assert info.status.value == "completed"
    assert info.progress == "2 accepted"


def test_infer_job_from_dir_zero_accepted(adapter, tmp_path):
    d = _job(tmp_path)
    (d / "output" / "summary.csv").write_text("scope,metric\n", encoding="utf-8")
    info = adapter.infer_job_from_dir(d)
    assert info.status.value == "completed"
    assert info.progress == "finished"


def test_infer_job_from_dir_no_outputs(adapter, tmp_path):
    info = adapter.infer_job_from_dir(_job(tmp_path))
    assert info.status.value == "failed"


def test_manifest_extras_shape(adapter):
    extras = adapter.manifest_extras()
    assert extras["tool_outputs"]["ranked"].endswith("3_Ranked/!_Ranked.csv")
    assert extras["tool_outputs"]["summary"] == "summary.csv"
    assert "job://<job_id>[/<subdir>]" in extras["input_uri_schemes"]
    assert extras["weights"]["alphafold_params_dir"]
    assert len(extras["weights"]["expected_alphafold_models"]) == 7


def test_endpoint_examples_cover_all_six(adapter):
    examples = adapter.endpoint_examples()
    for path in (
        "/api/design",
        "/api/tasks/design",
        "/api/rank",
        "/api/tasks/rank",
        "/api/filter",
        "/api/tasks/filter",
    ):
        assert path in examples, f"missing examples for {path}"
        assert examples[path], f"empty examples for {path}"
        assert any(e.curl for e in examples[path]), f"no curl for {path}"
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cd services/bindcraft2-server && uv run --group dev python -m pytest tests/test_adapter.py -q
```

Expected：`ModuleNotFoundError: No module named 'server.adapter'`。

- [ ] **Step 3: 写 adapter.py**

```python
"""bindcraft2-server 的服务级策略。

三处偏离框架默认：

  * `detect_outputs` 接受 design / rank / filter 三类产物中的任意一种——一个
    adapter 服务三组 endpoint。
  * `subprocess_env` 注入 BC2 必需的环境变量（AF2 参数目录、XLA 编译缓存、
    worker 上限），否则上游会尝试下载 5.3 GB 参数或按节点内存过度 pack worker。
  * `subprocess_cwd` 返回上游源码树：`settings/` 与 `scaffolds/` 是相对仓库根
    解析的，必须在根目录运行。
"""

from __future__ import annotations

import logging
from pathlib import Path

from bioq_service import EndpointExample, JobAdapter, JobInfo, JobStatus

from .settings import Bindcraft2Settings
from .tools import (
    CAMPAIGN_METADATA_JSON,
    CAMPAIGN_MODELS,
    DESIGN_RANKED_CSV,
    DESIGN_SUMMARY_CSV,
    FILTER_OUTPUT,
    TRAJECTORIES_CSV,
)

logger = logging.getLogger(__name__)


class Bindcraft2Adapter(JobAdapter):
    name = "bindcraft2"

    settings: Bindcraft2Settings  # 收窄类型，便于 IDE 提示

    def __init__(self, settings: Bindcraft2Settings) -> None:
        super().__init__(settings)

    # ---- 子进程环境 / 工作目录 ----

    def subprocess_cwd(self) -> Path | None:
        return self.settings.root

    def subprocess_env(self) -> dict[str, str]:
        cache_dir = self.settings.compile_cache_dir
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            # NAS 不可写时不能让整个 job 失败——只是每次冷启动都要重编译。
            logger.warning(
                "compile cache dir %s is not writable (%s); JAX will recompile every run",
                cache_dir,
                exc,
            )
        return {
            # 必须显式设置：缺省时上游会去下载 alphafold_params_2022-12-06.tar。
            "BINDCRAFT_AF2_PARAMS": str(self.settings.alphafold_params_dir),
            "JAX_COMPILATION_CACHE_DIR": str(cache_dir),
            # BC2 默认吃掉所有可见卡并按显存 pack worker；FC 一实例一卡，锁死 1。
            "BINDCRAFT_WORKERS_PER_GPU": str(self.settings.workers_per_gpu),
            "BINDCRAFT_MAX_WORKERS_PER_GPU": str(self.settings.max_workers_per_gpu),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }

    # ---- 产物判定 ----

    def detect_outputs(self, job_dir: Path) -> bool:
        """design / rank / filter 任一类产物存在且非空即算成功。

        必须包含 `summary.csv`：一场合法完成但 accepted=0 的 campaign 不会写
        `3_Ranked/!_Ranked.csv`。放宽是安全的——rc != 0 仍由框架 `finalize_job`
        判失败，这些文件只在 rc == 0 时才是成功证据。
        """
        out = self.output_dir(job_dir)
        for candidate in (
            out / DESIGN_RANKED_CSV,
            out / DESIGN_SUMMARY_CSV,
            out / FILTER_OUTPUT,
        ):
            if candidate.is_file() and candidate.stat().st_size > 0:
                return True
        return any(
            path.is_file() and path.stat().st_size > 0
            for path in out.glob("ranked_by_*.csv")
        )

    def infer_job_from_dir(self, job_dir: Path) -> JobInfo:
        """从磁盘恢复 job；progress 报告 accepted 数量（rank/filter 报 finished）。"""
        job_id = job_dir.name
        if not self.detect_outputs(job_dir):
            return JobInfo(
                job_id=job_id,
                status=JobStatus.FAILED,
                service=self.name,
                message="Recovered from disk (no outputs)",
            )

        ranked = self.output_dir(job_dir) / DESIGN_RANKED_CSV
        if ranked.is_file():
            rows = ranked.read_text(encoding="utf-8", errors="replace").splitlines()
            accepted = max(0, len(rows) - 1)  # 减表头
            return JobInfo(
                job_id=job_id,
                status=JobStatus.COMPLETED,
                service=self.name,
                progress=f"{accepted} accepted",
                message=f"Recovered from disk ({accepted} accepted designs)",
            )
        return JobInfo(
            job_id=job_id,
            status=JobStatus.COMPLETED,
            service=self.name,
            progress="finished",
            message="Recovered from disk (no accepted designs)",
        )

    # ---- Agent 面向的元数据 ----

    def manifest_extras(self) -> dict:
        return {
            "tool_outputs": {
                "ranked": DESIGN_RANKED_CSV,
                "summary": DESIGN_SUMMARY_CSV,
                "trajectories": TRAJECTORIES_CSV,
                "metadata": CAMPAIGN_METADATA_JSON,
                "rank": "ranked_by_<metric>.csv",
                "filter": FILTER_OUTPUT,
            },
            "input_uri_schemes": {
                "upload": "multipart/form-data UploadFile（target）",
                "job://<job_id>[/<subdir>]": "上一 job 的 output/ 目录（零拷贝，共享 NAS）",
                "file:///abs/path": "NAS 绝对路径（网关把 oss:// 改写成 /mnt/oss/... 后走这条）",
                "oss://<bucket>/<key>": "仅 target 单文件输入支持",
                "http(s)://...": "仅 target 单文件输入支持",
            },
            "chaining_tip": (
                "design 完成后用 campaign_uri=job://<design_job_id> 调 /api/rank 或 "
                "/api/filter，无需重新上传或重跑设计。rank 只读源目录、产物写进新 job。"
            ),
            "campaign_knobs": {
                "modality": (
                    "binder / large_binder / peptide / cyclic_peptide / homo_oligomer / "
                    "multidomain / VHH / ARP / scFv / Fab / induced_fit / fold_switch"
                ),
                "properties": [
                    "forced_targeting",
                    "humanize",
                    "protease_stable",
                    "disulfide_staple",
                    "mixed_topology",
                    "termini_together",
                    "termini_accessible",
                    "initial_guess",
                    "bigbang",
                ],
                "note": "modality 可逗号组合；兼容性由上游 preflight 校验，失败原因见 job 日志。",
            },
            "weights": {
                "alphafold_params_dir": str(self.settings.alphafold_params_dir),
                "expected_alphafold_models": [f"params_{m}.npz" for m in CAMPAIGN_MODELS],
                "expected_files": "params/params_<model>.npz 或 params_<model>.npz",
                "proteinmpnn": "随包（<root>/bindcraft/weights/proteinmpnn/weights_{neutral,negative,positive}/v_48_020.npz）",
            },
            "gpu": (
                "无 CPU 模式。cuda13 extra 需 compute capability >= 7.5；"
                "单 worker 显存预算 2.0*(3.4GB + 38kB*N^2)，N=256 时约 11.8 GB。"
            ),
            "license_notice": (
                "BindCraft2 采用 Source-Available License (Hosting-Restricted)。"
                "本服务仅限组织内部使用；不得作为 hosted/API/agent action 提供给第三方。"
            ),
        }

    def endpoint_examples(self) -> dict[str, list[EndpointExample]]:
        return {
            "/api/design": [
                EndpointExample(
                    title="shipped target，小 campaign",
                    curl=(
                        "curl -X POST $URL/api/design "
                        "-F target_name=hPDL1 "
                        "-F 'binder_lengths=[80,80]' "
                        "-F number_of_final_designs=10 "
                        "-F max_trajectories=200"
                    ),
                    notes=(
                        "shipped target 见 `bindcraft design --list-targets`。"
                        "HTTP 上复杂字段（binder_lengths / on / where）必须写成合法 JSON "
                        "字符串；逗号写法只对 CLI 有效，表单里发 80,80 会 422 json_invalid。"
                    ),
                ),
                EndpointExample(
                    title="上传自己的 target + hotspot",
                    curl=(
                        "curl -X POST $URL/api/design "
                        "-F target=@target.pdb "
                        "-F target_chains=A "
                        "-F 'hotspots=54,56,66-70' "
                        "-F modality=VHH "
                        "-F humanize=true "
                        "-F number_of_final_designs=10"
                    ),
                    python=(
                        "import httpx\n"
                        "with open('target.pdb', 'rb') as fh:\n"
                        "    r = httpx.post(\n"
                        "        f'{base_url}/api/design',\n"
                        "        files={'target': fh},\n"
                        "        data={\n"
                        "            'target_chains': 'A',\n"
                        "            'hotspots': '54,56,66-70',\n"
                        "            'modality': 'VHH',\n"
                        "            'humanize': 'true',\n"
                        "            'number_of_final_designs': '10',\n"
                        "        },\n"
                        "        timeout=120,\n"
                        "    )\n"
                        "job_id = r.json()['job_id']"
                    ),
                    notes="target / target_name / target_uri 三选一，多给或多不给都是 422。",
                ),
            ],
            "/api/tasks/design": [
                EndpointExample(
                    title="FC 异步任务模式（推荐入口）",
                    curl=(
                        "curl -X POST $URL/api/tasks/design "
                        "-H 'X-Fc-Invocation-Type: Async' "
                        "-H 'bioagent-session-id: demo-session' "
                        "-F target_uri=oss://bioagent-inputs/target.pdb "
                        "-F target_chains=A "
                        "-F max_trajectories=200"
                    ),
                    notes=(
                        "campaign 是小时级，必须走异步任务模式。上传体积超过 FC 128 KiB "
                        "事件上限时只能用 *_uri。"
                    ),
                ),
            ],
            "/api/rank": [
                EndpointExample(
                    title="接续上一场 campaign，换指标重排",
                    curl=(
                        "curl -X POST $URL/api/rank "
                        "-F campaign_uri=job://<design_job_id> "
                        "-F 'on=[\"i_pTM\",\"i_pDAE\"]' "
                        "-F table=accepted"
                    ),
                    notes="产物写到新 job 的 output/ranked_by_i_pTM.csv，源目录只读。",
                ),
            ],
            "/api/tasks/rank": [
                EndpointExample(
                    title="task 模式重排",
                    curl=(
                        "curl -X POST $URL/api/tasks/rank "
                        "-H 'X-Fc-Invocation-Type: Async' "
                        "-F campaign_uri=job://<design_job_id> "
                        "-F 'on=[\"i_pTM\"]'"
                    ),
                ),
            ],
            "/api/filter": [
                EndpointExample(
                    title="用新阈值重放候选集",
                    curl=(
                        "curl -X POST $URL/api/filter "
                        "-F campaign_uri=job://<design_job_id> "
                        "-F 'where=[\"i_pAE=0.45\",\"Interface_Residues>=7\"]'"
                    ),
                    notes="where 为空等价于用 campaign 自身阈值重放。",
                ),
            ],
            "/api/tasks/filter": [
                EndpointExample(
                    title="task 模式重放阈值",
                    curl=(
                        "curl -X POST $URL/api/tasks/filter "
                        "-H 'X-Fc-Invocation-Type: Async' "
                        "-F campaign_uri=job://<design_job_id>"
                    ),
                ),
            ],
        }
```

- [ ] **Step 4: 跑测试确认通过**

```bash
cd services/bindcraft2-server && uv run --group dev python -m pytest tests/test_adapter.py -q
```

Expected：`16 passed`。

- [ ] **Step 5: Commit**

```bash
git add services/bindcraft2-server/adapter.py services/bindcraft2-server/tests/test_adapter.py
git commit -m "feat(bindcraft2-server): add job adapter with output detection and manifest extras"
```

---

### Task 6: 离线 stub + app.py + test_app.py

**Files:**
- Create: `services/bindcraft2-server/tests/data/fake_bindcraft.sh`
- Modify: `services/bindcraft2-server/tests/conftest.py`（追加离线 fixture；别名在 Task 1 已建）
- Create: `services/bindcraft2-server/app.py`
- Test: `services/bindcraft2-server/tests/test_app.py`

- [ ] **Step 1: 写离线 stub**

`services/bindcraft2-server/tests/data/fake_bindcraft.sh`：

```bash
#!/usr/bin/env bash
# `python -m bindcraft.cli` 的离线替身：不碰 GPU，只写各子命令本该写出的产物。
# 由 tests/conftest.py 复制到 tmp_path 并 chmod +x，通过 BINDCRAFT2_PYTHON 指进来。
set -euo pipefail

if [ "${1:-}" = "-c" ]; then
  # /healthz/detail 的 GPU 探针：第一行 backend，第二行 devices。
  echo gpu
  echo cuda:0
  exit 0
fi

sub="${1:-}"
shift || true

if [ "$sub" = "design" ]; then
  json="$1"
  out="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_folder"])' "$json")"
  mkdir -p "$out/3_Ranked" "$out/1_Trajectories"
  printf 'design,i_pDAE\ndemo_1,0.72\n' > "$out/3_Ranked/!_Ranked.csv"
  printf 'design,terminated\ndemo_1,\n' > "$out/1_Trajectories/!_Trajectories.csv"
  printf 'scope,metric,samples\ncampaign,final,1\n' > "$out/summary.csv"
  printf '{"source_revision":"fake"}\n' > "$out/campaign_metadata.json"
  exit 0
fi

if [ "$sub" = "rank" ] || [ "$sub" = "filter" ]; then
  out=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --output) out="$2"; shift 2 ;;
      *) shift ;;
    esac
  done
  [ -n "$out" ] || { echo "fake_bindcraft: --output is required" >&2; exit 2; }
  mkdir -p "$(dirname "$out")"
  printf 'design,rank\n' > "$out"
  exit 0
fi

echo "fake_bindcraft: unknown subcommand '${sub}'" >&2
exit 2
```

- [ ] **Step 2: 往 tests/conftest.py 追加离线 fixture**

Task 1 已建好 conftest（`server` 别名 + fc marker，且已 `import pytest`）。本步只在文件
**末尾追加**下面的内容，并补三处 import：顶部 import 区加 `import shutil` 与
`from pydantic_settings import SettingsConfigDict`，`# noqa: E402` 那一段加
`from server.settings import Bindcraft2Settings`（`OfflineSettings` 要继承它——
计划初稿漏了这一条，实测加载 conftest 时直接 `NameError`）。

**追加内容里有两处不是可选的**（计划初稿漏了，实测暴露）：
1. `(tmp_path / "root").mkdir(...)`——`adapter.subprocess_cwd()` 返回 `settings.root`，
   生产里 `/opt/bindcraft` 必然存在，离线 tmp 目录不存在时 `Popen(cwd=...)` 直接
   `ENOENT`，每个 job 都以 `rc=127` 失败。
2. `shipped_weights_dir=tmp_path / "root"`——生产默认值 `shipped_weights_dir` 与 `root`
   同为 `/opt/bindcraft`，只覆盖 `root` 会让权重探针去查宿主的 `/opt/bindcraft`（本地
   不存在），`/healthz/detail` 的断言就失去意义。

另有一条与 Task 6 无关、但在本机实测暴露的**既有框架问题**（不属于本计划范围，不要在
本任务里修）：本机 `uv sync` 装到的是 `mcp` 2.x，其 `mcp.server.fastmcp` 已更名为
`MCPServer`，因此 `bioq_service/app.py` 的 `attach_mcp` 会走降级分支（记 warning 后返回
`None`，不抛错）。所有服务的 MCP 挂载都受影响，与 BindCraft2 无关；`app.py` 里
`attach_mcp(app)` 仍必须放在最后，顺序契约不受影响。

```python
# ---------------------------------------------------------------------------
# 离线 fixture：只把子进程换成 stub，其余全走真实框架
# （真实 JobRunner、真实 HTTP 路由、真实 job 目录与日志）。
# ---------------------------------------------------------------------------


class OfflineSettings(Bindcraft2Settings):
    """离线测试用：不读 .env，python 指向 stub，module 置空。"""

    model_config = SettingsConfigDict(
        env_prefix="BINDCRAFT2_TEST_", env_file=None, extra="ignore", case_sensitive=False
    )

    @classmethod
    def build(cls, tmp_path: Path, **overrides) -> "OfflineSettings":
        stub = tmp_path / "fake_bindcraft"
        shutil.copy2(SERVICE_DIR / "tests" / "data" / "fake_bindcraft.sh", stub)
        stub.chmod(0o755)
        # 子进程 cwd 是上游源码树（adapter.subprocess_cwd → settings.root）：生产里
        # /opt/bindcraft 必然存在，离线时必须先建出来，否则 Popen(cwd=...) 直接 ENOENT。
        (tmp_path / "root").mkdir(parents=True, exist_ok=True)
        params = dict(
            jobs_base_dir=tmp_path / "jobs",
            root=tmp_path / "root",
            # ProteinMPNN 随包在源码树里；生产默认值二者同为 /opt/bindcraft，
            # 离线时显式指向同一个 tmp 目录，否则探针会去查宿主的 /opt/bindcraft。
            shipped_weights_dir=tmp_path / "root",
            python=str(stub),
            module="",
            alphafold_params_dir=tmp_path / "af",
            compile_cache_dir=tmp_path / "cache",
            max_concurrent_jobs=1,
            gpu_probe_ttl_seconds=0,
        )
        params.update(overrides)
        return cls(**params)


@pytest.fixture
def offline_settings(tmp_path: Path) -> OfflineSettings:
    return OfflineSettings.build(tmp_path)
```

`env_prefix="BINDCRAFT2_TEST_"` 与生产前缀隔离，避免 CI 上真实的 `BINDCRAFT2_*` 变量污染测试。

- [ ] **Step 3: 写 test_app.py（失败）**

```python
"""离线 HTTP 端到端：真实 server.app + 真实 runner，子进程换成 stub。"""

from __future__ import annotations

import importlib
import time

from fastapi.testclient import TestClient


def _client(settings, monkeypatch, module: str = "") -> TestClient:
    # server.app 在 import 期用自己的 settings 构造 app；把 env 指到 tmp，
    # 再 reload，避免写到 /data/bindcraft2_jobs。
    monkeypatch.setenv("BINDCRAFT2_JOBS_BASE_DIR", str(settings.jobs_base_dir))
    monkeypatch.setenv("BINDCRAFT2_ROOT", str(settings.root))
    monkeypatch.setenv("BINDCRAFT2_SHIPPED_WEIGHTS_DIR", str(settings.shipped_weights_dir))
    monkeypatch.setenv("BINDCRAFT2_PYTHON", settings.python)
    monkeypatch.setenv("BINDCRAFT2_MODULE", module)
    monkeypatch.setenv("BINDCRAFT2_ALPHAFOLD_PARAMS_DIR", str(settings.alphafold_params_dir))
    monkeypatch.setenv("BINDCRAFT2_COMPILE_CACHE_DIR", str(settings.compile_cache_dir))
    monkeypatch.setenv("BINDCRAFT2_GPU_PROBE_TTL_SECONDS", "0")
    server_app = importlib.reload(importlib.import_module("server.app"))
    return TestClient(server_app.app)


def _wait(client: TestClient, job_id: str, timeout_s: float = 20.0) -> dict:
    deadline = time.monotonic() + timeout_s
    body: dict = {}
    while time.monotonic() < deadline:
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["status"] in ("completed", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish: {body}")


def test_health_and_detail(offline_settings, monkeypatch):
    client = _client(offline_settings, monkeypatch)
    health = client.get("/healthz").json()
    assert health["status"] == "ok"
    assert health["service"] == "bindcraft2"

    detail = client.get("/healthz/detail").json()
    assert detail["service"] == "bindcraft2"
    assert detail["gpu_backend"] == "gpu"  # stub 探针
    assert detail["weights_loaded"] is False
    assert len(detail["weights_missing"]) == 7


def test_healthz_detail_reports_job_counters(offline_settings, monkeypatch):
    """跑批并发度是 /healthz/detail 的契约字段；缺了它们调用方无法判断容量。"""
    client = _client(offline_settings, monkeypatch)
    detail = client.get("/healthz/detail").json()
    assert detail["active_jobs"] == 0
    assert detail["max_concurrent_jobs"] == 1


def test_detail_reports_loaded_weights(offline_settings, monkeypatch, tmp_path):
    af = offline_settings.alphafold_params_dir / "params"
    af.mkdir(parents=True)
    from server.tools import CAMPAIGN_MODELS

    for name in CAMPAIGN_MODELS:
        (af / f"params_{name}.npz").write_bytes(b"x" * (101 << 20))
    # ProteinMPNN 随包：在 settings.root 下造三变体
    for variant in ("neutral", "negative", "positive"):
        d = offline_settings.root / "bindcraft" / "weights" / "proteinmpnn" / f"weights_{variant}"
        d.mkdir(parents=True)
        (d / "v_48_020.npz").write_bytes(b"x" * (2 << 20))

    client = _client(offline_settings, monkeypatch)
    detail = client.get("/healthz/detail").json()
    assert detail["weights_loaded"] is True
    assert detail["weights_missing"] == []
    assert detail["proteinmpnn_weights_loaded"] is True


def test_design_with_shipped_target(offline_settings, monkeypatch):
    client = _client(offline_settings, monkeypatch)
    r = client.post(
        "/api/design",
        data={"target_name": "hPDL1", "number_of_final_designs": "1", "max_trajectories": "1"},
    )
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    body = _wait(client, job_id)
    assert body["status"] == "completed", body

    files = client.get(f"/api/jobs/{job_id}/files").json()
    assert "3_Ranked/!_Ranked.csv" in files["files"]
    # 落盘的 campaign 内容可核对
    campaign = (
        offline_settings.jobs_base_dir / job_id / "input" / "campaign.json"
    ).read_text(encoding="utf-8")
    assert '"target": "hPDL1"' in campaign
    assert '"max_trajectories": 1' in campaign


def test_design_with_upload(offline_settings, monkeypatch):
    client = _client(offline_settings, monkeypatch)
    r = client.post(
        "/api/design",
        data={"target_chains": "A", "hotspots": "54,56"},
        files={"target": ("target.pdb", b"ATOM\n", "chemical/x-pdb")},
    )
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    body = _wait(client, job_id)
    assert body["status"] == "completed", body

    campaign = (
        offline_settings.jobs_base_dir / job_id / "input" / "campaign.json"
    ).read_text(encoding="utf-8")
    assert '"name": "target"' in campaign
    assert (offline_settings.jobs_base_dir / job_id / "input" / "target.pdb").is_file()


def test_design_with_file_uri_target(offline_settings, monkeypatch, tmp_path):
    """`target_uri=file://...` 的 happy path（此前完全没测）。"""
    pdb = tmp_path / "bait.pdb"
    pdb.write_text("ATOM      1  CA  ALA A   1\n", encoding="utf-8")

    client = _client(offline_settings, monkeypatch)
    r = client.post(
        "/api/design",
        data={"target_uri": f"file://{pdb}", "max_trajectories": "1"},
    )
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    body = _wait(client, job_id)
    assert body["status"] == "completed", body
    assert (offline_settings.jobs_base_dir / job_id / "input" / "target.pdb").is_file()


def test_design_rejects_unresolvable_target_uri(offline_settings, monkeypatch):
    """客户端输入错误必须是 4xx；未包装时 FastAPI 会把它们当服务端故障报 500。"""
    client = _client(offline_settings, monkeypatch)
    for uri in ("oss://bucket/key", "file:///etc", "http://"):
        r = client.post("/api/design", data={"target_uri": uri})
        assert r.status_code == 422, f"{uri} -> {r.status_code}: {r.text}"


def test_task_design_rejects_unresolvable_target_uri(offline_settings, monkeypatch):
    """孪生端点必须和 /api/design 同样映射成 422。

    FC 异步任务模式走的正是这条；只包装 submit/poll 那条路时，同样的坏 URI 会在这里
    变成 500——两个入口共用 `_resolve_target_input` 就是为了不让它再分叉。
    """
    client = _client(offline_settings, monkeypatch)
    for uri in ("oss://bucket/key", "file:///etc", "http://"):
        r = client.post(
            "/api/tasks/design",
            data={"target_uri": uri},
            headers={"bioagent-session-id": "s1"},
        )
        assert r.status_code == 422, f"{uri} -> {r.status_code}: {r.text}"


def test_design_upload_keeps_cif_suffix(offline_settings, monkeypatch):
    """落盘后缀必须跟着上传文件走：上游按后缀识别 mmCIF，改成常量就废掉这条。"""
    client = _client(offline_settings, monkeypatch)
    r = client.post(
        "/api/design",
        data={"target_chains": "A"},
        files={"target": ("target.cif", b"data_demo\n", "chemical/x-cif")},
    )
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    body = _wait(client, job_id)
    assert body["status"] == "completed", body

    saved = offline_settings.jobs_base_dir / job_id / "input" / "target.cif"
    assert saved.is_file()
    assert not (offline_settings.jobs_base_dir / job_id / "input" / "target.pdb").exists()


def test_design_rejects_no_target(offline_settings, monkeypatch):
    client = _client(offline_settings, monkeypatch)
    r = client.post("/api/design", data={"number_of_final_designs": "1"})
    assert r.status_code == 422
    assert "Exactly one of" in r.text


def test_design_rejects_target_name_plus_upload(offline_settings, monkeypatch):
    client = _client(offline_settings, monkeypatch)
    r = client.post(
        "/api/design",
        data={"target_name": "hPDL1"},
        files={"target": ("target.pdb", b"ATOM\n", "chemical/x-pdb")},
    )
    assert r.status_code == 422


def test_design_rejects_descending_binder_lengths(offline_settings, monkeypatch):
    client = _client(offline_settings, monkeypatch)
    r = client.post(
        "/api/design",
        data={"target_name": "hPDL1", "binder_lengths": "[100,60]"},
    )
    assert r.status_code == 422


def test_design_rejects_invalid_json_in_complex_field(offline_settings, monkeypatch):
    client = _client(offline_settings, monkeypatch)
    r = client.post("/api/design", data={"target_name": "hPDL1", "binder_lengths": "80,80"})
    # model_form_depends 对复杂字段要求合法 JSON，逗号写法是 CLI 专用。
    assert r.status_code == 422
    assert "Invalid JSON" in r.text


def test_rank_reads_previous_job(offline_settings, monkeypatch):
    client = _client(offline_settings, monkeypatch)
    design = client.post(
        "/api/design", data={"target_name": "hPDL1", "max_trajectories": "1"}
    ).json()
    _wait(client, design["job_id"])

    r = client.post(
        "/api/rank",
        data={"campaign_uri": f"job://{design['job_id']}", "on": '["i_pTM"]'},
    )
    assert r.status_code == 200, r.text
    body = _wait(client, r.json()["job_id"])
    assert body["status"] == "completed", body

    files = client.get(f"/api/jobs/{r.json()['job_id']}/files").json()
    assert "ranked_by_i_pTM.csv" in files["files"]


def test_filter_reads_previous_job(offline_settings, monkeypatch):
    client = _client(offline_settings, monkeypatch)
    design = client.post(
        "/api/design", data={"target_name": "hPDL1", "max_trajectories": "1"}
    ).json()
    _wait(client, design["job_id"])

    r = client.post(
        "/api/filter",
        data={"campaign_uri": f"job://{design['job_id']}", "where": '["i_pAE=0.45"]'},
    )
    assert r.status_code == 200, r.text
    body = _wait(client, r.json()["job_id"])
    assert body["status"] == "completed", body
    files = client.get(f"/api/jobs/{r.json()['job_id']}/files").json()
    assert "filtered.csv" in files["files"]


def test_rank_rejects_bad_campaign_uri(offline_settings, monkeypatch):
    client = _client(offline_settings, monkeypatch)
    r = client.post("/api/rank", data={"campaign_uri": "job://does-not-exist"})
    assert r.status_code == 404


def test_task_design_runs_atomically(offline_settings, monkeypatch):
    client = _client(offline_settings, monkeypatch)
    r = client.post(
        "/api/tasks/design",
        data={"target_name": "hPDL1", "max_trajectories": "1"},
        headers={"bioagent-session-id": "s1"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "completed"


def test_task_rank_and_filter(offline_settings, monkeypatch):
    client = _client(offline_settings, monkeypatch)
    design = client.post(
        "/api/tasks/design", data={"target_name": "hPDL1", "max_trajectories": "1"}
    ).json()
    job_id = design["job_id"]

    rank = client.post("/api/tasks/rank", data={"campaign_uri": f"job://{job_id}"})
    assert rank.status_code == 200, rank.text
    assert rank.json()["status"] == "completed"

    filt = client.post("/api/tasks/filter", data={"campaign_uri": f"job://{job_id}"})
    assert filt.status_code == 200, filt.text
    assert filt.json()["status"] == "completed"


def test_manifest_lists_six_endpoints_with_examples(offline_settings, monkeypatch):
    client = _client(offline_settings, monkeypatch)
    body = client.get("/api/manifest").json()
    eps = {e["path"]: e for e in body["endpoints"]}
    for path in (
        "/api/design",
        "/api/tasks/design",
        "/api/rank",
        "/api/tasks/rank",
        "/api/filter",
        "/api/tasks/filter",
    ):
        assert path in eps, f"missing {path}"
        assert eps[path]["examples"], f"{path} has no examples"
    assert body["service"] == "bindcraft2"
    assert body["service_specific"]["tool_outputs"]["ranked"].endswith("!_Ranked.csv")


def test_openapi_exposes_six_body_schemas(offline_settings, monkeypatch):
    client = _client(offline_settings, monkeypatch)
    models = set(client.get("/openapi.json").json()["components"]["schemas"].keys())
    expected = {
        "Body_run_design_api_design_post",
        "Body_run_design_task_api_tasks_design_post",
        "Body_run_rank_api_rank_post",
        "Body_run_rank_task_api_tasks_rank_post",
        "Body_run_filter_api_filter_post",
        "Body_run_filter_task_api_tasks_filter_post",
    }
    assert not (expected - models), sorted(expected - models)


def test_upload_field_names_follow_convention(offline_settings, monkeypatch):
    client = _client(offline_settings, monkeypatch)
    eps = {e["path"]: e for e in client.get("/api/manifest").json()["endpoints"]}
    for path in ("/api/design", "/api/tasks/design"):
        fields = {f["name"]: f for f in eps[path]["request_fields"]}
        assert fields["target"]["is_file"] is True
        assert "target_uri" in fields
        # 不得出现旧式裸 input_uri
        assert "input_uri" not in fields


def test_gpu_probe_runs_out_of_process(offline_settings, monkeypatch):
    """GPU 探针必须走子进程：HTTP 进程绝不能 import jax。

    在 HTTP 进程里 import jax 会初始化 CUDA context、占掉 campaign 需要的显存；
    无卡时还会静默回退 CPU。stub 只在 `-c` 分支打印 `gpu`，因此
    `gpu_backend == "gpu"` 本身就证明探针是在子进程里跑的。
    """
    import sys

    client = _client(offline_settings, monkeypatch)
    detail = client.get("/healthz/detail").json()
    assert detail["gpu_backend"] == "gpu"
    assert "jax" not in sys.modules


def test_gpu_probe_ignores_upstream_module(offline_settings, monkeypatch):
    """生产 settings 带 module="bindcraft.cli"，探针仍必须拿到 backend。

    探针 argv 若拼成 `python -m bindcraft.cli -c <code>`，`-c <code>` 只会被当成
    上游 CLI 的参数（stub 走 unknown subcommand 分支、什么都不打印），
    gpu_backend 永远停在 "probe_failed"——这是生产里唯一能发现静默跑 CPU 的信号。
    """
    client = _client(offline_settings, monkeypatch, module="bindcraft.cli")
    detail = client.get("/healthz/detail").json()
    assert detail["gpu_backend"] == "gpu"
    assert detail["gpu_devices"] == "cuda:0"


def test_healthz_detail_warns_when_probe_cannot_run(offline_settings, monkeypatch):
    """探针跑不起来（解释器不存在）也要 200 + warning，而不是 500。"""
    broken = offline_settings.model_copy(update={"python": "/nonexistent/python-fixture"})
    client = _client(broken, monkeypatch)
    r = client.get("/healthz/detail")
    assert r.status_code == 200, r.text
    detail = r.json()
    assert detail["gpu_backend"] == "probe_failed"
    assert "warning" in detail
    assert "GPU" in detail["warning"]
```

- [ ] **Step 4: 跑测试确认失败**

```bash
cd services/bindcraft2-server && uv run --group dev python -m pytest tests/test_app.py -q 2>&1 | tail -20
```

Expected：collection error / `ModuleNotFoundError: No module named 'server.app'`。

- [ ] **Step 5: 写 app.py**

```python
"""bindcraft2-server 的 FastAPI app。

暴露三组 endpoint（design / rank / filter），每组都有 `/api/tasks/*` 孪生。
框架负责 /healthz、/api/jobs/{id}、/files、/log、/download、/file/{path}、
DELETE /api/jobs/{id}、/api/manifest —— 这里只注册服务专属路由和自定义
/healthz/detail。

FC 部署要点：
  - 监听 0.0.0.0:9000（CAPort）
  - 启动 120 s 内必须能应答 /healthz
  - keep-alive >= 15 min；campaign 是小时级，必须走异步任务模式
"""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import Optional

from bioq_service import (
    JobInfo,
    attach_mcp,
    create_app,
    execute_task,
    model_form_depends,
    read_version_file,
    resolve_task_id,
)
from bioq_service.uris import resolve_input
from fastapi import Depends, File, Form, Header, HTTPException, Request, UploadFile

from .adapter import Bindcraft2Adapter
from .campaigns import resolve_campaign_dir
from .models import (
    DesignRequest,
    FilterRequest,
    RankRequest,
    validate_target_selection,
)
from .settings import Bindcraft2Settings
from .tools import (
    CAMPAIGN_MODELS,
    design_argv,
    filter_argv,
    missing_alphafold_params,
    missing_proteinmpnn_weights,
    prepare_design,
    rank_argv,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

settings = Bindcraft2Settings()
adapter = Bindcraft2Adapter(settings=settings)

app = create_app(
    adapter,
    settings,
    title="BindCraft2 Server",
    version=read_version_file(__file__, default="0.0.1"),
)


# ---------------------------------------------------------------------------
# 自定义 /healthz/detail
# ---------------------------------------------------------------------------

_KNOWN_SUFFIXES = (".mmcif", ".fasta", ".pdb", ".cif", ".fas", ".fa")

# GPU 探针在子进程里跑：绝不能在 HTTP 进程内 import jax——那会初始化 CUDA
# context 并占掉 campaign 需要的显存。
_GPU_PROBE_CODE = (
    "import jax; print(jax.default_backend()); "
    "print(','.join(sorted({d.platform for d in jax.devices()})))"
)
_probe_cache: dict[str, object] = {"at": 0.0, "value": ("", "")}


def _gpu_probe(settings: Bindcraft2Settings) -> tuple[str, str]:
    """返回 (backend, devices)；结果按 gpu_probe_ttl_seconds 缓存。"""
    now = time.monotonic()
    cached = _probe_cache["value"]
    if isinstance(cached, tuple) and cached[0]:
        if now - float(_probe_cache["at"]) < settings.gpu_probe_ttl_seconds:
            return cached  # type: ignore[return-value]
    backend, devices = "probe_failed", ""
    # 探针只做 `import jax`，与上游 CLI 无关，绝不能拼 `-m <module>`：生产里
    # module="bindcraft.cli"，`python -m bindcraft.cli -c <code>` 会把 `-c <code>`
    # 当成 CLI 的普通参数，代码永不执行（CLI 报 usage 并退出 2），于是
    # gpu_backend 永远停在 "probe_failed"——唯一能发现静默回退 CPU 的信号被毁掉。
    argv = [settings.python, "-c", _GPU_PROBE_CODE]
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            # 探针卡死不能拖住健康检查；30 s 足够解释器 import jax。
            timeout=30,
            cwd=str(settings.root) if settings.root.is_dir() else None,
        )
        lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        if proc.returncode == 0 and lines:
            backend = lines[0]
            devices = lines[1] if len(lines) > 1 else ""
        else:
            backend = "probe_failed"
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("GPU probe failed: %s", exc)
    _probe_cache["at"] = now
    _probe_cache["value"] = (backend, devices)
    return backend, devices


# 摘掉框架的通用 /healthz/detail，让服务专属探针接管。
def _strip_route(router, path: str, method: str) -> None:
    router.routes = [
        r
        for r in router.routes
        if not (getattr(r, "path", None) == path and method in getattr(r, "methods", set()))
    ]
    # FastAPI 的 included-router 层按 route-version 计数器缓存有效候选；直接赋值
    # routes 不会 bump 计数器，缓存已热时这次剥离会被静默忽略（框架的通用
    # /healthz/detail 先匹配，还不报错）。有该方法就防御性地 bump 一次。
    mark = getattr(router, "_mark_routes_changed", None)
    if callable(mark):
        mark()
    for r in router.routes:
        inner = getattr(r, "original_router", None)
        if inner is not None:
            _strip_route(inner, path, method)


_strip_route(app.router, "/healthz/detail", "GET")


@app.get("/healthz/detail")
def healthz_detail(request: Request) -> dict:
    """扩展健康检查：权重是否就位 + JAX 是否真的拿到了 GPU。

    权重缺失时返回 HTTP 200 + `weights_loaded=false`，不在 import 期 raise。
    `gpu_backend != "gpu"` 是唯一能暴露"静默跑 CPU"的信号——镜像内没有
    nvidia-smi，JAX 取不到卡时只警告一次然后慢两个数量级。
    """
    params_dir = settings.alphafold_params_dir
    missing = missing_alphafold_params(params_dir)
    mpnn_missing = missing_proteinmpnn_weights(settings.shipped_weights_dir)
    backend, devices = _gpu_probe(settings)

    body = {
        "status": "ok",
        "service": adapter.name,
        "version": request.app.version,
        "weights_dir": str(params_dir),
        "weights_loaded": not missing and not mpnn_missing,
        "weights_missing": missing,
        "proteinmpnn_weights_loaded": not mpnn_missing,
        "proteinmpnn_weights_missing": mpnn_missing,
        "expected_alphafold_models": [f"params_{m}.npz" for m in CAMPAIGN_MODELS],
        "gpu_backend": backend,
        "gpu_devices": devices,
        "active_jobs": request.app.state.runner.active_job_count,
        "max_concurrent_jobs": settings.max_concurrent_jobs,
    }
    if backend != "gpu":
        body["warning"] = (
            "JAX did not report a GPU backend; a campaign would run on CPU "
            "(orders of magnitude slower). Check the FC GPU allocation."
        )
    return body


# ---------------------------------------------------------------------------
# 服务专属 endpoint
# ---------------------------------------------------------------------------


def _input_suffix(filename: Optional[str]) -> str:
    """按文件名/URI 后缀决定落盘扩展名；上游按后缀识别 PDB / mmCIF / FASTA。"""
    if filename:
        lower = filename.lower()
        for suffix in _KNOWN_SUFFIXES:
            if lower.endswith(suffix):
                return suffix
    return ".pdb"


def _resolve_target_input(
    target: Optional[UploadFile],
    target_uri: Optional[str],
    dest: Path,
    settings: Bindcraft2Settings,
) -> Path:
    """`resolve_input` 的包装：客户端输入错误一律 422，不要漏成 500。

    submit/poll 与 FC 异步任务两条入口共用（孪生端点若各写一份，很容易只修一处）。
    """
    try:
        return resolve_input(target, target_uri, dest, settings, field_name="target")
    except HTTPException:
        # 框架已给出合理的 4xx/502，原样透传。
        raise
    except Exception as exc:
        # 其余都是客户端输入错误（目录、坏 URI、缺凭证……），映射成 422，
        # 否则 FastAPI 会把它当服务端故障返回 500。
        raise HTTPException(
            status_code=422, detail=f"Could not resolve target: {exc}"
        ) from exc


@app.post("/api/design", response_model=JobInfo)
def run_design(
    target: Optional[UploadFile] = File(
        None, description="目标结构（PDB/mmCIF/FASTA）。与 `target_name`、`target_uri` 三选一。"
    ),
    target_uri: Optional[str] = Form(
        None,
        description=(
            "目标结构的 URI 代替上传。schemes: job://<id>/<file>, file:///path, "
            "oss://<bucket>/<key>, http(s)://..."
        ),
    ),
    params: DesignRequest = Depends(model_form_depends(DesignRequest)),
) -> JobInfo:
    """提交一场 BindCraft2 campaign（submit/poll 模式）。"""

    def _build(job_id: str, job_dir: Path) -> list[str]:
        validate_target_selection(params.target_name, target is not None, target_uri)
        target_path = None
        if not params.target_name:
            dest = job_dir / "input" / f"target{_input_suffix(target_uri or getattr(target, 'filename', None))}"
            target_path = _resolve_target_input(target, target_uri, dest, settings)
        campaign_file = prepare_design(
            params, job_dir=job_dir, target_path=target_path, settings=settings
        )
        return design_argv(campaign_file, job_dir, settings)

    return app.state.runner.submit(
        build_argv=_build, label="design", input_params=params.model_dump(mode="json")
    )


@app.post("/api/rank", response_model=JobInfo)
def run_rank(
    params: RankRequest = Depends(model_form_depends(RankRequest)),
) -> JobInfo:
    """对已有 campaign 目录按指定指标重排（submit/poll 模式）。"""

    def _build(job_id: str, job_dir: Path) -> list[str]:
        campaign_dir = resolve_campaign_dir(params.campaign_uri, settings)
        return rank_argv(params, campaign_dir, job_dir, settings)

    return app.state.runner.submit(
        build_argv=_build, label="rank", input_params=params.model_dump(mode="json")
    )


@app.post("/api/filter", response_model=JobInfo)
def run_filter(
    params: FilterRequest = Depends(model_form_depends(FilterRequest)),
) -> JobInfo:
    """对已有 campaign 目录重放 / 覆盖 acceptance 阈值（submit/poll 模式）。"""

    def _build(job_id: str, job_dir: Path) -> list[str]:
        campaign_dir = resolve_campaign_dir(params.campaign_uri, settings)
        return filter_argv(params, campaign_dir, job_dir, settings)

    return app.state.runner.submit(
        build_argv=_build, label="filter", input_params=params.model_dump(mode="json")
    )


# ---------------------------------------------------------------------------
# FC 异步任务模式孪生
# ---------------------------------------------------------------------------

if settings.task_endpoints_enabled:

    @app.post("/api/tasks/design", response_model=JobInfo)
    def run_design_task(
        request: Request,
        target: Optional[UploadFile] = File(None),
        target_uri: Optional[str] = Form(None),
        params: DesignRequest = Depends(model_form_depends(DesignRequest)),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        """campaign 作为单个原子任务执行（小时内阻塞，需 FC 异步任务模式）。"""
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        paths: dict[str, Path] = {}

        def _save(_req, input_dir: Path) -> None:
            validate_target_selection(params.target_name, target is not None, target_uri)
            if params.target_name:
                return
            dest = input_dir / f"target{_input_suffix(target_uri or getattr(target, 'filename', None))}"
            paths["target"] = _resolve_target_input(target, target_uri, dest, settings)

        def _build(req, _job_id: str, job_dir: Path) -> list[str]:
            campaign_file = prepare_design(
                req, job_dir=job_dir, target_path=paths.get("target"), settings=settings
            )
            return design_argv(campaign_file, job_dir, settings)

        return execute_task(
            request, job_id=job_id, label="design", params=params,
            build_argv=_build, save_inputs=_save,
        )

    @app.post("/api/tasks/rank", response_model=JobInfo)
    def run_rank_task(
        request: Request,
        params: RankRequest = Depends(model_form_depends(RankRequest)),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        paths: dict[str, Path] = {}

        def _save(_req, _input_dir: Path) -> None:
            paths["campaign"] = resolve_campaign_dir(params.campaign_uri, settings)

        def _build(req, _job_id: str, job_dir: Path) -> list[str]:
            return rank_argv(req, paths["campaign"], job_dir, settings)

        return execute_task(
            request, job_id=job_id, label="rank", params=params,
            build_argv=_build, save_inputs=_save,
        )

    @app.post("/api/tasks/filter", response_model=JobInfo)
    def run_filter_task(
        request: Request,
        params: FilterRequest = Depends(model_form_depends(FilterRequest)),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        paths: dict[str, Path] = {}

        def _save(_req, _input_dir: Path) -> None:
            paths["campaign"] = resolve_campaign_dir(params.campaign_uri, settings)

        def _build(req, _job_id: str, job_dir: Path) -> list[str]:
            return filter_argv(req, paths["campaign"], job_dir, settings)

        return execute_task(
            request, job_id=job_id, label="filter", params=params,
            build_argv=_build, save_inputs=_save,
        )


# MCP server 必须在所有 POST 路由注册之后挂载（自动发现会遍历全部路由）。
attach_mcp(app)
```

- [ ] **Step 6: 跑测试确认通过**

```bash
cd services/bindcraft2-server && uv run --group dev python -m pytest tests/test_app.py -q 2>&1 | tail -25
```

Expected：`24 passed`（含钉住 GPU 探针进程隔离、孪生端点 422 映射的那两条）。若 `test_health_and_detail` 报 `gpu_backend != "gpu"`，检查
stub 是否被 chmod +x。

- [ ] **Step 7: 跑全量离线测试**

```bash
cd services/bindcraft2-server && uv run --group dev python -m pytest tests/ -q -m "not fc"
```

Expected：`test_models` + `test_campaigns` + `test_tools` + `test_adapter` + `test_app` 全绿。

- [ ] **Step 8: Commit**

```bash
git add services/bindcraft2-server/app.py services/bindcraft2-server/tests/conftest.py \
        services/bindcraft2-server/tests/data/fake_bindcraft.sh \
        services/bindcraft2-server/tests/test_app.py
git commit -m "feat(bin### Task 7: __main__.py（CLI 批处理）+ test_cli.py

**Files:**
- Create: `services/bindcraft2-server/__main__.py`
- Test: `services/bindcraft2-server/tests/test_cli.py`

- [ ] **Step 1: 写失败测试**

```python
"""CLI 批处理模式：endpoint 注册、argv 回调、create_cli 端到端。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from bioq_service.cli import create_cli
from server.adapter import Bindcraft2Adapter
from server.settings import Bindcraft2Settings
from server.tools import build_campaign_json

import server.__main__ as cli_main


@pytest.fixture
def offline(tmp_path: Path, offline_settings):
    return offline_settings


def test_endpoint_registry_covers_three_subcommands():
    assert set(cli_main.endpoints.keys()) == {"design", "rank", "filter"}


def test_design_endpoint_declares_optional_target_input():
    ep = cli_main.endpoints["design"]
    assert "target" in ep.inputs
    help_text, required = ep.inputs["target"]
    assert required is False, "target 可选：也可以走 --target-name"
    assert "target-name" not in ep.inputs


def test_rank_and_filter_have_no_file_inputs():
    assert cli_main.endpoints["rank"].inputs == {}
    assert cli_main.endpoints["filter"].inputs == {}


def test_design_build_writes_campaign(tmp_path, offline):
    job_dir = tmp_path / "run"
    (job_dir / "output").mkdir(parents=True)
    req = cli_main.endpoints["design"].request_model(target_name="hPDL1", max_trajectories=3)
    argv = cli_main.endpoints["design"].build_argv(req, {}, job_dir, offline)
    assert argv[-2] == "design"
    campaign = json.loads(Path(argv[-1]).read_text(encoding="utf-8"))
    assert campaign["target"] == "hPDL1"
    assert campaign["max_trajectories"] == 3


def test_design_build_uses_target_input(tmp_path, offline):
    job_dir = tmp_path / "run"
    (job_dir / "output").mkdir(parents=True)
    target = tmp_path / "t.pdb"
    target.write_text("ATOM\n", encoding="utf-8")
    req = cli_main.endpoints["design"].request_model(max_trajectories=1)
    argv = cli_main.endpoints["design"].build_argv(req, {"target": target}, job_dir, offline)
    campaign = json.loads(Path(argv[-1]).read_text(encoding="utf-8"))
    assert campaign["targets"][0]["target_path"] == str(target.resolve())


def test_rank_build_resolves_campaign_uri(tmp_path, offline):
    campaign = tmp_path / "jobs" / "abc" / "output"
    campaign.mkdir(parents=True)
    job_dir = tmp_path / "run"
    (job_dir / "output").mkdir(parents=True)
    req = cli_main.endpoints["rank"].request_model(campaign_uri="job://abc", on="i_pTM")
    argv = cli_main.endpoints["rank"].build_argv(req, {}, job_dir, offline)
    assert str(campaign) in argv
    assert argv[-1] == str((job_dir / "output" / "ranked_by_i_pTM.csv").resolve())


def test_cli_design_success(tmp_path, offline):
    adapter = Bindcraft2Adapter(settings=offline)
    output_dir = tmp_path / "run" / "output"
    with patch.object(sys, "argv", [
        "prog", "design", "--target-name", "hPDL1", "--max-trajectories", "1",
        "--output-dir", str(output_dir),
    ]):
        with pytest.raises(SystemExit) as exc:
            create_cli(adapter, offline, cli_main.endpoints, version="0.0.1")
    assert exc.value.code == 0
    assert (output_dir / "3_Ranked" / "!_Ranked.csv").is_file()


def test_cli_rank_json_output(tmp_path, offline, capsys):
    adapter = Bindcraft2Adapter(settings=offline)
    campaign = tmp_path / "jobs" / "abc" / "output"
    campaign.mkdir(parents=True)
    output_dir = tmp_path / "run" / "output"
    with patch.object(sys, "argv", [
        "prog", "rank", "--campaign-uri", "job://abc", "--on", "i_pTM",
        "--json", "--output-dir", str(output_dir),
    ]):
        with pytest.raises(SystemExit) as exc:
            create_cli(adapter, offline, cli_main.endpoints, version="0.0.1")
    assert exc.value.code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "completed"


def test_cli_no_subcommand_exits_2(tmp_path, offline):
    adapter = Bindcraft2Adapter(settings=offline)
    with patch.object(sys, "argv", ["prog"]):
        with pytest.raises(SystemExit, match="2"):
            create_cli(adapter, offline, cli_main.endpoints)


def test_cli_requires_target_or_target_name(tmp_path, offline):
    adapter = Bindcraft2Adapter(settings=offline)
    output_dir = tmp_path / "run" / "output"
    with patch.object(sys, "argv", ["prog", "design", "--output-dir", str(output_dir)]):
        with pytest.raises(SystemExit) as exc:
            create_cli(adapter, offline, cli_main.endpoints)
    assert exc.value.code != 0
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cd services/bindcraft2-server && uv run --group dev python -m pytest tests/test_cli.py -q 2>&1 | tail -10
```

Expected：`ModuleNotFoundError: No module named 'server.__main__'`。

- [ ] **Step 3: 写 __main__.py**

```python
"""bindcraft2-server 的 CLI 批处理入口。

用法::

    python -m server design --target-name hPDL1 --max-trajectories 200 \\
        --output-dir /scratch/$SLURM_JOB_ID/
    python -m server design --target /data/target.pdb --target-chains A \\
        --output-dir /scratch/$SLURM_JOB_ID/
    python -m server rank --campaign-uri file:///scratch/run1/output --on i_pTM \\
        --output-dir /scratch/run2/
    python -m server filter --campaign-uri file:///scratch/run1/output \\
        --where 'i_pAE=0.45' --output-dir /scratch/run3/
"""

from __future__ import annotations

import sys
from pathlib import Path

from bioq_service.cli import CLIEndpoint, create_cli

from .adapter import Bindcraft2Adapter
from .campaigns import resolve_campaign_dir
from .models import DesignRequest, FilterRequest, RankRequest
from .settings import Bindcraft2Settings
from .tools import design_argv, filter_argv, prepare_design, rank_argv

settings = Bindcraft2Settings()
adapter = Bindcraft2Adapter(settings=settings)


def _design_build(req: DesignRequest, inputs: dict[str, Path], job_dir: Path, settings) -> list[str]:
    target_path = inputs.get("target")
    # "二选一"这种条件必填只能在 build 回调里查：create_cli 只强制 argparse 层的
    # required，表达不了跨字段条件（`inputs["target"]` 声明为可选，见 CLIEndpoint）。
    # 不查的话会一路走到 prepare_design 抛 ValueError，用户看到的是一整坨
    # traceback 而不是用法提示。仓库既有先例：services/diffdock-server/cli_impl.py。
    if target_path is None and not req.target_name:
        print("error: one of --target or --target-name is required", file=sys.stderr)
        raise SystemExit(2)
    campaign_file = prepare_design(
        req, job_dir=job_dir, target_path=target_path, settings=settings
    )
    return design_argv(campaign_file, job_dir, settings)


def _rank_build(req: RankRequest, _inputs: dict[str, Path], job_dir: Path, settings) -> list[str]:
    return rank_argv(req, resolve_campaign_dir(req.campaign_uri, settings), job_dir, settings)


def _filter_build(req: FilterRequest, _inputs: dict[str, Path], job_dir: Path, settings) -> list[str]:
    return filter_argv(req, resolve_campaign_dir(req.campaign_uri, settings), job_dir, settings)


endpoints = {
    "design": CLIEndpoint(
        name="design",
        help="Run a full BindCraft2 design campaign",
        request_model=DesignRequest,
        build_argv=_design_build,
        inputs={
            "target": (
                "Target structure (PDB/mmCIF/FASTA); omit when using --target-name",
                False,
            )
        },
    ),
    "rank": CLIEndpoint(
        name="rank",
        help="Re-rank an existing campaign on other measurements",
        request_model=RankRequest,
        build_argv=_rank_build,
    ),
    "filter": CLIEndpoint(
        name="filter",
        help="Re-apply acceptance thresholds to an existing campaign",
        request_model=FilterRequest,
        build_argv=_filter_build,
    ),
}

if __name__ == "__main__":
    # 守卫是必需的，不是风格问题：tests/test_cli.py 直接 import 本模块来复用
    # endpoints 注册表（避免像 rfantibody 那样在测试里重复一份）。无守卫时 import
    # 会立刻执行 create_cli，按 pytest 的 sys.argv 解析 → argparse SystemExit(2)。
    # `python -m server <endpoint>` 路径下 __name__ == "__main__"，行为不变。
    create_cli(adapter, settings, endpoints, version="0.0.1")
```

- [ ] **Step 4: 跑测试确认通过**

```bash
cd services/bindcraft2-server && uv run --group dev python -m pytest tests/test_cli.py -q 2>&1 | tail -20
```

Expected：全部通过（`tests/test_cli.py` 共 10 个用例）。

另外单独确认 `import` 无副作用（守卫生效）：

```bash
cd services/bindcraft2-server && uv run --group dev \
  python -c "import server.__main__ as m; print('import side-effect free:', sorted(m.endpoints))"
```

Expected：`import side-effect free: ['design', 'filter', 'rank']`，退出码 0（没有 SystemExit）。

- [ ] **Step 5: lint**

```bash
uvx ruff check services/bindcraft2-server/
```

Expected：`All checks passed!`。有问题就地修（常见：未使用的 import）。

- [ ] **Step 6: Commit**

```bash
git add services/bindcraft2-server/__main__.py services/bindcraft2-server/tests/test_cli.py
git commit -m "feat(bindcraft2-server): add CLI batch-mode entry with three subcommands"
```

---

 -m "feat(bindcraft2-server): add CLI batch-mode entry with three subcommands"
```

---

### Task 8: vendor.sh

**Files:**
- Create: `services/bindcraft2-server/scripts/vendor.sh`

- [ ] **Step 1: 写 vendor.sh**

```bash
#!/usr/bin/env bash
# 把上游 BindCraft2 源码 vendor 到 services/bindcraft2-server/upstream/，
# 固定 SHA，使 `docker build` 不访问网络。每次构建前跑一次（升级时重跑）：
#
#   ./services/bindcraft2-server/scripts/vendor.sh
#
# CN 网络下可用镜像：
#   BINDCRAFT2_REPO=https://ghproxy.cn/https://github.com/PacesaLab/BindCraft2 \
#       ./services/bindcraft2-server/scripts/vendor.sh
#
# 升级 pin：改 BINDCRAFT2_SHA。
#
# 注意：**不要**排除 bindcraft/weights/proteinmpnn/ —— 三个变体的 .npz 是
# 上游 pyproject 的 package-data（26 MB/变体，共 ~78 MB），随包分发；排除它们会让 design 在
# ProteinMPNN 重设计阶段才失败。

set -euo pipefail

BINDCRAFT2_REPO="${BINDCRAFT2_REPO:-https://github.com/PacesaLab/BindCraft2}"
BINDCRAFT2_SHA="${BINDCRAFT2_SHA:-5342aefa18dedad653f7a5f6dbee1e566ca24d8f}"  # v1.0.1

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
DST="$PROJECT_ROOT/services/bindcraft2-server/upstream"
TMP="$(mktemp -d -t bc2-vendor.XXXXXX)"
trap "rm -rf '$TMP'" EXIT

mkdir -p "$DST"

for i in 1 2 3 4 5; do
    rm -rf "$TMP/repo"
    if git clone --filter=blob:none --no-checkout "$BINDCRAFT2_REPO" "$TMP/repo"; then
        break
    fi
    [ "$i" = "5" ] && {
        echo "ERROR: git clone failed after 5 attempts" >&2
        exit 1
    }
    echo "  clone failed, retrying in $((i*10))s ..."
    sleep $((i*10))
done

cd "$TMP/repo"
git checkout "$BINDCRAFT2_SHA"
actual="$(git rev-parse HEAD)"
if [[ "$actual" != "$BINDCRAFT2_SHA" ]]; then
    echo "ERROR: HEAD mismatch after checkout (got $actual, expected $BINDCRAFT2_SHA)" >&2
    exit 1
fi
rm -rf .git

# --delete 清掉上一次 vendor 的陈旧文件。docs/ 与 containers/ 不进镜像
# （Dockerfile 会再剪一次），但留在 upstream/ 便于本地比对上游文档。
rsync -a --delete \
    --exclude='__pycache__' --exclude='*.pyc' --exclude='.venv' --exclude='results' \
    "$TMP/repo/" "$DST/"

# 自检：ProteinMPNN 权重必须在（少了会在重设计阶段才炸）
for variant in neutral negative positive; do
    f="$DST/bindcraft/weights/proteinmpnn/weights_${variant}/v_48_020.npz"
    [ -f "$f" ] || { echo "ERROR: missing shipped ProteinMPNN weights: $f" >&2; exit 1; }
done
# 自检：运行期需要的 preset 与 scaffold 树必须在
for d in settings/core settings/modality settings/property settings/target scaffolds; do
    [ -d "$DST/$d" ] || { echo "ERROR: missing upstream tree: $DST/$d" >&2; exit 1; }
done

echo "Vendored $BINDCRAFT2_REPO @ $BINDCRAFT2_SHA"
echo "  -> $DST"
du -sh "$DST"
```

- [ ] **Step 2: 跑 vendor 并验证产物**

```bash
chmod +x services/bindcraft2-server/scripts/vendor.sh
./services/bindcraft2-server/scripts/vendor.sh
ls services/bindcraft2-server/upstream/ | head
ls services/bindcraft2-server/upstream/bindcraft/weights/proteinmpnn/
```

Expected：末尾打印 `Vendored ... @ 5342aefa...` + 体积；`upstream/` 下有 `bindcraft/ settings/ scaffolds/ pyproject.toml` 等；`proteinmpnn/` 下有三个 `weights_*` 目录。

- [ ] **Step 3: 确认 pin 版本正确**

```bash
grep -n '^version' services/bindcraft2-server/upstream/pyproject.toml
grep -n 'CAMPAIGN_MODELS' services/bindcraft2-server/upstream/bindcraft/model_weights.py
```

Expected：`version = "1.0.1"`；`CAMPAIGN_MODELS = tuple(f'model_{index}_multimer_v3' for index in range(1, 6)) + ('model_1_ptm', 'model_2_ptm')`——与 `tools.CAMPAIGN_MODELS` 一致。

- [ ] **Step 4: 确认 upstream/ 未被 git 跟踪**

```bash
git status --short services/bindcraft2-server/ | head
```

Expected：只看到 `scripts/vendor.sh` 是 untracked；**没有** `upstream/...` 条目。

- [ ] **Step 5: Commit**

```bash
git add services/bindcraft2-server/scripts/vendor.sh
git commit -m "build(bindcraft2-server): add upstream vendoring script pinned to v1.0.1"
```

---

### Task 9: fetch_weights.sh

**Files:**
- Create: `services/bindcraft2-server/scripts/fetch_weights.sh`

- [ ] **Step 1: 写 fetch_weights.sh**

```bash
#!/usr/bin/env bash
# 预取 BindCraft2 需要的 7 个 AlphaFold 检查点（上游称 ~5.3 GB，实测归档
# 5,587,968,000 B ≈ 5.59 GB），只解出这 7 个文件，其余丢弃。
#
# 默认落到 services/bindcraft2-server/weights/（stage 目录）；
# 正式部署直接下到 NAS：
#
#   WEIGHTS_DST=/data/models/bindcraft2/alphafold \
#       ./services/bindcraft2-server/scripts/fetch_weights.sh
#
# 若 NAS 上 /data/models/alphafold 已由 alphafold-server 就位，则不必跑本脚本：
#   ln -s /data/models/alphafold /data/models/bindcraft2/alphafold
# （上游查找顺序：<dir>/params/params_<model>.npz 或 <dir>/params_<model>.npz。）
#
# 本脚本**不**处理 ProteinMPNN 权重：它们随包分发（vendor.sh 已校验）。
#
# 归档布局（实测 alphafold_params_2022-12-06.tar）：16 个成员——15 个
# params_model_*.npz + LICENSE，全部在归档**顶层**（没有 params/ 前缀），
# 每个 npz 约 356 MiB，远高于上游 100 MiB 的"未完成"下限。

set -euo pipefail

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
DST="${WEIGHTS_DST:-$PROJECT_ROOT/services/bindcraft2-server/weights/alphafold}"
URL="${ALPHAFOLD_PARAMS_URL:-https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar}"

MODELS=(
    model_1_multimer_v3 model_2_multimer_v3 model_3_multimer_v3
    model_4_multimer_v3 model_5_multimer_v3
    model_1_ptm model_2_ptm
)

# 目标已存在但不是目录：立刻失败，避免白下 5.59 GB 才在 mkdir 处报错。
if [ -e "$DST" ] && [ ! -d "$DST" ]; then
    echo "ERROR: WEIGHTS_DST exists but is not a directory: $DST" >&2
    exit 1
fi

TMP="$(mktemp -d -t bc2-weights.XXXXXX)"
trap "rm -rf '$TMP'" EXIT

echo "Downloading AlphaFold parameters -> $TMP"
echo "  $URL"
mkdir -p "$TMP/params"

# -C - 断点续传（对齐兄弟服务的 wget -c）；归档支持 Range（accept-ranges: bytes）。
for attempt in 1 2 3; do
    if curl -fL -C - --retry 3 --retry-delay 5 -o "$TMP/alphafold_params.tar" "$URL"; then
        break
    fi
    [ "$attempt" = "3" ] && { echo "ERROR: download failed after 3 attempts" >&2; exit 1; }
    echo "  download failed, retrying ..."
    sleep $((attempt * 10))
done

# 只解出需要的 7 个成员。成员名在归档顶层，故用精确名匹配；**不加** `*/` 前缀
# 候选——GNU tar 对任一未命中的模式都会以退出码 2 失败（实测），会让下面的
# `||` 分支在下载完全成功时也误触发。
members=()
for model in "${MODELS[@]}"; do
    members+=(--wildcards "params_${model}.npz")
done

echo "Extracting 7 of the archive's checkpoints ..."
tar -xf "$TMP/alphafold_params.tar" -C "$TMP/params" "${members[@]}" 2>/dev/null || {
    echo "ERROR: none of the expected members matched; archive layout changed?" >&2
    tar -tf "$TMP/alphafold_params.tar" | head -20 >&2
    exit 1
}

# tar 会把 members 解到各自的目录层级；统一摊平到 params/。
find "$TMP/params" -mindepth 2 -name 'params_*.npz' -exec mv -f {} "$TMP/params/" \;
find "$TMP/params" -mindepth 1 -type d -exec rm -rf {} +

mkdir -p "$DST/params"
for model in "${MODELS[@]}"; do
    src="$TMP/params/params_${model}.npz"
    [ -f "$src" ] || { echo "ERROR: missing params_${model}.npz in archive" >&2; exit 1; }
    # 上游用 100 MB 下限判"未完成"——小于此值的文件视为损坏
    size="$(stat -c%s "$src")"
    [ "$size" -ge $((100 << 20)) ] || { echo "ERROR: ${model} truncated ($size bytes)" >&2; exit 1; }
    mv -f "$src" "$DST/params/params_${model}.npz"
done

echo "Done. 7 checkpoints at $DST/params/"
du -sh "$DST"
```

- [ ] **Step 2: 语法检查（不实际下载 5.3 GB）**

```bash
chmod +x services/bindcraft2-server/scripts/fetch_weights.sh
bash -n services/bindcraft2-server/scripts/fetch_weights.sh && echo "syntax OK"
shellcheck services/bindcraft2-server/scripts/fetch_weights.sh 2>/dev/null || echo "(shellcheck 未安装，跳过)"
```

Expected：`syntax OK`。

- [ ] **Step 3: 验证 P1：NAS 上是否已有可复用的参数**

```bash
ls /data/models/alphafold/params/params_model_{1,2,3,4,5}_multimer_v3.npz \
   /data/models/alphafold/params/params_model_{1,2}_ptm.npz 2>&1 | head
```

Expected 二选一：
- 7 个路径全部打印 → **走软链分支**：`ln -s /data/models/alphafold /data/models/bindcraft2/alphafold`，不跑下载；
- `No such file` → 跑 Step 4 的下载。

（本步是设计文档 P1 的核验动作；结果记进 Task 13 的设计文档回写。）

- [ ] **Step 4: （仅当 Step 3 缺失时）下载到 NAS**

```bash
WEIGHTS_DST=/data/models/bindcraft2/alphafold ./services/bindcraft2-server/scripts/fetch_weights.sh
```

Expected：`Done. 7 checkpoints at /data/models/bindcraft2/alphafold/params/`。

- [ ] **Step 5: Commit**

```bash
git add services/bindcraft2-server/scripts/fetch_weights.sh
git commit -m "build(bindcraft2-server): add AlphaFold parameter staging script"
```

---

**为什么 tar 的成员模式只写一种（`params_<model>.npz`）而不是再加一条 `*/params_<model>.npz`：**
GNU tar 只要有**任意一个**模式匹配不到就以 2 退出（哪怕其余文件都成功解出）。实测该归档是
**扁平**的（16 个成员全在顶层，无 `params/` 前缀），所以 `*/...` 那条永远匹配不到 → tar 退出 2
→ 计划初稿的 `|| { echo ERROR...; exit 1; }` 会在**下载完全成功**的情况下误报失败，`$DST/params`
根本不会被创建。因此只保留能命中的那一种模式，同时保留"布局变了就大声失败"的行为（实测把
归档换成 `params/params_*.npz` 布局后，脚本确实以 1 退出并打印诊断）。
### Task 10: Dockerfile + 本地构建验证

**Files:**
- Create: `services/bindcraft2-server/Dockerfile`

- [ ] **Step 1: 写 Dockerfile**

```dockerfile
# 从仓库根构建：
#   docker build --platform linux/amd64 -t bindcraft2-server \
#       -f services/bindcraft2-server/Dockerfile .
#
# 构建前置（各跑一次；升级 pin 时重跑）：
#   ./services/bindcraft2-server/scripts/vendor.sh
#   ./services/bindcraft2-server/scripts/fetch_weights.sh   # 或 NAS 软链，见设计文档 P1
#
# 上游源码来自 services/bindcraft2-server/upstream/（vendor.sh 产物，固定 SHA）。
# AlphaFold 参数 5.3 GB **不烘焙**——挂 NAS 到 /data/models/bindcraft2/。
#
# 基础镜像用纯 ubuntu:24.04（与上游 containers/Dockerfile 一致）：jax cuda13
# wheels 自带 CUDA/cuDNN，宿主机只需提供 NVIDIA 驱动 + container toolkit。
# 用 nvidia/cuda base 不但冗余，还可能引入版本冲突。
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
ENV UV_HTTP_TIMEOUT=600
ENV UV_CONCURRENT_DOWNLOADS=4

# python3 = 3.12（ubuntu:24.04 默认），满足上游 requires-python >=3.12。
# 不装 git：上游源码已 vendor；不装 build-essential：amd64 上全是 wheel。
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/

# --- 上游源码 ---
# editable install：settings/ 与 scaffolds/ 在仓库根（不在包里），运行期按仓库根
# 相对定位，所以源码树必须留在 /opt/bindcraft 并留在 import path 上。
COPY services/bindcraft2-server/upstream /opt/bindcraft

# docs/ 与 containers/ 是构建期无关的大块，剪掉省体积；results/ 不该存在。
RUN rm -rf /opt/bindcraft/docs /opt/bindcraft/containers /opt/bindcraft/.github \
    && find /opt/bindcraft -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

RUN uv venv /opt/venv --python python3

# cuda13 extra：需要 compute capability >= 7.5（T4/A10/L20/A100）。
# V100(7.0) 及更老改用 [cuda12]。
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python -e "/opt/bindcraft[cuda13]"

# jax 的 cuda 库在 site-packages/nvidia/*/lib 下，动态链接器默认不查那里；
# 同时镜像必须显式声明这两个变量，NVIDIA container hook 才会注入宿主机驱动
# （纯 ubuntu base 不会像 CUDA base 那样预置）。
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN python3 -c "import nvidia, pathlib; print('\n'.join(sorted(str(p) for root in nvidia.__path__ for p in pathlib.Path(root).glob('*/lib'))))" \
        > /etc/ld.so.conf.d/bindcraft-cuda.conf \
 && ldconfig && ldconfig -p | grep -q libcupti

# --- 共享服务框架 ---
# COPY（不是 bind-mount）：bind mount 的内容不进 BuildKit 层缓存键，会发出陈旧框架。
COPY framework /tmp/service-framework
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python \
        "/tmp/service-framework[mcp]" httpx alibabacloud-oss-v2

# --- 服务代码 ---
# 显式文件清单：测试/文档改动不破坏这一层缓存。
COPY services/bindcraft2-server/__init__.py \
     services/bindcraft2-server/__main__.py \
     services/bindcraft2-server/VERSION \
     services/bindcraft2-server/app.py \
     services/bindcraft2-server/adapter.py \
     services/bindcraft2-server/campaigns.py \
     services/bindcraft2-server/models.py \
     services/bindcraft2-server/settings.py \
     services/bindcraft2-server/tools.py \
     /opt/bindcraft2/server/

# --- 构建期自检 ---
# 让"wheel 没对上"和"包没带上权重"在这里失败，而不是在别人第一次 campaign 时。
RUN test -f /opt/bindcraft/bindcraft/weights/proteinmpnn/weights_neutral/v_48_020.npz \
 && test -f /opt/bindcraft/bindcraft/weights/proteinmpnn/weights_negative/v_48_020.npz \
 && test -f /opt/bindcraft/bindcraft/weights/proteinmpnn/weights_positive/v_48_020.npz \
 && test -d /opt/bindcraft/settings/target \
 && test -f /opt/bindcraft/bindcraft/cli.py \
 && /opt/venv/bin/python -c "import jax; print('jax', jax.__version__)" \
 && /opt/venv/bin/python -c "import bindcraft.proteinmpnn; print('proteinmpnn ok')" \
 && cd / && /opt/venv/bin/python -m bindcraft.cli --help > /dev/null

# --- 运行期环境 ---
ENV PYTHONPATH=/opt/bindcraft2
ENV BINDCRAFT2_ROOT=/opt/bindcraft
ENV BINDCRAFT2_PYTHON=/opt/venv/bin/python
ENV BINDCRAFT2_JOBS_BASE_DIR=/data/bindcraft2_jobs
ENV BINDCRAFT2_ALPHAFOLD_PARAMS_DIR=/data/models/bindcraft2/alphafold
ENV BINDCRAFT2_COMPILE_CACHE_DIR=/data/models/bindcraft2/xla_cache
ENV BINDCRAFT2_MAX_CONCURRENT_JOBS=1
ENV BINDCRAFT2_WORKERS_PER_GPU=1
ENV BINDCRAFT2_MAX_WORKERS_PER_GPU=1
# 输出回传：网关带 X-Bioagent-Oss-Prefix 时框架把完成的 job 目录镜像到这里。
# FC 控制台需要把数据面 OSS bucket 挂到 /mnt/oss（RW）；没挂则是 no-op。
ENV BINDCRAFT2_OSS_OUTPUT_MOUNT=/mnt/oss
# FC 会话亲和：在返回 job_id 的 POST 响应里回显这个 header。
ENV BINDCRAFT2_SESSION_HEADER_NAME=bioagent-session-id
RUN mkdir -p /data/bindcraft2_jobs

WORKDIR /opt/bindcraft

# ldconfig 必须在访问 GPU 之前跑过（非 root 场景下 /sbin/ldconfig 需要 setuid）。
RUN echo $'#!/bin/bash\nldconfig\nexec "$@"' > /opt/entrypoint.sh \
 && chmod +x /opt/entrypoint.sh
ENTRYPOINT ["/opt/entrypoint.sh"]

EXPOSE 9000
CMD ["/opt/venv/bin/python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", "--port", "9000", "--timeout-keep-alive", "900"]
```

- [ ] **Step 2: 构建镜像**

```bash
cd /home/zhiheng/Documents/Work/Noah/Tasks/Project/bioq-services && make build-bindcraft2-server 2>&1 | tail -30
```

Expected：构建成功（首次可能十几分钟：jax cuda13 wheels + cuDNN 数 GB）。若 `make` 未发现
该服务，先确认 `services/bindcraft2-server/Dockerfile` 存在且 `make list` 里有
`bindcraft2-server`。

- [ ] **Step 3: 离线 smoke — 镜像内 CLI 可应答（不碰 GPU）**

```bash
docker run --rm bindcraft2-server:v0.0.1 /opt/venv/bin/python -m server --help
docker run --rm bindcraft2-server:v0.0.1 /opt/venv/bin/python -m server design --help
```

Expected：打印三个子命令 / design 的参数列表，退出码 0。（`--help` 不建 JAX runtime。）

- [ ] **Step 4: manifest 路由 sanity**

```bash
docker run --rm -p 9000:9000 bindcraft2-server:v0.0.1 &
sleep 20
curl -s http://localhost:9000/openapi.json | python3 -c "import sys,json; print('\n'.join(sorted(p for p in json.load(sys.stdin)['paths'] if 'api' in p)))"
curl -s http://localhost:9000/api/manifest | python3 -c "import sys,json; d=json.load(sys.stdin); print([e['path'] for e in d['endpoints']])"
kill %1
```

Expected：`/api/design`、`/api/rank`、`/api/filter`、`/api/tasks/design`、
`/api/tasks/rank`、`/api/tasks/filter` 六条齐全；manifest 的 endpoints 中每条都有非空
`examples`。

- [ ] **Step 5: 确认没有 git clone / weights COPY**

```bash
grep -nE 'git clone|COPY [^ ]*weights' services/bindcraft2-server/Dockerfile || echo "clean"
```

Expected：`clean`。

- [ ] **Step 6: Commit**

```bash
git add services/bindcraft2-server/Dockerfile
git commit -m "build(bindcraft2-server): add dual-mode docker image with jax cuda13"
```

---

### Task 11: 部署描述、注册表、第三方声明、README

**Files:**
- Create: `services/bindcraft2-server/deploy/fc.yaml`
- Create: `services/bindcraft2-server/README.md`
- Modify: `services.yaml`（新增条目）
- Modify: `THIRD_PARTY_NOTICES`（新增 BC2 行）

- [ ] **Step 1: 写 deploy/fc.yaml**

以 `services/alphafold-server/deploy/fc.yaml` 为模板，改这些字段：

```yaml
edition: 3.0.0
name: fc3-bindcraft2
access: default
resources:
  fcDemo:
    component: fc3
    props:
      region: cn-hangzhou
      handler: index.handler
      diskSize: 10240
      instanceLifecycleConfig:
        preStop:
          handler: 'true'
          timeout: 3
      runtime: custom-container
      cpu: 8
      memorySize: 32768
      instanceConcurrency: 200
      vpcConfig:
        securityGroupId: <照 alphafold 填>
        anytunnelViaENI: false
        vpcId: <照 alphafold 填>
        vSwitchIds:
          - <照 alphafold 填>
      role: acs:ram::1713785072839647:role/bioresearch-fc
      disableOndemand: false
      nasConfig:
        groupId: 0
        mountPoints:
          - enableTLS: false
            serverAddr: 0620a487b6-gum23.cn-hangzhou.nas.aliyuncs.com:/fc
            mountDir: /data
        userId: 0
      description: 'BindCraft2 protein binder design campaigns (internal use only)'
      # campaign 是小时级：对齐 alphafold / boltzgen 的 36000s。
      timeout: 36000
      sessionAffinityConfig:
        affinityHeaderFieldName: bioagent-session-id
        disableSessionIdReuse: false
        enableAutoPause: false
        sessionConcurrencyPerInstance: 1
        sessionIdleTimeoutInSeconds: 1800
        sessionTTLInSeconds: 86400
      internetAccess: true
      gpuConfig:
        # P2 核验后确认卡型；cuda13 extra 需 compute capability >= 7.5。
        gpuMemorySize: 24576
        gpuType: fc.gpu.ada.1
      ossMountConfig:
        mountPoints:
          - bucketName: bio-gateway
            endpoint: http://oss-cn-hangzhou-internal.aliyuncs.com
            bucketPath: ''
            mountDir: /mnt/oss
```

VPC 三件套与 `gpuConfig` 的合法取值**必须**照 FC 控制台当前值填写——`deploy/fc.yaml` 是
部署描述，不参与镜像构建，但它要能 `s deploy` 成功。若只是归档用，把不认识的行删掉并留
注释（说明实际值在控制台上）。

- [ ] **Step 2: 验证 YAML 可解析**

```bash
python3 -c "import yaml,sys; yaml.safe_load(open('services/bindcraft2-server/deploy/fc.yaml')); print('fc.yaml OK')"
```

Expected：`fc.yaml OK`。

- [ ] **Step 3: 写 README.md**

```markdown
# bindcraft2-server

把 [BindCraft2](https://github.com/PacesaLab/BindCraft2)（BC2 v1.0.1）包装成 bioq 服务：
一次 `design` 请求跑一场完整的蛋白 binder 设计 campaign（AlphaFold 2 hallucination 设计 →
ProteinMPNN 重设计 → 独立 AlphaFold 验证 → 过滤排序），并可用 `rank` / `filter` 对已有
campaign 换指标、换阈值，无需重跑设计。

## ⚠️ 许可证红线（先读这段）

BC2 采用 **BindCraft2 Source-Available License (Hosting-Restricted)**
（Copyright (c) 2026 Martin Pacesa, University of Zurich），**不是** OSI 开源许可证。

- **仅限组织内部使用**：本地 / 私有基础设施 / 私有云 / 为使用组织单独运营的基础设施；
  由本组织自己的自动化与 agent 工具链驱动执行属于允许范围。
- **不得对外提供工具功能**：未经 UZH 单独商业许可，不得把本服务或其修改版作为
  hosted / managed / cloud / API / web application / workflow platform / SaaS 提供给第三方
  （许可证点名了"作为 hosted platform 里可调用的 tool / agent action / connector"这一形态）。
- **不得分发预配置部署**：不得把本镜像 / VM 镜像 / 安装器交给第三方去托管。
- **命名**：不得用 "BindCraft2" 标识、宣传 Hosted Service 或衍生作品，以免暗示与 UZH 的
  官方关联或等效性；"基于 BindCraft2" 这类准确陈述是允许的。

完整条款见上游 `LICENSE`，以及仓库根 [`THIRD_PARTY_NOTICES`](../../THIRD_PARTY_NOTICES)。

## Endpoints

| Endpoint | 说明 |
|---|---|
| `POST /api/design` | 提交一场 campaign（submit/poll） |
| `POST /api/tasks/design` | 同上，FC 异步任务模式（**推荐**，campaign 小时级） |
| `POST /api/rank` | 对已有 campaign 按指标重排 → `output/ranked_by_<metric>.csv` |
| `POST /api/tasks/rank` | 同上，task 模式 |
| `POST /api/filter` | 对已有 campaign 重放/覆盖阈值 → `output/filtered.csv` |
| `POST /api/tasks/filter` | 同上，task 模式 |

请求参数、curl 示例与字段表以运行中实例的 `GET /api/manifest` 为准（`endpoint_examples()`
是单一真源）。

### 快速上手

```bash
# shipped target
curl -X POST $URL/api/design -F target_name=hPDL1 -F 'binder_lengths=[80,80]' \
     -F number_of_final_designs=10 -F max_trajectories=200

# 自己的 target
curl -X POST $URL/api/design -F target=@target.pdb -F target_chains=A \
     -F 'hotspots=54,56,66-70' -F modality=VHH -F humanize=true

# 对上一场结果换指标重排（零拷贝，共享 NAS）
curl -X POST $URL/api/rank -F campaign_uri=job://<design_job_id> \
     -F 'on=["i_pTM","i_pDAE"]'

# 换阈值重新筛选
curl -X POST $URL/api/filter -F campaign_uri=job://<design_job_id> \
     -F 'where=["i_pAE=0.45","Interface_Residues>=7"]'
```

`binder_lengths` / `on` / `where` 在 HTTP 上以 **JSON 字符串**传入（`[80,80]`），
在 CLI 上也接受逗号写法（`80,80`）。

### CLI 批处理（Slurm）

```bash
apptainer exec --nv --bind /data:/data bindcraft2-server.sif \
    /opt/venv/bin/python -m server design --target-name hPDL1 \
    --max-trajectories 200 --output-dir /data/runs/$SLURM_JOB_ID/
```

## 运行约束

- **无 CPU 模式**。cuda13 extra 需 compute capability ≥ 7.5；V100(7.0) 及更老要改成
  `[cuda12]`（改 Dockerfile 一行）。
- **显存**：单 worker 预算 `2.0*(3.4GB + 38kB*N²)`，N=256 约 11.8 GB、N=384 约 18 GB、
  N=512 约 26.7 GB，另留 4 GB headroom。服务锁死 `workers_per_gpu=1`（BC2 的宿主内存
  上限读节点可用内存，会过度 pack）。
- **时长**：FC `timeout: 36000`（10 h）是硬上限。`max_trajectories` 是唯一的主动兜底。
  超时后任务 FAILED，但 `1_Trajectories/`、`2_Refolded/` 的中间结果仍在 NAS 上。
  **v0.0.1 无自动 resume**（框架每次 submit 建新 job_dir，BC2 原生续跑跨 job 失效）。
- **进度**：无实时百分比。看 `/log` 尾部，或直接读
  `job://<id>/1_Trajectories/!_Trajectories.csv` 行数。
- **冷启动**：JAX runtime + 模型编译。`JAX_COMPILATION_CACHE_DIR` 指到 NAS 可缓解；
  缓存按卡型分目录，跨卡型不可复用。

## 输出

```
<jobs_base_dir>/<job_id>/
├── input/campaign.json          # 服务生成的 campaign（唯一真源）
├── input/target.pdb             # 上传/URI 落盘（shipped target 时不存在）
├── output/                      # == 上游 project_folder
│   ├── 1_Trajectories/!_Trajectories.csv
│   ├── 2_Refolded/!_Refolded.csv
│   ├── 3_Ranked/!_Ranked.csv    # 主产物：accepted，best-first by i_pDAE
│   ├── summary.csv
│   ├── campaign_metadata.json   # resolved 设置 + checkpoint SHA-256
│   └── workers/worker_<NN>_gpu_<id>.log
└── logs/run.log
```

`detect_outputs` 接受 `3_Ranked/!_Ranked.csv`、`summary.csv`、`ranked_by_*.csv`、
`filtered.csv` 任一非空。**包含 `summary.csv` 是刻意的**：accepted=0 的合法 campaign
不会写 `!_Ranked.csv`。

## Weights

AlphaFold 参数 5.3 GB，**外置 NAS**（不烘焙进镜像）：

```
/data/models/bindcraft2/alphafold/
└── params/
    ├── params_model_1_multimer_v3.npz     # 5 个 multimer（设计 + 验证）
    ├── params_model_2_multimer_v3.npz
    ├── params_model_3_multimer_v3.npz
    ├── params_model_4_multimer_v3.npz
    ├── params_model_5_multimer_v3.npz
    ├── params_model_1_ptm.npz             # 2 个 monomer（验证）
    └── params_model_2_ptm.npz
```

上游接受 `<dir>/params/params_<model>.npz` 或 `<dir>/params_<model>.npz`；服务用
`BINDCRAFT2_ALPHAFOLD_PARAMS_DIR` 指定，并把它注入子进程的 `BINDCRAFT_AF2_PARAMS`
（**必须**——否则上游会去下载 5.3 GB）。

pre-stage：

```bash
# 方式 A：NAS 上已有 alphafold-server 的参数 → 软链，零下载
ln -s /data/models/alphafold /data/models/bindcraft2/alphafold

# 方式 B：下载（只解出这 7 个文件）
WEIGHTS_DST=/data/models/bindcraft2/alphafold \
    ./services/bindcraft2-server/scripts/fetch_weights.sh
```

ProteinMPNN 三变体（`weights_{neutral,negative,positive}/v_48_020.npz`，各 ~6.6 MB）
**随包**分发，由 `vendor.sh` 校验、Dockerfile 构建期自检。

FC 验证：

```bash
curl -s $URL/healthz/detail | python3 -m json.tool
# 期望：weights_loaded=true, weights_missing=[], gpu_backend="gpu"
```

SIF / HPC：

```bash
apptainer exec --nv --bind /scratch/models:/data/models bindcraft2-server.sif \
    /opt/venv/bin/python -m server design --target-name hPDL1 --output-dir /scratch/run1/output
```

## 配置

`env_prefix = BINDCRAFT2_`，全走 pydantic-settings。

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `BINDCRAFT2_JOBS_BASE_DIR` | `/data/bindcraft2_jobs` | 任务目录根 |
| `BINDCRAFT2_ROOT` | `/opt/bindcraft` | 上游源码树（editable install 依赖它） |
| `BINDCRAFT2_PYTHON` | `/opt/venv/bin/python` | 解释器 |
| `BINDCRAFT2_MODULE` | `bindcraft.cli` | 上游 CLI 模块；置空则只跑 `python <args>` |
| `BINDCRAFT2_ALPHAFOLD_PARAMS_DIR` | `/data/models/bindcraft2/alphafold` | AF2 参数目录 |
| `BINDCRAFT2_COMPILE_CACHE_DIR` | `/data/models/bindcraft2/xla_cache` | JAX 图缓存 |
| `BINDCRAFT2_MAX_CONCURRENT_JOBS` | `1` | 整实例独占 GPU |
| `BINDCRAFT2_WORKERS_PER_GPU` | `1` | 覆盖 BC2 自动 pack |
| `BINDCRAFT2_MAX_WORKERS_PER_GPU` | `1` | 同上 |
| `BINDCRAFT2_DEFAULT_MAX_TRAJECTORIES` | `500` | `max_trajectories` 未指定时的兜底 |
| `BINDCRAFT2_GPU_PROBE_TTL_SECONDS` | `300` | `/healthz/detail` GPU 探针缓存 |

## v0.0.1 不做

`score` / `archive` / `unarchive` / `fetch-weights` / `campaign_output` / 参数 sweep /
多 target（同源对、detarget、多链受体）/ 自定义 scaffold / wall-clock 强杀 / 跨 job resume /
实时百分比进度。边界理由见
[设计文档](../../docs/specs/2026-09-22-bindcraft2-server-design.md)。

## 设计文档

[`docs/specs/2026-09-22-bindcraft2-server-design.md`](../../docs/specs/2026-09-22-bindcraft2-server-design.md)
```

- [ ] **Step 4: services.yaml 新增条目**

在 `services:` 下按字母序（`boltzgen-server` 之后、`chembounce-server` 之前）插入：

```yaml
  bindcraft2-server:
    url: https://fc-bindcraft2-<hash>.cn-hangzhou-vpc.fcapp.run
    tier: warm
    function: fc_bindcraft2
    gpu: fc.gpu.ada.1
    oss_mount: true
```

`url` / `function` / `gpu` 在 FC 部署（Task 12）拿到真实值后回填。

- [ ] **Step 5: THIRD_PARTY_NOTICES 新增 BC2 行**

在 License summary 表里 `alphafold-server` 之后、`bindflow-server` 之前插入：

```markdown
| bindcraft2-server | BindCraft2 | PacesaLab/BindCraft2 | Source-Available (Hosting-Restricted) — 仅限内部使用，禁止作为 Hosted Service 对外提供 |
```

- [ ] **Step 6: 校验**

```bash
python3 -c "import yaml; d=yaml.safe_load(open('services.yaml')); print(d['services']['bindcraft2-server'])"
grep -n 'bindcraft2-server' THIRD_PARTY_NOTICES
uvx ruff check services/bindcraft2-server/
```

Expected：打印条目；NOTICES 有且仅有一行；ruff 通过。

- [ ] **Step 7: Commit**

```bash
git add services/bindcraft2-server/deploy/fc.yaml services/bindcraft2-server/README.md \
        services.yaml THIRD_PARTY_NOTICES
git commit -m "docs(bindcraft2-server): add deploy descriptor, README, registry and license notice"
```

---

### Task 12: FC 集成测试 + 部署

**Files:**
- Create: `services/bindcraft2-server/tests/test_fc.py`
- Create: `services/bindcraft2-server/tests/test_fc_task.py`

- [ ] **Step 1: 写 test_fc.py**

```python
"""针对已部署 FC 实例的集成测试（默认 skip；`-m fc` 或 RUN_FC_TESTS=1 才跑）。

⚠️ design 是真实 GPU campaign。smoke 用 shipped target hPDL1 + 1 trailer + 1 个
final design，仍然需要数分钟；不是单元测试。
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import pytest
from bioq_service.fc_testing import fc_url, make_retrying_client

pytestmark = pytest.mark.fc

SERVICE = "bindcraft2-server"
TIMEOUT = 900.0

# 跨测试传递的 job id（模块级 dict，不往 pytest 模块上挂属性）。
STATE: dict[str, str] = {}


@pytest.fixture(scope="module")
def client():
    """会话亲和要求 submit 与之后的每次 poll 都带同一个 header 值——所以 headers
    挂在 client 上，而不是逐个请求传（否则 poll 会被当成新会话，FC 起一堆实例）。"""
    url = fc_url(SERVICE, start=Path(__file__))
    headers = {"bioagent-session-id": f"fc-bindcraft2-{uuid.uuid4().hex[:8]}"}
    with make_retrying_client(
        url, timeout=TIMEOUT, max_retries=10, backoff_s=20.0, headers=headers
    ) as c:
        yield c


def _poll(client, job_id: str, timeout_s: float = 3600.0) -> dict:
    deadline = time.monotonic() + timeout_s
    body: dict = {}
    while time.monotonic() < deadline:
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["status"] in ("completed", "failed"):
            return body
        time.sleep(15)
    raise AssertionError(f"job {job_id} did not finish within {timeout_s}s: {body}")


def test_healthz(client):
    assert client.get("/healthz").json()["status"] == "ok"


def test_healthz_detail_reports_weights_and_gpu(client):
    detail = client.get("/healthz/detail").json()
    assert detail["service"] == "bindcraft2"
    assert detail["weights_loaded"] is True, detail["weights_missing"]
    assert detail["gpu_backend"] == "gpu", detail


def test_manifest_lists_endpoints_with_examples(client):
    body = client.get("/api/manifest").json()
    eps = {e["path"]: e for e in body["endpoints"]}
    for path in (
        "/api/design",
        "/api/tasks/design",
        "/api/rank",
        "/api/tasks/rank",
        "/api/filter",
        "/api/tasks/filter",
    ):
        assert eps[path]["examples"], f"{path} missing examples"


def test_design_shipped_target_smoke(client):
    """一个 trajectory 的最小 campaign（shipped target，保证结构可解析）。"""
    resp = client.post(
        "/api/design",
        data={
            "target_name": "hPDL1",
            "modality": "binder",
            "binder_lengths": "[60,60]",
            "number_of_final_designs": "1",
            "max_trajectories": "1",
            "campaign_name": "fc_smoke",
        },
    )
    resp.raise_for_status()
    job_id = resp.json()["job_id"]

    body = _poll(client, job_id)
    assert body["status"] == "completed", body

    files = client.get(f"/api/jobs/{job_id}/files").json()["files"]
    assert "summary.csv" in files
    assert "campaign_metadata.json" in files
    assert "1_Trajectories/!_Trajectories.csv" in files

    # 记住这个 job 供 rank / filter 接续
    STATE["design_job_id"] = job_id


def test_design_rejects_bad_target_selection(client):
    resp = client.post("/api/design", data={})
    assert resp.status_code == 422
    assert "Exactly one of" in resp.text


def test_rank_on_previous_campaign(client):
    design_job_id = STATE.get("design_job_id")
    if not design_job_id:
        pytest.skip("design smoke did not run")
    resp = client.post(
        "/api/rank",
        data={"campaign_uri": f"job://{design_job_id}", "on": '["i_pTM"]'},
    )
    resp.raise_for_status()
    body = _poll(client, resp.json()["job_id"], timeout_s=900)
    assert body["status"] == "completed", body
    files = client.get(f"/api/jobs/{resp.json()['job_id']}/files").json()["files"]
    assert "ranked_by_i_pTM.csv" in files


def test_filter_on_previous_campaign(client):
    design_job_id = STATE.get("design_job_id")
    if not design_job_id:
        pytest.skip("design smoke did not run")
    resp = client.post(
        "/api/filter",
        data={
            "campaign_uri": f"job://{design_job_id}",
            "where": '["i_pAE=0.45","Interface_Residues>=7"]',
        },
    )
    resp.raise_for_status()
    body = _poll(client, resp.json()["job_id"], timeout_s=900)
    assert body["status"] == "completed", body
    files = client.get(f"/api/jobs/{resp.json()['job_id']}/files").json()["files"]
    assert "filtered.csv" in files


def test_rank_rejects_unknown_campaign(client):
    resp = client.post(
        "/api/rank",
        data={"campaign_uri": "job://definitely-not-a-job"},
    )
    assert resp.status_code == 404
```

- [ ] **Step 2: 写 test_fc_task.py**

```python
"""FC 异步任务模式（`/api/tasks/*`）的集成测试。"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import pytest
from bioq_service.fc_testing import fc_url, make_retrying_client

pytestmark = pytest.mark.fc

SERVICE = "bindcraft2-server"
TIMEOUT = 900.0

# 跨测试传递的 job id（模块级 dict，不往 pytest 模块上挂属性）。
STATE: dict[str, str] = {}


@pytest.fixture(scope="module")
def client():
    """会话亲和要求 submit 与之后的每次 poll 都带同一个 header 值——所以 headers
    挂在 client 上，而不是逐个请求传（否则 poll 会被当成新会话，FC 起一堆实例）。"""
    url = fc_url(SERVICE, start=Path(__file__))
    headers = {"bioagent-session-id": f"fc-bindcraft2-{uuid.uuid4().hex[:8]}"}
    with make_retrying_client(
        url, timeout=TIMEOUT, max_retries=10, backoff_s=20.0, headers=headers
    ) as c:
        yield c


def test_task_design_is_atomic(client):
    """task 端点同步执行到底，返回时已是终态。"""
    started = time.monotonic()
    resp = client.post(
        "/api/tasks/design",
        data={
            "target_name": "hPDL1",
            "binder_lengths": "[60,60]",
            "number_of_final_designs": "1",
            "max_trajectories": "1",
            "campaign_name": "fc_task_smoke",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] in ("completed", "running", "pending")
    assert time.monotonic() - started >= 0

    job_id = body["job_id"]
    if body["status"] != "completed":
        deadline = time.monotonic() + 3600
        while time.monotonic() < deadline:
            body = client.get(f"/api/jobs/{job_id}").json()
            if body["status"] in ("completed", "failed"):
                break
            time.sleep(15)
    assert body["status"] == "completed", body
    STATE["task_design_job_id"] = job_id


def test_task_rank_is_atomic(client):
    job_id = STATE.get("task_design_job_id")
    if not job_id:
        pytest.skip("task design smoke did not run")
    resp = client.post(
        "/api/tasks/rank",
        data={"campaign_uri": f"job://{job_id}"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] in ("completed", "running", "pending")


def test_task_filter_is_atomic(client):
    job_id = STATE.get("task_design_job_id")
    if not job_id:
        pytest.skip("task design smoke did not run")
    resp = client.post(
        "/api/tasks/filter",
        data={"campaign_uri": f"job://{job_id}"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] in ("completed", "running", "pending")


def test_task_endpoints_are_registered(client):
    paths = client.get("/openapi.json").json()["paths"]
    for path in ("/api/tasks/design", "/api/tasks/rank", "/api/tasks/filter"):
        assert path in paths
```

- [ ] **Step 3: 部署到 FC（人工，控制台）**

按 [`docs/adding-a-new-service/deploy.zh.md`](../../docs/adding-a-new-service/deploy.zh.md)：

1. `make build-bindcraft2-server && make push-bindcraft2-server`
2. FC 控制台创建函数：custom-container、GPU 实例、`timeout=36000`、内存 32 GB、
   磁盘 10 GB；NAS `mountDir=/data`；OSS `bio-gateway` → `/mnt/oss`（RW）。
   （`bioagent-inputs` 是计划初稿的笔误：仓库里 30 个部署描述符一律用 `bio-gateway`，与本服务的 `deploy/fc.yaml` 一致。以控制台实际值为准。）
3. **开启异步任务模式**、清空 keepalive URL、会话亲和 = HeaderField `bioagent-session-id`、
   `sessionConcurrencyPerInstance=1`。
4. 把真实 URL / function 名 / gpuType 回填 `services.yaml` 与 `deploy/fc.yaml`。

- [ ] **Step 4: 从 VPC 内跑 smoke（不跑 design）**

```bash
cd services/bindcraft2-server
RUN_FC_TESTS=1 uv run --group dev \
  python -m pytest -m fc tests/test_fc.py -v -k "healthz or manifest or bad_target_selection or unknown_campaign"
```

Expected：**5 项**通过（`test_healthz` 与 `test_healthz_detail_reports_weights_and_gpu` 都匹配 `healthz`；计划初稿写 4 是漏数了。已用 `--collect-only` 核对）。`weights_loaded` 若为 false → 按 `weights_missing` 补 NAS（Task 9）。
`gpu_backend` 若不是 `"gpu"` → **停**，先解决 FC GPU 分配，否则 campaign 会跑 CPU。

- [ ] **Step 5: 跑 design smoke（分钟级）**

```bash
RUN_FC_TESTS=1 uv run --group dev \
  python -m pytest -m fc tests/test_fc.py -v -k "design_shipped_target_smoke"
```

Expected：PASS，`summary.csv` / `campaign_metadata.json` / `1_Trajectories/!_Trajectories.csv`
都在。若失败：`curl $URL/api/jobs/<id>/log` 看尾部，`campaign refused:` 后是上游逐条原因。

- [ ] **Step 6: 跑 rank / filter（验证 P3：源目录不被污染）**

**不要**用 `pytest -k "rank or filter"` 来验 P3——那是新起一个进程，模块级 `STATE` 里的
`design_job_id` 为空，两个 rank/filter 用例会直接 `pytest.skip`，于是"前后无差异"是**假阳性**
（已在本地实测到这个 skip 行为）。用下面的 curl 版本，它自己找回 design 的 job：

```bash
cd services/bindcraft2-server
export UV_CACHE_DIR=<可写 cache>

# 1) 先单独跑 design smoke
RUN_FC_TESTS=1 uv run --group dev python -m pytest -m fc tests/test_fc.py -v \
  -k "design_shipped_target_smoke"

# 2) 找到它刚落下的 campaign 目录
JOB=$(ls -1dt /data/bindcraft2_jobs/*/ | head -1 | xargs basename); echo "design job = $JOB"

# 3) 给源目录拍快照（文件集合 + mtime）
find /data/bindcraft2_jobs/$JOB/output -type f | sort > /tmp/p3_before_files.txt
ls -la --time-style=full-iso /data/bindcraft2_jobs/$JOB/output/ > /tmp/p3_before_ls.txt

# 4) 用 curl 驱动 rank / filter（HTTP 上复杂字段必须是 JSON 字符串）
URL=$(python -c "import sys; sys.path.insert(0,'../..'); \
  from bioq_service.service_registry import fc_url; print(fc_url('bindcraft2-server'))")
curl -s -X POST "$URL/api/rank"   -H "bioagent-session-id: p3-manual" \
  -F "campaign_uri=job://$JOB" -F 'on=["i_pTM"]'
curl -s -X POST "$URL/api/filter" -H "bioagent-session-id: p3-manual" \
  -F "campaign_uri=job://$JOB" -F 'where=["i_pAE=0.45"]'
# 各自轮询到 completed：curl -s "$URL/api/jobs/<job_id>"

# 5) 比对（文件集合 **和** mtime 都要一致）
find /data/bindcraft2_jobs/$JOB/output -type f | sort > /tmp/p3_after_files.txt
diff /tmp/p3_before_files.txt /tmp/p3_after_files.txt \
  && echo "P3: 源目录文件集合未变" || echo "P3: 源目录被改动 → 改 copy 模式"
diff /tmp/p3_before_ls.txt <(ls -la --time-style=full-iso /data/bindcraft2_jobs/$JOB/output/) \
  && echo "P3: mtime 也未变（零拷贝安全）" || echo "P3: mtime 变化 → 即使文件集合不变也不算零拷贝"
```

Expected：两次 diff 都无差异 → `P3: 零拷贝安全`。**判定规则**：文件集合与 mtime 都一致才算通过；
否则在 Task 13 记录偏差，并给 `campaigns.resolve_campaign_dir` 增加
`copy_to=<job_dir>/input/campaign` 分支（rank/filter 改为拷贝后操作）。

### Task 13: 回写设计文档（三处偏差 + P1/P2/P3 实测结果）

**Files:**
- Modify: `docs/specs/2026-09-22-bindcraft2-server-design.md`

- [ ] **Step 1: 回写偏差 A1 / A2 / A3**

在 §请求 Schema 末尾把"JSON 字符串"那段改成：

```markdown
`binder_lengths` / `on` / `where` 是复杂类型。HTTP 上以 **JSON 字符串**表单字段传入
（`-F 'binder_lengths=[80,80]'`）——`model_form_depends` 对复杂字段执行 `json.loads`；
CLI 上以**逗号分隔字符串**传入（`--binder-lengths 80,80`）——框架的 CLI 层把 list 字段
当 `type=str` 直接交给 pydantic。模型用 `field_validator(mode="before")` 同时解码两种
写法（`[80,80]` 与 `80,80` 等价）。
```

把"`model_validator(mode="after")` 交叉校验"那句改成：

```markdown
目标结构三选一由 `models.validate_target_selection()` 在路由层校验：上传是路由级
`File(...)` / `Form(...)`，不是 model 字段，`model_validator` 看不到它们。
```

把 §测试策略里 `design` smoke 的 fixture 说明改成：

```markdown
fixture：FC smoke 用 **shipped target `hPDL1` + `binder_lengths=[60,60]` +
`max_trajectories=1` + `number_of_final_designs=1`**——手写 PDB 有被上游 preflight
判为畸形结构的风险，会让 FC 测试以误导性方式失败；shipped target 保证可解析。
`tests/data/fake_bindcraft.sh` 供离线 HTTP / CLI 成功路径使用。
```

- [ ] **Step 2: 回写 P1 / P2 / P3 实测结果**

把 §部署前置核验的表改成"核验结果 + 处置"，逐条填实测值：

```markdown
| # | 核验项 | 实测结果 | 处置 |
|---|---|---|---|
| P1 | NAS `/data/models/alphafold` 是否含 BC2 需要的 7 个 npz | <软链 / 已下载> | <`alphafold_params_dir` 指向软链 / 跑 fetch_weights.sh> |
| P2 | FC 卡型与 compute capability | <卡型，cc=X.Y> | <cuda13 / cuda12>；worker 与实例内存参数 <值> |
| P3 | `rank` / `filter` 是否只写 `--output` | <未改动 / 被改动> | <保持零拷贝 / 改为 copy 模式> |
```

- [ ] **Step 3: 把状态改成已实现**

```markdown
- **状态**：已实现（v0.0.1，镜像 tag 见 `services/bindcraft2-server/VERSION`）
```

- [ ] **Step 4: 自检 + 提交**

```bash
grep -n "TODO\|TBD\|待定\|占位" docs/specs/2026-09-22-bindcraft2-server-design.md || echo "no placeholders"
python3 - <<'PY'
import re, pathlib
p = pathlib.Path("docs/specs/2026-09-22-bindcraft2-server-design.md")
base = p.parent
bad = [t.split('#')[0] for t in re.findall(r"\]\(([^)]+)\)", p.read_text())
       if not t.startswith(("http://", "https://", "#")) and not (base / t.split('#')[0]).exists()]
print("broken links:", bad or "(none)")
PY
git add docs/specs/2026-09-22-bindcraft2-server-design.md
git commit -m "docs(bindcraft2-server): record implemented design and verified deployment prerequisites"
```

Expected：`no placeholders` + `broken links: (none)` + 提交成功。

---

## 自检结果

**Spec coverage**（设计文档章节 → 任务）：

| 设计文档章节 | 落地任务 |
|---|---|
| §2 许可证与合规约束 | Task 11（README 红线 + THIRD_PARTY_NOTICES + manifest extras 的 `license_notice`） |
| §3 设计目标 1 纯 argv 包装 | Task 4（`tools.py`）、Task 10（Dockerfile 无 patch） |
| §3 目标 2 单 campaign + rank/filter | Task 4 / Task 6（六条路由） |
| §3 目标 3 权重外置 | Task 9（fetch_weights.sh）、Task 10（`ENV` 注入）、Task 11（README Weights） |
| §3 目标 4 `max_trajectories` 兜底 | Task 1（`default_max_trajectories`）、Task 4（`prepare_design`）、Task 2（字段） |
| §3 目标 5 GPU 静默降级可探测 | Task 6（`_gpu_probe` + `/healthz/detail`），由 `tests/test_app.py::test_health_and_detail` 与 `test_fc.py::test_healthz_detail_reports_weights_and_gpu` 双重覆盖 |
| §3 目标 6 双模式对齐 | Task 6（app.py）+ Task 7（`__main__.py` 复用同一 `tools.py`） |
| §4 Endpoint 拓扑（6 条） | Task 6 Step 5；由 `test_openapi_exposes_six_body_schemas` 与 `test_manifest_lists_six_endpoints_with_examples` 断言 |
| §4 v0.0.1 不做清单 | Task 11（README 章节）；无任务实现它们 = 无 scope creep |
| §5 请求 Schema（含 9 个 bool） | Task 2 全字段 + `test_properties_lists_only_enabled_flags` |
| §6 输出树 + `detect_outputs` 含 `summary.csv` | Task 5（含 `test_detect_outputs_true_on_summary_only`）、Task 1（常量在 `tools.py`） |
| §7 实现要点（包装决策 / 零拷贝 / argv / 探针 / 并发） | Task 3（零拷贝）、Task 4（argv）、Task 5（env/cwd/detect）、Task 6（探针） |
| §8 配置表（8 字段） | Task 1（全部字段，含 `gpu_probe_ttl_seconds`） |
| §9 部署目标 | Task 10（镜像）、Task 11（fc.yaml + services.yaml） |
| §10 测试策略四层 | Task 2–7（三张离线表）+ Task 12（FC 两张表） |
| §11 风险（时长 / 演进 / 零拷贝副作用 / 显存 / 冷启动 / 许可） | Task 12 Step 6（P3 实测）、Task 13（回写）；其余由 README「运行约束」承载 |
| §成功标准 7 条 | 1→Task 1/11、2→Task 7 Step 5、3→Task 10 Step 4、4→Task 12 Step 4、5→Task 12 Step 5/6、6→Task 12 Step 7、7→Task 11 |
| §部署前置核验 P1/P2/P3 | P1→Task 9 Step 3、P2→Task 11 Step 1、P3→Task 12 Step 6；实测回写 Task 13 |
| §Sources | 已固化在 Task 8 的 pin SHA 与 README 链接 |

**Placeholder scan**：无 `TODO` / `TBD` / "fill in" / "similar to Task N"。所有代码步骤都带完整
代码块；所有命令步骤都带期望输出。唯一两处需要现场取值的地方是
`deploy/fc.yaml` 的 VPC 三件套与 `gpuConfig`（FC 控制台真实值）——Task 11 Step 1 明确写了
"照 alphafold 填"并给出取值来源，不是占位符；`services.yaml` 的 `url`/`function` 由
Task 12 Step 3 部署后回填，Step 4/8 是回填动作。

**Type consistency**：`Bindcraft2Settings` 字段名在 Task 1 定义，Task 3/4/5/6/7 引用一致
（`root` / `python` / `module` / `alphafold_params_dir` / `compile_cache_dir` /
`default_max_trajectories` / `gpu_probe_ttl_seconds` / `workers_per_gpu` /
`max_workers_per_gpu`）；`tools.py` 的 15 个导出名在 Task 4 定义、
Task 5/6/7 引用一致；`resolve_campaign_dir(uri, settings)` 签名在 Task 3 定义、
Task 6/7 引用一致；adapter 的 `name = "bindcraft2"` 与镜像名 `bindcraft2-server` 的区分在
全程一致（服务名 `bindcraft2`、目录/镜像 `bindcraft2-server`、env 前缀 `BINDCRAFT2_`）。
