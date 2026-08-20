# E12 Uniform-Contract Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 E12 审计的三处残余缺口（`array[file]` 文件识别、endpoint summary、`default` 语义化），把 `docs_text`/`file_fields`/`defaults` 的 task 通过率从 61.5%/98.5%/16.9% 提到 100%，服务 ≥0.9 达到 28/28，且不修改 bioq-paper 的 E12 检查逻辑。

**Architecture:** 分三层递进。先改共享框架 `bioq_service`（新增 `fields.py` 的 `default_semantics` helper、`manifest.py` 的 `array[file]` 识别 + `default_kind`/`default_note` 语义、`forms.py` 的 `json_schema_extra` 透传、`task_endpoint.py` 的 summary 透传），全部用 TDD 覆盖；再给 30 个 endpoint 补 summary；最后对 24 个服务的 `Optional[X] = None` 字段做机械的语义标注，让 manifest 的 `default` 恒非 null。

**Tech Stack:** Python 3、FastAPI 0.136、pydantic 2.12、pytest、uv。框架改动在 `framework/`，服务改动在 `services/<svc>-server/`。

---

## 前置约定

- 工作目录为仓库根（`bioq-services/`）。框架离线测试：
  `cd framework && uv run --extra dev python -m pytest tests/ -q`。
- 服务离线测试：`cd services/<svc>-server && uv run --group dev python -m pytest tests/ -q`
  （各服务目录自己有 test 命令；若某服务无 `tests/` 则跳过，改跑 manifest 校验脚本，见 Phase 3）。
- 提交信息用英文，**不加** AI co-author trailer。
- 每改一处 endpoint 的 summary（改动 manifest），同步该服务 `README.md` 里对应 endpoint 的一句话描述（若存在），保持文档一致。

---

## Phase 1 — 框架层（先落地，后续全部依赖它）

### Task 1: 新增 `default_semantics` helper

**Files:**
- Create: `framework/src/bioq_service/fields.py`
- Modify: `framework/src/bioq_service/__init__.py:51,83-108`
- Test: `framework/tests/test_fields.py`（新建）

- [ ] **Step 1: Write failing test**

`framework/tests/test_fields.py`：

```python
"""Tests for bioq_service.fields.default_semantics."""

from bioq_service import default_semantics


def test_default_semantics_returns_bioq_default_marker() -> None:
    marker = default_semantics("auto", "auto-select CUDA if available")
    assert marker == {"bioq_default": {"kind": "auto", "note": "auto-select CUDA if available"}}


def test_default_semantics_unset() -> None:
    assert default_semantics("unset", "only used when explicitly provided") == {
        "bioq_default": {"kind": "unset", "note": "only used when explicitly provided"}
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd framework && uv run --extra dev python -m pytest tests/test_fields.py -v`
Expected: FAIL（`ImportError: cannot import name 'default_semantics'`）。

- [ ] **Step 3: Write implementation**

`framework/src/bioq_service/fields.py`：

```python
"""Field-annotation helpers for Optional request fields.

An ``Optional[X] = None`` field surfaces in the manifest with ``default: null``
(FastAPI emits no ``default`` key for a None default), which the E12 contract
audit reads as "undeclared".  When a field actually means "omission lets the
service/tool supply a default" or "omission leaves the feature off", the service
annotates it with ``default_semantics(...)`` so the manifest carries a non-null,
machine-readable omission semantics token instead of a bare null.
"""

from typing import Any, Literal

DefaultKind = Literal["auto", "unset"]


def default_semantics(kind: DefaultKind, note: str) -> dict[str, Any]:
    """Marker dict for ``Field(json_schema_extra=...)`` on an Optional field.

    kind  "auto"  -> omitting lets the service/tool supply a default value
                      (selected at runtime or a fixed upstream default).
          "unset" -> omitting leaves the parameter inactive / unused.
    note  one-line prose describing the omission semantics.
    """
    return {"bioq_default": {"kind": kind, "note": note}}


__all__ = ["default_semantics"]
```

`framework/src/bioq_service/__init__.py` — 在 `from bioq_service.errors import ...`
之后加：

```python
from bioq_service.fields import default_semantics
```

在 `__all__` 里（`"extract_error_summary",` 之前）加 `"default_semantics",`。

- [ ] **Step 4: Run test to verify it passes**

Run: `cd framework && uv run --extra dev python -m pytest tests/test_fields.py -v`
Expected: PASS（2 passed）。

- [ ] **Step 5: Commit**

```bash
git add framework/src/bioq_service/fields.py framework/src/bioq_service/__init__.py framework/tests/test_fields.py
git commit -m "feat(framework): add default_semantics field marker helper"
```

### Task 2: `_is_file_schema` 识别 `array[file]`（C）

**Files:**
- Modify: `framework/src/bioq_service/manifest.py:239-246`
- Test: `framework/tests/test_manifest.py`

- [ ] **Step 1: Write failing test**

在 `framework/tests/test_manifest.py` 末尾追加：

```python
def test_extract_fields_marks_file_arrays_as_files() -> None:
    """list[UploadFile] (array of binary) must come out is_file=True / type='array[file]'."""
    from bioq_service.manifest import _extract_fields

    body_schema = {
        "type": "object",
        "properties": {
            "model_stats_files": {
                "type": "array",
                "items": {"type": "string", "format": "binary"},
                "title": "Model Stats Files",
            },
        },
    }
    fields = {f.name: f for f in _extract_fields(body_schema)}
    assert fields["model_stats_files"].is_file is True
    assert fields["model_stats_files"].type == "array[file]"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd framework && uv run --extra dev python -m pytest tests/test_manifest.py::test_extract_fields_marks_file_arrays_as_files -v`
Expected: FAIL（`assert True is False`，因为 `is_file` 现为 False）。

- [ ] **Step 3: Write minimal implementation**

`framework/src/bioq_service/manifest.py:239-246` 替换为：

```python
def _is_file_schema(schema: dict[str, Any]) -> bool:
    """Multipart file uploads: format=binary / octet-stream, scalar or array of them."""
    if schema.get("type") == "array":
        return _is_file_schema(schema.get("items") or {})
    if schema.get("type") != "string":
        return False
    return (
        schema.get("format") == "binary"
        or schema.get("contentMediaType") == "application/octet-stream"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd framework && uv run --extra dev python -m pytest tests/test_manifest.py -v`
Expected: 全 PASS。

- [ ] **Step 5: Commit**

```bash
git add framework/src/bioq_service/manifest.py framework/tests/test_manifest.py
git commit -m "fix(framework): recognize array-of-file uploads in manifest"
```

### Task 3: manifest 支持 `default_kind` / `default_note`（B-①）

**Files:**
- Modify: `framework/src/bioq_service/manifest.py:50-73,261-278`
- Test: `framework/tests/test_manifest.py`

- [ ] **Step 1: Write failing test**

在 `framework/tests/test_manifest.py` 末尾追加：

```python
def test_extract_fields_semantic_default_tokens() -> None:
    """A field annotated with bioq_default surfaces default=token + kind + note."""
    from bioq_service.manifest import _extract_fields

    body_schema = {
        "type": "object",
        "properties": {
            "device": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "title": "Device",
                "bioq_default": {"kind": "auto", "note": "auto-select CUDA if available"},
            },
            "extra": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "title": "Extra",
                "bioq_default": {"kind": "unset", "note": "only used when explicitly provided"},
            },
            "plain": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "title": "Plain",
            },
            "speed": {"type": "integer", "title": "Speed", "default": 3},
        },
    }
    fields = {f.name: f for f in _extract_fields(body_schema)}
    assert fields["device"].default == "auto"
    assert fields["device"].default_kind == "auto"
    assert fields["device"].default_note == "auto-select CUDA if available"
    assert fields["extra"].default == "unset"
    assert fields["extra"].default_kind == "unset"
    assert fields["plain"].default is None
    assert fields["plain"].default_kind is None
    assert fields["speed"].default == 3
    assert fields["speed"].default_kind == "literal"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd framework && uv run --extra dev python -m pytest tests/test_manifest.py::test_extract_fields_semantic_default_tokens -v`
Expected: FAIL（`FieldInfo` 没有 `default_kind` 属性 / `default` 断言失败）。

- [ ] **Step 3: Write implementation**

`framework/src/bioq_service/manifest.py` 的 `FieldInfo`（`default` 字段之后）加两个字段：

```python
    default_kind: str | None = Field(
        default=None,
        description=(
            "'literal' | 'auto' | 'unset'. 'auto'/'unset' are semantics tokens "
            "(not sendable literals); 'literal' means `default` is a real value."
        ),
    )
    default_note: str | None = Field(
        default=None,
        description="One-line prose for the 'auto'/'unset' omission semantics.",
    )
```

并把 `default` 的描述改为：

```python
    default: Any = Field(
        default=None,
        description=(
            "Effective default when omitted: a literal (default_kind='literal') "
            "or a semantics token 'auto'/'unset'."
        ),
    )
```

`framework/src/bioq_service/manifest.py` 的 `_extract_fields` 循环体（现 L266-277）替换为：

```python
    for fname, fschema in properties.items():
        actual = _peel_optional(fschema)
        marker = fschema.get("bioq_default") or {}
        kind = marker.get("kind")
        note = marker.get("note")
        literal = fschema.get("default")
        if literal is not None:
            default, default_kind = literal, "literal"
        elif kind in ("auto", "unset"):
            default, default_kind = kind, kind
        else:
            default, default_kind = None, None
        out.append(
            FieldInfo(
                name=fname,
                type=_format_type(actual),
                required=fname in required_set,
                description=fschema.get("description") or actual.get("description"),
                is_file=_is_file_schema(actual),
                default=default,
                default_kind=default_kind,
                default_note=note,
            )
        )
```

> 必须从外层 `fschema` 读 `default` 与 `bioq_default`；`_peel_optional` 会解包到
> `anyOf` 内的非 null 分支，丢掉外层 marker。

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd framework && uv run --extra dev python -m pytest tests/test_manifest.py -v`
Expected: 全 PASS。

- [ ] **Step 5: Commit**

```bash
git add framework/src/bioq_service/manifest.py framework/tests/test_manifest.py
git commit -m "feat(framework): surface default_kind/default_note for optional fields"
```

### Task 4: `model_form_depends` 透传 `json_schema_extra`

**Files:**
- Modify: `framework/src/bioq_service/forms.py:89-104`
- Test: `framework/tests/test_forms.py`

- [ ] **Step 1: Write failing test**

在 `framework/tests/test_forms.py` 末尾追加：

```python
def test_json_schema_extra_forwarded_to_openapi():
    """A model Field(json_schema_extra=...) must land in the endpoint's OpenAPI body."""
    from bioq_service import default_semantics

    class WithMarker(BaseModel):
        name: str = "run"
        device: Optional[str] = Field(
            default=None,
            json_schema_extra=default_semantics("auto", "auto-select CUDA if available"),
        )

    client = _make_app(WithMarker)
    spec = client.app.openapi()
    body = next(iter(spec["paths"]["/x"]["post"]["requestBody"]["content"].values()))
    body_schema = spec["components"]["schemas"][body["schema"]["$ref"].rsplit("/", 1)[-1]]
    assert body_schema["properties"]["device"]["bioq_default"] == {
        "kind": "auto",
        "note": "auto-select CUDA if available",
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd framework && uv run --extra dev python -m pytest tests/test_forms.py::test_json_schema_extra_forwarded_to_openapi -v`
Expected: FAIL（`KeyError: 'bioq_default'`，因为字段的 `json_schema_extra` 被丢弃）。

- [ ] **Step 3: Write implementation**

`framework/src/bioq_service/forms.py:89-104`，四个 `Form(...)` 都追加透传参数：

复杂字段分支（L89-95）：

```python
        if is_complex:
            complex_field_names.add(field_name)
            param_annotation: Any = str if is_required else Optional[str]
            if is_required:
                form_default = Form(..., description=field.description,
                                    json_schema_extra=field.json_schema_extra)
            else:
                form_default = Form(None, description=field.description,
                                    json_schema_extra=field.json_schema_extra)
```

标量分支（L96-104）：

```python
        else:
            param_annotation = annotation
            if is_required:
                form_default = Form(..., description=field.description,
                                    json_schema_extra=field.json_schema_extra)
            else:
                default_value = (
                    field.default if field.default is not PydanticUndefined else None
                )
                form_default = Form(default_value, description=field.description,
                                    json_schema_extra=field.json_schema_extra)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd framework && uv run --extra dev python -m pytest tests/test_forms.py -v`
Expected: 全 PASS。

- [ ] **Step 5: Commit**

```bash
git add framework/src/bioq_service/forms.py framework/tests/test_forms.py
git commit -m "fix(framework): forward field json_schema_extra through form models"
```

### Task 5: `register_task_endpoint` 透传 `summary`/`description`（A1）

**Files:**
- Modify: `framework/src/bioq_service/task_endpoint.py:216-224,275-280`
- Test: `framework/tests/test_task_endpoint.py`

- [ ] **Step 1: Write failing test**

在 `framework/tests/test_task_endpoint.py` 末尾追加：

```python
def test_register_task_endpoint_summary_surfaces_in_manifest(tmp_path: Path) -> None:
    settings = _EchoSettings(jobs_base_dir=tmp_path / "jobs", keepalive_interval_s=0)
    adapter = _EchoAdapter(settings=settings)
    app = create_app(adapter, settings, title="Echo Summary")
    register_task_endpoint(
        app,
        path="/api/tasks/echo-summary",
        label="echo",
        request_model=_EchoRequest,
        build_argv=_echo_argv,
        summary="Echo a message (single atomic task).",
    )
    client = TestClient(app)
    body = client.get("/api/manifest").json()
    ep = next(e for e in body["endpoints"] if e["path"] == "/api/tasks/echo-summary")
    assert ep["summary"] == "Echo a message (single atomic task)."
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd framework && uv run --extra dev python -m pytest tests/test_task_endpoint.py::test_register_task_endpoint_summary_surfaces_in_manifest -v`
Expected: FAIL（`TypeError: register_task_endpoint() got an unexpected keyword argument 'summary'`）。

- [ ] **Step 3: Write implementation**

`framework/src/bioq_service/task_endpoint.py:216-224` 签名加两个参数：

```python
def register_task_endpoint(
    app: FastAPI,
    *,
    path: str,
    label: str,
    request_model: type[BaseModel],
    build_argv: BuildArgvForTask,
    save_inputs: Optional[Callable[[BaseModel, Path], None]] = None,
    summary: Optional[str] = None,
    description: Optional[str] = None,
) -> None:
```

`framework/src/bioq_service/task_endpoint.py:275-280` 的 `add_api_route` 调用改为：

```python
    app.add_api_route(
        path,
        _task_handler,
        methods=["POST"],
        response_model=JobInfo,
        summary=summary,
        description=description,
    )
```

> 本模块不得添加 `from __future__ import annotations`（硬约束）；本改动不涉及。

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd framework && uv run --extra dev python -m pytest tests/test_task_endpoint.py -v`
Expected: 全 PASS。

- [ ] **Step 5: Commit**

```bash
git add framework/src/bioq_service/task_endpoint.py framework/tests/test_task_endpoint.py
git commit -m "feat(framework): pass summary/description through register_task_endpoint"
```

### Task 6: 框架全量回归 + 提交检查点

- [ ] **Step 1: Run whole framework suite**

Run: `cd framework && uv run --extra dev python -m pytest tests/ -q`
Expected: 全 PASS（无回归）。

- [ ] **Step 2: 确认 git 状态干净**

Run: `git status --porcelain`
Expected: 除 Phase 1 已提交外无未提交改动。

---

## Phase 2 — 补 endpoint summary（A2，30 个）

每个 Step 完成后单独 commit。装饰器用两行形式（超 88 列则换行），
`register_task_endpoint` 调用里在 `build_argv=...` 行后新增 `summary=...` 行。

### Task 7: reinvent（5 legacy + 5 task = 10 个）

**Files:**
- Modify: `services/reinvent-server/app.py:191,202,213,227,243,262,277,292,310,330`

Legacy 端点（卸下 task 措辞）：

| 行 | old | new |
|---|---|---|
| 191 | `@app.post("/api/sampling", response_model=JobInfo)` | `@app.post("/api/sampling", response_model=JobInfo,\n          summary="De novo sampling from a Reinvent generator.")` |
| 202 | `@app.post("/api/scoring", response_model=JobInfo)` | `@app.post("/api/scoring", response_model=JobInfo,\n          summary="Score SMILES with a scoring function.")` |
| 213 | `@app.post("/api/enumeration", response_model=JobInfo)` | `@app.post("/api/enumeration", response_model=JobInfo,\n          summary="Peptide enumeration with pepinvent.")` |
| 227 | `@app.post("/api/transfer-learning", response_model=JobInfo)` | `@app.post("/api/transfer-learning", response_model=JobInfo,\n          summary="Fine-tune a generative prior on target molecules (long-running).")` |
| 243 | `@app.post("/api/staged-learning", response_model=JobInfo)` | `@app.post("/api/staged-learning", response_model=JobInfo,\n          summary="Staged learning: RL / curriculum over multiple stages (long-running).")` |

Task 端点：

| 行 | summary= |
|---|---|
| 262 `/api/tasks/sampling` | `"De novo sampling from a Reinvent generator (single atomic task)."` |
| 277 `/api/tasks/scoring` | `"Score SMILES with a scoring function (single atomic task)."` |
| 292 `/api/tasks/enumeration` | `"Peptide enumeration with pepinvent (single atomic task)."` |
| 310 `/api/tasks/transfer-learning` | `"Fine-tune a generative prior on target molecules (single atomic task; long-running)."` |
| 330 `/api/tasks/staged-learning` | `"Staged learning: RL / curriculum over multiple stages (single atomic task; long-running)."` |

（每个 task 端点同样用两行装饰器形式：`@app.post("/api/tasks/sampling", response_model=JobInfo,\n          summary="...")`。）

- [ ] **Step 1: 应用 10 处编辑**
- [ ] **Step 2: 运行测试**

Run: `cd services/reinvent-server && uv run --group dev python -m pytest tests/ -q`
Expected: PASS。

- [ ] **Step 3: 校验 manifest summary 已生效**（可选，用 `framework` 环境中的 TestClient 或直接跳过——以 E12 复跑为准）
- [ ] **Step 4: Commit**

```bash
git add services/reinvent-server/app.py
git commit -m "docs(reinvent): add endpoint summaries for legacy and task endpoints"
```

### Task 8: pocketxmol(6) + drughive(3) + diffdock(1) + openadmet(1) = 11 个

**Files:**
- Modify: `services/pocketxmol-server/app.py:502,533,557,585,613,644`
- Modify: `services/drughive-server/app.py:351,392,445`
- Modify: `services/diffdock-server/app.py:233`
- Modify: `services/openadmet-server/app.py:397`

全部用两行装饰器形式，`old` 均为 `@app.post("<path>", response_model=JobInfo)`，`new` 为
`@app.post("<path>", response_model=JobInfo,\n          summary="<summary>")`。逐行对照：

| 服务 | 行 | path | summary= |
|---|---|---|---|
| pocketxmol | 502 | `/api/tasks/dock` | `"Molecular docking, small-molecule or peptide (single atomic task)."` |
| pocketxmol | 533 | `/api/tasks/sbdd` | `"De novo structure-based drug design (single atomic task)."` |
| pocketxmol | 557 | `/api/tasks/linking` | `"Fragment linking / growing / PROTAC linker design (single atomic task)."` |
| pocketxmol | 585 | `/api/tasks/optimize` | `"Molecular optimization: local refinement of an input ligand (single atomic task)."` |
| pocketxmol | 613 | `/api/tasks/pepdesign` | `"Peptide design: linear/cyclic de novo, inverse folding, sc-packing (single atomic task)."` |
| pocketxmol | 644 | `/api/tasks/confidence` | `"Tuned-ranker confidence scoring on a previously completed job (single atomic task)."` |
| drughive | 351 | `/api/tasks/generate` | `"De novo ligand generation (single atomic task)."` |
| drughive | 392 | `/api/tasks/generate_spatial` | `"Substructure modification / scaffold hopping (single atomic task)."` |
| drughive | 445 | `/api/tasks/optimize` | `"Multi-cycle QVina2 property optimization (single atomic task; long-running)."` |
| diffdock | 233 | `/api/tasks/dock` | `"Protein-ligand docking (single atomic task)."` |
| openadmet | 397 | `/api/tasks/compare` | `"Post-hoc comparison of pre-trained models (Mode A) or their stats JSON (Mode B; single atomic task)."` |

- [ ] **Step 1: 应用 11 处编辑**
- [ ] **Step 2: 分别运行各服务测试**（pocketxmol、drughive、diffdock、openadmet）
- [ ] **Step 3: Commit（按服务分组，共 4 个 commit）**

```bash
git add services/pocketxmol-server/app.py && git commit -m "docs(pocketxmol): add task endpoint summaries"
git add services/drughive-server/app.py && git commit -m "docs(drughive): add task endpoint summaries"
git add services/diffdock-server/app.py && git commit -m "docs(diffdock): add dock task summary"
git add services/openadmet-server/app.py && git commit -m "docs(openadmet): add compare task summary"
```

### Task 9: register 式端点（9 个）

**Files:**
- Modify: `services/flowmol-server/app.py:146-152`
- Modify: `services/immunebuilder-server/app.py:194-200,201-207,208-214`
- Modify: `services/megalodon-server/app.py:154-160`
- Modify: `services/ppiflow-server/app.py:328-334`
- Modify: `services/rfdiffusion-server/app.py:310-316,317-323`
- Modify: `services/semlaflow-server/app.py:153-159`

每个调用在 `build_argv=<...>,` 之后新增一行 `summary="...",`。逐个：

1. flowmol `/api/tasks/generate`：`summary="Unconditional molecule generation (single atomic task).",`
2. immunebuilder `/api/tasks/predict_antibody`：`summary="Predict antibody structure from heavy + light chain sequences (single atomic task).",`
3. immunebuilder `/api/tasks/predict_nanobody`：`summary="Predict nanobody structure from heavy chain sequence (single atomic task).",`
4. immunebuilder `/api/tasks/predict_tcr`：`summary="Predict TCR structure from alpha + beta chain sequences (single atomic task).",`
5. megalodon `/api/tasks/generate`：`summary="Unconditional generation (single atomic task).",`
6. ppiflow `/api/tasks/sample/monomer`：`summary="Unconditional monomer generation at the requested lengths (single atomic task).",`
7. rfdiffusion `/api/tasks/generate/unconditional`：`summary="Unconditional monomer, or macrocycle with cyclic=true (single atomic task).",`
8. rfdiffusion `/api/tasks/generate/symmetry`：`summary="Symmetric oligomer: cyclic / dihedral / tetrahedral (single atomic task).",`
9. semlaflow `/api/tasks/generate`：`summary="Unconditional generation (single atomic task).",`

示例（flowmol，改动后完整调用）：

```python
register_task_endpoint(
    app,
    path="/api/tasks/generate",
    label="generate",
    request_model=GenerateRequest,
    build_argv=_task_build,
    summary="Unconditional molecule generation (single atomic task).",
)
```

- [ ] **Step 1: 应用 9 处编辑**
- [ ] **Step 2: 分别运行各服务测试**（flowmol、immunebuilder、megalodon、ppiflow、rfdiffusion、semlaflow）
- [ ] **Step 3: Commit（按服务分组）**

```bash
git add services/flowmol-server/app.py services/immunebuilder-server/app.py services/megalodon-server/app.py services/ppiflow-server/app.py services/rfdiffusion-server/app.py services/semlaflow-server/app.py
git commit -m "docs: add summaries to register_task_endpoint call sites"
```

---

## Phase 3 — defaults 语义标注（B-①，24 个服务）

**统一规则（每个字段二选一）：**

- `auto`：省略后由服务/工具给出默认值（运行时自选或固定上游默认）。
  命中名单（按字段名）：`seed`、`random_seed`、`device`、`model`、`cores`、`n_cpu`、
  `maxthreads`、`analysis_num_processes`、`num_devices`、`model_file`、`prior_file`、
  `agent_file`、`input_model_file`、`guide_scale`、`guide_decay`。
  注解 note 按名称用下表中的固定文案。
- `unset`：省略即不启用/不传（可选约束、过滤、JSON 覆盖、URI、序列等）。
  注解 note = `"only used when explicitly provided"`。

**固定 note 文案：**

| 字段名 | kind | note |
|---|---|---|
| `seed` / `random_seed` | auto | `random seed selected by the tool at runtime` |
| `device` | auto | `auto-select CUDA if available` |
| `model` | auto | `auto-select by the tool from the request inputs` |
| `cores` / `n_cpu` / `maxthreads` | auto | `use all available cores` |
| `analysis_num_processes` / `num_devices` | auto | `auto-detect available devices/processes` |
| `model_file` / `prior_file` / `agent_file` / `input_model_file` | auto | `use the tool's default model when omitted` |
| `guide_scale` / `guide_decay` | auto | `use the tool's default when omitted` |
| 其它所有 | unset | `only used when explicitly provided` |

**统一编辑变换（三种既有形态）：**

形态 A（裸 `= None`）：
```python
device: Optional[str] = None
```
→
```python
device: Optional[str] = Field(default=None, json_schema_extra=default_semantics("auto", "auto-select CUDA if available"))
```

形态 B（已有 `Field(default=None, ...)`）：在最后一个参数后追加
`, json_schema_extra=default_semantics("<kind>", "<note>")`。

形态 C（`Field(default_factory=...)`，如 boltz 的 `sequences`/`constraints`/`templates`）：
`default` 本应为空集合但 FastAPI 不输出，视为 `unset`，同样追加
`, json_schema_extra=default_semantics("unset", "empty when omitted")`。

**每个服务还需要**：在 `models.py` 顶部（或既有 `from bioq_service import ...` 行）加
`from bioq_service import default_semantics`。

**通用 Step 骨架（每服务重复）：**
- [ ] 定位 `Optional[...] = None` / `Field(default=None,...)` / `Field(default_factory=...)` 字段：

  `grep -nE 'Optional\[|default=None|default_factory=' services/<svc>-server/models.py`
- [ ] 按下表对每个命中字段做编辑变换 + 加 import
- [ ] 运行该服务测试（若有 `tests/`），否则跑 manifest 校验（Task 14）
- [ ] Commit：`docs(<svc>): annotate Optional field omission semantics`

### Task 10: rfdiffusion + rfdiffusion2（参考样例，完整给出）

**Files:**
- Modify: `services/rfdiffusion-server/models.py`
- Modify: `services/rfdiffusion2-server/models.py`

rfdiffusion `models.py` 加 import `from bioq_service import default_semantics`（在
`from bioq_service import FailureKind, JobInfo, JobStatus` 之后）。8 个字段：

| 字段（现态行） | kind | 变换后 |
|---|---|---|
| `model` (L87) | auto | `model: Optional[str] = Field(default=None, description=_MODEL_DESC, json_schema_extra=default_semantics("auto", "auto-select by the tool from the request inputs"))` |
| `length` (L116) | unset | 追加 `, json_schema_extra=default_semantics("unset", "only used when explicitly provided")` |
| `inpaint_seq` (L120) | unset | 同上追加 |
| `hotspots` (L137) | unset | 同上追加 |
| `guiding_potentials` (L161) | unset | 同上追加 |
| `guide_scale` (L168) | auto | 追加 `, json_schema_extra=default_semantics("auto", "use the tool's default when omitted")` |
| `guide_decay` (L171) | auto | 同上追加 |
| `extra_overrides` (L189) | unset | 同上追加 `"only used when explicitly provided"` |

rfdiffusion2 `models.py` 同样加 import，字段：`model`(auto)、`only_guidepost_positions`、
`partially_fixed_ligand`、`inpaint_seq`、`length`、`ligand`、`extra_overrides` 全为 unset。

> 注：rfdiffusion/rfdiffusion2 的 `input_uri` 是 `app.py` 里的裸 `Form(None)`（不在 models.py），
> 属于跨服务 URI 命名 spec 的改造对象，本计划**不处理**（见 Out of Scope）。

- [ ] **Step 1: rfdiffusion 编辑 + 测试**
- [ ] **Step 2: rfdiffusion2 编辑 + 测试**
- [ ] **Step 3: Commit**

```bash
git add services/rfdiffusion-server/models.py && git commit -m "docs(rfdiffusion): annotate optional-field omission semantics"
git add services/rfdiffusion2-server/models.py && git commit -m "docs(rfdiffusion2): annotate optional-field omission semantics"
```

### Task 11: reinvent + pocketxmol（高优先级）

**Files:**
- Modify: `services/reinvent-server/models.py`
- Modify: `services/pocketxmol-server/models.py`

各加 `from bioq_service import default_semantics`。

reinvent 字段（auto 者标 *）：
`agent_file`(auto)、`agent_file_uri`(unset)、`device`(auto)、`diversity_filter`(unset)、
`inception`(unset)、`input_model_file`(auto)、`input_model_uri`(unset)、
`learning_strategy`(unset)、`model_file`(auto)、`pairs`(unset)、`prior_file`(auto)、
`prior_file_uri`(unset)。

pocketxmol 字段：`fix_pos_res_bb`、`fix_pos_res_sc`、`fix_type_res_bb`、`fix_type_res_sc`、
`pep_length`、`pep_sequence`、`pocket_coord`、`smiles` 为 unset；`seed`(auto)。

- [ ] **Step 1: 两服务编辑 + 测试**
- [ ] **Step 2: Commit**

```bash
git add services/reinvent-server/models.py && git commit -m "docs(reinvent): annotate optional-field omission semantics"
git add services/pocketxmol-server/models.py && git commit -m "docs(pocketxmol): annotate optional-field omission semantics"
```

### Task 12: 其余 20 个服务（机械批量）

每个服务：加 `from bioq_service import default_semantics` + 按下表标注。auto 已标注，其余均 unset（note `"only used when explicitly provided"`）。

| 服务 | 字段（`*`=auto） |
|---|---|
| boltz | constraints、sequences、templates（三个 `default_factory=list`，kind=`unset`、note=`"empty when omitted"`）、msa_files（`unset`）、raw_yaml（`unset`）、raw_yaml_uri（`unset`）、seed*（auto）、step_scale（`unset`）、template_files（`unset`） |
| chembounce | core_smiles、h_acceptor_max、h_acceptor_min、h_donor_max、h_donor_min、input_smiles_uri、logp_max、logp_min、mw_max、mw_min、overall_max_n、qed_max、qed_min、sa_max、sa_min、scaffold_top_n |
| boltzgen | alpha、analysis_num_processes*、diffusion_batch_size、inverse_fold_avoid、noise_scale、ref_files_zip_uri、step_scale |
| openadmet | aq_fxns、best_y、beta、input_col、input_smiles、label_types、labels、model_names、mt_id、task_names、xi |
| drughive | mol_filter、substruct_modify_pattern、temps、zbetas |
| proteinmpnn | bias_AA、bias_by_res、chains_to_design、fixed_positions、omit_AA_per_chain、tied_positions |
| dockq | mapping、models、models_zip_uri、n_cpu* |
| genie3 | num_devices*、selections |
| flowmol | hc_thresh、n_atoms_per_mol、seed*、stochasticity |
| iggm | epitope、seed* |
| plip | intra_chain、maxthreads*、peptide_chains、report_formats |
| ppiflow | length_subset、specified_hotspots |
| diffdock | ligand_description、protein_sequence、seed* |
| openbpmd | equil_steps、sim_ns、system_format |
| megalodon | n_atoms_per_mol、seed* |
| rfantibody | hotspots、seed* |
| alphafold | random_seed* |
| diffdock-pp | seed*（`int \| None = Field(...)` 形态，追加 marker） |
| lightdock | cores* |
| semlaflow | seed* |

> boltz 的三个 `default_factory=list` 字段（`sequences`/`constraints`/`templates`）用 note
> `"empty when omitted"` + kind `unset`；`msa_files`/`template_files` 若在 `app.py` 声明，则在该
> `Form(...)` 上追加 `json_schema_extra=default_semantics("unset", "only used when explicitly provided")`。

- [ ] **Step 1: 逐服务编辑**（每个服务独立 commit：`git add services/<svc>-server/models.py && git commit -m "docs(<svc>): annotate optional-field omission semantics"`）
- [ ] **Step 2: 每个服务跑测试**（有 `tests/` 的跑 pytest；无的进 Task 14 校验）

### Task 13: B 的验收闭环（grep + 服务测试）

- [ ] **Step 1: 逐服务确认无漏网字段**。对 24 个服务依次运行：

```bash
grep -nE 'Optional\[[^]]*\] = None|= Field\(default=None|default_factory=' services/<svc>-server/models.py
```

  Expected：每个命中行都应在相邻位置带 `json_schema_extra=default_semantics(...)`；若有
  `= None` / `default=None` 且无 marker 的行，按 Task 10-12 补上。

- [ ] **Step 2: 确认已加 import**：

```bash
grep -l 'default_semantics' services/*-server/models.py
```

  Expected：至少覆盖 Task 10-12 表中的所有服务。

- [ ] **Step 3: 运行受影响服务的离线测试**（至少有 `tests/` 的 reinvent、pocketxmol、
  rfdiffusion、rfdiffusion2、openadmet、drughive、flowmol、megalodon、semlaflow、dockq、
  proteinmpnn）：

```bash
uv run --group dev python -m pytest services/reinvent-server/tests -q
uv run --group dev python -m pytest services/pocketxmol-server/tests -q
uv run --group dev python -m pytest services/rfdiffusion-server/tests -q
uv run --group dev python -m pytest services/openadmet-server/tests -q
```

  Expected：全 PASS（标注只增 `json_schema_extra`，不改变解析/默认行为，不应引入回归）。

- [ ] **Step 4: Commit 剩余改动**（未提交服务逐个）：

```bash
git add services/<svc>-server/models.py && git commit -m "docs(<svc>): annotate optional-field omission semantics"
```

---

## 明确不做（Out of Scope）

- bioq-paper 的 E12 `audit.py`/`score.py` 口径不变；本计划只改 bioq-services。
- bioq CLI 的 array（同名多文件）`--file` 上传侧支持（bioq 仓库跟进）。
- 跨服务 URI 字段命名 spec（rfdiffusion/rfdiffusion2 的 `input_uri → input_pdb_uri`）——独立调整。
- B1 字面默认值推算（如 `guide_scale: float = 10.0`）：需要逐个核对上游默认 + 行为等价，是独立后续项，本计划用 `auto`/`unset` token 已满足 E12 `defaults` 检查，不改行为。
- 版本号 bump / 镜像构建部署 / E12 重采集（发布与验证期动作）。
- `esmfold2`/`promera` 可达性、`operation_id` 填充（不计分项）。

## Success Criteria（复核用）

1. `cd framework && uv run --extra dev python -m pytest tests/ -q` 全绿。
2. 受影响服务离线测试全绿。
3. `bioq describe openadmet` 的 `compare` 把 `model_stats_files` 列进 `files:` 区且 `is_file=True`。
4. `bioq describe reinvent` 每 endpoint 有 summary。
5. 24 个服务的 task endpoint 无任何可选非文件参数的 `default` 为 null（全部为字面量或 `auto`/`unset` token）。
6. （跨仓库，验证期）重跑 E12 后 `docs_text`/`file_fields`/`defaults` task 通过率均 100%，服务 ≥0.9 达 28/28。