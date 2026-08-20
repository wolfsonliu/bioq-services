# E12 uniform-contract 修复设计（C `array[file]` + A endpoint summary + B defaults 语义化）

## Problem Statement

E12 审计（28 服务 / 65 task endpoints）的三处残余缺口，均落在元数据与文件识别的边界上：

| 结构项 | 现状 | 缺口性质 |
|---|---|---|
| `typed_params` / `machine_view` | 100% | 无需改动 |
| `file_fields` | 98.5% | openadmet `array[file]` 漏标 `is_file`（C） |
| `docs_text` | 61.5% | 25 个 task endpoint + 5 个 reinvent legacy 无 summary（A） |
| `defaults` | 16.9% | 176 个 `Optional[X] = None` 在 manifest 里显示 `default: null`（B） |

**C 根因**：`manifest.py::_is_file_schema` 只认标量 `string` 的 binary/octet-stream，不递归
`array`，于是 `list[UploadFile] = File(...)` 在 `_format_type` 里已渲染成 `array[file]`，但
`is_file` 仍为 False，对 `bioq run --file` 上传面不可见。

**A 根因**：9 个无上传 task endpoint 走 `register_task_endpoint`，其 `app.add_api_route(...)`
不带 `summary`，服务侧无文案可透传；另 16 个显式 `@app.post` 装饰器漏写 `summary`；reinvent
还有 5 个 legacy 孪生端点同款。

**B 根因（真正的瓶颈）**：FastAPI 对 `Optional[X] = None` 的字段**不输出 `default` 键**
（审计抽查：112 个非空默认值正常输出、0 个显式 null、66 个无键）。`_extract_fields` 读
`fschema.get("default")` 得到 None，manifest 于是显示 `default: null`，E12 按"保守口径"判为
未声明。服务代码普遍只透传参数，真正的默认值在内/上游工具内部，无法从服务代码机械推导。

## Proposed Solution

- **C / A1 / A2**：机械修复，低风险，照改 + 补测试。
- **B**：采用 **语义化 `default` + 新增 `default_kind`/`default_note`**（方案 B-①）。让 manifest
  的 `default` 字段对每个可选非文件参数都非 null——若是真字面量就写字面量，若是"运行时自选/
  省略即不启"就用 `"auto"`/`"unset"` 语义 token，并用 `default_kind` 区分 token 不可当字面量发送。
  如此 **`defaults` 16.9% → 100%，且不改 bioq-paper 的 E12 检查口径**。

## Detailed Design

### 1 — C：`framework/src/bioq_service/manifest.py`

`_is_file_schema` 增加 array 递归（`_format_type` 已能把 `array[binary]` 渲染成 `array[file]`，
这里只补 `is_file`）：

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

### 2 — A1：`framework/src/bioq_service/task_endpoint.py`

`register_task_endpoint` 增加两个可选参数并透传（`summary=None`/`description=None` 时行为与现状
完全一致，向后兼容）：

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
    ...
    app.add_api_route(
        path,
        _task_handler,
        methods=["POST"],
        response_model=JobInfo,
        summary=summary,
        description=description,
    )
```

> 注意：此模块**不得**加 `from __future__ import annotations`（硬约束 #13），本改动不涉及。

### 3 — A2：30 个 endpoint 补 summary

规则：显式 `@app.post` 在装饰器加 `summary="..."`；`register_task_endpoint(...)` 在调用里加
`summary="..."`（依赖 A1）。`(single atomic task)` 表示阻塞式一次完成。行号以实现时 grep 为准。

**显式 `@app.post`（16 个）**

| 服务 | endpoint | 文件:行 | summary= |
|---|---|---|---|
| diffdock | `/api/tasks/dock` | `app.py:233` | `"Protein-ligand docking (single atomic task)."` |
| drughive | `/api/tasks/generate` | `app.py:351` | `"De novo ligand generation (single atomic task)."` |
| drughive | `/api/tasks/generate_spatial` | `app.py:392` | `"Substructure modification / scaffold hopping (single atomic task)."` |
| drughive | `/api/tasks/optimize` | `app.py:445` | `"Multi-cycle QVina2 property optimization (single atomic task; long-running)."` |
| openadmet | `/api/tasks/compare` | `app.py:397` | `"Post-hoc comparison of pre-trained models (Mode A) or their stats JSON (Mode B; single atomic task)."` |
| pocketxmol | `/api/tasks/dock` | `app.py:502` | `"Molecular docking, small-molecule or peptide (single atomic task)."` |
| pocketxmol | `/api/tasks/sbdd` | `app.py:533` | `"De novo structure-based drug design (single atomic task)."` |
| pocketxmol | `/api/tasks/linking` | `app.py:557` | `"Fragment linking / growing / PROTAC linker design (single atomic task)."` |
| pocketxmol | `/api/tasks/optimize` | `app.py:585` | `"Molecular optimization: local refinement of an input ligand (single atomic task)."` |
| pocketxmol | `/api/tasks/pepdesign` | `app.py:613` | `"Peptide design: linear/cyclic de novo, inverse folding, sc-packing (single atomic task)."` |
| pocketxmol | `/api/tasks/confidence` | `app.py:644` | `"Tuned-ranker confidence scoring on a previously completed job (single atomic task)."` |
| reinvent | `/api/tasks/sampling` | `app.py:262` | `"De novo sampling from a Reinvent generator (single atomic task)."` |
| reinvent | `/api/tasks/scoring` | `app.py:277` | `"Score SMILES with a scoring function (single atomic task)."` |
| reinvent | `/api/tasks/enumeration` | `app.py:292` | `"Peptide enumeration with pepinvent (single atomic task)."` |
| reinvent | `/api/tasks/transfer-learning` | `app.py:310` | `"Fine-tune a generative prior on target molecules (single atomic task; long-running)."` |
| reinvent | `/api/tasks/staged-learning` | `app.py:330` | `"Staged learning: RL / curriculum over multiple stages (single atomic task; long-running)."` |

**`register_task_endpoint(...)`（9 个，依赖 A1）**

| 服务 | endpoint | 文件:行 | summary= |
|---|---|---|---|
| flowmol | `/api/tasks/generate` | `app.py:146` | `"Unconditional molecule generation (single atomic task)."` |
| immunebuilder | `/api/tasks/predict_antibody` | `app.py:194` | `"Predict antibody structure from heavy + light chain sequences (single atomic task)."` |
| immunebuilder | `/api/tasks/predict_nanobody` | `app.py:201` | `"Predict nanobody structure from heavy chain sequence (single atomic task)."` |
| immunebuilder | `/api/tasks/predict_tcr` | `app.py:208` | `"Predict TCR structure from alpha + beta chain sequences (single atomic task)."` |
| megalodon | `/api/tasks/generate` | `app.py:154` | `"Unconditional generation (single atomic task)."` |
| ppiflow | `/api/tasks/sample/monomer` | `app.py:328` | `"Unconditional monomer generation at the requested lengths (single atomic task)."` |
| rfdiffusion | `/api/tasks/generate/unconditional` | `app.py:310` | `"Unconditional monomer, or macrocycle with cyclic=true (single atomic task)."` |
| rfdiffusion | `/api/tasks/generate/symmetry` | `app.py:317` | `"Symmetric oligomer: cyclic / dihedral / tetrahedral (single atomic task)."` |
| semlaflow | `/api/tasks/generate` | `app.py:153` | `"Unconditional generation (single atomic task)."` |

**reinvent legacy（5 个，去掉 task 措辞）**

| endpoint | 文件:行 | summary= |
|---|---|---|
| `/api/sampling` | `app.py:191` | `"De novo sampling from a Reinvent generator."` |
| `/api/scoring` | `app.py:202` | `"Score SMILES with a scoring function."` |
| `/api/enumeration` | `app.py:213` | `"Peptide enumeration with pepinvent."` |
| `/api/transfer-learning` | `app.py:227` | `"Fine-tune a generative prior on target molecules (long-running)."` |
| `/api/staged-learning` | `app.py:243` | `"Staged learning: RL / curriculum over multiple stages (long-running)."` |

### 4 — B-①：defaults 语义化（框架三处 + 服务标注）

#### 4.1 `manifest.py::FieldInfo` 新增两个字段

```python
default: Any = Field(
    default=None,
    description=(
        "Effective default when the field is omitted: a literal (see default_kind='literal') "
        "or a short semantics token ('auto' / 'unset') when no single literal exists."
    ),
)
default_kind: str | None = Field(
    default=None,
    description="'literal' | 'auto' | 'unset'; 'auto'/'unset' tokens are descriptive, never sent verbatim.",
)
default_note: str | None = Field(
    default=None,
    description="One-line prose for 'auto'/'unset' omission semantics."
)
```

#### 4.2 `forms.py::model_form_depends` 透传 `json_schema_extra`（原草稿缺失的一环）

现在该函数只转发 pydantic Field 的 `description` 与默认值，**`json_schema_extra` 被丢弃**。
标量分支与复杂字段分支都要转发（`Form(...)` 接受 `json_schema_extra`，会以顶层 key 落到
OpenAPI；已实测）：

```python
# 复杂字段（dict/list/nested BaseModel）
form_default = Form(None, description=field.description,
                    json_schema_extra=field.json_schema_extra)   # Optional 分支
form_default = Form(..., description=field.description,
                    json_schema_extra=field.json_schema_extra)   # 必填分支
# 标量分支
form_default = Form(default_value, description=field.description,
                    json_schema_extra=field.json_schema_extra)
```

> 副作用：若有服务此前在模型字段上用了 `json_schema_extra`（此前被静默丢弃），透传后其内容
> 会正确浮现到 manifest——属于对既有隐蔽丢失的修复，在验证阶段扫一遍即可。

#### 4.3 `manifest.py::_extract_fields` 读标记并合成 `default`/`default_kind`

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
    out.append(FieldInfo(
        name=fname,
        type=_format_type(actual),
        required=fname in required_set,
        description=fschema.get("description") or actual.get("description"),
        is_file=_is_file_schema(actual),
        default=default,
        default_kind=default_kind,
        default_note=note,
    ))
```

> 必须从**外层** `fschema` 读标记与 `default`（`_peel_optional` 会把可选字段解包成
> `anyOf` 内的非 null 分支，丢掉外层的 `bioq_default`/`default`）。

#### 4.4 标注约定：框架提供 `bioq_service/fields.py` helper

```python
"""Field-annotation helpers for declaring omission semantics on Optional fields."""
from typing import Any, Literal

DefaultKind = Literal["auto", "unset"]

def default_semantics(kind: DefaultKind, note: str) -> dict[str, Any]:
    """Marker for ``Field(json_schema_extra=...)``: declares what omitting an
    Optional field means when no single literal default exists."""
    return {"bioq_default": {"kind": kind, "note": note}}
```

从 `bioq_service/__init__.py` 再导出 `default_semantics`。服务用法（例子）：

```python
from bioq_service import default_semantics

device: Optional[str] = Field(
    default=None,
    description="CUDA device string (e.g. 'cuda:0').",
    json_schema_extra=default_semantics("auto", "auto-select CUDA if available"),
)
```

#### 4.5 处置规则 + token 词表

对每个 `Optional[X] = None` 参数，三选一：

| 省略时的实际行为 | 处置 | `default` | `default_kind` | 备注 |
|---|---|---|---|---|
| 有安全字面量，且"显式传 L == 省略"（上游默认） | **B1**：字段改成具体默认值（去掉 `Optional`） | 字面量 L | `literal` | 例如 `guide_scale: float = 10.0` |
| 运行时自选，无单一字面量 | 标注 `default_semantics("auto", n)` | `"auto"` | `auto` | device / model / seed / cores |
| 省略即不启用 / 不传 | 标注 `default_semantics("unset", n)` | `"unset"` | `unset` | 可选约束、过滤、JSON 覆盖 |

**代表性映射**（实现时按此归类，不再逐个臆造字面量）：

| 参数模式 | 例子 | 处置 |
|---|---|---|
| `seed` / `random_seed` | flowmol、iggm、diffdock、diffdock-pp、pocketxmol、semlaflow、megalodon、alphafold | `auto`（"工具自选随机种子"）——**绝不 B1**，否则随机→固定，改行为 |
| `device` | reinvent | `auto`（"auto-select CUDA"） |
| `model` / checkpoint override | rfdiffusion、rfdiffusion2、boltzgen | `auto`（"由上游按输入自选"） |
| `cores` / `n_cpu` / `maxthreads` | lightdock、dockq、plip | `auto`（"用尽可用核"），或 B1 若上游有明确默认 |
| `guide_scale` / `guide_decay`（上游有明确默认） | rfdiffusion | B1 → `literal`（10.0 / "constant"） |
| 可选约束/过滤（None=off） | chembounce 的 `mw_min`/`logp_max` 等 | `unset` |
| 可选 dict/JSON 覆盖 | reinventing `pairs`、rfdiffusion `extra_overrides`、boltz `constraints` | `unset` |

> **B1 只做"显式传 == 省略"这一种**：错误字面量比缺失更糟，拿不准一律走 `auto`/`unset`。

#### 4.6 服务标注范围（task 端点，176 参数 / 24 服务）

按 fix-suggestions 优先级（reinvent → pocketxmol → rfdiffusion → drughive …）推进；下表为
`audit.py` 同口径的完整枚举，作为实现 checklist。

> **标注落在模型字段上**：同一请求模型被 legacy `/api/*` 与 `/api/tasks/*` 孪生端点共用，
> 所以在模型字段上做 B1/标注会**同时修复两个孪生端点**；下表仅按 task 口径统计（对应打分
> 分母），非 task 孪生端点由此一并转正，无需单独列出。

| 服务 | 参数数 | 可选 `default=null` 参数 |
|---|---|---|
| boltz | 18 | constraints, msa_files, raw_yaml, raw_yaml_uri, seed, sequences, step_scale, template_files, templates |
| chembounce | 16 | core_smiles, h_acceptor_max, h_acceptor_min, h_donor_max, h_donor_min, input_smiles_uri, logp_max, logp_min, mw_max, mw_min, overall_max_n, qed_max, qed_min, sa_max, sa_min, scaffold_top_n |
| reinvent | 16 | agent_file, agent_file_uri, device, diversity_filter, inception, input_model_file, input_model_uri, learning_strategy, model_file, pairs, prior_file, prior_file_uri |
| rfdiffusion | 15 | extra_overrides, guide_decay, guide_scale, guiding_potentials, hotspots, inpaint_seq, input_uri, length, model |
| boltzgen | 14 | alpha, analysis_num_processes, diffusion_batch_size, inverse_fold_avoid, noise_scale, ref_files_zip_uri, step_scale |
| pocketxmol | 14 | fix_pos_res_bb, fix_pos_res_sc, fix_type_res_bb, fix_type_res_sc, pep_length, pep_sequence, pocket_coord, seed, smiles |
| rfdiffusion2 | 12 | extra_overrides, inpaint_seq, input_uri, length, ligand, model, only_guidepost_positions, partially_fixed_ligand |
| openadmet | 11 | aq_fxns, best_y, beta, input_col, input_smiles, label_types, labels, model_names, mt_id, task_names, xi |
| drughive | 10 | mol_filter, substruct_modify_pattern, temps, zbetas |
| proteinmpnn | 8 | bias_AA, bias_by_res, chains_to_design, fixed_positions, omit_AA_per_chain, tied_positions |
| dockq | 6 | mapping, models, models_zip_uri, n_cpu |
| genie3 | 6 | num_devices, selections |
| flowmol | 4 | hc_thresh, n_atoms_per_mol, seed, stochasticity |
| iggm | 4 | epitope, seed |
| plip | 4 | intra_chain, maxthreads, peptide_chains, report_formats |
| ppiflow | 4 | length_subset, specified_hotspots |
| diffdock | 3 | ligand_description, protein_sequence, seed |
| openbpmd | 3 | equil_steps, sim_ns, system_format |
| megalodon | 2 | n_atoms_per_mol, seed |
| rfantibody | 2 | hotspots, seed |
| alphafold | 1 | random_seed |
| diffdock-pp | 1 | seed |
| lightdock | 1 | cores |
| semlaflow | 1 | seed |

### 5 — 回归测试（TDD）

**框架层**（先写、确认在改动前失败）：

- `framework/tests/test_manifest.py`：
  - 新增 `test_extract_fields_marks_file_arrays_as_files`（原草稿代码，断言 `is_file=True` 且
    `type=='array[file]'`）。
  - 新增 `test_extract_fields_semantic_default_auto_unset`：构造带 `bioq_default` 标记的 schema，
    断言 `default`/`default_kind`/`default_note` 三者正确；未标记的 `Optional` 仍为 `default=None`。
- `framework/tests/test_task_endpoint.py`：`register_task_endpoint(..., summary=...)` 后 manifest
  endpoint 的 `summary` 生效。
- `framework/tests/test_forms.py`（若无则建）：`model_form_depends` 把模型字段的
  `json_schema_extra` 透传进 OpenAPI schema（即 4.2 的回归）。

**服务层**（代表性 manifest 回归）：

- openadmet：`/api/tasks/compare` 的 `model_stats_files` 出现在 `request_fields` 且
  `is_file=True`/`type='array[file]'`。
- reinvent / rfdiffusion：受影响 endpoint 的每个可选非文件参数 `default != null`，且
  `default_kind` 在 `{"literal","auto","unset"}` 内。

## 明确不做（Out of Scope）

- **bioq-paper 的 E12 `audit.py`/`score.py` 检查口径**——本设计在 bioq-services 侧闭环，
  `defaults` 检查仍按 `default != null`，不改审计逻辑（重跑 E12 只是"重新采集 + 评分"，属验证）。
- **bioq CLI 的 array 上传侧支持**：`array[file]` 转正后 `bioq describe openadmet` 会把它列进
  `files:` 区；CLI 若对"同名多文件（array 语义）"的 `--file` 上传不支持，属 bioq 仓库跟进。
- `esmfold2` / `promera` 审计时不可达（冷启动/运维问题），不计分，不属本设计。
- `operation_id` manifest 未填充（审计仅作信息列，不计分）。
- 版本号 bump / 镜像重新构建部署 / E12 重跑（发布与验证期动作）。

## Success Criteria

1. 框架测试 + 全部受影响服务离线测试通过：
   `cd framework && uv run --extra dev python -m pytest tests/ -q`，
   以及 `uv run python -m pytest services/{openadmet,reinvent,rfdiffusion,...}-server/tests`。
2. `docs_text` task 通过率 61.5% → **100%**（65/65）。
3. `file_fields` 98.5% → **100%**。
4. `defaults` task 通过率 16.9% → **100%**（176 个参数全数 `default != null`，`default_kind`
   覆盖 `literal`/`auto`/`unset`，无残留"未声明"）。
5. 服务中位数 ≥0.9 达标 **28/28**；task endpoint 全通过 **65/65**。
6. 抽查：`bioq describe reinvent` 每 endpoint 出现 summary；`bioq describe openadmet` 的
   `compare` 把 `model_stats_files` 列进 `files:` 区且带 `default`/`default_kind` 语义。

## Open Questions

- `default` 承载 `"auto"`/`"unset"` token 后，需核对 **bioq CLI（本仓库之外）** 是否把
  `default` 当字面量自动回填请求——若有，需让 CLI 依据 `default_kind` 跳过非 literal 项（属
  bioq 仓库跟进；本仓库内 manifest 无此类消费方）。
- `default_kind` 取值是否要扩充第四类 `upstream`（语义"工具内部默认，未显式声明"）？当前
  并入 `auto`/`unset` 已够用，若后续要区分"工具默认"与"运行时自选"再扩，属向后兼容新增。
- 复用已有 description 里"default 10 / leave unset ..."的散文：是否要把它们同步搬进
  `default_note`（避免同义双写），还是保留现状仅新增 `default_note` 增量？倾向后者（最小 diff）。