# 跨服务 URI 字段命名修复设计（`input_uri` → `<upload>_uri`）

## Problem Statement

bioq CLI 的 `--file <field>=<path>` 约定把上传字段映射为请求体里的 `<field>_uri`
（`bioq/upload.py`）。因此每个「文件上传 / URI 二选一」的输入，其 URI 表单字段名必须
严格等于 `<upload_field_name>_uri`，否则 CLI 生成的字段会被 FastAPI 丢弃，端点两侧
`upload=None, uri=None` → 422。

对全部服务的 `resolve_input` / `maybe_resolve_input` 调用与所有 `File()` / `*_uri Form()`
声明做全量排查后，确认 rfantibody（已修复）之外还有 **3 个服务、16 个端点** 存在同类
命名错配：

| 服务 | 上传字段 | 错误 URI 字段 | 应改为 | 端点 |
|---|---|---|---|---|
| `rfdiffusion-server` | `input_pdb` | `input_uri` | `input_pdb_uri` | 6（3 同步 + 3 task） |
| `rfdiffusion2-server` | `input_pdb` | `input_uri` | `input_pdb_uri` | 6（3 同步 + 3 task） |
| `bindflow-server` | `custom_ff_zip` | `custom_ff_uri` | `custom_ff_zip_uri` | 2（fep + mmpbsa） |
| `bindflow-server` | `topology_zip` | `topology_uri` | `topology_zip_uri` | 2（fep + mmpbsa） |

另发现上一轮 rfantibody 修复漏掉了 README 同步（该文件仍多处写 `input_uri`），本次一并补。

## Proposed Solution

沿用 rfantibody 的解法（方案 A：彻底重命名，无向后兼容别名；均为 v0.x）。把 URI 字段名
改成与上传字段严格一致，并给所有 `resolve_input` 调用补 `field_name` 以便 422 报错定位。

## Detailed Design

### 1 — `rfdiffusion-server`：`input_uri` → `input_pdb_uri`

`app.py`（16 处）：3 同步端点（motif / binder / custom）的 docstring、`Form` 签名、
`resolve_input` 调用，以及 3 task 端点同款；全部 `resolve_input` 补 `field_name="input_pdb"`
（custom 端点的 resolve 位于 `if input_pdb is not None or input_uri:` 守卫内，一并改名守卫条件）。
- `adapter.py:162` `downstream_pipeline_tip` 的 `` `input_uri=job://...` `` → `input_pdb_uri=...`；
  `:135` 的 `input_uri_schemes` 键保留。
- `README.md:136,185` 同步。
- `tests/test_app.py`：新增 manifest 回归测试。

### 2 — `rfdiffusion2-server`：`input_uri` → `input_pdb_uri`

`app.py`（16 处）：active_site / small_molecule_binder / custom 的同步 + task，补 `field_name="input_pdb"`。
- `adapter.py`：无改动（仅有 `:132` 的 `input_uri_schemes` 键，保留）。
- `README.md:81,115,159–162` 同步。
- `tests/test_app.py`：新增 manifest 回归测试。

### 3 — `bindflow-server`：`custom_ff_uri`→`custom_ff_zip_uri`、`topology_uri`→`topology_zip_uri`

`app.py`（12 处）：`_stage_inputs` 签名、2 个 `resolve_dir_zip` 调用、fep/mmpbsa 两端点的
`Form` 声明与 `_build` kwargs。**`uris.py` 不改**——`resolve_dir_zip(upload, zip_uri, ...)`
的参数名是通用 `zip_uri`，错配只发生在 app.py 的表单字段名上。
- `_stage_inputs` 里的 `field_name="custom_ff"/"topology"` 仅用于内部临时 zip 文件名，
  非用户可见，保持不动（最小 diff）。
- `adapter.py:78` 键保留；`README.md:56,57` 同步。
- `tests/test_app.py`：新增 manifest 回归测试。

### 4 — 补 rfantibody README 遗漏

`services/rfantibody-server/README.md:35,57,67,77,83` 的 `input_uri` → `input_quiver_uri`
（`:104` 的 `input_uri_schemes` 是 manifest 键名，保留）。

### 回归测试（TDD）

三个服务各新增一条 manifest 回归测试，镜像 rfantibody 的
`test_quiver_uri_field_matches_upload_field`：断言受影响端点的 `request_fields` 含
上传字段（`is_file=True`）+ `<upload>_uri`，且不含旧的错误字段名。先写测试并确认其在
当前（buggy）代码上失败，再实现改名让它转绿。

## 明确不做（Out of Scope）

- 类别 2（`ref_files`/`models`/`ligands`/`msa_files` 等 `List[UploadFile]` 上传 +
  `*_zip_uri` 命名偏离）——语义不同，可能是「多传文件或传一个 zip URI」的有意设计，另开 issue。
- 共享框架 `uris.py` 默认 422 文案、gateway / bioq CLI。
- 版本号 bump（发布期动作，`make bump-<svc>`）。

## Success Criteria

1. 三个服务的 `pytest tests/test_app.py` 通过（离线）。
2. `bioq describe rfdiffusion` / `rfdiffusion2` / `bindflow` 显示 `--file input_pdb=<path>` /
   `--file custom_ff_zip=<path>` 等，且不再出现冗余的 `--set input_uri` / `--set custom_ff_uri`。
3. 缺输入时 422 详情带 `input_pdb:` / `custom_ff_zip:` 语境的字段名。
4. 文档与实现一致：全仓库 `grep -r 'input_uri'` 仅剩 `input_uri_schemes`（manifest 键名）。

## Open Questions

- `bindflow` 的两个 zip 上传字段：是否值得把上传名里的 `_zip` 去掉（`custom_ff_zip`→`custom_ff`、
  `topology_zip`→`topology`）以简化语义？本次按最小改动原则仅改 URI 字段名，上传名保持不动。