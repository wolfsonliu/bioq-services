# RFantibody URI 字段命名修复设计（`input_uri` → `input_quiver_uri`）

## Problem Statement

RFantibody 服务的 `proteinmpnn` / `rf2` 端点（同步 `/api/proteinmpnn`、`/api/rf2` 与
task `/api/tasks/proteinmpnn`、`/api/tasks/rf2`）把「Quiver 输入的 URI 表单字段」命名成了
`input_uri`，违反了全平台的 `<upload_field>_uri` 命名约定。

bioq CLI 的 `--file <field>=<path>` 约定（`bioq/upload.py` 的 `upload_files`）会把字段映射为
`{field}_uri` 写入请求体：

```
--file input_quiver=<path>  →  body["input_quiver_uri"] = "job://.../1_rfdiffusion.qv"
```

但端点签名只声明了 `input_uri`。FastAPI 对 multipart 里未声明的字段直接丢弃，导致：

```
input_quiver=None, input_uri=None
resolve_input(...) → 422 "Either an upload or input_uri is required."
```

这正是生产 FC 日志里的 422（`app.py` `_save` 第 238 行 → `uris.py` `resolve_input`）。
后果：bioq CLI 无法用 `--file input_quiver` 把 ProteinMPNN / RF2 链到前一步的 `.qv` 输出，
三阶段 RFantibody 流水线（rfdiffusion → proteinmpnn → rf2）在 CLI 侧走不通。

## Proposed Solution

方案 A（本服务为 v0.x，接受 breaking change）：把 4 个端点的 `input_uri` 统一重命名为
`input_quiver_uri`，与上传字段 `input_quiver` 对齐、与 rfdiffusion 端点的
`target/target_uri`、`framework/framework_uri` 命名一致。同时给所有 `resolve_input` 调用补上
`field_name`，让 422 报错能指出是哪个字段缺失。

## Detailed Design

### 改动 1 — `services/rfantibody-server/app.py`（核心）

4 个端点的 `input_uri` → `input_quiver_uri`，docstring 同步：

| 端点 | 位置 | 改动 |
|---|---|---|
| `/api/proteinmpnn` | docstring / 签名 / resolve | `input_uri` → `input_quiver_uri` + `field_name="input_quiver"` |
| `/api/rf2` | docstring / 签名 / resolve | 同上 |
| `/api/tasks/proteinmpnn` | 签名 / resolve | 同上 |
| `/api/tasks/rf2` | 签名 / resolve | 同上 |

另外给 rfdiffusion 的 4 处 `resolve_input`（同步 `_build` 与 task `_save`）补上
`field_name="target"` / `field_name="framework"`，让缺输入时报错可定位。

`field_name` 是共享框架 `bioq_service/uris.py:resolve_input` 已支持的参数（默认 `None`），
传上后 422 详情变为 `<field>: either an upload or a URI is required.`。

### 改动 2 — `services/rfantibody-server/adapter.py`

- `endpoint_examples` 中 `/api/proteinmpnn`、`/api/rf2` 的 curl / python 示例共 4 处
  `input_uri=` → `input_quiver_uri=`。
- `manifest_extras["chaining_tip"]` 中 2 处 `input_uri=job://…` → `input_quiver_uri=job://…`。
- `manifest_extras["input_uri_schemes"]` 键名**保留不动**：它描述的是「URI 方案列表」这一通用
  概念（job://、file://、oss://、http(s)://），不是表单字段名。

### 改动 3 — 测试

- `tests/test_app.py`：`test_proteinmpnn_task_endpoint_accepts_uri_fallback` 的请求体
  `"input_uri"` → `"input_quiver_uri"`；新增一条 manifest 回归测试，断言 4 个端点的
  `request_fields` 含 `input_quiver` + `input_quiver_uri` 且不含 `input_uri`。
- `tests/test_fc.py`：190、219 行 `"input_uri"` → `"input_quiver_uri"`。
- `tests/test_fc_task.py`：254、286、520 行 `"input_uri"` → `"input_quiver_uri"`。

`test_app.py:145-146`、`test_fc.py:317` 中的 `input_uri_schemes` 是 manifest 键名，保留不动。

### 版本

`VERSION` 当前 `v0.0.20`。字段改名属于 breaking change，发布时通过
`make bump-rfantibody` 再弹版本号，不在本次代码改动内处理。另记录一个既有不一致：
`pyproject.toml` 与 `app.py` 默认版本为 `0.2.0`，而 `VERSION`（实际镜像标签来源）为
`v0.0.20`，建议后续单独对齐。

## 明确不做（Out of Scope）

- 不改共享框架 `framework/src/bioq_service/uris.py` 的默认 422 文案（影响 40+ 服务）；
  `field_name` 参数已存在，本服务传上即可避免命中默认文案。
- 不改 gateway / bioq CLI（CLI 的 `--file` → `{field}_uri` 映射本身是对的）。
- 不动输出命名 `1_rfdiffusion.qv / 2_proteinmpnn.qv / 3_rf2.qv` 与 `output/` 内部约定。
- 同类 bug 也存在于 `rfdiffusion-server` / `rfdiffusion2-server`（`input_pdb` 配 `input_uri`
  而非 `input_pdb_uri`），本次不处理，另开 issue。

## Success Criteria

1. `pytest services/rfantibody-server/tests/test_app.py` 通过（离线）。
2. `bioq describe rfantibody` 显示 `--file input_quiver=<path>`，且不再出现冗余的
   `--set input_uri`。
3. 缺输入时 422 详情为 `input_quiver: either an upload or a URI is required.`
4. FC 集成测试（opt-in）`test_fc.py` / `test_fc_task.py` 在重新部署后通过。

> 验证备注：本地离线跑 `test_app.py` 时，`test_openapi_lists_service_request_models`
> 会因本机 framework venv 的 FastAPI 0.136.1（服务 `uv.lock` 锁定的是 0.139.2，组件
> schema 的 `$ref` 生成行为不同）而失败；该用例在改动前/后都失败（已用 `git stash` 验证），
> 与本次字段改名无关，属于既有的环境版本差异，非本次回归。