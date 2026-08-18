# 服务解剖学

[English](service-anatomy.md) | 中文

> **适用**：在 `services/<svc>-server/` 内创建或修改文件时。
> **来源**：由真实服务（见参考表）与 [`../adding-a-new-service/index.zh.md`](../adding-a-new-service/index.zh.md) 的清单归纳。
> **刷新/删除条件**：必备文件契约变化（如清单新增了强制文件）时。

## 必备文件

```
services/<svc>-server/
├── __init__.py          # 包标记，通常空
├── app.py               # create_app + 服务专属 endpoint + /healthz/detail + task endpoint
├── __main__.py          # CLI 批处理入口（python -m server <endpoint>）
├── adapter.py           # JobAdapter 子类（name + detect_outputs + manifest_extras + endpoint_examples）
├── settings.py          # ServiceSettings 子类（env_prefix=<SVC>_）；weights_dir 默认 /data/models/<svc>/
├── models.py            # 请求 pydantic models
├── tools.py             # argv builders（可选，视复杂度）
├── Dockerfile           # COPY framework + services/<svc>/upstream/ + 算法栈
├── pyproject.toml       # 离线测试/开发依赖（仅 uv venv 骨架需要；部分服务无此文件）
├── README.md            # endpoint / 配置 / 部署说明 + «Weights» 章节
├── VERSION              # 镜像 tag（Makefile 读取，如 "v0.0.1"）
├── scripts/
│   ├── vendor.sh        # 必备：clone 上游源码到 upstream/ at pinned SHA
│   └── fetch_weights.sh # 可选：预下载权重到 weights/ 或直接 NAS（WEIGHTS_DST 覆盖）
└── tests/
    ├── __init__.py / conftest.py   # server 模块注册（importlib）+ fc marker
    ├── test_app.py      # 离线 TestClient 单测
    ├── test_cli.py      # CLI 批处理单测
    ├── test_fc.py       # FC sync 集成测试（@pytest.mark.fc，默认 skip）
    ├── test_fc_task.py  # FC 异步 task 模式测试（默认 skip）
    └── data/            # 测试 fixture（小 PDB/JSON 等）
```

`upstream/`（vendor.sh 产物）与 `weights/`（fetch_weights.sh 产物）不入库。

## 最省事的起点

把一个结构相近的现有服务整目录拷贝改名，再逐文件改。

| 场景 | 参考 |
|---|---|
| uv venv + 序列设计 + 权重外置；vendor.sh 单上游标准示范 | `services/proteinmpnn-server/` |
| CPU-only uv-venv slim | `services/dockq-server/`、`services/diamond-server/` |
| conda/micromamba 多阶段 | `services/deeprank-ab-server/`、`services/pocketxmol-server/` |
| manifest_extras + endpoint_examples 完整示范 | `services/rfantibody-server/`、`services/genie3-server/` |
| 多 endpoint + config YAML 驱动 | `services/genie3-server/`、`services/drughive-server/` |
| vendor.sh 多 upstream / symlink 权重 | `services/promera-server/` |
| fetch_weights.sh + 大镜像瘦身 | `services/boltzgen-server/` |

## 备注

- 可选 per-service 文件（`configs.py`、`datasets.py`、`patches/`、额外脚本）见
  [`../adding-a-new-service/index.zh.md`](../adding-a-new-service/index.zh.md)。
- `adapter.py` / `app.py` 背后的 job 生命周期模型见 [framework-api.md](./framework-api.md) 与
  [mental-model.md](./mental-model.md)。
