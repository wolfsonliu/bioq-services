# PPIFlow Server

基于 FastAPI 的 PPIFlow HTTP 服务，**仅暴露 PPIFlow 自身的结构生成功能**：

| Endpoint | 用途 | 权重 |
|---|---|---|
| `POST /api/sample/binder` | PPI binder design 对靶蛋白 | `binder.ckpt` |
| `POST /api/sample/antibody` | 抗体 CDR 设计 (heavy + light) | `antibody.ckpt` |
| `POST /api/sample/nanobody` | VHH CDR 设计 (heavy-only) | `nanobody.ckpt` |
| `POST /api/sample/monomer` | 无条件单体生成 | `monomer.ckpt` |
| `POST /api/sample/scaffolding` | Motif 支架生成 | `monomer.ckpt` |

**不在此服务范围内**（PPIFlow 完整 pipeline 的其他步骤将作为单独的 bioagent service 开发）：

- 序列设计（ProteinMPNN / AbMPNN）→ 另起 `proteinmpnn-server`
- 侧链 packing（Flowpacker）→ 另起 `flowpacker-server`
- 评分（AF3Score）→ 另起 `af3score-server`
- Rosetta refinement / DockQ → 另起 `rosetta-server`
- Partial-flow 局部重设计（`sample_*_partial.py`）→ 未来在 v0.0.2+ 加 endpoint

**v0.2 起构建在 [bioagent-service-framework](../_framework/) 之上**：HTTP / job 生命周期 /
错误处理 / 持久化 / 多实例一致性 / Agent manifest 由框架统一提供。

## 架构

```
客户端 / Agent
  ↓ HTTP
┌────────────────────────────────────────────────────────────────┐
│  FastAPI + bioagent-service-framework  (port 9000)             │
│                                                                │
│  服务专属 (5 个 sampler)                                       │
│    POST /api/sample/{binder,antibody,nanobody,monomer,scaffolding}
│                                                                │
│  框架统一提供                                                  │
│    GET  /api/manifest      (Agent 协议描述 + 5 个端点示例)      │
│    GET  /openapi.json                                          │
│    GET  /healthz, /healthz/detail                              │
│    GET  /api/jobs/{id} (+ /files /log /download /file/{path})  │
│    DELETE /api/jobs/{id}                                       │
└────────────────────────────────────────────────────────────────┘
  ↓ subprocess (cwd=/opt/ppiflow, 单卡串行)
python sample_<mode>.py --model_weights checkpoint/<mode>.ckpt ...
  ↓
NAS: /data/ppiflow_jobs/<job_id>/output/<name>/*.pdb + ... + job.json
```

## API 速览

每个 POST 端点接 `multipart/form-data`：文件字段 + pydantic 参数（`Annotated[..., Form()]`）。
返回 `JobInfo`（含 `job_id`），实际计算在后台异步执行。

### binder

```bash
curl -X POST http://localhost:9000/api/sample/binder \
  -F target=@target.pdb \
  -F target_chain=B \
  -F binder_chain=A \
  -F 'specified_hotspots=B119,B141,B200' \
  -F samples_min_length=75 \
  -F samples_max_length=120 \
  -F samples_per_target=5 \
  -F name=IL7Ra
```

输出：`output/IL7Ra/*.pdb`

### antibody / nanobody

抗体框架 PDB 需**先去除 CDR 环 + IMGT 编号**（详见 PPIFlow README 的 `framework_pdb` 说明）。

```bash
# antibody (heavy + light)
curl -X POST http://localhost:9000/api/sample/antibody \
  -F antigen=@antigen.pdb \
  -F framework=@framework.pdb \
  -F antigen_chain=C -F heavy_chain=A -F light_chain=B \
  -F 'specified_hotspots=C11,C14,C15' \
  -F 'cdr_length=CDRH1,8-8,CDRH2,8-8,CDRH3,10-20,CDRL1,6-9,CDRL2,3-3,CDRL3,9-11' \
  -F samples_per_target=5 \
  -F name=1IJZ_IL13

# nanobody (heavy-only)
curl -X POST http://localhost:9000/api/sample/nanobody \
  -F antigen=@antigen.pdb \
  -F framework=@nanobody_framework.pdb \
  -F antigen_chain=C -F heavy_chain=A \
  -F 'specified_hotspots=C101,C135,C171,C198' \
  -F 'cdr_length=CDRH1,8-8,CDRH2,8-8,CDRH3,9-21' \
  -F samples_per_target=5 \
  -F name=1CVS_FGFR1
```

### monomer / scaffolding

```bash
# unconditional monomer at multiple lengths
curl -X POST http://localhost:9000/api/sample/monomer \
  -F 'length_subset=[50, 100]' \
  -F samples_per_target=5 \
  -F name=monomer_test

# motif scaffolding from CSV
curl -X POST http://localhost:9000/api/sample/scaffolding \
  -F motif_csv=@motif_metadata.csv \
  -F 'motif_names=["01_1LDB"]' \
  -F samples_per_target=5 \
  -F name=motif_test
```

### 支持的 input URI schemes

可上传文件，**也可** 用 URI 字段（每个文件字段都有 `<name>_uri` 对应项）引用 NAS 上已有文件，
避免重复上传：

| scheme | 用途 |
|---|---|
| `job://<job_id>/<filename>` | 拉同 service 历史 job 的输出 |
| `file:///abs/path` | NAS 上绝对路径（跨 service 共享挂载） |
| `oss://<bucket>/<key>` | OSS 对象（需 `OSS_ACCESS_KEY_ID` / `_SECRET` 环境变量） |
| `http(s)://...` | 任意 URL，含 OSS 签名 URL |

### Agent 友好接口

```bash
curl http://localhost:9000/api/manifest    # 含 5 个 endpoint 的 schema 引用 + curl 示例
curl http://localhost:9000/openapi.json    # 完整字段 schema
```

manifest 的 `service_specific` 段含：
- `tool_outputs` —— 每个 endpoint 的输出文件命名约定
- `input_uri_schemes` —— 上面那张表
- `weights` —— 权重清单 + 调试提示
- `config_tips` —— cdr_length / length_subset / specified_hotspots / framework_pdb 四个常见坑
- `endpoints_summary` —— 5 个 endpoint 一句话说明
- `not_in_scope_v0_0_1` —— 明确告诉 agent partial-flow / 后处理在别的 service

### 失败响应

`JobInfo` 自动填充：

| 字段 | 含义 |
|---|---|
| `failure_kind` | `subprocess_error` / `no_outputs` / `interrupted` / `dataset_invalid` |
| `error_summary` | log 中提取的最后一行异常 |
| `error_tail` | log 尾部 ~4 KB |

## 配置

环境变量前缀 `PPIFLOW_`：

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `PPIFLOW_JOBS_BASE_DIR` | `/data/ppiflow_jobs` | NAS 上的 job 根目录 |
| `PPIFLOW_ROOT` | `/opt/ppiflow` | PPIFlow `tool/PPIFlow` 源码根（subprocess cwd） |
| `PPIFLOW_CKPT_DIR` | `/data/models/ppiflow/checkpoint` | 权重目录（v0.0.11 起 NAS 挂载，~1.1 GB 外置）|
| `PPIFLOW_CONFIG_DIR` | `/opt/ppiflow/configs` | 推理 YAML 目录 |
| `PPIFLOW_PORT` | `9000` | uvicorn 端口（FC CAPort） |
| `PPIFLOW_KEEP_ALIVE_SEC` | `900` | uvicorn `--timeout-keep-alive` |
| `PPIFLOW_MAX_CONCURRENT_JOBS` | `1` | 单实例并发（单卡保持 1） |
| `PPIFLOW_DISK_LIMIT_MB` | `8000` | 自动清理已完成 job 的阈值 |
| `PPIFLOW_ERROR_TAIL_CHARS` | `4000` | JobInfo.error_tail 字节数 |
| `PPIFLOW_OSS_REGION` | `cn-hangzhou` | OSS URI 下载用的区域 |

OSS 凭证按阿里云 SDK 约定走 `OSS_ACCESS_KEY_ID` + `OSS_ACCESS_KEY_SECRET`。

## 本地开发

PPIFlow 依赖 conda（pytorch + CUDA + 一堆 bioconda 包，纯 pip 装不齐）。Dockerfile 用
`nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` 作 base（与 genie3-server 一致，共享 CI 缓存
层），在其上手动安装 micromamba 来跑 PPIFlow 的 environment.yml；PyTorch wheels 用 CUDA
12.1 编译，在 12.4 runtime 上向前兼容。本地开发同样用 micromamba：

```bash
# 1. 创建 PPIFlow 的 conda 环境
cd opensource/PPIFlow/tool/PPIFlow
micromamba env create -y -n ppiflow -f environment.yml
micromamba activate ppiflow

# 2. 装服务框架
pip install ../../../../services/_framework httpx alibabacloud-oss-v2

# 3. 软链 server 包让 uvicorn 找到
ln -s $(pwd)/../../../../services/ppiflow-server $(pwd)/server

# 4. 跑
export PPIFLOW_ROOT=$(pwd)
export PPIFLOW_CKPT_DIR=$(pwd)/../../checkpoint
export PPIFLOW_JOBS_BASE_DIR=/tmp/ppiflow_jobs
uvicorn server.app:app --host 0.0.0.0 --port 9000 --reload
```

## Weights

v0.0.11 起 4 个 `.ckpt` 权重（~1.1 GB）**不再 baked 到镜像**，从 NAS 加载。
期望布局：

```
/data/models/ppiflow/
└── checkpoint/
    ├── binder.ckpt
    ├── antibody.ckpt
    ├── nanobody.ckpt
    └── monomer.ckpt
```

### Pre-stage（一次性）

把上游下好的 4 个 `.ckpt` 放到本地 stage 目录或直接上传到 NAS：

```bash
# 1. 准备：手动下载 4 个 .ckpt 到 services/ppiflow-server/checkpoint/
#    （上游 PPIFlow 没有公开下载脚本，权重需从作者处获取）

# 2. 上传到 NAS
rsync -av services/ppiflow-server/checkpoint/ \
    <NAS-mount>:/data/models/ppiflow/checkpoint/
```

### FC

NAS 自动挂载到 `/data/models/ppiflow/`。验证：

```bash
curl https://fc-ppiflow-lufflhmlaw.cn-hangzhou-vpc.fcapp.run/healthz/detail
# 期望：{"status":"ok","weights_loaded":true,"ckpts_found":4}
```

### SIF / HPC

```bash
apptainer run --nv \
    --bind /scratch/models/ppiflow:/data/models/ppiflow \
    ppiflow-server.sif python -m server sample binder ...
```

详见 [weights externalization design](../../engineering/decisions/2026-06-26-service-weights-externalization.md)。

## Docker 构建与运行

```bash
# 1. Vendor 上游源（一次性，重跑可升级 SHA）
./services/ppiflow-server/scripts/vendor.sh

# 2. 构建（~5.5 GB 镜像，无权重）
docker build --platform linux/amd64 -t ppiflow-server -f services/ppiflow-server/Dockerfile .

# 或通过 Makefile（用 services/ppiflow-server/VERSION 里的 tag）
make build-ppiflow-server

# 本地运行（需 GPU + NAS / 本地 --bind 注入权重）
docker run --gpus all -p 9000:9000 --memory 16g \
    -v $(pwd)/checkpoint:/data/models/ppiflow/checkpoint \
    ppiflow-server
```

构建上下文必须是项目根目录，因为 Dockerfile 同时 `COPY services/_framework`
安装框架包 + `COPY services/ppiflow-server/upstream/` 取 vendored 源码。

## 阿里云函数计算部署

参考 [services/genie3-server/README.md §阿里云函数计算部署](../genie3-server/README.md)
的步骤；对 ppiflow-server 的具体差异：

| 配置项 | 推荐 |
|---|---|
| GPU 配置 | `fc.gpu.tesla.1` 起步（16 GB 显存：抗体 / VHH 设计长输入会用更多） |
| 函数超时 | ≥ 7200 秒（5 个 sample 通常 5–20 分钟） |
| 内存 | `32768` MB（PPIFlow 加载几个 ckpt + transformer 模型，单卡较吃 RAM） |
| 磁盘 | `10240` MB |
| NAS 挂载 | 推荐 `/fc → /data`，跨实例 / 跨服务共享 job 文件 |

镜像 push 与 FC 函数更新：

```bash
make push-ppiflow-server    # 自动用 services/ppiflow-server/VERSION 里的 tag
# 然后到 FC 控制台把函数镜像更新到 harbor.ruosheng.bio/aliyun_fc/ppiflow-server:vX.Y.Z
```

## 设计要点

- **仅生成**：本服务只跑 PPIFlow 的 5 个 sampler。完整 pipeline 的后处理步骤（MPNN / AF3 /
  Rosetta）有各自的 bioagent service。
- **5 个独立端点**：每个 endpoint 对应一种生成模式 + 一份权重，agent 不必猜该用哪个 ckpt。
- **共享 NAS 链式调用**：生成 PDB 写到 `/data/ppiflow_jobs/<id>/output/<name>/`，下游
  `proteinmpnn-server` / `af3score-server` 等可直接 `Path(...).read_bytes()` 读，无 HTTP 上下行。
- **错误信息丰富**：失败 job 的 `JobInfo` 自带 `error_summary` + `error_tail` + `failure_kind`，
  快速判断是 ckpt 缺失还是 input PDB 格式问题。
- **快速启动**：`/healthz` 端点无模型加载，FC 120 s 启动探测通过；权重烘焙到镜像避免冷启动下载。

## 已知限制 / 后续

- v0.0.1 不含 `sample_*_partial.py` 的 partial-flow 端点。需要时在 v0.0.2 加。
- PPIFlow conda env 较大（~10 GB），镜像总大小可能逼近 FC 15 GB 上限。监控 Docker 镜像大小，
  必要时 squash 层或剔除不必要的 conda 包。
- 单实例 GPU 串行；不适合大批量并行，需要时升级到多实例 FC + NAS 共享。

## 相关文档

- [新增 bioagent service cookbook](../../engineering/guides/adding-a-new-service.md) —— 本 service 按这份建
- [bioagent-service-framework](../_framework/README.md) —— 通用 HTTP / job / 错误处理 / manifest 层
- [调用 bioagent service](../../engineering/guides/calling-bioagent-services.md) —— Agent / client 调用协议
- [PPIFlow 上游 README](../../opensource/PPIFlow/tool/PPIFlow/README.md) —— 算法本身的参数细节
