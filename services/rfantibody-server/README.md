# RFantibody Server

基于 FastAPI 的 RFantibody HTTP 服务，**v0.2 起构建在 [bioq-service-framework](../_framework/) 之上**：
HTTP 层 / job 生命周期 / 错误处理 / 持久化 / 多实例一致性 / Agent 协议描述均由框架统一提供，服务自身只
负责 RFantibody 三个工具（RFdiffusion / ProteinMPNN / RF2）的 argv 拼装与 URI 解析。

## 架构

```
客户端 / Agent
  ↓ HTTP
┌────────────────────────────────────────────────────────────────┐
│  FastAPI + bioq-service-framework  (port 9000)             │
│                                                                │
│  服务专属（rfantibody-server 注册）                            │
│    POST /api/rfdiffusion              (单步：骨架设计)         │
│    POST /api/proteinmpnn              (单步：序列设计)         │
│    POST /api/rf2                      (单步：结构预测)         │
│                                                                │
│  框架统一提供                                                  │
│    GET  /api/manifest                 (Agent 协议描述)         │
│    GET  /openapi.json                 (字段 schema)            │
│    GET  /healthz, /healthz/detail                              │
│    GET  /api/jobs/{id}                (JobInfo 含 error 详情)  │
│    GET  /api/jobs/{id}/files / log / download / file/{path}    │
│    DELETE /api/jobs/{id}                                       │
└────────────────────────────────────────────────────────────────┘
  ↓ subprocess (单卡串行)
RFantibody CLI: rfdiffusion_inference.py / proteinmpnn_interface_design.py / rf2_predict.py
  ↓ 输出
NAS: <jobs_base_dir>/<job_id>/{input/, output/{1,2,3}_*.qv, logs/run.log, job.json}
```

**没有 `/api/pipeline`**。三步组合（v0.1 的 `/api/pipeline`）已剥离到客户端 / Agent 编排层 ——
借助共享 NAS，把上一步的 `job://<id>/<file>` 直接作为下一步 `input_uri` 即可，无需重新上传 .qv。

## API 速览

每个 POST 端点接受 `multipart/form-data`：文件字段 + pydantic 参数（`Annotated[..., Form()]` 自动校验）。
返回 `JobInfo`（含 `job_id`），实际计算在后台异步执行。

### Step 1 — RFdiffusion

```bash
curl -X POST http://localhost:9000/api/rfdiffusion \
  -F "target=@antigen.pdb" \
  -F "framework=@framework.pdb" \
  -F "num_designs=100" \
  -F "design_loops=H1:7,H2:6,H3:5-13" \
  -F "hotspots=B146,B170,B177"
```

输出：`output/1_rfdiffusion.qv`

### Step 2 — ProteinMPNN

可上传 Quiver 文件，**也可** 用 `input_uri` 直接引用上一步的 NAS 路径（推荐 —— 零拷贝）：

```bash
# 方式 A: 上传
curl -X POST http://localhost:9000/api/proteinmpnn \
  -F "input_quiver=@1_rfdiffusion.qv" \
  -F "seqs_per_struct=4"

# 方式 B: NAS 共享路径 (推荐)
curl -X POST http://localhost:9000/api/proteinmpnn \
  -F "input_uri=job://<rfdiffusion_job_id>/1_rfdiffusion.qv" \
  -F "seqs_per_struct=4"
```

输出：`output/2_proteinmpnn.qv`

### Step 3 — RF2

```bash
curl -X POST http://localhost:9000/api/rf2 \
  -F "input_uri=job://<proteinmpnn_job_id>/2_proteinmpnn.qv" \
  -F "num_recycles=10"
```

输出：`output/3_rf2.qv`

### 支持的 `input_uri` schemes

| 形式 | 说明 |
|---|---|
| `job://<job_id>/<filename>` | 从本服务的另一个 job 的 `output/` 拉文件（**推荐用于链式调用**） |
| `file:///abs/path` 或 `/abs/path` | NAS 上的绝对路径（同一挂载下跨 service 共享） |
| `oss://<bucket>/<key>` | 阿里云 OSS 对象（需 `OSS_ACCESS_KEY_ID` / `OSS_ACCESS_KEY_SECRET` 环境变量） |
| `http(s)://...` | 任意 HTTP(S) URL，包括 OSS 签名 URL |

### Agent 友好接口

```bash
# 一次拿到协议描述：endpoints / job 生命周期 / NAS 布局 / 链式调用提示
curl http://localhost:9000/api/manifest

# 详细字段 schema
curl http://localhost:9000/openapi.json
```

`/api/manifest` 的 `service_specific` 段包含：
- `tool_outputs` —— 三个工具的输出文件名（agent 用它构造下游 `job://<id>/<file>`）
- `input_uri_schemes` —— 上面那张表的运行时版本
- `chaining_tip` —— 一句话教 agent 串联三步
- `weights` —— 权重清单

### 任务管理（框架提供）

```bash
# 查询状态（含 status / progress / error_summary / error_tail / failure_kind）
curl http://localhost:9000/api/jobs/{job_id}

# 列出 output/ 下所有文件
curl http://localhost:9000/api/jobs/{job_id}/files

# 完整 subprocess 日志
curl http://localhost:9000/api/jobs/{job_id}/log

# 打包 zip 下载
curl -O http://localhost:9000/api/jobs/{job_id}/download

# 下载单文件（防 path traversal）
curl -O http://localhost:9000/api/jobs/{job_id}/file/3_rf2.qv

# 删除任务（同步清理 NAS 目录）
curl -X DELETE http://localhost:9000/api/jobs/{job_id}
```

### 失败响应

job 失败时 `JobInfo` 自动填充：

| 字段 | 含义 |
|---|---|
| `status` | `"failed"` |
| `failure_kind` | `subprocess_error` (rc ≠ 0) / `no_outputs` (rc = 0 但无输出) / `interrupted` (被重启打断) |
| `error_summary` | 从 log 中提取的最后一行异常（如 `ValueError: bad input`） |
| `error_tail` | log 文件尾部 ~4 KB |

不必单独调 `/log` 即可获得足够信息做诊断。

## 配置

服务通过 `pydantic_settings.BaseSettings` 读环境变量，env_prefix=`RFANTIBODY_`：

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `RFANTIBODY_JOBS_BASE_DIR` | `/data/rfantibody_jobs` | NAS 上的 job 根目录；多实例共享时所有实例指向同一路径 |
| `RFANTIBODY_ROOT` | `/opt/rfantibody` | RFantibody 源码根（subprocess cwd） |
| `RFANTIBODY_WEIGHTS_DIR` | `/data/models/rfantibody/weights` | 权重目录（v0.0.17 起 NAS 挂载，~1.6 GB 外置）|
| `RFANTIBODY_SCRIPTS_DIR` | `/opt/rfantibody/scripts` | 三个工具的 entry point 脚本 |
| `RFANTIBODY_PORT` | `9000` | uvicorn 监听端口（FC CAPort） |
| `RFANTIBODY_KEEP_ALIVE_SEC` | `900` | uvicorn `--timeout-keep-alive` |
| `RFANTIBODY_MAX_CONCURRENT_JOBS` | `1` | 单实例并发 job 数（单卡建议保持 1） |
| `RFANTIBODY_DISK_LIMIT_MB` | `8000` | 触发自动清理已完成 job 的阈值 |
| `RFANTIBODY_ERROR_TAIL_CHARS` | `4000` | JobInfo.error_tail 字节数 |
| `RFANTIBODY_OSS_REGION` | `cn-hangzhou` | OSS URI 下载用的区域 |

OSS 凭证按阿里云 SDK 约定走 `OSS_ACCESS_KEY_ID` + `OSS_ACCESS_KEY_SECRET`，**不**走 `RFANTIBODY_` 前缀。

## 持久化与多实例 (框架能力)

- **重启恢复**：每个 `create` / `update` 同步写 `<job_id>/job.json` sidecar；服务启动时扫盘恢复。
  重启时仍在 RUNNING 的 job 自动降级为 FAILED + `failure_kind=interrupted`。
- **跨实例**：多个 FC 实例挂同一块 NAS 时，`GET /api/jobs/{id}` 在本地无缓存时回查 sidecar
  （read-through + mtime 检查），所以 client 提交到实例 A、轮询路由到实例 B 也能拿到 job。
- **跨服务文件共享**：本服务的 output 文件路径（`<jobs_base_dir>/<id>/output/*.qv`）可被同 NAS 上
  其他 bioagent service 直接 `Path(...).read_bytes()` 读取，无需走 HTTP。

详见 [engineering/decisions/2026-05-12-service-framework-design.md](../../engineering/decisions/2026-05-12-service-framework-design.md)。

## 本地开发

```bash
# 1. 准备 RFantibody（在 bioagent 项目根目录）
cd opensource/RFantibody
uv sync

# 2. 安装服务框架
uv pip install ../../framework

# 3. 软链 server 包，让 uvicorn 通过 `server.app:app` 找到
ln -s ../../services/rfantibody-server server
export RFANTIBODY_ROOT=$(pwd)
export RFANTIBODY_JOBS_BASE_DIR=/tmp/rfantibody_jobs

# 4. 启动
uv run uvicorn server.app:app --host 0.0.0.0 --port 9000 --reload
```

## Weights

v0.0.17 起权重（~1.6 GB）**不再 baked 到镜像**，从 NAS 加载。期望布局：

```
/data/models/rfantibody/
└── weights/
    ├── RFdiffusion_Ab.pt    ← 抗体专用 RFdiffusion 检查点
    ├── ...
```

### 下载（一次性）

```bash
# Stage 到本地
./services/rfantibody-server/scripts/fetch_weights.sh    # → ./weights/

# 上传到 NAS
rsync -av services/rfantibody-server/weights/ \
    <NAS-mount>:/data/models/rfantibody/weights/
```

或直接下到 NAS（如果本地能挂）：

```bash
WEIGHTS_DST=/mnt/nas/data/models/rfantibody/weights \
    ./services/rfantibody-server/scripts/fetch_weights.sh
```

### FC

NAS 自动挂载到 `/data/models/rfantibody/`。验证：

```bash
curl https://fc-rfantibody-guekbpucdo.cn-hangzhou-vpc.fcapp.run/healthz/detail
# 期望：{"status":"ok","weights_loaded":true,"files_found":N}
```

### SIF / HPC

```bash
apptainer run --nv \
    --bind /scratch/models/rfantibody:/data/models/rfantibody \
    rfantibody-server.sif python -m server design ...
```

详见 [weights externalization design](../../engineering/decisions/2026-06-26-service-weights-externalization.md)。

## Docker 构建与运行

```bash
# 1. Vendor 上游源（一次性，重跑可升级 SHA）
./services/rfantibody-server/scripts/vendor.sh

# 2. 构建（CUDA 12.1 base + cu121 torch/dgl，~3.5 GB 镜像）
# 支持 sm_89 (RTX 4090) 及以下；sm_120 (RTX 5090 / Blackwell) 暂不支持
docker build --platform linux/amd64 -t rfantibody-server -f services/rfantibody-server/Dockerfile .

# 或通过 Makefile
make build-rfantibody-server

# 本地运行（需 GPU + NAS / 本地 --bind 注入权重）
docker run --gpus all -p 9000:9000 --memory 16g \
    -v $(pwd)/weights:/data/models/rfantibody/weights \
    rfantibody-server
```

构建上下文必须是项目根目录，因为 Dockerfile 同时 `COPY framework` 安装框架包。

## 阿里云函数计算部署

### 前置条件

- ACR 个人版或企业版（非经济版），与函数计算同地域、同账号
- GPU 实例可用地域：华东 1/2、华北 2/3、华南 1、日本、美国

### 部署步骤

1. 构建并推送镜像：

   ```bash
   docker login --username=<阿里云账号> registry.cn-hangzhou.aliyuncs.com
   docker build --platform linux/amd64 -t rfantibody-server -f services/rfantibody-server/Dockerfile .
   docker tag rfantibody-server registry.cn-hangzhou.aliyuncs.com/<namespace>/rfantibody-server:latest
   docker push registry.cn-hangzhou.aliyuncs.com/<namespace>/rfantibody-server:latest
   ```

2. 创建函数：

   | 配置项 | 推荐值 |
   |---|---|
   | 运行时 | 自定义容器镜像 |
   | 镜像地址 | ACR VPC 地址（加速拉取） |
   | GPU 配置 | `fc.gpu.tesla.1` 起步（8 GB 显存） |
   | 监听端口 CAPort | `9000` |
   | 函数超时 | ≥ 600 秒（单步 RF2 可能数百秒） |
   | 内存 | `16384` MB |
   | CPU | `4` vCPU |
   | 磁盘 | `10240` MB |
   | NAS 挂载 | 推荐挂载到 `/data`，跨实例 / 跨服务共享 job 文件 |

3. HTTP 触发器，绑定自定义域名（解除 `content-disposition: attachment` 限制）

### FC 平台限制与适配

| 限制项 | 值 | 适配 |
|---|---|---|
| 启动超时 | 120 s | `/healthz` 不加载模型，立即响应 |
| Keep-alive | ≥ 15 min | `RFANTIBODY_KEEP_ALIVE_SEC=900` |
| 监听地址 | `0.0.0.0:CAPort` | uvicorn `--host 0.0.0.0 --port 9000` |
| 可写磁盘 | 512 MB ~ 10 GB | 自动清理已完成 job + 推荐挂载 NAS |
| GPU 镜像大小 | ≤ 15 GB | 模型权重烘焙到镜像 |
| 镜像架构 | AMD64 only | `--platform linux/amd64` |

## 设计要点

- **单功能 API**：每个端点一次只跑一个工具。Pipeline 编排留给上层（[Tool 抽象层设计](../../engineering/decisions/2026-04-23-tool-abstraction-design.md) 的 HTTPRunner / Agent）。
- **共享 NAS 链式调用**：相邻步骤通过 `job://<id>/<file>` 引用上一步输出，避免 Quiver 文件
  在 HTTP 上下行（10–100 MB 量级的 zero-copy）。
- **错误信息丰富**：失败 job 的 `JobInfo` 自带 `error_summary` + `error_tail`，无需额外 `/log`
  调用即可判定原因（如 weight 缺失、CUDA OOM、参数错误）。
- **多实例 FC 友好**：sidecar 持久化 + read-through cache，submit/poll 落到不同实例都 OK。
- **快速启动**：`/healthz` 端点无模型加载，FC 120s 启动探测通过；权重烘焙到镜像避免冷启动下载。

## 相关文档

- [bioq-service-framework](../_framework/README.md) — 通用 HTTP / job / 错误处理 / manifest 层
- [Service 框架抽象设计](../../engineering/decisions/2026-05-12-service-framework-design.md) — 设计决策
- [Tool 抽象层设计](../../engineering/decisions/2026-04-23-tool-abstraction-design.md) — Client 端 Tool + Runner（消费本 service 的 manifest）
