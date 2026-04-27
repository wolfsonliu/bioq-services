# RFantibody Web Server

基于 FastAPI 的 RFantibody HTTP 服务，用于阿里云函数计算（FC）GPU 实例部署。

## 架构

```
客户端
  ↓ HTTP
┌──────────────────────────────────────┐
│  FastAPI Server (port 9000)          │
│                                      │
│  POST /api/rfdiffusion    (Step 1)   │
│  POST /api/proteinmpnn    (Step 2)   │
│  POST /api/rf2            (Step 3)   │
│  POST /api/pipeline       (全流程)    │
│                                      │
│  GET  /api/jobs/{id}      (查询状态)  │
│  GET  /api/jobs/{id}/download (下载)  │
└──────────────────────────────────────┘
  ↓ subprocess
┌──────────────────────────────────────┐
│  RFantibody CLI (GPU)                │
│  rfdiffusion → proteinmpnn → rf2     │
└──────────────────────────────────────┘
```

## API 接口

### 健康检查

```
GET /health
```

### 单步运行

每个端点接受 PDB 文件上传和参数，返回 `job_id`，后台异步执行。

**Step 1 — RFdiffusion（骨架设计）**

```bash
curl -X POST http://localhost:9000/api/rfdiffusion \
  -F "target=@antigen.pdb" \
  -F "framework=@framework.pdb" \
  -F "num_designs=100" \
  -F "design_loops=H1:7,H2:6,H3:5-13" \
  -F "hotspots=B146,B170,B177"
```

**Step 2 — ProteinMPNN（序列设计）**

```bash
curl -X POST http://localhost:9000/api/proteinmpnn \
  -F "input_quiver=@1_rfdiffusion.qv" \
  -F "seqs_per_struct=4" \
  -F "temperature=0.2"
```

**Step 3 — RF2（结构预测）**

```bash
curl -X POST http://localhost:9000/api/rf2 \
  -F "input_quiver=@2_proteinmpnn.qv" \
  -F "num_recycles=10"
```

### 全流程

```bash
curl -X POST http://localhost:9000/api/pipeline \
  -F "target=@antigen.pdb" \
  -F "framework=@framework.pdb" \
  -F 'config={"rfdiffusion":{"num_designs":100,"design_loops":"H1:7,H2:6,H3:5-13","hotspots":"B146,B170,B177"},"proteinmpnn":{"seqs_per_struct":4},"rf2":{"num_recycles":10}}'
```

### 任务管理

```bash
# 查询状态
curl http://localhost:9000/api/jobs/{job_id}

# 列出输出文件
curl http://localhost:9000/api/jobs/{job_id}/files

# 下载全部结果（zip）
curl -O http://localhost:9000/api/jobs/{job_id}/download

# 下载单个文件
curl -O http://localhost:9000/api/jobs/{job_id}/file/3_rf2.qv

# 删除任务
curl -X DELETE http://localhost:9000/api/jobs/{job_id}
```

## 本地开发

```bash
# 在 RFantibody 项目根目录（需要已安装 RFantibody 依赖）
cd opensource/RFantibody

# 安装依赖
uv sync
uv pip install fastapi uvicorn python-multipart

# 将 server 代码软链接到 RFantibody 目录
ln -s ../../services/rfantibody-server server

# 设置环境变量
export RFANTIBODY_ROOT=$(pwd)

# 启动服务
uv run uvicorn server.app:app --host 0.0.0.0 --port 9000 --reload
```

## Docker 构建与运行

```bash
# 在 bioagent 项目根目录构建（需要 opensource/RFantibody 已存在）
docker build --platform linux/amd64 -t rfantibody-server -f services/rfantibody-server/Dockerfile .

# 运行（需要 GPU）
docker run --gpus all -p 9000:9000 --memory 10g rfantibody-server
```

## 阿里云函数计算部署

### 前置条件

- 阿里云容器镜像服务（ACR）个人版或企业版（非经济版）
- ACR 实例与函数计算在同一地域、同一账号
- GPU 实例可用地域：华东1/2、华北2/3、华南1、日本、美国

### 部署步骤

1. **构建并推送镜像**到 ACR（同地域）：
   ```bash
   # 登录 ACR
   docker login --username=<阿里云账号> registry.cn-hangzhou.aliyuncs.com

   # 在 bioagent 项目根目录构建（必须 linux/amd64）
   docker build --platform linux/amd64 -t rfantibody-server -f services/rfantibody-server/Dockerfile .

   # 推送
   docker tag rfantibody-server registry.cn-hangzhou.aliyuncs.com/<namespace>/rfantibody-server:latest
   docker push registry.cn-hangzhou.aliyuncs.com/<namespace>/rfantibody-server:latest
   ```

2. **创建函数**：
   - 运行时：自定义容器镜像
   - 镜像地址：从 ACR 选择（推荐使用 VPC 地址以加速拉取）
   - GPU 配置：GPU 实例（如 `fc.gpu.tesla.1`，8GB 显存）
   - 监听端口（CAPort）：`9000`
   - 函数超时：`600` 秒以上（pipeline 完整运行可能需要数分钟）
   - 内存：`16384` MB (16GB)
   - CPU：`4` vCPU
   - 磁盘大小：`10240` MB（10GB，默认 512MB 不够用）

3. **配置触发器**：HTTP 触发器，绑定自定义域名（解除 `content-disposition: attachment` 限制）

### FC 平台限制与适配

| 限制项 | 值 | 适配措施 |
|--------|-----|---------|
| 启动超时 | 120s | `/health` 端点不加载模型，确保快速响应 |
| Keep-alive | ≥15 分钟 | uvicorn `--timeout-keep-alive 900` |
| 监听地址 | `0.0.0.0:CAPort` | 已配置，不可用 `127.0.0.1` |
| 可写磁盘 | 512MB ~ 10GB | 自动清理已完成任务；建议配置 10GB |
| GPU 镜像大小 | ≤15GB 未压缩 | 模型权重烘焙到镜像中 |
| 镜像架构 | AMD64 only | Dockerfile 指定 `--platform linux/amd64` |
| Response Header | ≤8KB | API 响应 header 较小，无需特殊处理 |

## 设计说明

- **异步执行**：GPU 任务耗时较长，所有计算端点异步提交后立即返回 `job_id`，客户端轮询 `/api/jobs/{id}` 获取进度
- **Quiver 格式**：中间步骤使用 RFantibody 原生的 Quiver 文件传递，减少格式转换
- **单线程 GPU**：`ThreadPoolExecutor(max_workers=1)` 确保 GPU 任务串行执行，避免显存竞争
- **磁盘管理**：任务文件存储在 `/tmp/rfantibody_jobs/`，当磁盘使用超过阈值时自动清理已完成任务，也可通过 `DELETE /api/jobs/{id}` 手动清理
- **快速启动**：`/health` 端点无重量级依赖，确保 FC 120s 启动探测通过；模型权重烘焙在镜像中避免冷启动下载
