# Genie3 Server

基于 FastAPI 的 Genie3 HTTP 服务，仅暴露 `genie3 generate` 功能（不含评估流程）。

镜像中只打包扩散模型生成所需的依赖（PyTorch + Genie3 包），**不包含** ColabFold / ESMFold /
FoldSeek / ProteinMPNN / TMscore / DSSP / IPSAE 等评估工具，因此镜像比官方 setup.sh
安装的环境小得多，适合阿里云函数计算（FC）GPU 实例部署。

## 架构

```
客户端
  ↓ HTTP
┌──────────────────────────────────────┐
│  FastAPI Server (port 9000)          │
│                                      │
│  POST /api/generate/unconditional    │
│  POST /api/generate/motif            │
│  POST /api/generate/binder           │
│  POST /api/generate    (自定义 yaml)  │
│                                      │
│  GET  /api/jobs/{id}      (查询状态)  │
│  GET  /api/jobs/{id}/files (文件列表) │
│  GET  /api/jobs/{id}/log  (运行日志)  │
│  GET  /api/jobs/{id}/download (下载)  │
└──────────────────────────────────────┘
  ↓ subprocess
┌──────────────────────────────────────┐
│  genie3 generate -c <yaml>           │
│  (pretrained v1, GPU)                │
└──────────────────────────────────────┘
```

## API 接口

### 健康检查

```
GET /health
GET /health/detail
```

### 无条件生成（unconditional）

无需上传任何文件，直接传参生成蛋白主链。

```bash
curl -X POST http://localhost:9000/api/generate/unconditional \
  -F "min_length=100" \
  -F "max_length=200" \
  -F "length_step=50" \
  -F "n_sample=8" \
  -F "direction_scale=0.8"   # 长度<=300 用 0.8，>300 用 0.0
```

### Motif 支架（motif scaffolding）

需上传一个 zip 包，结构如下（zip 顶层目录可有可无）：

```
binderbench/
  problems/
    01_*.json
    ...
  motifs/
    01_*.pdb
    ...
```

`problems/*.json` 中的 `motif_filepaths` 等路径会被服务器自动改写为绝对路径。

```bash
curl -X POST http://localhost:9000/api/generate/motif \
  -F "dataset=@motifbench.zip" \
  -F "selections=22_1BCF" \
  -F "n_sample=8" \
  -F "direction_scale=0.1"
```

### Binder 设计

需上传一个 zip 包，结构如下：

```
binderbench/
  problems/
    01_bhrf1.json
    ...
  targets/
    pdb/
    fasta/
    msa/
```

`problems/*.json` 中的 `target_pdb_filepath`、`target_fasta_filepath`、`target_msa_filepath`
等路径会被服务器自动改写为绝对路径，因此原 zip 中只要保持 `targets/...` 子目录结构即可。

```bash
curl -X POST http://localhost:9000/api/generate/binder \
  -F "dataset=@binderbench.zip" \
  -F "selections=01_bhrf1" \
  -F "n_sample=8" \
  -F "direction_scale=0.0"
```

### 自定义配置

完全自定义 YAML，附带可选的数据集 zip。`paths.rootdir`（以及 `paths.dataset` 如果上传了
数据集）会被服务器覆写为 job 本地目录。

```bash
curl -X POST http://localhost:9000/api/generate \
  -F 'config_yaml=experiment:
  name: custom
generation:
  dataset:
    source: target
    n_sample: 4
  sampler:
    sampler:
      direction_scale: 0.0' \
  -F "dataset=@my_dataset.zip"
```

### 任务管理

```bash
# 查询状态
curl http://localhost:9000/api/jobs/{job_id}

# 列出输出文件
curl http://localhost:9000/api/jobs/{job_id}/files

# 查看运行日志
curl http://localhost:9000/api/jobs/{job_id}/log

# 下载所有结果（zip）
curl -O http://localhost:9000/api/jobs/{job_id}/download

# 下载单个文件（支持子路径，例如 motif/binder 任务的 <problem>/pdbs/<name>.pdb）
curl -O http://localhost:9000/api/jobs/{job_id}/file/01_bhrf1/pdbs/sample_0.pdb

# 删除任务
curl -X DELETE http://localhost:9000/api/jobs/{job_id}
```

## 输出结构

`genie3 generate` 写入的目录结构：

- 无条件：`<output>/pdbs/<name>.pdb`
- Motif / binder：`<output>/<problem>/pdbs/<name>.pdb`

服务器把 `paths.rootdir` 设为 `<job_dir>/output/`，因此 `/api/jobs/{id}/files` 返回的
文件路径都相对于该目录。

## 本地开发

```bash
# 1. 在 bioagent 项目根目录创建 venv（若尚未创建）
uv venv .venv-genie3 --python 3.10
source .venv-genie3/bin/activate

# 2. 安装 genie3 与服务依赖（最小集合，不含评估工具）
pip install --upgrade pip setuptools wheel "numpy>=2.0.2,<3" Cython
pip install torch==2.7.1 tqdm scipy pandas lightning "biopython<1.86" \
            ml-collections zstandard huggingface_hub wandb tensorboard PyYAML
pip install --no-build-isolation -e ./opensource/genie3
pip install fastapi "uvicorn[standard]" python-multipart httpx

# 3. 设置环境变量
export GENIE3_ROOT=$(pwd)/opensource/genie3
export GENIE3_JOBS_DIR=/tmp/genie3_jobs

# 4. 软链 server 包到 genie3 目录（让 uvicorn 通过 server.app:app 找到）
ln -sf $(pwd)/services/genie3-server $GENIE3_ROOT/server

# 5. 启动
cd $GENIE3_ROOT
uvicorn server.app:app --host 0.0.0.0 --port 9000 --reload
```

## Docker 构建与运行

```bash
# 项目根目录构建（要求 opensource/genie3 已就绪 + pretrained/v1 权重已下载）
docker build --platform linux/amd64 -t genie3-server -f services/genie3-server/Dockerfile .

# 或通过 Makefile（自动发现 services/*/Dockerfile）
make build-genie3-server

# 运行（需要 GPU）
docker run --gpus all -p 9000:9000 --memory 16g genie3-server
```

## 阿里云函数计算部署

### 前置条件

- 阿里云容器镜像服务（ACR）个人版或企业版（非经济版）
- ACR 实例与函数计算同地域、同账号
- GPU 实例可用地域：华东1/2、华北2/3、华南1、日本、美国

### 部署步骤

1. **构建并推送镜像**到 ACR：

   ```bash
   docker login --username=<阿里云账号> registry.cn-hangzhou.aliyuncs.com

   docker build --platform linux/amd64 -t genie3-server -f services/genie3-server/Dockerfile .

   docker tag genie3-server registry.cn-hangzhou.aliyuncs.com/<namespace>/genie3-server:latest
   docker push registry.cn-hangzhou.aliyuncs.com/<namespace>/genie3-server:latest
   ```

2. **创建函数**：
   - 运行时：自定义容器镜像
   - 镜像地址：从 ACR 选择（推荐 VPC 地址）
   - GPU 配置：`fc.gpu.tesla.1` 起步（8GB 显存可生成 ≤512 残基的单体 / 中等长度 binder）
   - 监听端口（CAPort）：`9000`
   - 函数超时：`3600` 秒以上（生成多个长 binder 可能需要较久）
   - 内存：`16384` MB
   - 磁盘：`10240` MB（512MB 默认不够用）

3. **触发器**：HTTP 触发器，绑定自定义域名（解除 `content-disposition: attachment` 限制）。

### FC 平台限制与适配

| 限制项 | 值 | 适配措施 |
|--------|-----|---------|
| 启动超时 | 120s | `/health` 端点不加载模型 |
| Keep-alive | ≥15 分钟 | uvicorn `--timeout-keep-alive 900` |
| 监听地址 | `0.0.0.0:CAPort` | 已配置 |
| 可写磁盘 | 512MB ~ 10GB | 自动清理已完成任务；建议 10GB |
| GPU 镜像大小 | ≤15GB 未压缩 | 已剥离评估依赖；模型权重烘焙到镜像 |
| 镜像架构 | AMD64 only | Dockerfile 指定 `--platform linux/amd64` |

## 设计说明

- **仅生成**：本服务只跑 `genie3 generate`，**不**做 inverse fold / structure prediction /
  filtering。Genie3 自身的 `evaluate` 子命令需要 ColabFold/ESMFold/FoldSeek 等重依赖，
  打包到 FC GPU 镜像会超出 15GB 限制；如需评估，建议在另一个评估 server / 本地集群完成。
- **异步执行**：所有计算端点提交后立即返回 `job_id`，客户端轮询 `/api/jobs/{id}`。
- **单线程 GPU**：`ThreadPoolExecutor(max_workers=1)` 避免显存竞争。
- **磁盘管理**：任务文件存储在 `${GENIE3_JOBS_DIR:-/data/genie3_jobs}`，磁盘超阈值时
  自动清理已完成任务，也可 `DELETE /api/jobs/{id}` 手动清理。
- **路径改写**：上传的 binder/motif 数据集 zip 内 `problems/*.json` 含相对路径，服务器
  自动把它们重写为绝对路径，确保无论 zip 顶层目录命名如何都能正确解析。
- **自定义 cwd**：subprocess 执行 `genie3 generate` 时 cwd 设为 `/opt/genie3`，确保默认
  的 `pretrained/v1/checkpoints/step=600000.ckpt` 相对路径能解析到镜像中烘焙的权重。
