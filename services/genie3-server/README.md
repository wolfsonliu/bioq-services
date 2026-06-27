# Genie3 Server

基于 FastAPI 的 Genie3 HTTP 服务，仅暴露 `genie3 generate` 功能（不含评估流程）。
**v0.2 起构建在 [bioagent-service-framework](../_framework/) 之上**：HTTP / job / 错误 / 持久化
/ 多实例一致性 / Agent 协议描述由框架统一提供，本服务只负责数据集 zip 处理 + 配置 YAML 拼装 +
`genie3 generate` argv 构造。

镜像中只打包扩散模型生成所需的依赖（PyTorch + Genie3 包），**不包含** ColabFold / ESMFold /
FoldSeek / ProteinMPNN / TMscore / DSSP / IPSAE 等评估工具 —— 比官方 setup.sh 安装的环境小得多，
适合 FC GPU 镜像 15 GB 上限。

## 架构

```
客户端 / Agent
  ↓ HTTP
┌────────────────────────────────────────────────────────────────┐
│  FastAPI + bioagent-service-framework  (port 9000)             │
│                                                                │
│  服务专属（genie3-server 注册）                                │
│    POST /api/generate/unconditional                            │
│    POST /api/generate/motif         (含 dataset zip)           │
│    POST /api/generate/binder        (含 dataset zip)           │
│    POST /api/generate               (freeform YAML)            │
│                                                                │
│  框架统一提供                                                  │
│    GET  /api/manifest               (Agent 协议描述)           │
│    GET  /openapi.json               (字段 schema)              │
│    GET  /healthz, /healthz/detail                              │
│    GET  /api/jobs/{id}              (JobInfo 含 error 详情)    │
│    GET  /api/jobs/{id}/files / log / download / file/{path}    │
│    DELETE /api/jobs/{id}                                       │
└────────────────────────────────────────────────────────────────┘
  ↓ subprocess (cwd=/opt/genie3, 单卡串行)
genie3 generate -c <yaml>
  ↓ 输出
NAS: <jobs_base_dir>/<job_id>/{input/{experiment.yaml,dataset/}, output/<exp>/pdbs/*.pdb, logs/run.log, job.json}
```

## API 速览

### 无条件生成（unconditional）

不需要上传文件，直接传参生成蛋白主链。

```bash
curl -X POST http://localhost:9000/api/generate/unconditional \
  -F "min_length=100" \
  -F "max_length=200" \
  -F "length_step=50" \
  -F "n_sample=8" \
  -F "direction_scale=0.8"   # 长度 ≤ 300 用 0.8，> 300 用 0.0
```

输出：`output/unconditional/pdbs/*.pdb`

### Motif 支架生成

上传 zip，结构如下（zip 顶层目录可有可无）：

```
binderbench/             ← 顶层目录命名随意
  problems/
    01_*.json
  motifs/
    01_*.pdb
```

`problems/*.json` 中的 `motif_filepaths` 等路径会被服务器自动改写为绝对路径
（包括 `binder_framework` 指向的嵌套 motif config）。

```bash
curl -X POST http://localhost:9000/api/generate/motif \
  -F "dataset=@motifbench.zip" \
  -F "selections=22_1BCF" \
  -F "n_sample=8" \
  -F "direction_scale=0.1"
```

输出：`output/motif/<problem>/pdbs/*.pdb`

### Binder 设计

上传 zip，结构如下：

```
binderbench/
  problems/
    01_bhrf1.json
  targets/
    pdb/   01_bhrf1.pdb
    fasta/ 01_bhrf1.fasta
    msa/   01_bhrf1.a3m
```

`problems/*.json` 中的 `target_pdb_filepath` / `target_fasta_filepath` / `target_msa_filepath`
和 `target_*_filepath_by_chain` 等路径会被服务器自动改写为绝对路径。

```bash
curl -X POST http://localhost:9000/api/generate/binder \
  -F "dataset=@binderbench.zip" \
  -F "selections=01_bhrf1" \
  -F "n_sample=8" \
  -F "direction_scale=0.0"
```

输出：`output/binder/<problem>/pdbs/*.pdb`

### 自定义 YAML（高级）

完全自定义 YAML，附带可选的数据集 zip。`paths.rootdir`（以及 `paths.dataset` 如果上传了
数据集）会被服务器自动覆写为 job 本地路径，所以客户端不需要关心 job_dir。

```bash
curl -X POST http://localhost:9000/api/generate \
  -F 'config_yaml=experiment:
  name: trem1_vhh
generation:
  dataset:
    source: target
    selections: trem1_vhh
    n_sample: 4
    cond_strategy: hotspot     # ← 常见坑：默认 extended 找不到 interface 时改这个
  sampler:
    sampler:
      direction_scale: 0.0' \
  -F "dataset=@trem1.zip" \
  -F "num_devices=1"
```

**`cond_strategy` 坑**：genie3 默认 `cond_strategy: extended`，要求 problem JSON 提供
`extended` interface。如果你的 problem JSON 只定义了 `hotspot`，会直接抛
`ValueError: Interface mode 'extended' not found`。这种情况下覆写
`generation.dataset.cond_strategy: hotspot` 即可。

### Agent 友好接口

```bash
# 一次拿到协议描述：4 个 endpoint 摘要 + job 生命周期 + NAS 布局 + config_tips（含 cond_strategy 坑）
curl http://localhost:9000/api/manifest

# 详细 schema
curl http://localhost:9000/openapi.json
```

`/api/manifest` 的 `service_specific` 段包含：
- `tool_outputs` —— 所有模式都写到 `output/<experiment_name>/pdbs/*.pdb`，agent 用 `/api/jobs/{id}/files` 列文件
- `input_uri_schemes` —— 三种 dataset 形态的说明
- `endpoints_summary` —— 4 个端点的一句话用法
- `config_tips` —— `cond_strategy` 与 `direction_scale` 调参建议

### 任务管理（框架提供）

```bash
# 查询状态（含 error_summary / error_tail / failure_kind）
curl http://localhost:9000/api/jobs/{job_id}

# 列出 output/ 下所有文件（递归）
curl http://localhost:9000/api/jobs/{job_id}/files

# 完整 subprocess 日志
curl http://localhost:9000/api/jobs/{job_id}/log

# 打包 zip 下载
curl -O http://localhost:9000/api/jobs/{job_id}/download

# 下载单个 PDB（支持子路径）
curl -O http://localhost:9000/api/jobs/{job_id}/file/binder/01_bhrf1/pdbs/sample_0.pdb

# 删除任务（清理 NAS 目录）
curl -X DELETE http://localhost:9000/api/jobs/{job_id}
```

### 失败响应

job 失败时 `JobInfo` 自动填充：

| 字段 | 含义 |
|---|---|
| `status` | `"failed"` |
| `failure_kind` | `subprocess_error` (rc ≠ 0) / `no_outputs` (rc = 0 但无 .pdb) / `interrupted` (重启打断) / `dataset_invalid` (坏 zip) |
| `error_summary` | 从 log 提取的最后一行异常（如 `ValueError: Interface mode 'extended' not found...`） |
| `error_tail` | log 尾部 ~4 KB |

## 配置

服务通过 `pydantic_settings.BaseSettings` 读环境变量，env_prefix=`GENIE3_`：

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `GENIE3_JOBS_BASE_DIR` | `/data/genie3_jobs` | NAS 上的 job 根目录 |
| `GENIE3_ROOT` | `/opt/genie3` | genie3 源码根（subprocess cwd，让 `pretrained/v1/...` 相对路径解析） |
| `GENIE3_BIN` | `genie3` | CLI 入口（`pip install -e .` 后注册的命令） |
| `GENIE3_PRETRAINED_DIR` | `/data/models/genie3/pretrained/v1` | 预训练权重目录（v0.0.17 起 NAS 挂载，~512 MB 外置）|
| `GENIE3_PORT` | `9000` | uvicorn 端口（FC CAPort） |
| `GENIE3_KEEP_ALIVE_SEC` | `900` | uvicorn `--timeout-keep-alive` |
| `GENIE3_MAX_CONCURRENT_JOBS` | `1` | 单实例并发（单卡保持 1） |
| `GENIE3_DISK_LIMIT_MB` | `8000` | 自动清理已完成 job 的阈值 |
| `GENIE3_ERROR_TAIL_CHARS` | `4000` | `JobInfo.error_tail` 字节数 |
| `GENIE3_OSS_REGION` | `cn-hangzhou` | OSS URI 下载用的区域（保留接口，当前 4 个 endpoint 暂未使用） |

OSS 凭证按阿里云 SDK 约定走 `OSS_ACCESS_KEY_ID` + `OSS_ACCESS_KEY_SECRET`。

## 持久化与多实例（框架能力）

- **重启恢复**：`<job_id>/job.json` sidecar 持久化；启动时扫盘恢复，
  RUNNING 状态降级为 FAILED + `failure_kind=interrupted`。
- **跨实例 NAS 一致性**：多个 FC 实例共享 `GENIE3_JOBS_BASE_DIR` 时，`GET /api/jobs/{id}`
  走 read-through cache（stat sidecar mtime → 命中本地缓存 / 不命中或陈旧时回读 NAS）。
  client submit 到实例 A、poll 路由到实例 B 都能拿到 job。
- **跨服务文件共享**：本服务产物（`<jobs_base_dir>/<id>/output/...`）可被同 NAS 上其他
  bioagent service 直接 `Path(...).read_bytes()` 读取。

详见 [engineering/decisions/2026-05-12-service-framework-design.md](../../engineering/decisions/2026-05-12-service-framework-design.md)。

## 上游 patch

[`patches/`](patches/) 目录下保留对 `opensource/genie3/` 的本地修复（Dockerfile 构建时按文件名顺序 apply）：

- `0001-fix-binder-framework-prot-rep-mode.patch` —— 修复 `create_np_features_from_motif_config()`
  缺 `prot_rep_mode` 参数的上游 bug（应反馈给 genie3 项目）

新增 patch 时按 `NNNN-description.patch` 编号。

## 本地开发

```bash
# 1. 在 bioagent 项目根目录创建 venv
uv venv .venv-genie3 --python 3.10
source .venv-genie3/bin/activate

# 2. 装 genie3 与服务依赖
pip install --upgrade pip setuptools wheel "numpy>=2.0.2,<3" Cython
pip install torch==2.7.1 tqdm scipy pandas lightning "biopython<1.86" \
            ml-collections zstandard huggingface_hub wandb tensorboard PyYAML
pip install --no-build-isolation -e ./opensource/genie3
pip install ./services/_framework             # 服务框架
pip install httpx alibabacloud-oss-v2          # 远程 fetch 用

# 3. 软链 server 包到 genie3 目录（让 uvicorn 通过 server.app:app 找到）
ln -sf $(pwd)/services/genie3-server $(pwd)/opensource/genie3/server

# 4. 设置环境变量
export GENIE3_ROOT=$(pwd)/opensource/genie3
export GENIE3_JOBS_BASE_DIR=/tmp/genie3_jobs

# 5. 启动
cd $GENIE3_ROOT
uvicorn server.app:app --host 0.0.0.0 --port 9000 --reload
```

## Weights

v0.0.17 起预训练 v1 权重（~512 MB）**不再 baked 到镜像**，从 NAS 加载。
genie3 CLI 用相对路径 `<cwd>/pretrained/<version>/...` 查找权重，所以镜像里
`/opt/genie3/pretrained` 是个**软链**指向 NAS 挂载点。

期望布局：

```
/data/models/genie3/
└── pretrained/
    └── v1/
        ├── config.yaml
        └── checkpoints/
            └── step=600000.ckpt   ← 关键 checkpoint
```

### Pre-stage（一次性）

```bash
# 1. 下载到本地 stage 目录（手动下载，没有 fetch 脚本）
mkdir -p services/genie3-server/pretrained/v1
# <下载 config.yaml + checkpoints/step=600000.ckpt 到该目录>

# 2. 上传到 NAS
rsync -av services/genie3-server/pretrained/ \
    <NAS-mount>:/data/models/genie3/pretrained/
```

### FC

NAS 自动挂载到 `/data/models/genie3/`。验证：

```bash
curl https://fc-genie-icjpnieeiz.cn-hangzhou-vpc.fcapp.run/healthz/detail
# 期望：{"status":"ok","weights_loaded":true,"files_found":N}
```

### SIF / HPC

```bash
apptainer run --nv \
    --bind /scratch/models/genie3:/data/models/genie3 \
    genie3-server.sif python -m server generate unconditional ...
```

详见 [weights externalization design](../../engineering/decisions/2026-06-26-service-weights-externalization.md)。

## Docker 构建与运行

```bash
# 1. Vendor 上游源（一次性，重跑可升级 SHA）
./services/genie3-server/scripts/vendor.sh

# 2. 构建（~2.5 GB 镜像，无权重）
docker build --platform linux/amd64 -t genie3-server -f services/genie3-server/Dockerfile .

# 或通过 Makefile
make build-genie3-server

# 本地运行（需要 GPU + NAS / 本地 --bind 注入权重）
docker run --gpus all -p 9000:9000 --memory 16g \
    -v $(pwd)/pretrained:/data/models/genie3/pretrained \
    genie3-server
```

构建上下文必须是项目根目录，因为 Dockerfile 同时 `COPY services/_framework`
安装框架包，并 `COPY services/genie3-server/patches` 应用上游补丁。

## 阿里云函数计算部署

### 前置条件

- ACR 个人版或企业版（非经济版），同地域、同账号
- GPU 实例可用地域：华东 1/2、华北 2/3、华南 1、日本、美国

### 部署步骤

1. 构建并推送镜像：

   ```bash
   docker login --username=<阿里云账号> registry.cn-hangzhou.aliyuncs.com
   docker build --platform linux/amd64 -t genie3-server -f services/genie3-server/Dockerfile .
   docker tag genie3-server registry.cn-hangzhou.aliyuncs.com/<namespace>/genie3-server:latest
   docker push registry.cn-hangzhou.aliyuncs.com/<namespace>/genie3-server:latest
   ```

2. 创建函数：

   | 配置项 | 推荐值 |
   |---|---|
   | 运行时 | 自定义容器镜像 |
   | 镜像地址 | ACR VPC 地址 |
   | GPU 配置 | `fc.gpu.tesla.1` 起步（8 GB 显存：≤ 512 残基的单体 / 中长 binder） |
   | 监听端口 CAPort | `9000` |
   | 函数超时 | ≥ 3600 秒（生成多个长 binder 可能数十分钟） |
   | 内存 | `16384` MB |
   | 磁盘 | `10240` MB |
   | NAS 挂载 | 推荐挂载到 `/data`，跨实例 / 跨服务共享 |

3. HTTP 触发器，绑定自定义域名（解除 `content-disposition: attachment` 限制）

### FC 平台限制与适配

| 限制项 | 值 | 适配 |
|---|---|---|
| 启动超时 | 120 s | `/healthz` 不加载模型 |
| Keep-alive | ≥ 15 min | `GENIE3_KEEP_ALIVE_SEC=900` |
| 监听地址 | `0.0.0.0:CAPort` | uvicorn 已配置 |
| 可写磁盘 | 512 MB ~ 10 GB | 自动清理 + NAS 挂载 |
| GPU 镜像大小 | ≤ 15 GB | 已剥离评估依赖；权重烘焙到镜像 |
| 镜像架构 | AMD64 only | `--platform linux/amd64` |

## 设计要点

- **仅生成、不评估**：本服务只跑 `genie3 generate`，**不**做 inverse fold / structure prediction /
  filtering。Genie3 的 `evaluate` 子命令需要 ColabFold / ESMFold / FoldSeek 等重依赖，打包到
  FC GPU 镜像会超 15 GB；如需评估，建议在另一个评估 server / 本地集群完成。
- **路径自动改写**：上传的 binder/motif 数据集 zip 内 `problems/*.json` 用相对路径，服务器
  解压后自动改为绝对路径（含 `binder_framework` 指向的嵌套 motif config 链式改写），确保
  无论 zip 顶层目录命名如何都能正确解析。
- **YAML 注入**：`/api/generate` 自定义 YAML 路径中的 `paths.rootdir` / `paths.dataset`
  会被服务器覆写为 job 本地目录，客户端不需要预测 job_id。
- **自定义 subprocess cwd**：`genie3 generate` 默认从 `pretrained/v1/checkpoints/step=600000.ckpt`
  这种相对路径加载权重，所以 subprocess cwd 设为 `/opt/genie3`（即 `GENIE3_ROOT`）。
- **快速启动**：`/healthz` 端点无模型加载，FC 120s 启动探测通过；权重烘焙到镜像避免冷启动下载。

## 相关文档

- [bioagent-service-framework](../_framework/README.md) — 通用 HTTP / job / 错误处理 / manifest 层
- [Service 框架抽象设计](../../engineering/decisions/2026-05-12-service-framework-design.md) — 设计决策
- [Tool 抽象层设计](../../engineering/decisions/2026-04-23-tool-abstraction-design.md) — Client 端 Tool + Runner（消费本 service 的 manifest）
