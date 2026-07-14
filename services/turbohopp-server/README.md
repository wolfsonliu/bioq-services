# turbohopp-server

基于 FastAPI 的 [TurboHopp](https://github.com/orgw/TurboHopp) HTTP 服务
—— 把一致性模型加速的 **scaffold hopping** 功能包装成 FC GPU 服务。
**构建在 [bioagent-service-framework](../_framework/) 之上**。

TurboHopp：给定**蛋白口袋** + **参考配体**，通过 consistency-model 距离
蒸馏的 GVP 网络生成 N 个候选分子——同 DiffHopp 输入域但采样步数从 100+
降到个位数，速度提升 30×+（arXiv:2410.20660, NeurIPS 2024, MIT）。

镜像 base：`nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04`；conda env
(Python 3.10 + PyTorch 2.2.2 + cu121 + PyG 2.5 + rdkit + openmm + pdbfixer)。

## 与其他 scaffold hopping 服务的关系

项目里有 3 个 scaffold hopping 服务，覆盖不同工作流：

| | [chembounce-server](../chembounce-server/) | [diffusion-hopping-server](../diffusion-hopping-server/) | **turbohopp-server**（本服务） |
|---|---|---|---|
| 输入 | SMILES 字符串 | pocket PDB + ref ligand | pocket PDB + ref ligand |
| 蛋白结构约束 | ❌ | ✅ | ✅ |
| 方法 | 经典 fragmentation + 相似性 | 图扩散（100+ steps） | 一致性模型（5–40 steps） |
| 算力 | CPU | GPU | GPU |
| 单批耗时 | 秒 | 数分钟 | 秒–数十秒 |
| 适用 | 早期 hit-to-lead / 抗专利 / 无靶点结构 | 高质量离线 batch | 交互式迭代 / agent 循环 |
| 权重来源 | 内置 fragment DB | upstream 仓库自带 4 ckpt (~189 MB) | **需自训 / 从作者获取** |

## 架构

```
客户端 / Agent
  ↓ HTTP (multipart upload: protein.pdb + reference_ligand.sdf)
┌────────────────────────────────────────────────────────────────┐
│  FastAPI + bioagent-service-framework  (port 9000)             │
│                                                                │
│  服务专属                                                      │
│    POST /api/generate         (submit/poll)                    │
│    POST /api/tasks/generate   (FC async task mode)             │
│                                                                │
│  框架统一提供                                                  │
│    GET  /api/manifest                                          │
│    GET  /openapi.json                                          │
│    GET  /healthz, /healthz/detail   (后者带权重就位探针)       │
│    GET  /api/jobs/{id}/...                                     │
└────────────────────────────────────────────────────────────────┘
  ↓ subprocess (cwd=/opt/turbohopp, 单卡串行)
inference.py → ObabelTransform + ReduceTransform →
    LitConsistencyModel(student).load_state_dict(ckpt) →
    ConsistencySamplingAndEditing_DiffHopp(final_timesteps=K).__call__()
  ↓ 输出
NAS: <jobs_base_dir>/<job_id>/{input/, output/output_<i>.sdf, logs/run.log, job.json}
```

## API 速览

### `POST /api/generate`

multipart/form-data：

| 字段 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `protein` | file (.pdb) | ✓(或 `protein_uri`) | — | 蛋白口袋 |
| `reference_ligand` | file (.sdf / .mol2 / .pdb) | ✓(或 `reference_ligand_uri`) | — | 参考配体 |
| `num_samples` | int (1–100) | — | `10` | 候选数量（一次 batched sampling） |
| `num_sampling_steps` | int (1–100) | — | `40` | 一致性模型采样步数；5-10 交互场景够用 |
| `find_best` | bool | — | `false` | 后处理 QED+SA 综合分挑最优；有效样本 × 2 |
| `seed` | int (≥0) | — | `null` | 采样随机种子 |

示例：

```bash
# 交互式 5 步采样
curl -X POST $URL/api/generate \
    -F protein=@1abc_pocket.pdb \
    -F reference_ligand=@ref.sdf \
    -F num_samples=10 \
    -F num_sampling_steps=5 \
    -F seed=42
# → { "job_id": "...", "status": "pending", ... }

curl $URL/api/jobs/<job_id>
# completed 后：
curl $URL/api/jobs/<job_id>/files
# → ["output/output_0.sdf", "output/output_1.sdf", ...]
curl -O $URL/api/jobs/<job_id>/file/output/output_0.sdf
```

### `POST /api/tasks/generate`

FC 异步任务模式（HTTP 立即返回 202，FC 把请求和计算的生命周期绑在一起）。
控制台需要打开「异步任务模式」——详见 [新增服务 cookbook](../../engineering/guides/adding-a-new-service/deploy.md#fc-异步任务模式控制台配置)。

大文件（口袋 PDB 常见 > 128 KiB）需要走 URI fallback，因为 FC async
event payload cap 是 128 KiB。示例：

```bash
# 1. 先用 sync 端点上传一次（顺便跑一个短 job），或用别的方式把文件放
#    到 NAS 上 /data/turbohopp_jobs/<any>/input/ 或 /data/scratch/ 等；
curl -X POST $URL/api/generate \
    -H "X-Fc-Invocation-Type: Async" \
    -H "X-Bioagent-Job-Id: my-job-001" \
    -F protein_uri=file:///data/turbohopp_jobs/bootstrap/input/pocket.pdb \
    -F reference_ligand_uri=file:///data/turbohopp_jobs/bootstrap/input/ref.sdf \
    -F num_samples=10 \
    -F num_sampling_steps=20
```

支持的 URI schemes：
- `file:///abs/path` — 直接读 NAS 路径
- `job://<job_id>/<filename>` — 链接前一个 job 的 output
- `oss://<bucket>/<key>` — 阿里云 OSS
- `http(s)://...` — 通用 HTTP 下载

## Weights

**Upstream TurboHopp 不发布公开的 consistency-model checkpoint。**必须自行获取：

1. **自训**：使用 upstream 的 `train_consistency.py`，以 DiffHopp 作为
   teacher 进行知识蒸馏。参考配置在 `services/turbohopp-server/upstream/configs/config_consistency.yaml`。
2. **联系作者**（Yoo et al. 2024, NeurIPS 2024，arXiv:2410.20660）。

拿到 checkpoint 后，rsync 到 NAS：

```bash
# NAS 布局（FC 挂载）:
#   /data/models/turbohopp/checkpoints/v1/
#     └── turbohopp_consistency.ckpt   # 你的 consistency-model .ckpt
#
rsync -av path/to/turbohopp_consistency.ckpt \
    <nas>:/data/models/turbohopp/checkpoints/v1/
```

**在此之前**：`/healthz/detail` 会正常返回 200 + `weights_loaded: false`
（不 crash）。因此可以先把服务部署起来做 smoke 测，等有 ckpt 再做真跑。

多个候选 checkpoint 时可以用 `TURBOHOPP_CHECKPOINT_NAME=<filename>`
（相对 `weights_dir`）显式 pin 一个，否则服务会自动选 `weights_dir/`
下扫到的第一个 `*.ckpt`。

## 本地开发 / 部署命令

```bash
# 1. vendor 上游源码到 upstream/（pinned SHA e342350）
./services/turbohopp-server/scripts/vendor.sh

# 2. 构建 Docker 镜像
make build-turbohopp-server
# 或直接
docker build --platform linux/amd64 -t turbohopp-server \
    -f services/turbohopp-server/Dockerfile .

# 3. 本地跑一下（不做 GPU 推理，只测 HTTP 路由 / healthz）
docker run --rm -p 9000:9000 turbohopp-server
curl http://localhost:9000/healthz
# → 权重不存在时: {"weights_loaded": false, "files_found": 0}

# 4. 推到 harbor + 更新 FC 函数镜像
make push-turbohopp-server
```

FC 部署 checklist 见 [新增服务 cookbook](../../engineering/guides/adding-a-new-service/deploy.md#部署到-fc)。
架构决策：[engineering/decisions/2026-07-01-turbohopp-server-design.md](../../engineering/decisions/2026-07-01-turbohopp-server-design.md)。

## 配置项（环境变量）

| ENV | 默认 | 说明 |
|---|---|---|
| `TURBOHOPP_JOBS_BASE_DIR` | `/data/turbohopp_jobs` | Job 目录挂载点 |
| `TURBOHOPP_ROOT` | `/opt/turbohopp` | Upstream 源路径 |
| `TURBOHOPP_PYTHON` | `/opt/conda/bin/python` | Conda env python |
| `TURBOHOPP_INFERENCE_SCRIPT` | `/opt/turbohopp-server/server/inference.py` | Wrapper 脚本 |
| `TURBOHOPP_WEIGHTS_DIR` | `/data/models/turbohopp/checkpoints/v1` | NAS 权重目录 |
| `TURBOHOPP_CHECKPOINT_NAME` | *(unset)* | 显式 pin 一个 checkpoint 文件名 |
| `TURBOHOPP_MAX_CONCURRENT_JOBS` | `1` | 单实例并发 job 数（GPU 服务默认 1） |
| `TURBOHOPP_OSS_REGION` | `cn-hangzhou` | OSS URI 拉取区域 |
| `WANDB_MODE` | `disabled` | 强制关闭 wandb（upstream 默认开启） |

## CLI 批处理模式

同一 Docker 镜像可作为 sbatch 任务：

```bash
docker run --rm --gpus all \
    -v $PWD/inputs:/data/input:ro \
    -v $PWD/scratch:/scratch:rw \
    -v /mnt/nas/data/models/turbohopp/checkpoints/v1:/data/models/turbohopp/checkpoints/v1:ro \
    turbohopp-server \
    /opt/conda/bin/python -m server generate \
        --protein /data/input/pocket.pdb \
        --reference-ligand /data/input/ref.sdf \
        --output-dir /scratch/results/ \
        --params-json '{"num_samples": 10, "num_sampling_steps": 5, "find_best": false}'
```

详见 [CLI 批处理模式设计](../../engineering/decisions/2026-05-29-cli-batch-mode.md)。

## 测试

```bash
# offline (无需 GPU / 权重)
uv run python -m pytest services/turbohopp-server/tests/test_app.py -v
uv run python -m pytest services/turbohopp-server/tests/test_cli.py -v

# FC 集成 (需先部署)
RUN_FC_TESTS=1 uv run python -m pytest -m fc services/turbohopp-server/tests/test_fc.py -v
RUN_FC_TESTS=1 uv run python -m pytest -m fc services/turbohopp-server/tests/test_fc_task.py -v
```
