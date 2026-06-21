# ensemble-server

多方法生物分子任务聚合层（aggregator REST API）。客户给一条序列 + 一组方法名，
本 service 通过 [FCDispatcher](../../pipelines/framework/fc_dispatcher.py)
并行调用各底层 GPU FC service（alphafold-server / esmfold2-server / boltz-server / ...）
做翻译 + 标准化 + 排序，返回统一的 ensemble 结果。

**关键文档**：
- 架构设计：[engineering/decisions/2026-06-20-ensemble-service-design.md](../../engineering/decisions/2026-06-20-ensemble-service-design.md)
- Phase 1 实施计划：[engineering/decisions/2026-06-20-ensemble-service-plan.md](../../engineering/decisions/2026-06-20-ensemble-service-plan.md)
- 底层调用协议：[engineering/decisions/2026-06-17-fc-async-task-mode.md](../../engineering/decisions/2026-06-17-fc-async-task-mode.md)

## 当前状态

Phase 1 MVP folding aggregator 完成。已落地：

- `POST /v1/folding/ensemble` — 提交多方法并行 folding 任务
- `GET /v1/jobs/{task_id}` — 查询状态（含 sub-task 详情 + ensemble 排序）
- `GET /v1/jobs/{task_id}/structures/{method}/{filename}` — 下载结构文件
- `GET /v1/methods?task_kind=folding` — 列出已注册的方法 + options schema
- `GET /v1/manifest`、`GET /v1/healthz` — 元数据 / 健康检查
- API key 静态认证（`X-API-Key` header）

未做（待 Phase 3+）：cache、webhook callback、Tablestore auth、计费、design / scoring task kinds。

## 架构层次

```
HTTP /v1/folding/ensemble            ← 客户端入口（auth + validation）
  ↓
Orchestrator.submit                   ← TaskKind agnostic 调度核心
  ↓ per method
MethodAdapter.build_request           ← 翻译 normalized input → FC payload
  ↓
FCDispatcher.submit(async)            ← FC OpenAPI 异步入队（X-Fc-Invocation-Type: Async）
  ↓ (lazy poll via Orchestrator.refresh)
FCDispatcher.get_status(GetAsyncTask) ← 控制平面，不触发函数实例
  ↓ when terminal
FCDispatcher.fetch_result             ← 下载 zip + 解压到 NAS
  ↓
MethodAdapter.normalize_output        ← 翻译 FC 输出 → FoldingMethodResult
  ↓
folding.aggregator.aggregate_folding  ← cross-method 排序
  ↓
EnsembleJobStore.update               ← 写 NAS sidecar
  ↓
HTTP GET /v1/jobs/<task_id>           ← 客户端拿结果
```

## 配置（环境变量）

服务通过 `pydantic-settings` 读 `ENSEMBLE_*` 前缀的环境变量。嵌套字段用 `__` 分隔。

### FC OpenAPI 凭证（必填）

```bash
ENSEMBLE_FC_ACCESS_KEY_ID=<阿里云 AccessKey ID>
ENSEMBLE_FC_ACCESS_KEY_SECRET=<阿里云 AccessKey Secret>
```

### 底层 FC 方法注册（按需，至少一个）

每个方法独立配置，按嵌套字段名映射到 `FCMethodConfig`：

```bash
# alphafold
ENSEMBLE_FC_METHODS__ALPHAFOLD__FUNCTION=alphafold-server
ENSEMBLE_FC_METHODS__ALPHAFOLD__HTTP_BASE_URL=https://fc-alphafold-fcizetpnjc.cn-hangzhou-vpc.fcapp.run
ENSEMBLE_FC_METHODS__ALPHAFOLD__TASK_ENDPOINT=/api/tasks/fold

# esmfold2
ENSEMBLE_FC_METHODS__ESMFOLD2__FUNCTION=esmfold2-server
ENSEMBLE_FC_METHODS__ESMFOLD2__HTTP_BASE_URL=https://fc-esmfold-spayusioug.cn-hangzhou-vpc.fcapp.run
ENSEMBLE_FC_METHODS__ESMFOLD2__TASK_ENDPOINT=/api/tasks/fold

# boltz
ENSEMBLE_FC_METHODS__BOLTZ__FUNCTION=boltz-server
ENSEMBLE_FC_METHODS__BOLTZ__HTTP_BASE_URL=https://fc-boltz-kbioniejif.cn-hangzhou-vpc.fcapp.run
ENSEMBLE_FC_METHODS__BOLTZ__TASK_ENDPOINT=/api/tasks/predict_structure
```

可选字段：`__REGION`（默认 `cn-hangzhou`）、`__ENABLED`（默认 `true`，设 `false` 跳过注册）、
`__TIMEOUT_SECONDS`（默认 7200）。

### API key 名单（Phase 1 静态）

详细生成 + 配置 + 客户使用流程见下方 [API Key 配置](#api-key-配置) 一节。最简形态：

```bash
ENSEMBLE_API_KEYS__0__KEY_ID=ek_test_001
ENSEMBLE_API_KEYS__0__SECRET_HASH=<sha256(plaintext) 十六进制>
ENSEMBLE_API_KEYS__0__CUSTOMER_ID=internal_test
ENSEMBLE_API_KEYS__0__PLAN=internal               # optional
ENSEMBLE_API_KEYS__0__MONTHLY_QUOTA_CALLS=1000    # optional, 未生效（Phase 3）
```

多个 key 用 `__0__`、`__1__`、... 编号。

### NAS 路径

继承自 `bioagent_service.ServiceSettings`：

```bash
ENSEMBLE_JOBS_BASE_DIR=/data/jobs/ensemble    # default: /data/jobs
```

FC 部署时挂载 NAS 到 `/data` 即可。

## API Key 配置

每个 API key 三个字段：

| 字段 | 含义 | 公开/敏感 |
|---|---|---|
| `KEY_ID` | 客户可见的 key 标识符（写在合同 / 邮件里方便支持查询） | 公开 |
| `SECRET_HASH` | 验证用的 sha256 hex，**服务端永不存明文** | 公开（hash 而已）|
| `CUSTOMER_ID` | job 归属 + quota / 计费 key（同 CUSTOMER_ID 的多个 key 共享数据可见性） | 内部 |

### 一键生成 N 个 key

```bash
python <<'PY' > /tmp/ensemble_keys.txt
import secrets, hashlib

records = [
    # (KEY_ID,             CUSTOMER_ID,    PLAN)
    ("ek_internal_test",   "internal_test", "internal"),
    ("ek_acme_001",        "cust_acme",     "pro"),
]
for i, (key_id, cust, plan) in enumerate(records):
    secret = secrets.token_urlsafe(32)
    sha = hashlib.sha256(secret.encode()).hexdigest()
    print(f"# {key_id} ({cust}, {plan})")
    print(f"#   PLAINTEXT SECRET (give to customer + store safely): {secret}")
    print(f"ENSEMBLE_API_KEYS__{i}__KEY_ID={key_id}")
    print(f"ENSEMBLE_API_KEYS__{i}__SECRET_HASH={sha}")
    print(f"ENSEMBLE_API_KEYS__{i}__CUSTOMER_ID={cust}")
    print(f"ENSEMBLE_API_KEYS__{i}__PLAN={plan}")
    print()
PY

cat /tmp/ensemble_keys.txt
# 1. 把 PLAINTEXT SECRET 行单独发给对应客户（safely）
# 2. 把 ENSEMBLE_API_KEYS__* 行粘到 FC 控制台环境变量界面
# 3. 用完立即销毁 /tmp/ensemble_keys.txt（含明文）
rm /tmp/ensemble_keys.txt
```

### 手动算单个 hash（已有现成 secret 时）

```bash
SECRET='aB3kfP_xQ7yN-mLqRsT5vW8zA1bC4dE6gH9jKlMnOpQ'
echo -n "$SECRET" | sha256sum | awk '{print $1}'
```

⚠ 必须用 `echo -n`（无尾换行），否则算出来的 hash 不对。

### 客户怎么用

每次请求带 `X-API-Key: <明文 secret>`：

```bash
SECRET='<你给客户的明文>'
curl -H "X-API-Key: $SECRET" https://fc-ensemble-XXX.cn-hangzhou-vpc.fcapp.run/v1/healthz
```

服务端拿到 header → sha256 → 在 `api_keys` 列表里查匹配 → 取出 `CUSTOMER_ID` 用于 job 归属和后续 quota 计费。

### 本地验证 auth 工作正常

```bash
cd services/ensemble-server

SECRET='test_local_001'
SHA=$(echo -n "$SECRET" | sha256sum | awk '{print $1}')

export ENSEMBLE_JOBS_BASE_DIR=/tmp/ensemble_jobs
export ENSEMBLE_API_KEYS__0__KEY_ID=ek_local
export ENSEMBLE_API_KEYS__0__SECRET_HASH=$SHA
export ENSEMBLE_API_KEYS__0__CUSTOMER_ID=local_dev
mkdir -p $ENSEMBLE_JOBS_BASE_DIR

uv run python -m server &
sleep 2

# 提交 endpoint 没 header → 401
curl -s -o /dev/null -w "no header: %{http_code}\n" -X POST \
    -H "Content-Type: application/json" -d '{"input":{"sequences":[]}}' \
    http://localhost:9000/v1/folding/ensemble

# 错 secret → 401
curl -s -o /dev/null -w "wrong secret: %{http_code}\n" -X POST \
    -H "X-API-Key: wrong" -H "Content-Type: application/json" -d '{"input":{"sequences":[]}}' \
    http://localhost:9000/v1/folding/ensemble

# 对 secret + 没注册任何方法 → 422 或 503
curl -s -X POST -H "X-API-Key: $SECRET" \
    -H "Content-Type: application/json" \
    -d '{"input":{"sequences":[{"id":"A","sequence":"MKQH"}]}}' \
    http://localhost:9000/v1/folding/ensemble

kill %1
```

期望：第一个 401，第二个 401，第三个 503（没方法注册）—— 证明 auth 工作正常。

### 实战注意事项

- **不要把明文 secret 提交到 git**，只 commit `SECRET_HASH`。明文走 1password / Vault / 邮件密文等渠道发给客户。
- **每个客户独立 secret**：同一个 secret 分发给两个客户 = 无法独立 quota / 撤销。
- **撤销 key（Phase 1 静态机制下）**：FC 控制台删掉对应 `ENSEMBLE_API_KEYS__<N>__*` 项 → 触发函数重新部署。后续 Phase 3 上 Tablestore 后秒级撤销。
- **轮换 key**：先在 env 里追加一组 `__<N+1>__*` 给客户新 secret → 客户切到新的 → 删除旧的 `__<N>__*` 项 → 重新部署。
- **`PLAN` 和 `MONTHLY_QUOTA_CALLS` 字段当前只读、不强制执行**（Phase 3 才落实计费）—— 但**现在就把语义填对**，将来直接接 Stripe 不用回改。
- **FC 控制台环境变量界面**：建议把 `SECRET_HASH` 标成 secret 类型字段，避免出现在函数日志里（虽然 hash 本身泄露也不致命）。

## 本地开发

### 依赖安装

```bash
cd services/ensemble-server
uv venv
uv pip install -e ../_framework  # bioagent-service-framework editable
uv pip install -e .              # ensemble-server 自身
```

或者从仓库根直接跑（`pipelines/` 包通过 PYTHONPATH 可见）：

```bash
cd /path/to/bioagent
uv run python -m server   # 不行，需要先把 services/ensemble-server 注册为 server 模块
```

### 跑测试（offline，不依赖真实 FC）

```bash
uv run python -m pytest services/ensemble-server/tests/ -v
```

应该看到 32 个测试全部通过：
- 4 个 orchestrator 测试（用 fake adapter）
- 9 个 folding adapter 测试
- 12 个 route e2e 测试（FastAPI TestClient + mocked FCDispatcher）
- 7 个 auth 测试

### 本地启动 server（无真实 FC backends 也能跑）

```bash
# 准备最小环境（jobs_base_dir 临时目录就行）
export ENSEMBLE_JOBS_BASE_DIR=/tmp/ensemble_jobs
mkdir -p $ENSEMBLE_JOBS_BASE_DIR

cd services/ensemble-server
uv run python -m server  # 9000 端口
```

不配置 `ENSEMBLE_FC_METHODS__*` 时，没有方法注册，`GET /v1/manifest` 会返回空 methods 列表。

## Docker 构建 + 本地冒烟

镜像构建必须以 **仓库根** 作为构建上下文（因为要 COPY `services/_framework/` 和 `pipelines/`）：

```bash
cd /path/to/bioagent
docker build -f services/ensemble-server/Dockerfile -t ensemble-server:dev .
```

启动并冒烟：

```bash
docker run --rm -d --name ens_smoke -p 9001:9000 \
    -e ENSEMBLE_JOBS_BASE_DIR=/tmp/jobs \
    ensemble-server:dev
sleep 5
curl http://localhost:9001/v1/healthz
# 期望：{"status":"ok","service":"ensemble","version":"0.0.1"}
docker stop ens_smoke
```

镜像应该 < 200 MB（CPU only，无算法栈）。

## FC 部署

ensemble-server 是 CPU 函数（不跑 GPU 计算）。

### 推 ACR

```bash
docker tag ensemble-server:dev harbor.ruosheng.bio/aliyun_fc/ensemble-server:v0.0.1
docker push harbor.ruosheng.bio/aliyun_fc/ensemble-server:v0.0.1
```

### FC 控制台创建函数

- **实例规格**：CPU 1c2g（够用，本服务只编排，不算）
- **镜像**：`harbor.ruosheng.bio/aliyun_fc/ensemble-server:v0.0.1`
- **启动命令**：默认（Dockerfile CMD = `python -m server`）
- **监听端口**：9000
- **NAS 挂载**：`/data`（共享 ensemble jobs sidecar + 各底层 service 的输出文件）
- **环境变量**：参考上面「配置」一节，最少配齐 `FC_ACCESS_KEY_ID/SECRET` + 至少一组 `FC_METHODS__*` + 至少一个 `API_KEYS__0__*`

⚠ **重要**：
- **不要**开启 FC 控制台的「异步任务模式」—— 本 service 自己是 controller，不长跑。客户端用同步 HTTP 调用即可。
- **不需要** keepalive URL —— 本 service 状态都写 NAS，实例回收不丢数据。
- 函数 timeout 设 90s 够用（路由 + 转发都很快）。

### 部署后冒烟

```bash
SECRET=<你设的明文 secret>
TEST_URL=https://fc-ensemble-XXX.cn-hangzhou-vpc.fcapp.run

# 1. healthz（不需要 API key）
curl $TEST_URL/v1/healthz

# 2. 查注册的方法
curl -H "X-API-Key: $SECRET" "$TEST_URL/v1/methods?task_kind=folding"

# 3. 提交一个跑 esmfold2 + boltz 的 ensemble job
curl -X POST "$TEST_URL/v1/folding/ensemble" \
    -H "X-API-Key: $SECRET" \
    -H "Content-Type: application/json" \
    -d '{
      "input": {
        "sequences": [{"id": "A", "sequence": "MKQHKAMIVALIVICITAVVAALVTRKDLCEVHIRTGQTEVAVF"}],
        "msa_mode": "empty"
      },
      "methods": ["esmfold2", "boltz"]
    }'
# 期望：202 + {"task_id": "ens_fold_...", "status": "accepted", "requested_methods": [...]}

# 4. 轮询（每 30s 一次）
TASK_ID=ens_fold_...
curl -H "X-API-Key: $SECRET" $TEST_URL/v1/jobs/$TASK_ID

# 5. 期待约 10-15 min 后 status=completed，aggregated_output 含 ensemble_ranking
# 6. 下载 rank-0 结构
curl -H "X-API-Key: $SECRET" -OJ \
    "$TEST_URL/v1/jobs/$TASK_ID/structures/esmfold2/prediction_0.cif"
```

部署成功后请把 ensemble-server URL 追加到 [`services/aliyun_fc_url.md`](../aliyun_fc_url.md)。

## API 参考（快速）

### `POST /v1/folding/ensemble`

请求体：

```json
{
  "input": {
    "sequences": [
      {"id": "A", "sequence": "MKQH...", "type": "protein"}
    ],
    "msa_mode": "empty"          // "auto" | "empty" | "search"
  },
  "methods": ["alphafold", "esmfold2", "boltz"],   // 缺省则跑全部已注册
  "method_options": {
    "alphafold": {"db_preset": "reduced_dbs", "model_preset": "monomer_ptm"},
    "esmfold2":  {"num_loops": 3, "num_sampling_steps": 50},
    "boltz":     {"recycling_steps": 3, "sampling_steps": 200}
  }
}
```

每个方法的 options schema 可通过 `GET /v1/methods?task_kind=folding` 查到。

响应：`202 + {"task_id": "...", "status": "accepted", "requested_methods": [...]}`

### `GET /v1/jobs/{task_id}`

返回完整 `EnsembleJob` 状态，包括：
- `status`: pending / running / completed / partial / failed
- `sub_tasks`: `{method_name: SubTaskRecord}`，每个含 `status` / `runtime_seconds` / `output`（含 structures + confidence）/ `error_summary`
- `aggregated_output`: 含 `ensemble_ranking`（按 pLDDT 降序的跨方法结构列表）+ `ensemble_score`

每次 GET 触发 lazy refresh —— 调 FC GetAsyncTask 查每个 sub-task 状态，发现终态就拉文件 + 解析 + 持久化。

### `GET /v1/jobs/{task_id}/structures/{method}/{filename}`

直接流式下载 NAS 上的结构文件。`filename` 必须是 basename（不能含 `/` 或 `..`）。

### 错误码

| 状态 | 含义 |
|---|---|
| 401 | 缺 `X-API-Key` 或 key 不匹配 |
| 404 | task_id 不存在 / 不属于当前 customer |
| 422 | 入参 validation 失败（未知 method、错的 task_kind 等） |
| 503 | 没有方法注册（服务端 `FC_METHODS__*` 没配） |

## 加新 service / task_kind

加新 folding 方法：
1. 在 `services/ensemble-server/adapters/folding/<method>.py` 新增 `MethodAdapter` 子类
2. 在 `app.py` 的 `_FOLDING_ADAPTER_CLASSES` 元组里加入
3. 部署时配 `ENSEMBLE_FC_METHODS__<METHOD_UPPER>__*` 环境变量
4. 重启服务（FC 控制台触发函数重新部署或代码刷新）

加新 task kind（design / scoring / ...）：
1. 在 `task_kind.py` 加 enum 项
2. 新增 `<task_kind>/schemas.py`（Input / Output / MethodResult）
3. 新增 `<task_kind>/aggregator.py`
4. 新增 `adapters/<task_kind>/`，每个方法一个 adapter
5. 新增 `routes/<task_kind>.py`
6. 在 `app.py` 注册 adapter + mount route

详见 [架构设计文档](../../engineering/decisions/2026-06-20-ensemble-service-design.md) 的「通用性扩展机制」一节。

## 监控 / 排查

任务卡住时：

```bash
# 看服务端 sidecar 看 sub_tasks 各自的状态
cat /data/jobs/<task_id>/job.json | jq .sub_tasks

# 如果某个 sub_task 一直 running，去对应底层 service 查
SUB_TASK_ID=<task_id>__esmfold2
curl -H "X-Bioagent-Session-Id: $SUB_TASK_ID" \
    https://fc-esmfold-spayusioug.cn-hangzhou-vpc.fcapp.run/api/jobs/$SUB_TASK_ID
```

429 throttling 风险：HTTP polling 在 high concurrency 下会触发 FC HTTP 限流，但 ensemble-server
用 FC GetAsyncTask 控制平面（不走 HTTP），所以理论上不受影响。详见
[FC HTTP 轮询并发不可靠 memory](../../engineering/decisions/2026-06-17-fc-async-task-mode-plan.md)
里的描述。

## 后续路线（不在 Phase 1 内）

- Phase 2：Python SDK + CLI 包装 REST API
- Phase 3：cache + webhook + Tablestore auth + 管理后台
- Phase 4：design ensemble（boltzgen / rfdiffusion / odesign）
- Phase 5：scoring ensemble（dockq / deeprank-ab）
- Phase 6：BYOK 模式 + Stripe 计费
