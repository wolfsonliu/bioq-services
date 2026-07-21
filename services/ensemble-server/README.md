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

# promera (cofold endpoint — protein structure prediction; design endpoint not surfaced via ensemble)
ENSEMBLE_FC_METHODS__PROMERA__FUNCTION=promera-server
ENSEMBLE_FC_METHODS__PROMERA__HTTP_BASE_URL=https://fc-promera-adkrlhmlcq.cn-hangzhou-vpc.fcapp.run
ENSEMBLE_FC_METHODS__PROMERA__TASK_ENDPOINT=/api/tasks/cofold
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

继承自 `bioq_service.ServiceSettings`：

```bash
ENSEMBLE_JOBS_BASE_DIR=/data/jobs/ensemble    # default: /data/jobs
```

FC 部署时挂载 NAS 到 `/data` 即可。

## 认证配置（VPC bypass + JWT + API Key）

ensemble-server 应用层实现三层 fallthrough 认证链：

```
HTTP request → require_auth dependency
  ↓
  1. VPC bypass？检查 Host header → 匹配 `*-vpc.fcapp.run` 或 localhost → 放行
  ↓ 未匹配
  2. JWT？`Authorization: Bearer <token>` → 验签 → 取 `sub` 作 customer_id
  ↓ 未提供 / 验签失败
  3. API Key？`X-API-Key` header → sha256 → 静态 allowlist 查表
  ↓ 都未通过
  401 unauthorized
```

任一通过即可。三种 auth 可同时配置（默认全部启用），客户端选最方便的用。

设计原因 + 三层取舍详见
[engineering/decisions/2026-06-21-ensemble-server-auth.md](../../engineering/decisions/2026-06-21-ensemble-server-auth.md)。

### 三种 auth 各自的使用场景

| 来源 | 怎么调 | 后端识别为 |
|---|---|---|
| **VPC 内部脚本** | 直接打 VPC URL，无任何 auth header | `customer_id=internal_vpc`, `method=vpc_bypass` |
| **企业客户**（先拿 JWT） | `curl ... -H "Authorization: Bearer <jwt>"` | `customer_id=<jwt.sub>`, `method=jwt` |
| **SaaS 客户**（静态 API Key） | `curl ... -H "X-API-Key: <secret>"` | `customer_id=<APIKeyConfig.customer_id>`, `method=api_key` |

### 配置环境变量

#### 1. VPC bypass

```bash
# 默认开。希望关闭则设 false（这样从 VPC URL 来的请求也必须带 JWT 或 API Key）
ENSEMBLE_AUTH__BYPASS_VPC=true
ENSEMBLE_AUTH__VPC_CUSTOMER_ID=internal_vpc    # 内部请求归属的 customer_id
```

VPC 检测基于 `Host` header：匹配 `*-vpc.fcapp.run` 或 `localhost` / `127.0.0.1` 视为 VPC 内访问。
外部攻击者无法路由到 VPC URL（物理隔离），所以 Host 检查可靠。

#### 2. JWT（可选；不配则禁用 JWT 路径）

复用 `services/jwt/` 的 JWKS：

```bash
ENSEMBLE_AUTH__JWT_JWKS_URL=https://fc-jwt-XXX.cn-hangzhou-vpc.fcapp.run/.well-known/jwks.json
ENSEMBLE_AUTH__JWT_AUDIENCE=ensemble-server
ENSEMBLE_AUTH__JWT_ISSUER=                          # 留空 = 不校验 iss
ENSEMBLE_AUTH__JWT_JWKS_CACHE_TTL_SEC=3600
ENSEMBLE_AUTH__JWT_SUB_IS_CUSTOMER=true             # true: 直接把 sub 当 customer_id
```

`sub → customer_id` 映射（`JWT_SUB_IS_CUSTOMER=false` 时启用）：

```bash
# 把 sub="ext-acme-001" 映射成内部 customer_id="cust_acme"
ENSEMBLE_AUTH__JWT_SUB_TO_CUSTOMER='{"ext-acme-001": "cust_acme"}'
```

JWKS 默认 cache 1 小时；遇到 kid miss（key rotation 场景）自动 force-refresh 一次再判定失败。

#### 3. 静态 API Key（Phase 1）

每个 key 三个必填字段：

| 字段 | 含义 | 公开/敏感 |
|---|---|---|
| `KEY_ID` | 客户可见的 key 标识符（写在合同 / 邮件里方便支持查询） | 公开 |
| `SECRET_HASH` | 验证用的 sha256 hex，**服务端永不存明文** | 公开（hash 而已）|
| `CUSTOMER_ID` | job 归属 + quota / 计费 key（同 CUSTOMER_ID 的多个 key 共享数据可见性） | 内部 |

##### 一键生成 N 个 key

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

##### 手动算单个 hash（已有现成 secret 时）

```bash
SECRET='aB3kfP_xQ7yN-mLqRsT5vW8zA1bC4dE6gH9jKlMnOpQ'
echo -n "$SECRET" | sha256sum | awk '{print $1}'
```

⚠ 必须用 `echo -n`（无尾换行），否则算出来的 hash 不对。

### 客户端使用示例

```bash
# 1. 内部 VPC 脚本（什么都不用带）
curl https://fc-ensemble-XXX.cn-hangzhou-vpc.fcapp.run/v1/folding/ensemble \
    -X POST -H "Content-Type: application/json" -d '{...}'

# 2. 企业客户先去 JWT service 拿 token，再带 Bearer
TOKEN=$(curl -X POST https://fc-jwt-XXX/api/token \
    -H "X-API-Key: $JWT_SVC_ADMIN_KEY" \
    -d '{"sub":"customer-acme"}' | jq -r .token)
curl https://fc-ensemble-XXX.cn-hangzhou.fcapp.run/v1/folding/ensemble \
    -X POST -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" -d '{...}'

# 3. SaaS 客户用 API Key
curl https://fc-ensemble-XXX.cn-hangzhou.fcapp.run/v1/folding/ensemble \
    -X POST -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" -d '{...}'
```

### 优先级 / fallthrough 细节

- VPC bypass **优先于** JWT/API Key：从 VPC URL 来的请求即使带了 Bearer 或 X-API-Key 也走 VPC bypass 路径，customer_id 是 `internal_vpc`。生产中如果想让 VPC URL 也走 JWT/API Key（比如要按客户身份记账），设 `ENSEMBLE_AUTH__BYPASS_VPC=false`。
- **JWT 验签失败会 fallthrough 到 API Key**：客户同时带了 `Authorization: Bearer ...` 和 `X-API-Key: ...` 时，先试 JWT；JWT 验签失败（过期 / 签名错 / 网络拉 JWKS 失败）则继续试 API Key。这是为了兼容老客户从静态 API Key 平滑迁移到 JWT。如果想严格要求 JWT 必须通过、不允许 fallthrough，把 API Key allowlist 设空即可。
- **3 种都没通过 → 401**，响应 body：`{"detail": "missing or invalid credentials (provide Authorization: Bearer or X-API-Key)"}`。

### 本地验证 auth 工作正常

```bash
cd services/ensemble-server

SECRET='test_local_001'
SHA=$(echo -n "$SECRET" | sha256sum | awk '{print $1}')

export ENSEMBLE_JOBS_BASE_DIR=/tmp/ensemble_jobs
export ENSEMBLE_AUTH__BYPASS_VPC=false       # 关闭 VPC bypass 强制走 API Key
export ENSEMBLE_API_KEYS__0__KEY_ID=ek_local
export ENSEMBLE_API_KEYS__0__SECRET_HASH=$SHA
export ENSEMBLE_API_KEYS__0__CUSTOMER_ID=local_dev
mkdir -p $ENSEMBLE_JOBS_BASE_DIR

uv run python -m server &
sleep 2

# 提交 endpoint 没 header → 401
curl -s -o /dev/null -w "no auth: %{http_code}\n" -X POST \
    -H "Content-Type: application/json" -d '{"input":{"sequences":[]}}' \
    http://localhost:9000/v1/folding/ensemble

# 错 secret → 401
curl -s -o /dev/null -w "wrong api key: %{http_code}\n" -X POST \
    -H "X-API-Key: wrong" -H "Content-Type: application/json" -d '{"input":{"sequences":[]}}' \
    http://localhost:9000/v1/folding/ensemble

# 对 secret + 没注册任何方法 → 422 或 503
curl -s -X POST -H "X-API-Key: $SECRET" \
    -H "Content-Type: application/json" \
    -d '{"input":{"sequences":[{"id":"A","sequence":"MKQH"}]}}' \
    http://localhost:9000/v1/folding/ensemble

# 验证 VPC bypass：开启 + 用 localhost host → 没 key 也通
export ENSEMBLE_AUTH__BYPASS_VPC=true
# 重启 server（kill + 重跑）后：
# curl -H "Host: localhost" ... → 不带 X-API-Key 也走通

kill %1
```

### FC 平台层 JWT 验证（可选启用，与应用层不冲突）

FC HTTP 触发器本身支持 JWT 验签作为请求前置处理。如果你希望让 FC 网关在请求到达函数**之前**就过滤无效 token，可在 FC 控制台启用：

- 函数详情 → 触发器 → HTTP 触发器 → JWT 鉴权配置
- JWKS URL：填 `services/jwt/` 的 `.well-known/jwks.json` 地址
- 期望 audience：填 `ensemble-server`

启用后，平台层和应用层会**各自验一遍**（多花 3-5 ms / 请求），但能挡住打到函数的恶意流量。
应用层验证不删除，因为：

1. K8s 迁移时 FC 平台层这一层消失，应用层必须能独立承担
2. 本地开发没有 FC 网关
3. 应用层要从 sub 取出 customer_id 做 quota / 计费
4. VPC bypass 是业务逻辑，平台层不知道我们的策略

详见 [设计文档「为什么应用层仍然要写 JWT 验证逻辑」一节](../../engineering/decisions/2026-06-21-ensemble-server-auth.md)。

### 实战注意事项

- **不要把明文 secret / JWT 提交到 git**。明文 secret 走 1password / Vault / 邮件密文等渠道发给客户；JWT 走客户端代码 env var。
- **每个客户独立凭证**：同一份 secret / JWT key 给两个客户 = 无法独立 quota / 撤销。
- **撤销 API Key（Phase 1 静态机制下）**：FC 控制台删掉对应 `ENSEMBLE_API_KEYS__<N>__*` 项 → 触发函数重新部署。后续 Phase 3 上 Tablestore 后秒级撤销。
- **撤销 JWT**：服务端短期内无 token 黑名单机制，只能等 token 自然过期（建议签发时 `exp` ≤ 24h）。需要紧急撤销时旋转 JWKS kid。
- **轮换 API Key**：先在 env 里追加一组 `__<N+1>__*` 给客户新 secret → 客户切到新的 → 删除旧的 `__<N>__*` → 重新部署。
- **JWT 与 API Key 同时配置时**：客户两个 header 都带，JWT 优先生效；JWT 失败时降级到 API Key（fallthrough）。
- **`PLAN` 和 `MONTHLY_QUOTA_CALLS` 字段当前只读、不强制执行**（Phase 3 才落实计费）—— 但**现在就把语义填对**，将来直接接 Stripe 不用回改。

## 本地开发

### 依赖安装

```bash
cd services/ensemble-server
uv venv
uv pip install -e ../_framework  # bioq-service-framework editable
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

## Docker Swarm 部署（VPC 内自有服务器）

ensemble-server 是 CPU + 长闲置 + 频繁调用的工作负载，FC 上 cold start（60-90s）+ 按调用计费都是反向收益。
搬到 VPC 内自有服务器跑能消掉 cold start、简化调试、并为 Phase 3 的 cache / webhook / 后台 worker 留空间。
迁移代码 0 改动，部署层动作而已 —— 现 stack 文件就在仓内：

- [`compose.swarm.yml`](compose.swarm.yml) — Swarm stack 定义
- [`ensemble-server.env.example`](ensemble-server.env.example) — env 模板（拷成 `ensemble-server.env` 填值，已被根 `.gitignore` 覆盖）

推荐与 FC 并行运行一段时间（FC 作为热备）—— 两边挂同一个 NAS，env 里配同样的 API Key SHA256，就能任一 URL 都能服务。

### 部署步骤

```bash
# 1. 构建 + 推镜像（从仓库根，与 FC 一样）
docker build --platform linux/amd64 \
    -t harbor.ruosheng.bio/aliyun_fc/ensemble-server:v0.0.12 \
    -f services/ensemble-server/Dockerfile .
docker push harbor.ruosheng.bio/aliyun_fc/ensemble-server:v0.0.12

# 2. 在 swarm manager 节点上准备 env 文件 + NAS 路径
cd services/ensemble-server
cp ensemble-server.env.example ensemble-server.env
# 编辑 ensemble-server.env：填 ENSEMBLE_FC_METHODS__*（下游 GPU 服务的 VPC URL）
#                          + ENSEMBLE_API_KEYS__0__SECRET_HASH（与 FC 一致！）
#                          + 可选 ENSEMBLE_AUTH__JWT_JWKS_URL

export NAS_HOST_PATH=/mnt/nas      # host-side NAS 挂载点，必须已挂好
export IMAGE_TAG=v0.0.12

# 3. 部署
docker stack deploy \
    -c compose.swarm.yml \
    --with-registry-auth \
    ensemble

# 4. 验证
docker stack services ensemble
docker service logs -f ensemble_server
curl http://<swarm-node-ip>:9000/v1/healthz
# {"status":"ok","service":"ensemble","version":"0.0.12"}
```

冒烟测试同 [FC 部署后冒烟](#部署后冒烟) 一节，把 `TEST_URL` 换成 `http://<swarm-node-ip>:9000`（或 nginx 前端）即可。

### 拆除

```bash
docker stack rm ensemble
```

NAS 上 `/data/ensemble_jobs/` 内容不会被删（按设计），如需清理另行 `rm -rf`。

### 迁移注意事项

1. **VPC bypass 失效**：bypass 检测 `Host: *-vpc.fcapp.run`，新部署的 hostname 不匹配。原本依赖 bypass 的内部脚本必须改成 `X-API-Key` 或 JWT 调用。FC URL 仍可继续走 bypass。
2. **共享 NAS = 共享 job 可见性**：默认 `ENSEMBLE_JOBS_BASE_DIR=/data/ensemble_jobs` 与 FC 一致，jobs 互通（同一 task_id 在两边都能查）。如要在 bake-in 期间隔离两边，在 `ensemble-server.env` 设 `ENSEMBLE_JOBS_BASE_DIR=/data/ensemble_jobs_swarm`。
3. **API Key 双写**：两边 env 必须配同一组 `ENSEMBLE_API_KEYS__*__SECRET_HASH`，否则同一客户用 FC URL 行、用 swarm URL 401。JWT 同理（共用 JWKS）。
4. **下游 FC URL 用 VPC 版**（`*-vpc.fcapp.run`），不要走公网 URL —— 同 VPC 内调用更快、不消耗公网带宽、不暴露面。
5. **HA**：默认 `replicas=1`，是 SPOF。需 HA 时把 `REPLICAS=2` 并加 nginx / Traefik 反代 + 健康检查；EnsembleJobStore 写 NAS sidecar 是 idempotent、GetAsyncTask 是只读，多副本安全。
6. **不要同时配 FC 控制台的 JWT 平台层验证 + 又把流量切给 swarm** —— swarm 没有 FC 网关层，应用层 JWT 验证已经做完整套，不会有歧义；这条只是提醒别去 FC 控制台改 JWT 配置以为对 swarm 也生效。

### 监控

```bash
# 实时日志
docker service logs -f ensemble_server

# 副本健康 + 启动错误
docker service ps ensemble_server --no-trunc

# NAS 上 sidecar
ls -la /mnt/nas/ensemble_jobs/      # 注意：是 host 路径，不是容器内 /data/...
```

### 升级镜像

```bash
docker build --platform linux/amd64 \
    -t harbor.ruosheng.bio/aliyun_fc/ensemble-server:v0.0.13 \
    -f services/ensemble-server/Dockerfile .
docker push harbor.ruosheng.bio/aliyun_fc/ensemble-server:v0.0.13

IMAGE_TAG=v0.0.13 docker stack deploy \
    -c services/ensemble-server/compose.swarm.yml \
    --with-registry-auth \
    ensemble
# compose.swarm.yml 里 update_config.order=start-first，新 task healthy 之后才停旧的，无停机
```

回滚：

```bash
docker service rollback ensemble_server
```

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
