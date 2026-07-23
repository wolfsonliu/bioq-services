# Deploy — 部署到 FC + 控制台配置

日期: 2026-07-14
适用: [新增 bioagent service cookbook](./index.md) 的部署部分
相关: [testing](./testing.md) · 迁移到 OSS mount · [总览](./index.md)

> ← 返回 [新增 service cookbook 总览](./index.md)

本页覆盖部署到 FC 的通用约束、FC 控制台配置（异步任务模式 + OSS mount）、以及并发能力测试。

## 部署到 FC

详见 [services/genie3-server/README.md §阿里云函数计算部署](../../services/genie3-server/README.md) 的步骤；通用约束：

| 项 | 要求 |
|---|---|
| 启动 | `/healthz` 端点 ≤ 120 s 响应（**不要**在 import 时加载模型权重） |
| 监听 | `0.0.0.0:9000` (CAPort) |
| Keep-alive | uvicorn `--timeout-keep-alive 900` |
| 镜像大小 | GPU 镜像 ≤ 15 GB（权重外置后通常 1.5-5 GB） |
| 架构 | `--platform linux/amd64` |
| NAS 挂载（jobs） | `/fc → /data`，跨实例 / 跨服务共享 job 文件 |
| NAS 挂载（权重） | `/fc → /data/models/<svc>`，**只读**；首次部署前需先上传权重 |

**首次部署权重上传**（一次性）：

```bash
# 1. 本地下载到 stage 目录
./services/<svc>/scripts/fetch_weights.sh

# 2. rsync 到 NAS（路径与 settings.weights_dir default 一致）
rsync -av services/<svc>/weights/ <NAS-mount>:/data/models/<svc>/

# 3. 部署后验证
curl https://fc-<svc>-XXX.cn-hangzhou-vpc.fcapp.run/healthz/detail
# 期望：{"status":"ok", "weights_loaded": true, "weights_missing": {}}
```

Push 到 harbor + 更新 FC 函数镜像：

```bash
make push-<svc>                # build + tag + push (用 VERSION 文件里的版本)
# 然后到 FC 控制台把函数镜像更新到 harbor.ruosheng.bio/aliyun_fc/<svc>:vX.Y.Z
```

## FC 异步任务模式控制台配置

每个 service 镜像部署到 FC 后，控制台需要做以下设置才能让 task endpoint 真正按异步任务模式工作：

| 配置项 | 推荐值 | 说明 |
|---|---|---|
| 异步任务模式 | **启用** | 解锁 `X-Fc-Invocation-Type: Async` + `GetAsyncTask` + 去重 |
| 最大异步任务并发数 | 按 GPU 配额（如 5-10） | 超出自动排队，告别 429 |
| 异步任务重试次数 | 0 | 失败需要人工排查，避免烧 GPU 配额 |
| 单实例并发 | 1（GPU service 默认） | task endpoint 阻塞期间独占 GPU |
| Session Affinity | 保留 HeaderField 配置 | legacy submit/poll 仍需要；task endpoint 不依赖但不冲突 |
| Keepalive URL | **清空** | task endpoint HTTP 全程活跃，不再需要外部 keepalive |
| PreStop hook | 保留 | 实例销毁兜底，标记 running job 为 interrupted |
| 函数 timeout | 86400s | GPU 实例最长可达 24h |
| **OSS 挂载** | 数据面 bucket `bioagent-inputs` → `/mnt/oss`，**读写(RW)** | 经 gateway 调用必备：输入直读（gateway 把 `oss://` 输入改写成 `/mnt/oss/...`）+ 输出回传（output-sink 把 job dir 镜像到 `/mnt/oss`）。缺挂载则输入需下游有 OSS 凭证、download 回退到下游代理 |

启用方式：**函数详情页 → 异步配置 → 编辑「任务模式」**；OSS 挂载在 **函数详情页 → 配置 → 存储 → NAS/OSS 挂载**。

部署后 smoke 验证：

```bash
# 1. 同步调用（legacy）仍应 200 立即返回
curl -X POST https://fc-<svc>-XXX.cn-hangzhou-vpc.fcapp.run/api/<name> ...

# 2. 异步调用 task endpoint 应该 202 Accepted（不是 200）
curl -i -X POST https://fc-<svc>-XXX.cn-hangzhou-vpc.fcapp.run/api/tasks/<name> \
    -H "X-Fc-Invocation-Type: Async" \
    -H "X-Bioagent-Job-Id: smoke-001" \
    -F "..."
# 期望首行：HTTP/1.1 202 Accepted

# 3. 同步 GET 拉 JobInfo（带 affinity 减少 polling instance）
curl https://fc-<svc>-XXX.cn-hangzhou-vpc.fcapp.run/api/jobs/smoke-001 \
    -H "X-Bioagent-Session-Id: smoke-001"
```

如果第 2 步返回 200 而不是 202，说明控制台异步任务模式没启用，或 `X-Fc-Invocation-Type` header 没被网关识别。

上述 smoke 命令都覆盖在 [`tests/test_fc_task.py`](#12b-servicessvctestsstest_fc_taskpy) 里 ——
控制台配置改完之后跑一遍 `pytest -m fc services/<svc>/tests/test_fc_task.py` 就是完整回归。

详细操作手册见 fc-gpu-instance-keepalive.md。
