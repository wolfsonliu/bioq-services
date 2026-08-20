# describe 冷启动与静态自描述契约 设计

日期: 2026-08-20
状态: 方案（待实现）
适用: 网关 `/v1/services/{svc}`（describe）路径 · `gateway/discover.py`
`Discovery` · framework 的 manifest 物化入口 · build/release 物化管线与 CI 门禁
范围外: `bioq` CLI 仓库（client 超时 / `--wait` / `--output json` 对齐，见
文末「范围外 / 交底」）

## 概述

`bioq describe <svc>` 走网关 `GET /v1/services/{svc}`，当前实现会**实时
向下游容器抓取** `/api/manifest` 与 `/openapi.json` 两个端点再合并返回
（`gateway/discover.py`）。在 FC scale-to-zero 的服务族里，这一抓会触发下游
冷启动（镜像拉取 + 启动 + 重型 conda/torch import），叠加"单客户端 60s 超时 +
两次串行调用 + 失败不缓存"三者，导致两类已观测故障：

1. **挂死**（alphafold / diffdock / rfantibody）：下游冷启动 >60s，CLI 先超时
   放弃，且失败结果不进缓存，重试永远重触发冷启动——无暖路径的死循环。
2. **空契约**（esmfold2）：下游 `/api/manifest` + `/openapi.json` 本就
   404/空，是服务自身「框架自描述落地缺口」，与延迟无关。

本设计的根修是**把自描述契约变成注册服务的静态属性**：在 release 期把每个
服务的 manifest + openapi 一次性物化为 JSON、随网关镜像下发；`describe` 变成
内存/磁盘读取，冷启动免疫。作为过渡护栏，先加固 `Discovery`（限时超时 +
单飞 + 失败分类 + manifest/openapi 解耦），保证在静态契约尚未覆盖的服务上
也不会再挂死或静默返回空。

## 根因（已对照代码核实）

| 事实 | 位置 | 说明 |
|---|---|---|
| 下游超时 60s，两次串行 → 最坏 ~120s | `discover.py:14,28-29` | 单个 `httpx.Client(timeout=60)` |
| CLI 客户端超时 60s | `bioq/client.py:75-77`（跨仓库） | 总是输给网关最坏 ~120s |
| 仅全量成功才缓存 | `discover.py:34` `if manifest and openapi` | 失败不缓存 → 重试风暴 |
| 所有异常塌缩为 `{}` | `discover.py:38-44` `_get_json` | 404 / timeout / 5xx 无法区分 |
| 缓存 TTL 300s | `app.py:59` | 硬编码，未接入 settings |
| 线程池抬到 100 | `app.py:72-78`, `settings.py:92` | 全部 /v1 handler 是 sync def，冷启动突发会打满线程池 |
| `ServiceRecord` 无 manifest/openapi 字段 | `framework/.../service_registry.py:30-40` | `extra="forbid"` |

关键洞察：`framework/.../manifest.py:build_manifest()` 是**源码的确定性函数**
——只依赖 `app.openapi()`、`app.routes`、`adapter.endpoint_examples()` /
`manifest_extras()`，以及 `settings.jobs_base_dir`（仅落在 `nas_layout` 的一个
值）。框架把工具执行放在 subprocess，端点模块通常只 import pydantic 模型。
因此静态 manifest **不必跑镜像**：在 release 期 import 服务的 app 模块调用
`build_manifest()` + `app.openapi()` 即可 dump，比「跑打包镜像抓两个 endpoint」
更便宜、更不易翻车。

## 设计目标

1. **`describe` 冷启动免疫**：命中静态契约时零下游调用；未命中时也要在
   ~8s 内返回（成功或可行动的 warming 哨兵），绝不 60s 裸超时。
2. **契约是静态属性**：manifest/openapi 随服务 release 物化、随网关下发，
   manifest 内版本与服务 `VERSION` 绑定。
3. **失败可读**：把 404 / timeout / 5xx 分离开，给人类与 agent 一个可行动的
   失败分类（`ok / warming / partial / no_manifest / error`）。
4. **不破坏现有契约**：`--output json` 仍返回 manifest+openapi；pretty 视图
   仍只读 `manifest.endpoints`。
5. **低风险、可分行交付**：护栏阶段不改 schema、不改 services.yaml、
   不碰 CLI 也能单独上线。

## 方案总览（分阶段）

### 阶段一 — 加固实时发现（护栏，无 schema 改动）

只改 `gateway/discover.py` + `gateway/settings.py` + `gateway/app.py`：

- **限时超时接入 settings**（修隐藏 bug）：`Discovery` 不再硬编码 60s。
- **单飞合并**：同一 `svc` 的并发 describe 共享一次下游抓取。
- **失败分类 + 负缓存**：不再塌缩成 `{}`，返回 `status`/`detail`；对
  `warming`/`error` 做短 TTL 负缓存，避免重试风暴重新触发冷启动。
- **manifest/openapi 解耦**：manifest-first，manifest 失败即短接、不再抓
  openapi。

### 阶段二 — 静态契约根修

- **物化管线**：每个服务 release 时用 framework 提供的 `dump-manifest` 入口
  生成 `manifests/<svc>.manifest.json` + `manifests/<svc>.openapi.json`，提交
  进仓库；网关镜像 COPY `manifests/` 与 `services.yaml` 相邻。
- **describe 优先读静态**：`describe_service` 命中静态文件即返回（source=registry，
  零下游）；未命中回退阶段一的 live 路径（source=live）。
- **CI 门禁**：`make check-manifests` 强约束「每个已部署条目都有 manifest」且
  「重新生成 diff 为空」，同时抓出 esmfold2 式空契约与「静态契约过期」两件事。

## 详细设计

### 1. manifests/ 物化与命名约定

静态契约以**旁挂文件 + 命名约定**发现，**不改 `ServiceRecord`、不改
`services.yaml`**（相对调研笔记的方案 A 的"扩展 ServiceRecord 字段"更简单：
零 schema 迁移、零 `extra="forbid"` 升级摩擦、服务仍各自独立 release）。

```
<repo-root>/manifests/
  <svc>.manifest.json     # build_manifest() 的 JSON 序列化，按 registry key 命名
  <svc>.openapi.json      # app.openapi() 的 JSON 序列化
```

- 文件名用 **registry key**（网关 `describe` 的入参，总是 `*-server` 形式；
  CLI 已把用户短名 `alphafold` 补成 `alphafold-server`）。manifest 内部的
  `service` 字段（`adapter.name`，如 `alphafold`）与 registry key 不一定相等，
  由 CI 门禁校验一致性。
- 文件存在 = 该服务有静态契约；缺失 = 回退 live（阶段一路径）。同一份
  `services.yaml` 不加任何字段，向后兼容。
- 物化入口（repo 脚本 `scripts/gen_manifests.py`，复用 framework 既有
  `build_manifest()`，**零 framework 改动、零 per-service 文件改动**）：
  orchestrator 读 `services.yaml`，对每个服务 `uv run --project
  services/<svc>-server` 进入其 venv，把服务目录注册为 `server` 包后 import
  `server.app`，调 `build_manifest()` + `app.openapi()` 写两个 JSON。
  详见实施计划 `docs/plans/2026-08-20-describe-cold-start-static-manifest.md`。
- 在服务自己的完整 venv 里跑（release 主机已装齐依赖），即便个别服务在
  模块顶层 import torch 也能工作——这是一次性的离线步骤，不是按需冷启动。

Makefile 新增：

```
make gen-manifests      # 遍历 services.yaml，uv run → manifests/<svc>.(manifest|openapi).json
make check-manifests    # 重新生成到临时目录后 diff，任一差异或缺失即 fail（CI 门禁）
```

`check-manifests` 两项强约束（见「成功标准」）：
1. `scripts/` 遍历 `services.yaml` 的每个已部署条目，断言
   `manifests/<svc>.manifest.json` 与 `manifests/<svc>.openapi.json` 存在
   （抓 esmfold2 式缺口）；
2. 对每个服务重新 dump 并 diff 已提交文件（抓 VERSION 提升 / endpoint 变更后
   未重建的漂移——manifest 的 `version` 字段即 `VERSION`，天然绑定）。

### 2. 网关 describe_service 路径（app.py）

```
GET /v1/services/{svc}
  1. rec = registry.record(svc)                       # 未知服务 404 不变
  2. 若 manifests/<svc>.manifest.json 存在（内存缓存）:
       返回 {service, manifest, openapi, status:"ok", source:"registry"}
       openapi 读 manifests/<svc>.openapi.json；文件缺失则置 {}
       （均为磁盘/内存读，快）
  3. 否则回退:
       base = dispatch.describe_base_url(rec)
       返回 discover.describe(svc, base)                        # source:"live"
```

- 静态路径是磁盘/内存读，恒快；仍走 sync handler 即可（不引入 async 迁移）。
- `source` 字段让调用方分辨「registry 静态」vs「live 抓取」，便于 E6/E12 观测。

新增依赖：`ServiceRegistry`（`gateway/registry.py`）增加
`manifest(svc) -> dict | None` 与 `openapi(svc) -> dict | None`，按
`manifests_dir`（默认 `settings.registry_path` 的同级 `manifests/`）解析 + 读 +
进程内缓存；文件缺失/解析失败返回 None（回退 live），告警日志。

### 3. Discovery 加固（阶段一，discover.py）

**_get_json 改为返回结构化结果**：`(outcome, payload)`，
`outcome ∈ {ok, not_found, timeout, error}`，`httpx.TimeoutException` 单独归为
`timeout`，`HTTPStatusError.code==404` 归为 `not_found`，其余异常归 `error`。

**失败分类 → 载荷形状**（无 `response_model`，纯增量、向后兼容）：

| 下游结果 | `status` | `detail` 语义 |
|---|---|---|
| manifest(+openapi) 200 有效 | `ok` | — |
| 静态契约命中 | `ok` | `source:"registry"` |
| 冷启动（超时 / 502 / 504） / 连接错 | `warming` | "downstream cold-start timed out; retry in ~15s" |
| 404（framework 端点缺失） | `no_manifest` | 指向「服务未落地框架自描述」 |
| manifest 有效、openapi 缺失 | `partial` | openapi 不影响人读/CLI 路径 |
| 其它异常（其余 5xx / JSON 解析失败 / 未预期） | `error` | 含 `type(e).__name__` + message |

**负缓存**：`warming`/`error` 以短 TTL（`discovery_negative_ttl_sec`，默认 15s）
缓存哨兵，使重试突发不再重新触发冷启动；`ok` 以 `discovery_ttl_sec`（默认 300s）
缓存；`no_manifest` 不缓存（可重试，且阶段二 CI 会根除）。保留现有
`test_describe_does_not_cache_partial_failure` 的语义（部分失败不按成功缓存）。

**单飞合并**：每 `svc` 一把 `threading.Lock`（handler 是 sync、跑线程池）。
首个调用者取锁抓取并写缓存，后续调用者阻塞等待后读缓存。锁持有时间被限时
超时限制在 ~8s 内，不会长期占线程。**局限**：仅进程内合并——网关本身也是
FC 多实例，跨实例去重需共享存储（阶段一不做：短超时 + 负缓存已把爆炸半径
压得很小）。

**manifest/openapi 解耦（manifest-first + 失败短接）**：
`describe(svc, base_url)` 先抓 `GET /api/manifest`；manifest 一旦失败，**立即
返回哨兵（openapi 置 {}），不再发起第二次抓取**——冷启动最坏路径从两次串行
变成一次快失败。manifest 成功（实例已暖）才补抓 `/openapi.json`；此时同实例
内 openapi 也快，只需一个 read 超时兜底（`app.openapi()` 首次生成可能较慢）。

### 4. 配置（settings.py 新增，均设默认、可选）

| 字段 | 默认 | 说明 |
|---|---|---|
| `discovery_ttl_sec` | `300` | 成功缓存 TTL（现 `app.py:59` 硬编码） |
| `discovery_negative_ttl_sec` | `15` | warming/error 哨兵负缓存 TTL |
| `discovery_read_timeout_sec` | `8` | 下游 read 超时（冷启动绑定项） |
| `discovery_connect_timeout_sec` | `5` | 下游 connect 超时（网络病态） |

`Discovery` 构造改用 `httpx.Timeout(connect=…, read=…)`，不再用默认 60s。
describe 预算 ≈ read 超时（~8s），满足「≤8s 返回或明确哨兵」的验收。

### 5. 载荷契约

```json
{
  "service": "alphafold-server",
  "manifest": { "...": "build_manifest() 结果" },
  "openapi":  { "...": "manifest 成功或静态命中时出现；否则 {}" },
  "status": "ok | warming | partial | no_manifest | error",
  "source": "registry | live",
  "detail": "非 ok 时的单行可行动提示"
}
```

向后兼容：`manifest`/`openapi` 键与 `service` 不变，纯增加 `status`/`source`/
`detail`；CLI `--output json` 原样 dump 不受影响。

## 测试策略

| 层 | 文件 | 说明 |
|---|---|---|
| gateway discovery | `gateway/tests/test_discover.py` | 扩充：失败分类（404→no_manifest、timeout→warming、5xx→error）；负缓存命中不重发；manifest 失败短接、不再发 openapi 请求；单飞下 N 并发仅 1 次下游抓取（`threading` 并发 + `MockTransport` 计数） |
| gateway app | `gateway/tests/test_app.py` | `describe_service`：静态命中（source=registry、零下游调用）/ 回退 live（source=live）/ openapi 缺失降级为空 |
| gateway registry | `gateway/tests/test_registry.py` | `manifests_dir` 约定解析 + `manifest()/openapi()` 读取与缓存；文件缺失返回 None |
| framework dump | `framework/tests/test_manifest.py`（或新增 `test_dump_manifest.py`） | `dump_manifest` 写两个 JSON；manifest 含 `version`；openapi 含 `paths` |
| CI 门禁 | `make check-manifests` | 缺失 manifest 或 diff 非空即 fail（本地 gateway/框架测试之外的门禁） |

现有 `test_discover.py` 需随新语义调整（`Discovery` 构造新增
`negative_ttl_sec` / `connect_timeout_sec` / `read_timeout_sec`，默认 15/5/8）：
- `test_describe_does_not_cache_partial_failure`：旧语义「manifest 失败仍抓
  openapi」改为「manifest 成功 + openapi 失败 → `partial` 且不缓存」。
- `test_describe_does_not_cache_total_failure`：失败路径现在进负缓存；该测试
  显式 `negative_ttl_sec=0` 关负缓存（或改写为断言「短 TTL 内不重发」）。
- `test_describe_merges_manifest_and_openapi` / `test_describe_cached` /
  `test_describe_ttl_expiry_refetches` / `test_describe_errors_degrade`：
  语义与断言不变（皆 200，或单次 describe 不触发负缓存）。

## 成功标准

- `time bioq --profile ecs describe alphafold` 与 `... diffdock` 在 ≤ ~8s 内返回
  （成功 manifest 或明确 `warming` 哨兵），绝不 60s 裸超时（对应调研笔记 §6）。
- 阶段二后：`describe` 对 `services.yaml` 里**每个**条目返回完整契约——包括
  esmfold2（`make check-manifests` 使「缺 manifest」成为硬失败，esmfold2 必须先
  修好框架自描述才能过门禁）。
- N 个并发 describe 对一个冷服务触发 ≤1 次下游抓取（单飞，阶段一）。
- `--output json` 仍返回 openapi（阶段二静态 / 阶段一 manifest 成功后抓得），
  冷启动或缺失处降级为 `{}`（不报错）。

## 风险 / 限制

- **静态契约过期漂移**是本方案最大风险：若 release 后不重建 manifest，契约会
  静默变旧——正是要消灭的「agent 信错 schema」失败模式。由 `check-manifests`
  的「重新生成 diff」强约束兜底（manifest 内 `version`=VERSION）。
- **`jobs_base_dir` 被 bake 进静态 manifest**：`nas_layout.jobs_base_dir` 来自
  `ServiceSettings`（CI 里用确定性默认值生成）。它是信息性的（告诉 agent 输出
  落在 NAS 何处），运行时一般不随部署漂移；若未来出现部署相关取值，改为模板
  或生成时剥离。
- **单飞仅进程内**：跨网关实例的并发仍可能各自触发下游抓取；短超时 + 负缓存
  已把爆炸半径压小，v1 不上共享锁。
- **物化依赖服务源码可 import**：个别服务模块顶层 import 重型库会拉高 dump
  成本，但只发生在 release（离线、依赖齐全），不影响运行时。

## 范围外 / 交底

- **`bioq` CLI 仓库（跨仓库，仅交底，不在本仓库改）**：
  - client 超时与 `status/detail` 解析：长于网关最坏读超时（如 30s），读到
    `status`/`detail` 时打印 `"服务冷启动中，约 Ns 后重试"` 而非裸
    `ReadTimeout`（调研笔记 §C）。
  - 现有 `--wait`/`--timeout` 冷启动容忍已存在，无需动。
- **esmfold2-server 框架自描述落地**：独立任务；CI 门禁会在其缺失时暴露。
- **provisioned concurrency / 定期 warmer**：静态契约已移除对暖路径的依赖，
  本方案不做；`ServiceRecord.tier` 的 warm/hot 语义留给部署侧。

## 开放问题

- `openapi.json` 的消费方是否只有 `--output json`（以及未来的 MCP 工具）？
  若不是关键消费方，可在后续把 openapi 从 describe 载荷中移除、仅保留
  manifest。
- 物化是否纳入 `make bump-<svc>` 的同一提交/同一发布流程（推荐），还是作为
  独立 `make gen-manifests` 由 release 流水线显式调用？

## Sources

- 现状实现：`gateway/app.py`（`describe_service` / `_raise_thread_pool_limit`）、
  `gateway/discover.py`（`Discovery`）、`gateway/settings.py`、
  `framework/src/bioq_service/manifest.py`（`build_manifest`）、
  `framework/src/bioq_service/service_registry.py`（`ServiceRecord`）、
  `framework/src/bioq_service/adapter.py`（`endpoint_examples` /
  `manifest_extras`）、`gateway/Dockerfile`（`COPY services.yaml`）。
- 触发调研：团队 live 笔记 `bioq-services-describe-optimization.md`
  （E6 agent-drivability 实验观察，alphafold/diffdock 挂死、esmfold2 空契约）。
- CLI 侧证据（跨仓库，仅核对未改动）：`bioq/bioq/commands.py`
  （`_print_describe_cli` 只读 `manifest.endpoints`；`cmd_describe` json 路径
  `emit(info, fmt="json")`）、`bioq/bioq/client.py`（`describe`）、
  `bioq/docs/specs/2026-08-20-diffdock-cold-start-describe-design.md`
  （`--wait` 冷启动容忍）。