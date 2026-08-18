# Gateway（控制面）

[English](gateway.md) | 中文

> **适用**：改动 `gateway/` 下任何内容（认证、调度、存储、schema、配置）时。
> **来源**：`gateway/app.py`、`gateway/settings.py`、`gateway/dispatchers/`、`gateway/README.md`。
> **刷新/删除条件**：gateway 端点、后端或环境变量变化时。

`gateway/` 是常驻 API gateway（ECS / compose / kind），前接下游 FC/HTTP/OpenFaaS 服务。入口
`python -m server`（容器内）/ `python -m gateway`（仓库内）。FastAPI app 在 `gateway/app.py`；配置
schema 单一来源 `gateway/settings.py`（`GatewaySettings`，`env_prefix="GATEWAY_"`，嵌套用
`GATEWAY_AUTH__...`）。

## 端点（`/v1/*`）

`GET /v1/services`、`GET /v1/services/{svc}`、`POST /v1/run/{svc}/{endpoint}`、
`GET /v1/jobs/{job_id}`、`GET /v1/jobs/{job_id}/download`、`POST /v1/jobs/{job_id}/cancel`、
`POST /v1/uploads/prepare`、`GET /healthz`。

## 认证

两层：VPC bypass（localhost / 内网 break-glass）→ OIDC/JWT（`Authorization: Bearer`）。api key 已
退役；人类走 OIDC device flow / SSO，机器走 OIDC client-credentials。用户 JIT 落库，role 由 token
的 groups claim 推导（`GATEWAY_AUTH__JWT_ADMIN_GROUP`，默认 `bioq-admins` → admin）。生产必须设
`GATEWAY_AUTH__JWT_ISSUER` 且 `bypass_vpc=false`。管理控制台在 `/admin`（SSO，CSRF）。

## 调度后端

`GATEWAY_DISPATCH_BACKEND` = `fc`（阿里云 FC 异步任务模式，经 FC OpenAPI `GetAsyncTask` 轮询，
AK/SK 来自 `ALI_AK`/`ALI_SK`）/ `http`（compose，直接 submit/poll 各 service 的 in-process runner）
/ `openfaas`（kind + OpenFaaS）。dispatcher 协议在 `gateway/dispatchers/`（`base.py` + `fc.py` /
`local.py` / `openfaas.py`）。

## 存储

`GATEWAY_STORAGE_BACKEND` = `oss`（presign 直传，默认 bucket `bioagent-inputs`）或 `file`
（`/v1/files` 走共享卷 `GATEWAY_FILE_BASE_DIR`）。

## 数据库 / 迁移

用户+job 走 SQLAlchemy；schema 由 **Alembic** 管（非 `create_all()`）。容器 entrypoint 先
`alembic upgrade head` 再 uvicorn。本地 sqlite（`GATEWAY_DB_URL=sqlite:///<path>`），生产 Postgres
（`postgresql+psycopg://...`）。加 schema 变更：

```bash
cd gateway && GATEWAY_DB_URL=sqlite:///$PWD/gw.db uv run alembic revision --autogenerate -m "<change>"
```

## 部署配置

非密拓扑在 checked-in 的 `deploy/config/gateway.<target>.env`（`make gen-config` 从 schema 生成；
`make check-config` 是 CI drift 门禁），secrets 只在各 target 的 gitignored `.env`。gateway 启动时
校验配置，致命误配拒绝 boot。

## 相关

- 本地 kind 部署与 Keycloak：[local-dev.md](./local-dev.md)
- 构建/打 tag gateway 镜像：[build-deploy.md](./build-deploy.md)
