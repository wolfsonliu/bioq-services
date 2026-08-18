# 本地联调（kind + OpenFaaS）

[English](local-dev.md) | 中文

> **适用**：跑本地端到端联调栈（`make local-*`）时。
> **来源**：`Makefile` 的 local-* 目标、`deploy/openfaas/local-up.sh`、`README.md` 本地启动章节。
> **刷新/删除条件**：这些脚本/目标演进时。

一条命令把控制面 + 选定 worker 拉起在本地 kind 集群。前置只需 `docker`；`kind`/`kubectl`/`helm`
自动下载到 `$BIOQ_WORKDIR/bin`（默认 `~/.cache/bioq-local`）。

```bash
make local-up                                   # 默认 service：dockq-server
make local-up LOCAL_SERVICES="dockq-server plip-server"
make local-status / local-logs / local-info / local-test
make local-user ACCOUNT=alice PASSWORD=pw [ADMIN=1]   # 建 Keycloak 用户
make local-down / local-purge
```

## 端点与认证

- gateway 端口转发到 `http://127.0.0.1:9000`（`localhost` = VPC bypass，免凭据）。
- 内置 Keycloak：`http://localhost:8081`（realm `bioq`；master 控制台 `admin`/`admin`）；api key 已
  退役。`Makefile` 的 `BYPASS_VPC` 控制 localhost 免凭据访问（默认 `true`）。
- role 由组成员推导（`bioq-admins` → admin），首次登录 JIT 落库。
- 管理控制台 SSO：`http://127.0.0.1:9000/admin/login` → "Sign in with SSO" → Keycloak。

### OIDC 客户端登录

```bash
# 人类（device flow）
bioq --gateway-url http://127.0.0.1:9000 login --oidc \
     --issuer http://localhost:8081/realms/bioq --client-id bioq-cli

# 机器（client-credentials；先 `make local-svc CLIENT=ci` 建客户端）
export BIOQ_OIDC_CLIENT_SECRET=ci-secret
bioq --gateway-url http://127.0.0.1:9000 login --client-credentials \
     --issuer http://localhost:8081/realms/bioq --client-id ci
```

`BIOQ_KEYCLOAK=0` 禁用 Keycloak（此后只剩 localhost VPC bypass）。

## 改代码后重新部署

`make local-up` **不重建已存在镜像**。改代码后：

```bash
make local-up BIOQ_BUILD=always                 # 强制重建全部（worker + gateway）再部署
# 或只更新 gateway（更快）：make build-gateway → kind load docker-image gateway:latest → rollout restart
```

## 可调环境变量

`BIOQ_WORKDIR`、`BIOQ_GATEWAY_PORT`、`BIOQ_CLUSTER`、`BIOQ_BUILD`（`auto|always|never`）、
`BIOQ_DB_BACKEND`（`postgres|sqlite`）、`BIOQ_KEYCLOAK`（`0` 禁用）、`BIOQ_DOCKERHUB_MIRROR` 等。
细节见 `deploy/openfaas/local-up.sh` 头部注释。本地 GPU 运行的权重放
`$BIOQ_WORKDIR/shared/models/<svc>/`（映射到 `/data/models`）。

## 相关

- gateway 内部：[gateway.md](./gateway.md)
- 生产镜像构建/推送：[build-deploy.md](./build-deploy.md)
