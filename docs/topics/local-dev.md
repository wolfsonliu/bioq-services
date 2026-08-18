# Local Dev (kind + OpenFaaS)

English | [中文](local-dev.zh.md)

> **Read when**: you run the local end-to-end stack (`make local-*`).
> **Source**: `Makefile` local-* targets, `deploy/openfaas/local-up.sh`, `README.md` local-startup section.
> **Refresh/remove when**: those scripts/targets evolve.

One command brings up the control plane + selected workers in a local kind cluster. Prerequisite:
just `docker`; `kind`/`kubectl`/`helm` auto-download to `$BIOQ_WORKDIR/bin` (default
`~/.cache/bioq-local`).

```bash
make local-up                                   # default service: dockq-server
make local-up LOCAL_SERVICES="dockq-server plip-server"
make local-status / local-logs / local-info / local-test
make local-user ACCOUNT=alice PASSWORD=pw [ADMIN=1]   # create a Keycloak user
make local-down / local-purge
```

## Endpoints & auth

- Gateway port-forwards to `http://127.0.0.1:9000` (`localhost` = VPC bypass, no creds).
- Bundled Keycloak: `http://localhost:8081` (realm `bioq`; master console `admin`/`admin`); api key
  retired. `BYPASS_VPC` in the `Makefile` controls credential-free localhost access (`true` by default).
- Roles come from group membership (`bioq-admins` → admin), provisioned JIT on first login.
- Admin console SSO: `http://127.0.0.1:9000/admin/login` → "Sign in with SSO" → Keycloak.

### OIDC client logins

```bash
# human (device flow)
bioq --gateway-url http://127.0.0.1:9000 login --oidc \
     --issuer http://localhost:8081/realms/bioq --client-id bioq-cli

# machine (client-credentials; create the client first with `make local-svc CLIENT=ci`)
export BIOQ_OIDC_CLIENT_SECRET=ci-secret
bioq --gateway-url http://127.0.0.1:9000 login --client-credentials \
     --issuer http://localhost:8081/realms/bioq --client-id ci
```

`BIOQ_KEYCLOAK=0` disables Keycloak (then only localhost VPC bypass works).

## Redeploying after code changes

`make local-up` **does not rebuild existing images**. After a code change:

```bash
make local-up BIOQ_BUILD=always                 # force rebuild all (worker + gateway), then redeploy
# or, gateway-only, faster: make build-gateway → kind load docker-image gateway:latest → rollout restart
```

## Configurable env

`BIOQ_WORKDIR`, `BIOQ_GATEWAY_PORT`, `BIOQ_CLUSTER`, `BIOQ_BUILD` (`auto|always|never`),
`BIOQ_DB_BACKEND` (`postgres|sqlite`), `BIOQ_KEYCLOAK` (`0` disables), `BIOQ_DOCKERHUB_MIRROR`, etc.
Details: header comments of `deploy/openfaas/local-up.sh`. Weights for local GPU runs go in
`$BIOQ_WORKDIR/shared/models/<svc>/` (mapped to `/data/models`).

## Related

- Gateway internals: [gateway.md](./gateway.md)
- Production image build/push: [build-deploy.md](./build-deploy.md)
