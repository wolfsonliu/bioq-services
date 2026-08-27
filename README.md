# <img src="assets/bioq.svg" alt="bioq logo" width="40"> bioq-services

English | [中文](README.zh.md)

An AI drug discovery (AIDD) algorithm-service fleet plus a shared service framework. Each
`services/<name>-server/` packages a third-party bioinformatics / AIDD tool into a **dual-mode
Docker image**:

- **HTTP mode** (default): `uvicorn server.app:app` — FastAPI + an async job runner, deployed to
  Alibaba Cloud FC.
- **CLI batch mode**: `python -m server <endpoint> ...` — one-shot synchronous execution for
  Slurm/sbatch.

## Structure

Organized by responsibility:

```
framework/             — Shared service framework (a library, not a service — no Dockerfile); PyPI distribution name
│                        bioq-service-framework, import name bioq_service (JobAdapter / SubprocessRunner /
│                        CLIEndpoint / uris, etc.)
gateway/               — Control plane: auth / upload negotiation / FC async dispatch (ECS, docker-compose
│                        deployment; see deploy/ecs/)
edge/                  — Non-worker edge components
├── jwt/               — JWT signing helpers
└── protein-design-mcp/ — MCP protocol adapter
services/              — Compute workers (each is an FC/OpenFaaS function image, keeping the -server suffix)
└── <name>-server/
    ├── server package (app.py / models.py / tools.py / adapter.py / settings.py ...)
    ├── __main__.py    — CLI batch entry point
    ├── tests/         — offline unit tests (mock subprocess) + FC integration tests (@pytest.mark.fc, opt-in)
    ├── pyproject.toml — offline test/dev environment declaration (see below)
    ├── Dockerfile / VERSION
    └── deploy/        — platform deployment descriptors (fc.yaml; later openfaas.yaml / k8s.yaml)
services.yaml          — fleet registry (per-service url / tier / function / classification tag)
```

Each service's **heavy runtime dependencies (torch/cuda/conda/rdkit, etc.) live in its own
Dockerfile** — they are mutually exclusive, so they cannot share a single environment (which is
precisely why each service gets its own Dockerfile). `pyproject.toml` declares only the lightweight
pip dependencies required for **offline testing** plus the framework path dependency.

## Depending on the framework

Each service depends on the framework via a relative path — no publishing required:

```toml
[tool.uv.sources]
bioq-service-framework = { path = "../../framework", editable = true }
```

(`gateway/` sits at the top level, so it uses `path = "../framework"`.)

## Testing (offline)

Each service runs in isolation, independent of the others:

```bash
cd services/<name>-server
uv run --group dev python -m pytest tests/ -q          # offline unit tests
# FC integration tests (require a deployed service):
RUN_FC_TESTS=1 uv run --group dev python -m pytest -m fc tests/ -v
```

For the framework itself:

```bash
cd framework && uv run --extra dev python -m pytest tests/ -q
```

> A few services' tests read the vendored `upstream/` (git-ignored, not committed). Those services
> need to run their own `vendor.sh` to fetch the upstream source first, otherwise the related tests
> fail due to missing files.

## Building / pushing images

The `Makefile` auto-discovers across layers (`services/*/Dockerfile` + `gateway/Dockerfile` +
`edge/*/Dockerfile`; `framework/` has no Dockerfile so it isn't discovered), and each service uses
its own `VERSION` as the image tag (each releases independently). Image name = the last directory
segment (workers keep `-server`, so live FC references stay unchanged):

```bash
make list                       # list discovered services
make build-<service>            # build a single image (using its VERSION)
make build-<svc> TAG=v0.0.5     # override the tag
make push-<service>             # build + tag + push to harbor (REGISTRY overridable)
make bump-<service>             # patch version +1
make sif-<service>              # Docker → Apptainer SIF (HPC/Slurm)
```

The image build context is the repo root (`docker build -f <svc-dir>/Dockerfile .`, where
`<svc-dir>` is resolved across layers by name in the Makefile), and `.dockerignore` has been
trimmed accordingly.

## Local startup (kind + OpenFaaS)

One command brings up the whole control plane + the selected compute workers in a local kind
cluster for end-to-end integration.

**Prerequisites**: just `docker`; `kind` / `kubectl` / `helm` are auto-downloaded to
`$BIOQ_WORKDIR/bin` (default `~/.cache/bioq-local`). All state (kubeconfig, downloaded tools,
`gateway.db` + job dirs in the shared volume) lives under `BIOQ_WORKDIR`.

```bash
make local-up                                   # start the default service (dockq-server)
make local-up LOCAL_SERVICES="dockq-server plip-server"   # specify which workers to start
```

`local-up` idempotently brings up: kind cluster → OpenFaaS → bundled PostgreSQL → bioq gateway →
selected workers, and port-forwards the gateway to **`http://127.0.0.1:9000`** (default API key
`bioq-local-secret`).

### Common commands

```bash
make local-status              # view pods / services
make local-logs LOCAL_SVC=gateway   # tail logs (LOCAL_SVC=gateway for the gateway; defaults to dockq-server)
make local-info                # print gateway URL / API key / kubeconfig / shared directory
make local-forward             # re-establish the port-forward if it dropped
make local-test                # run the dockq functional tests against the local deployment
make local-user ACCOUNT=alice PASSWORD=pw          # create a normal user in Keycloak
make local-user ACCOUNT=root  PASSWORD=pw ADMIN=1  # create an admin (adds to the bioq-admins group)
make local-users               # list Keycloak users + bioq-admins members
make local-svc CLIENT=ci [ADMIN=1]  # create/rotate a machine account (client-credentials; secret defaults to <client>-secret)
make local-svcs                # list service-account clients
make local-down                # tear down the deployment (keeps BIOQ_WORKDIR state)
make local-purge               # tear down and wipe BIOQ_WORKDIR
```

### Authentication (Keycloak / OIDC)

The local deployment bundles **Keycloak** (realm `bioq`); auth uses OIDC/JWT (the API key has been
retired). With `BYPASS_VPC=true`: localhost is credential-free (break-glass); a non-localhost Host
requires an OIDC token.
- **Keycloak**: `http://localhost:8081` (master console `admin`/`admin`; bootstrap app admin
  `admin`/`admin` in realm `bioq`, group `bioq-admins`).
- **Users/permissions**: `make local-user ... [ADMIN=1]` creates users via kcadm; roles are
  determined by **group** (`bioq-admins` → `role=admin` in the gateway, provisioned JIT on first
  login).
- **Admin console SSO**: open `http://127.0.0.1:9000/admin/login` → "Sign in with SSO" → Keycloak
  login (admin/admin) → back to the console (localhost can also enter directly without login).
- **bioq CLI** (human, device flow):
  ```bash
  bioq --gateway-url http://127.0.0.1:9000 login --oidc \
       --issuer http://localhost:8081/realms/bioq --client-id bioq-cli
  bioq services       # with a Bearer JWT → gateway validates (JWKS in-cluster / issuer=localhost:8081)
  ```
- **Machine/CI** (client-credentials, service account): `make local-svc CLIENT=ci [ADMIN=1]` creates
  a confidential client, then
  ```bash
  export BIOQ_OIDC_CLIENT_SECRET=ci-secret        # default <client>-secret; never write it into config files
  bioq --gateway-url http://127.0.0.1:9000 login --client-credentials \
       --issuer http://localhost:8081/realms/bioq --client-id ci
  bioq services       # exchanges client_id+secret for a fresh token each time (unattended)
  ```
  The realm ships with a sample `bioq-svc` (regular permissions, secret `bioq-svc-secret`) that can
  be used directly.
- `BIOQ_KEYCLOAK=0` disables Keycloak (then only the localhost VPC bypass is available, no OIDC).

> Key mechanism: Keycloak uses `KC_HOSTNAME=http://localhost:8081` (the frontend issuer reachable
> from the browser/bioq) + `KC_HOSTNAME_BACKCHANNEL_DYNAMIC=true` (the gateway pod fetches
> token/jwks over the cluster DNS); one issuer, two reachability paths.

### Redeploying after code changes

`make local-up` **does not rebuild existing images** (it reuses the local `<svc>:latest` /
`gateway:latest`). To make code changes take effect:

```bash
make local-up BIOQ_BUILD=always            # force-rebuild all images (worker + gateway) then redeploy; slower

# update only the gateway (faster when gateway/ changed): rebuild → load into kind → restart
make build-gateway
export KUBECONFIG=$HOME/.cache/bioq-local/kubeconfig PATH="$HOME/.cache/bioq-local/bin:$PATH"
kind load docker-image gateway:latest --name bioq
kubectl -n bioq rollout restart deploy/bioq-gateway
```

### Configurable environment variables

`BIOQ_WORKDIR` (state dir), `BIOQ_API_KEY`, `BIOQ_GATEWAY_PORT`, `BIOQ_CLUSTER` (kind cluster name,
default `bioq`), `BIOQ_BUILD` (`auto|always|never`), `BIOQ_DB_BACKEND` (`postgres|sqlite`),
`BIOQ_KEYCLOAK` (`1` default bundled Keycloak; `0` disable and fall back to api-key), `BIOQ_KC_PORT`
(default 8081), `BIOQ_DOCKERHUB_MIRROR` (Docker Hub mirror/accelerator). See the header comments of
`deploy/openfaas/local-up.sh` for details.

## Adding a new service

**Read [`docs/adding-a-new-service/index.zh.md`](docs/adding-a-new-service/index.zh.md) first** — the
authoritative process (design doc first → skeleton → the two Dockerfile approaches / conda pitfalls →
test skeleton → FC deployment / submission checklist), plus `bioq_service` naming and concrete
commands for vendor / testing / building / registry. Chinese for now; English translations are on
the way.

> Design docs are archived in [`docs/specs/`](docs/specs/).

## Related repositories

- **`bioq`** — thin-client CLI (gateway REST client)
- **`bioagent`** — research knowledge base (`wiki/`), pipeline orchestration (`pipelines/`),
  engineering docs (`engineering/`)
