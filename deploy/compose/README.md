# Local deployment (Docker Compose)

Run the gateway + workers on a single machine with **no Alibaba FC / OSS**:

- **Dispatch = `http`** — the gateway submits to each worker's `POST /api/<endpoint>`
  and polls `GET /api/jobs/<id>` (the worker's own in-process async runner). No FC.
- **Storage = `file`** — inputs are uploaded through the gateway's `PUT /v1/files/...`
  onto a `shared` volume; workers read them as `file://` paths. Results are streamed
  back by proxying the worker's `GET /api/jobs/<id>/download`.

The same worker images run here and on Alibaba — only the gateway's backend env
(`GATEWAY_DISPATCH_BACKEND` / `GATEWAY_STORAGE_BACKEND`) and the registry differ.

## 1. Build images

From the repo root:

```bash
make build-gateway
make build-dockq-server        # example CPU worker
```

## 2. Bring the stack up

```bash
docker compose -f deploy/compose/docker-compose.yml up -d
```

The gateway entrypoint runs `alembic upgrade head` then starts uvicorn on `:9000`.

## 3. Auth (Keycloak / OIDC)

Auth is **OIDC/JWT** (bundled Keycloak, realm `bioq`) — API keys were retired.
localhost is VPC-bypassed by default (`BYPASS_VPC=true`), so **local dev needs no
credentials**; OIDC is the real path (and what to use with `BYPASS_VPC=false`).

- **Keycloak**: `http://localhost:8081` (master console `admin`/`admin`;
  realm `bioq` bootstrap admin `admin`/`admin` in group `bioq-admins`).
- **Create users** (role = group; `bioq-admins` → admin) via the shared kcadm helper:
  ```bash
  docker compose -f deploy/compose/docker-compose.yml exec -T keycloak \
      bash -s -- alice pw       < deploy/openfaas/kc-user.sh   # normal user
  docker compose -f deploy/compose/docker-compose.yml exec -T keycloak \
      bash -s -- root  pw admin < deploy/openfaas/kc-user.sh   # admin
  ```
- **Admin console**: `http://localhost:9000/admin` (localhost → bypass, straight in;
  or `/admin/login` → "Sign in with SSO").

## 4. Run a job

localhost is bypassed, so no credential is needed locally:

```bash
bioq --gateway-url http://localhost:9000 services
bioq --gateway-url http://localhost:9000 run dockq-server score \
    --file model=./a.pdb --file native=./b.pdb
```

To exercise real OIDC (or when `BYPASS_VPC=false`), log in first (device flow):

```bash
bioq --gateway-url http://localhost:9000 login --oidc \
    --issuer http://localhost:8081/realms/bioq --client-id bioq-cli
bioq services      # now sends Authorization: Bearer <JWT>
```

Or with `curl` (localhost bypass; job-centric flow):

```bash
JOB=$(uuidgen | tr -d - | cut -c1-20)
curl -X PUT --data-binary @a.pdb \
    "http://localhost:9000/v1/files/users/local/$JOB/input/a.pdb"
curl -X POST -H "x-bioagent-job-id: $JOB" -H "content-type: application/json" \
    -d '{"model_uri":"file:///shared/users/local/'"$JOB"'/input/a.pdb"}' \
    "http://localhost:9000/v1/run/dockq-server/score"
curl "http://localhost:9000/v1/jobs/$JOB"
curl -L -o results.zip "http://localhost:9000/v1/jobs/$JOB/download"
```

## Adding a worker

1. Add a service block to `docker-compose.yml` (copy the `dockq-server` one).
2. Mount `shared:/shared` so it can read staged `file://` inputs.
3. Add an entry to `services.local.yaml` (`url: http://<name>:8000`).
4. For a **GPU** worker, uncomment the `deploy.resources` block (needs the
   NVIDIA Container Toolkit on the host).

## Notes

- **Concurrency**: the `http` backend has no per-task instance spawning — a worker
  runs jobs on its own in-process runner (see each service's `max_concurrent_jobs`).
  Scale by running replicas, or move to the Kubernetes/OpenFaaS deployment when you
  need elastic per-task scaling.
- **Results**: downloads proxy the worker's `results.zip` through the gateway. If a
  worker also mounts `shared` at its output mount and mirrors results to
  `/shared/users/<acct>/<job>/`, the gateway serves them directly via a `/v1/files`
  redirect instead (both paths are supported).
- **Alibaba Cloud FC**: auth (Keycloak/OIDC) is independent of the dispatch/storage
  backend, so it works the same when this stack fronts FC. To front FC, drop the
  local worker services and set on the gateway: `GATEWAY_DISPATCH_BACKEND=fc` +
  `GATEWAY_STORAGE_BACKEND=oss` (+ the FC/OSS credentials and a `services.yaml`
  pointing at the FC VPC URLs). Keep the Keycloak/OIDC env as-is; for a public
  entrypoint set `BYPASS_VPC=false` (see the production-hardening checklist).
