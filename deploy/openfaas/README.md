# OpenFaaS deployment

Run the workers as OpenFaaS functions (async task mode) for elastic, scale-to-zero
execution on your own Kubernetes cluster (or Alibaba ACK). The gateway stays a
normal Deployment and dispatches here via `GATEWAY_DISPATCH_BACKEND=openfaas`.

**How it maps** (see the [OpenFaaS backend plan](../../)* for the design):
- **submit** → `POST {gw}/async-function/<fn>/api/tasks/<endpoint>` — NATS-queued;
  the worker's `execute_task` runs the job synchronously and keys it by the
  `X-Bioagent-Job-Id` the gateway sends.
- **status / download** → `GET {gw}/function/<fn>/api/jobs/<id>[/download]` —
  routed to any replica; reliable because `JobStore` reads the `job.json` sidecar
  fresh from disk.

> \* design doc lives in the team engineering repo, not this repo.

## Quick start: one-command local deploy (kind)

`local-up.sh` spins up a full local stack on **kind + OpenFaaS** (no manual K8s
steps): kind cluster, OpenFaaS, a bundled **PostgreSQL** for the gateway DB, the
bioq gateway (openfaas + `file` storage over a shared volume), and the selected
worker services wrapped as OpenFaaS functions — then seeds an API key and
port-forwards the gateway.

```bash
cd deploy/openfaas

./local-up.sh                         # default: just dockq-server
./local-up.sh dockq-server plip-server   # pick specific services

# tear down (add --purge to also delete the work dir):
./local-down.sh
```

Only `docker` is required — `kind`/`kubectl`/`helm` are auto-downloaded to
`~/.cache/bioq-local/bin`. Docker Hub images (kind node, NATS) are pulled via a
mirror (`BIOQ_DOCKERHUB_MIRROR`, default `docker.m.daocloud.io`) since Docker Hub
is often unreachable; ghcr.io images are pulled directly. Base service images are
built via `make build-<svc>` if missing (some need a one-time `vendor.sh`).

The script is **idempotent**: re-run to switch services or pick up rebuilt
images. By default it **prunes** worker functions not in the set you pass (so
`./local-up.sh plip-server` frees a previously-deployed `dockq-server`) — handy
on a memory-limited host where running one worker at a time is best. Set
`BIOQ_PRUNE=0` for additive re-runs that keep prior workers.
Key env overrides: `BIOQ_CLUSTER`, `BIOQ_WORKDIR`, `BIOQ_API_KEY`,
`BIOQ_GATEWAY_PORT`, `BIOQ_DOCKERHUB_MIRROR`, `BIOQ_BUILD` (auto|always|never),
`BIOQ_DB_BACKEND` (postgres|sqlite), `BIOQ_MODELS_DIR`, `BIOQ_GPU` (0|1).

**Gateway DB:** defaults to a bundled **PostgreSQL** pod (`bioq-postgres` in the
`bioq` namespace), mirroring the ECS `gateway/deploy/docker-compose.yml` default;
its data persists on the shared hostPath (`$BIOQ_WORKDIR/shared/pgdata`) across a
non-purge `local-down`. Pass `BIOQ_DB_BACKEND=sqlite` for the old single-file DB
(`$BIOQ_WORKDIR/shared/gateway.db`) instead. The postgres image is pulled via the
Docker Hub mirror; override with `BIOQ_PG_IMAGE` / `BIOQ_PG_PASSWORD`.

**GPU services + model weights:** worker images ship **no weights** (see the
weights-externalization decision); each reads them from `/data/models/<svc>/`
via its baked `<PREFIX>_WEIGHTS_DIR`. `local-up.sh` mounts a host dir there
(`BIOQ_MODELS_DIR`, default `$BIOQ_WORKDIR/shared/models`, read-only). To run a
GPU service locally:

1. Fetch its weights once into `$BIOQ_MODELS_DIR/<svc>/`:
   `services/<svc>/scripts/fetch_weights.sh` (layout must match the baked
   `<PREFIX>_WEIGHTS_DIR` — e.g. `.../models/boltz/`, `.../models/diffdock/`).
2. Deploy with GPU scheduling: `make local-up LOCAL_SERVICES=<svc> BIOQ_GPU=1`
   — this adds `nvidia.com/gpu: 1` + `runtimeClassName: nvidia` to the pod. It
   requires the **NVIDIA device plugin / GPU operator** on the cluster and a
   GPU-capable node (kind alone does not provide GPUs). This path is provided
   but untested in CI here (no GPU); CPU services need neither step.

The default weights dir lives under the shared volume, so it works on an
already-running cluster. A custom `BIOQ_MODELS_DIR` outside that tree needs its
own kind mount, applied only on a freshly-created cluster (re-`local-down` /
`local-up`).

On success it prints the gateway URL + API key and the command to run the
functional test (`gateway/tests/test_local_openfaas.py`).

> **Single-node note:** the script gives each function one replica on a hostPath
> shared volume, so the "shared RWX" requirement below is satisfied trivially. For
> a real multi-node cluster, use a shared RWX PVC as described next.

## ⚠️ Critical requirement: shared RWX job store

Status polls are routed to **any** replica, so every replica must see every job's
state. Put `jobs_base_dir` on a **shared ReadWriteMany volume** (e.g. an NFS/CephFS
PVC) mounted into all function replicas — the same volume the gateway uses for
`file` storage. Without this, polls hit a replica that doesn't have the job and
return 404. (This mirrors how the FC backend relies on shared NAS.)

## Manual / real-cluster path (faas-cli)

The steps below are the **manual** path for a real cluster (multi-node / Alibaba
ACK) where worker images are pushed to a public/reachable registry. They use two
**reference-only** files that `local-up.sh` does **not** use:

- **`stack.yaml`** — a faas-cli stack (one `functions:` block per service). You
  extend it per service; it is not a per-service file.
- **`services.local.yaml`** — the gateway's downstream registry (`function:` names).

`local-up.sh` instead generates the function Deployments/Services and the gateway
registry ConfigMap dynamically from the services you pass — so for **local dev you
don't need either file**; use `make local-up` and skip this whole section.

> **Why the split:** OpenFaaS CE rejects local/kind-loaded images
> (`faas-cli deploy` → "only allows public images"), so this faas-cli path needs
> images pushed to a registry the cluster can pull. `local-up.sh` sidesteps that
> by applying raw K8s Deployments instead of going through `faas-cli`.

## 1. Wrap worker images with of-watchdog

OpenFaaS functions need the watchdog. `Dockerfile.watchdog` adds it over an
existing worker image without touching the worker's code/deps:

```bash
docker build -f Dockerfile.watchdog \
  --build-arg BASE_IMAGE=harbor.ruosheng.bio/aliyun_fc/dockq-server:latest \
  -t <registry>/dockq-server-fn:latest .
docker push <registry>/dockq-server-fn:latest
```

(Or let `faas-cli up -f stack.yaml` build+push from `Dockerfile.watchdog`.)

## 2. Wire the shared PVC (+ GPU) via OpenFaaS profiles

Create profiles the functions reference (`com.openfaas.profile` annotation in
`stack.yaml`) — one mounts the shared RWX PVC at `/shared`, another selects GPU
nodes. Example (apply to the `openfaas-fn` namespace):

```yaml
kind: Profile
apiVersion: openfaas.com/v1
metadata: { name: shared-jobs, namespace: openfaas }
spec:
  podSecurityContext: {}
  # volumes/volumeMounts for the RWX PVC "bioq-shared" at /shared
---
kind: Profile
apiVersion: openfaas.com/v1
metadata: { name: gpu, namespace: openfaas }
spec:
  tolerations: [{ key: nvidia.com/gpu, operator: Exists, effect: NoSchedule }]
  runtimeClassName: nvidia
  # resources.limits: { nvidia.com/gpu: 1 }
```

Then uncomment the `annotations` block in `stack.yaml` for the relevant functions.

## 3. Deploy the functions

```bash
export OPENFAAS_GATEWAY=http://<openfaas-gateway>:8080
faas-cli deploy -f stack.yaml            # pre-built images
# or: faas-cli up -f stack.yaml          # build + push + deploy
```

## 4. Deploy the gateway pointed at OpenFaaS

Run the gateway (Deployment or Compose) with:

```
GATEWAY_DISPATCH_BACKEND=openfaas
GATEWAY_OPENFAAS_GATEWAY_URL=http://<openfaas-gateway>:8080
GATEWAY_STORAGE_BACKEND=file        # or oss
GATEWAY_FILE_BASE_DIR=/shared       # same shared volume as the functions
GATEWAY_REGISTRY_PATH=/opt/gateway/services.yaml   # mount services.local.yaml
```

## 5. Run a job

Same client flow as any backend (see `../compose/README.md`): `bioq run <svc> ...`
→ `bioq status <job>` → `bioq download <job>`. Submits enqueue on NATS; watch
scaling with `kubectl -n openfaas-fn get deploy` or the OpenFaaS UI.

## Adding a worker

1. Wrap + push its image (step 1).
2. Add a function block to `stack.yaml` (copy `dockq-server`), set its env prefix's
   `*_JOBS_BASE_DIR=/shared/jobs` + `*_OSS_OUTPUT_MOUNT=/shared`.
3. Add an entry to `services.local.yaml` with `function: <name>`.
4. For GPU, reference the `gpu` profile.
