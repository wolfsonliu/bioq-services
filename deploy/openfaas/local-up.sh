#!/usr/bin/env bash
# local-up.sh — deploy bioq-services locally on kind + OpenFaaS (async mode).
#
# Brings up (idempotently): a kind cluster, OpenFaaS, the bioq gateway
# (GATEWAY_DISPATCH_BACKEND=openfaas + file storage over a shared volume), and
# one or more worker services wrapped as OpenFaaS functions. Then seeds an API
# key and port-forwards the gateway to localhost.
#
# Usage:
#   ./local-up.sh                      # default: just dockq-server
#   ./local-up.sh dockq-server plip-server
#
# Env overrides (with defaults):
#   BIOQ_CLUSTER=bioq                  kind cluster name
#   BIOQ_WORKDIR=~/.cache/bioq-local   kubeconfig / shared vol / tools / manifests
#   BIOQ_DOCKERHUB_MIRROR=docker.m.daocloud.io   mirror for Docker Hub images
#   BIOQ_NODE_IMAGE=kindest/node:v1.31.0
#   BIOQ_API_KEY=bioq-local-secret     seeded gateway API key
#   BIOQ_GATEWAY_PORT=9000             host port the gateway is forwarded to
#   BIOQ_BUILD=auto                    auto|always|never — build missing base images
#   BIOQ_PRUNE=1                       1|0 — remove worker functions not in the
#                                      requested set (frees memory; set 0 for additive)
#   BIOQ_DB_BACKEND=postgres           postgres|sqlite — gateway DB (postgres bundles
#                                      a PostgreSQL pod, mirroring the ECS compose default)
#   BIOQ_PG_IMAGE=postgres:18.4-trixie postgres image (pulled via the Docker Hub mirror)
#   BIOQ_PG_PASSWORD=bioq-local-pg     bundled postgres password
#   BIOQ_MODELS_DIR=<workdir>/shared/models   host weights dir mounted at
#                                      /data/models (GPU services read /data/models/<svc>/)
#   BIOQ_GPU=0                         1|0 — request a GPU + nvidia runtime for workers
#
# Requires: docker. kind/kubectl/helm are auto-downloaded to $BIOQ_WORKDIR/bin
# if not already on PATH. Docker Hub is assumed unreachable directly, so Docker
# Hub images are pulled via the mirror.
set -euo pipefail

# --- config ---------------------------------------------------------------
CLUSTER="${BIOQ_CLUSTER:-bioq}"
WORKDIR="${BIOQ_WORKDIR:-$HOME/.cache/bioq-local}"
MIRROR="${BIOQ_DOCKERHUB_MIRROR:-docker.m.daocloud.io}"
NODE_IMAGE="${BIOQ_NODE_IMAGE:-kindest/node:v1.31.0}"
API_KEY="${BIOQ_API_KEY:-bioq-local-secret}"
GATEWAY_PORT="${BIOQ_GATEWAY_PORT:-9000}"
BUILD_MODE="${BIOQ_BUILD:-auto}"
ACCOUNT="local"

# Gateway DB: postgres (bundled pod, like the ECS compose) or sqlite (single file
# on the shared volume). Postgres tunables mirror gateway/deploy/.env.example.
DB_BACKEND="${BIOQ_DB_BACKEND:-postgres}"
PG_IMAGE="${BIOQ_PG_IMAGE:-postgres:18.4-trixie}"
PG_USER="${BIOQ_PG_USER:-bioagent}"
PG_PASSWORD="${BIOQ_PG_PASSWORD:-bioq-local-pg}"
PG_DB="${BIOQ_PG_DB:-gateway}"

# In-cluster Keycloak (OIDC IdP). BIOQ_KEYCLOAK=0 skips it (api-key-only local mode).
# Browser/bioq reach it via port-forward at localhost:$KC_PORT; the gateway pod
# reaches it via cluster DNS. KC_HOSTNAME pins the frontend issuer to the host URL
# while KC_HOSTNAME_BACKCHANNEL_DYNAMIC lets token/jwks follow the caller's host.
KEYCLOAK="${BIOQ_KEYCLOAK:-1}"
KC_IMAGE="${BIOQ_KC_IMAGE:-quay.io/keycloak/keycloak:26.0}"
KC_PORT="${BIOQ_KC_PORT:-8081}"
KC_FRONTEND="http://localhost:${KC_PORT}"
KC_ISSUER_FRONTEND="${KC_FRONTEND}/realms/bioq"
KC_INCLUSTER="http://keycloak.bioq.svc.cluster.local:8080/realms/bioq"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # repos/bioq-services
BIN_DIR="$WORKDIR/bin"
SHARED_DIR="$WORKDIR/shared"
MANIFEST_DIR="$WORKDIR/manifests"
export KUBECONFIG="$WORKDIR/kubeconfig"
export PATH="$BIN_DIR:$PATH"

# Model weights. GPU services read weights from /data/models/<svc>/ (baked
# <PREFIX>_WEIGHTS_DIR); the image ships none (see the weights-externalization
# decision). We mount a host dir there. Default lives UNDER the shared volume so
# it needs no extra kind mount and works on an already-running cluster; a custom
# BIOQ_MODELS_DIR outside the shared tree gets its own kind extraMount (which
# only takes effect on a freshly-created cluster). Populate per service with
# services/<svc>/scripts/fetch_weights.sh into $MODELS_DIR/<svc>/.
MODELS_DIR="${BIOQ_MODELS_DIR:-$SHARED_DIR/models}"
case "$MODELS_DIR" in
  "$SHARED_DIR"/*)  MODELS_NODE_PATH="/shared/${MODELS_DIR#"$SHARED_DIR"/}"; MODELS_EXTRA_MOUNT=0 ;;
  *)                MODELS_NODE_PATH="/data/models"; MODELS_EXTRA_MOUNT=1 ;;
esac
# GPU scheduling for worker pods (untested here — no GPU). When 1, requests one
# nvidia.com/gpu + selects the nvidia runtime. Requires the NVIDIA device plugin
# / GPU operator on the cluster and a GPU-capable node. The snippets are spliced
# into the worker manifest (empty = CPU pod, unchanged); keep the trailing
# newlines so the following YAML keys stay correctly indented.
GPU="${BIOQ_GPU:-0}"
GPU_POD_YAML="" GPU_RES_YAML="" GPU_LOG=""
if [ "$GPU" = 1 ]; then
  GPU_LOG=", gpu"
  GPU_POD_YAML="      runtimeClassName: nvidia
      tolerations: [ { key: nvidia.com/gpu, operator: Exists, effect: NoSchedule } ]
"
  GPU_RES_YAML="          resources: { limits: { nvidia.com/gpu: 1 } }
"
fi

SERVICES=("$@"); [ "${#SERVICES[@]}" -eq 0 ] && SERVICES=(dockq-server)

log()  { printf '\033[1;34m[local-up]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[local-up]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[local-up] error:\033[0m %s\n' "$*" >&2; exit 1; }

mkdir -p "$BIN_DIR" "$SHARED_DIR" "$MANIFEST_DIR" "$MODELS_DIR"
chmod 777 "$SHARED_DIR"

# --- tool bootstrap -------------------------------------------------------
command -v docker >/dev/null 2>&1 || die "docker is required and must be running"
docker info >/dev/null 2>&1 || die "cannot reach the docker daemon"

ensure_kind() {
  command -v kind >/dev/null 2>&1 && return
  [ -x "$BIN_DIR/kind" ] && return
  log "downloading kind -> $BIN_DIR/kind"
  curl -fsSL -o "$BIN_DIR/kind" https://github.com/kubernetes-sigs/kind/releases/download/v0.24.0/kind-linux-amd64
  chmod +x "$BIN_DIR/kind"
}
ensure_helm() {
  command -v helm >/dev/null 2>&1 && return
  [ -x "$BIN_DIR/helm" ] && return
  log "downloading helm -> $BIN_DIR/helm"
  curl -fsSL -o "$WORKDIR/helm.tgz" https://get.helm.sh/helm-v3.16.2-linux-amd64.tar.gz
  tar -xzf "$WORKDIR/helm.tgz" -C "$WORKDIR" linux-amd64/helm
  mv "$WORKDIR/linux-amd64/helm" "$BIN_DIR/helm"; rm -rf "$WORKDIR/linux-amd64" "$WORKDIR/helm.tgz"
  chmod +x "$BIN_DIR/helm"
}
ensure_node_image() {
  docker image inspect "$NODE_IMAGE" >/dev/null 2>&1 && return
  log "pulling $NODE_IMAGE via $MIRROR"
  docker pull "$MIRROR/${NODE_IMAGE}" >/dev/null
  docker tag "$MIRROR/${NODE_IMAGE}" "$NODE_IMAGE"
}
ensure_kubectl() {
  command -v kubectl >/dev/null 2>&1 && return
  [ -x "$BIN_DIR/kubectl" ] && return
  # Extract kubectl from the kind node image (avoids a separate, often-slow download).
  ensure_node_image
  log "extracting kubectl from $NODE_IMAGE -> $BIN_DIR/kubectl"
  local cid; cid="$(docker create "$NODE_IMAGE")"
  docker cp "$cid:/usr/bin/kubectl" "$BIN_DIR/kubectl"; docker rm "$cid" >/dev/null
  chmod +x "$BIN_DIR/kubectl"
}

ensure_kind; ensure_kubectl; ensure_helm

# --- kind cluster ---------------------------------------------------------
# A custom BIOQ_MODELS_DIR (outside the shared tree) needs its own node mount,
# which can only be added at cluster-creation time.
models_extra_mount_yaml=""
if [ "$MODELS_EXTRA_MOUNT" = 1 ]; then
  models_extra_mount_yaml="      - hostPath: $MODELS_DIR
        containerPath: /data/models"
fi

if kind get clusters 2>/dev/null | grep -qx "$CLUSTER"; then
  log "kind cluster '$CLUSTER' already exists"
  kind export kubeconfig --name "$CLUSTER" --kubeconfig "$KUBECONFIG" >/dev/null 2>&1 || true
  if [ "$MODELS_EXTRA_MOUNT" = 1 ]; then
    warn "custom BIOQ_MODELS_DIR=$MODELS_DIR won't be mounted on the EXISTING cluster"
    warn "(node mounts are set at creation). Run ./local-down.sh then ./local-up.sh,"
    warn "or use the default location under $SHARED_DIR/models."
  fi
else
  ensure_node_image
  log "creating kind cluster '$CLUSTER' (shared: $SHARED_DIR -> /shared; models: $MODELS_DIR -> /data/models)"
  cat > "$MANIFEST_DIR/kind.yaml" <<EOF
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
name: $CLUSTER
nodes:
  - role: control-plane
    image: $NODE_IMAGE
    extraMounts:
      - hostPath: $SHARED_DIR
        containerPath: /shared
$models_extra_mount_yaml
EOF
  kind create cluster --config "$MANIFEST_DIR/kind.yaml" --kubeconfig "$KUBECONFIG" --wait 120s
fi

# --- OpenFaaS -------------------------------------------------------------
if kubectl get deploy gateway -n openfaas >/dev/null 2>&1; then
  log "OpenFaaS already installed"
else
  log "installing OpenFaaS"
  export HELM_CACHE_HOME="$WORKDIR/helm/cache" HELM_CONFIG_HOME="$WORKDIR/helm/config" HELM_DATA_HOME="$WORKDIR/helm/data"
  helm repo add openfaas https://openfaas.github.io/faas-netes/ >/dev/null 2>&1 || true
  helm repo update openfaas >/dev/null
  kubectl create namespace openfaas >/dev/null 2>&1 || true
  kubectl create namespace openfaas-fn >/dev/null 2>&1 || true
  # Preload the one Docker Hub image OpenFaaS CE needs (NATS) via the mirror.
  NATS_IMG="$(helm show values openfaas/openfaas 2>/dev/null | grep -E 'image:\s*nats-streaming' | head -1 | awk '{print $2}')"
  NATS_IMG="${NATS_IMG:-nats-streaming:0.25.6}"
  log "preloading $NATS_IMG via $MIRROR"
  docker pull "$MIRROR/library/$NATS_IMG" >/dev/null
  docker tag "$MIRROR/library/$NATS_IMG" "$NATS_IMG"
  kind load docker-image "$NATS_IMG" --name "$CLUSTER" >/dev/null
  # basic_auth off (we invoke the data plane only) + prometheus off (no autoscaling needed).
  helm upgrade --install openfaas openfaas/openfaas -n openfaas \
    --set functionNamespace=openfaas-fn --set basic_auth=false \
    --set prometheus.create=false --set alertmanager.create=false --set async=true >/dev/null
  log "waiting for OpenFaaS core to become ready..."
  kubectl -n openfaas rollout status deploy/gateway --timeout=180s
  kubectl -n openfaas rollout status deploy/nats --timeout=120s || true
fi

# --- helpers --------------------------------------------------------------
# Base image tag for a service (matches the Makefile: <svc>:latest).
svc_image() { echo "$1:latest"; }

ensure_base_image() {
  local svc="$1" img; img="$(svc_image "$svc")"
  if docker image inspect "$img" >/dev/null 2>&1 && [ "$BUILD_MODE" != always ]; then return; fi
  [ "$BUILD_MODE" = never ] && die "base image $img missing (BIOQ_BUILD=never)"
  if [ -x "$REPO_ROOT/services/$svc/scripts/vendor.sh" ]; then
    log "[$svc] vendoring upstream"
    (cd "$REPO_ROOT" && bash "services/$svc/scripts/vendor.sh") >/dev/null
  fi
  log "[$svc] building base image (make build-$svc) — may take a while"
  (cd "$REPO_ROOT" && make "build-$svc")
}

# Wrap a base image with of-watchdog, deriving fprocess + port from its CMD.
build_fn_image() {
  local svc="$1" base fn workdir port fprocess cmdjson
  base="$(svc_image "$svc")"; fn="$svc-fn:latest"
  cmdjson="$(docker inspect --format '{{json .Config.Cmd}}' "$base")"
  workdir="$(docker inspect --format '{{.Config.WorkingDir}}' "$base")"
  read -r port fprocess < <(python3 - "$cmdjson" "$workdir" <<'PY'
import json, sys, shlex
cmd = json.loads(sys.argv[1]); workdir = sys.argv[2].rstrip("/")
if cmd and workdir and not cmd[0].startswith("/"):
    # Resolve a relative launcher (e.g. ".venv/bin/python") against WORKDIR.
    # NB: strip only a leading "./" prefix — not lstrip("./"), which would eat
    # the dot of ".venv".
    rel = cmd[0][2:] if cmd[0].startswith("./") else cmd[0]
    cmd[0] = workdir + "/" + rel
port = "8000"
for i, a in enumerate(cmd):
    if a == "--port" and i + 1 < len(cmd):
        port = cmd[i + 1]
print(port, " ".join(shlex.quote(c) for c in cmd))
PY
)
  log "[$svc] wrapping with of-watchdog (upstream :$port)"
  docker build -q -f "$REPO_ROOT/deploy/openfaas/Dockerfile.watchdog" \
    --build-arg "BASE_IMAGE=$base" --build-arg "UPSTREAM_PORT=$port" \
    --build-arg "FPROCESS=$fprocess" -t "$fn" "$REPO_ROOT/deploy/openfaas" >/dev/null
  kind load docker-image "$fn" --name "$CLUSTER" >/dev/null
}

# Settings env prefix for a service (e.g. DOCKQ_), read from its settings.py.
svc_env_prefix() {
  local svc="$1" p
  p="$(grep -ohP 'env_prefix\s*=\s*"\K[^"]+' "$REPO_ROOT/services/$svc"/settings.py "$REPO_ROOT/services/$svc"/server/settings.py 2>/dev/null | head -1)"
  echo "${p:-$(echo "${svc%-server}" | tr 'a-z-' 'A-Z_')_}"
}

deploy_function() {
  local svc="$1" prefix; prefix="$(svc_env_prefix "$svc")"
  log "[$svc] deploying function (env prefix ${prefix}${GPU_LOG})"
  cat > "$MANIFEST_DIR/fn-$svc.yaml" <<EOF
apiVersion: apps/v1
kind: Deployment
metadata: { name: $svc, namespace: openfaas-fn, labels: { faas_function: $svc } }
spec:
  replicas: 1
  selector: { matchLabels: { faas_function: $svc } }
  template:
    metadata: { labels: { faas_function: $svc } }
    spec:
$GPU_POD_YAML      volumes:
        - { name: shared, hostPath: { path: /shared, type: DirectoryOrCreate } }
        - { name: models, hostPath: { path: $MODELS_NODE_PATH, type: DirectoryOrCreate } }
      containers:
        - name: $svc
          image: $svc-fn:latest
          imagePullPolicy: IfNotPresent
          ports: [ { containerPort: 8080 } ]
          env:
            - { name: ${prefix}JOBS_BASE_DIR, value: /shared/jobs }
            - { name: ${prefix}OSS_OUTPUT_MOUNT, value: /shared }
          volumeMounts:
            - { name: shared, mountPath: /shared }
            # Weights read-only at the baked <PREFIX>_WEIGHTS_DIR root (/data/models/<svc>/).
            - { name: models, mountPath: /data/models, readOnly: true }
$GPU_RES_YAML          readinessProbe: { httpGet: { path: /_/health, port: 8080 }, initialDelaySeconds: 3, periodSeconds: 5 }
          livenessProbe:  { httpGet: { path: /_/health, port: 8080 }, initialDelaySeconds: 5, periodSeconds: 15 }
---
apiVersion: v1
kind: Service
metadata: { name: $svc, namespace: openfaas-fn, labels: { faas_function: $svc } }
spec:
  selector: { faas_function: $svc }
  ports: [ { name: http, port: 8080, targetPort: 8080 } ]
EOF
  kubectl apply -f "$MANIFEST_DIR/fn-$svc.yaml" >/dev/null
  # Re-runs reload the image under the same :latest tag; restart to pick it up.
  kubectl -n openfaas-fn rollout restart "deploy/$svc" >/dev/null 2>&1 || true
}

# Bundled PostgreSQL in the bioq namespace (mirrors the ECS compose default).
# Idempotent: skips if already deployed. Data lives on the shared hostPath so it
# survives a `local-down` (non-purge), like the SQLite file did. The image runs
# as root first so its entrypoint chowns the hostPath data dir to the postgres uid.
deploy_postgres() {
  kubectl create namespace bioq >/dev/null 2>&1 || true
  if kubectl -n bioq get deploy bioq-postgres >/dev/null 2>&1; then
    log "postgres already deployed"; return
  fi
  if ! docker image inspect "$PG_IMAGE" >/dev/null 2>&1; then
    log "pulling $PG_IMAGE via $MIRROR"
    docker pull "$MIRROR/library/$PG_IMAGE" >/dev/null
    docker tag "$MIRROR/library/$PG_IMAGE" "$PG_IMAGE"
  fi
  kind load docker-image "$PG_IMAGE" --name "$CLUSTER" >/dev/null
  log "deploying postgres ($PG_IMAGE)"
  cat > "$MANIFEST_DIR/postgres.yaml" <<EOF
apiVersion: apps/v1
kind: Deployment
metadata: { name: bioq-postgres, namespace: bioq, labels: { app: bioq-postgres } }
spec:
  replicas: 1
  # Single hostPath volume — never run two pods against it.
  strategy: { type: Recreate }
  selector: { matchLabels: { app: bioq-postgres } }
  template:
    metadata: { labels: { app: bioq-postgres } }
    spec:
      volumes: [ { name: pgdata, hostPath: { path: /shared/pgdata, type: DirectoryOrCreate } } ]
      containers:
        - name: postgres
          image: $PG_IMAGE
          imagePullPolicy: IfNotPresent
          env:
            - { name: POSTGRES_USER, value: "$PG_USER" }
            - { name: POSTGRES_PASSWORD, value: "$PG_PASSWORD" }
            - { name: POSTGRES_DB, value: "$PG_DB" }
          ports: [ { containerPort: 5432 } ]
          # PG 18+ keeps data in a major-version subdir, so mount the parent.
          volumeMounts: [ { name: pgdata, mountPath: /var/lib/postgresql } ]
          readinessProbe:
            exec: { command: ["pg_isready", "-U", "$PG_USER", "-d", "$PG_DB"] }
            initialDelaySeconds: 5
            periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata: { name: bioq-postgres, namespace: bioq }
spec:
  selector: { app: bioq-postgres }
  ports: [ { name: pg, port: 5432, targetPort: 5432 } ]
EOF
  kubectl apply -f "$MANIFEST_DIR/postgres.yaml" >/dev/null
  kubectl -n bioq rollout status deploy/bioq-postgres --timeout=180s
}

# In-cluster Keycloak (OIDC IdP). Realm imported from keycloak-realm.json via a
# ConfigMap; H2 data persisted on the shared hostPath so users created with
# `make local-user` survive pod restarts. See KC_HOSTNAME notes above.
deploy_keycloak() {
  kubectl create namespace bioq >/dev/null 2>&1 || true
  if ! docker image inspect "$KC_IMAGE" >/dev/null 2>&1; then
    log "pulling $KC_IMAGE via $MIRROR"
    docker pull "$KC_IMAGE" >/dev/null 2>&1 \
      || docker pull "$MIRROR/keycloak/keycloak:26.0" >/dev/null && docker tag "$MIRROR/keycloak/keycloak:26.0" "$KC_IMAGE" \
      || die "cannot obtain $KC_IMAGE"
  fi
  kind load docker-image "$KC_IMAGE" --name "$CLUSTER" >/dev/null
  log "deploying keycloak ($KC_IMAGE)"
  kubectl -n bioq create configmap keycloak-realm \
    --from-file=realm.json="$REPO_ROOT/deploy/openfaas/keycloak-realm.json" \
    --dry-run=client -o yaml | kubectl apply -f - >/dev/null
  cat > "$MANIFEST_DIR/keycloak.yaml" <<EOF
apiVersion: apps/v1
kind: Deployment
metadata: { name: keycloak, namespace: bioq, labels: { app: keycloak } }
spec:
  replicas: 1
  strategy: { type: Recreate }
  selector: { matchLabels: { app: keycloak } }
  template:
    metadata: { labels: { app: keycloak } }
    spec:
      volumes:
        - { name: kcdata, hostPath: { path: /shared/keycloak, type: DirectoryOrCreate } }
        - { name: realm, configMap: { name: keycloak-realm } }
      # hostPath is root-owned; Keycloak runs as uid 1000 and needs to write H2 +
      # transaction-logs under /opt/keycloak/data. chmod it writable first.
      initContainers:
        - name: fix-perms
          image: $KC_IMAGE
          imagePullPolicy: IfNotPresent
          command: ["sh", "-c", "mkdir -p /data && chmod -R 777 /data"]
          securityContext: { runAsUser: 0 }
          volumeMounts: [ { name: kcdata, mountPath: /data } ]
      containers:
        - name: keycloak
          image: $KC_IMAGE
          imagePullPolicy: IfNotPresent
          args: ["start-dev", "--import-realm"]
          env:
            - { name: KC_BOOTSTRAP_ADMIN_USERNAME, value: "admin" }
            - { name: KC_BOOTSTRAP_ADMIN_PASSWORD, value: "admin" }
            - { name: KC_HOSTNAME, value: "$KC_FRONTEND" }
            - { name: KC_HOSTNAME_BACKCHANNEL_DYNAMIC, value: "true" }
            - { name: KC_HTTP_ENABLED, value: "true" }
          ports: [ { containerPort: 8080 } ]
          volumeMounts:
            - { name: kcdata, mountPath: /opt/keycloak/data }
            - { name: realm, mountPath: /opt/keycloak/data/import }
          readinessProbe:
            httpGet: { path: /realms/bioq, port: 8080 }
            initialDelaySeconds: 20
            periodSeconds: 5
            failureThreshold: 40
---
apiVersion: v1
kind: Service
metadata: { name: keycloak, namespace: bioq }
spec:
  selector: { app: keycloak }
  ports: [ { name: http, port: 8080, targetPort: 8080 } ]
EOF
  kubectl apply -f "$MANIFEST_DIR/keycloak.yaml" >/dev/null
  kubectl -n bioq rollout status deploy/keycloak --timeout=240s
}

# --- per-service build + deploy -------------------------------------------
for svc in "${SERVICES[@]}"; do
  [ -d "$REPO_ROOT/services/$svc" ] || die "unknown service: $svc (no services/$svc)"
  ensure_base_image "$svc"
  build_fn_image "$svc"
  deploy_function "$svc"
done

# Prune worker functions NOT in the requested set (default on). On a memory-
# limited host, running every previously-deployed worker at once is costly, so
# re-running with a single service frees the others. Set BIOQ_PRUNE=0 to keep
# old functions (additive re-runs).
if [ "${BIOQ_PRUNE:-1}" != 0 ]; then
  want=" ${SERVICES[*]} "
  existing="$(kubectl -n openfaas-fn get deploy -l faas_function \
    -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || true)"
  for fn in $existing; do
    case "$want" in
      *" $fn "*) ;;  # keep — requested
      *) log "pruning worker not in requested set: $fn"
         kubectl -n openfaas-fn delete deploy,svc "$fn" >/dev/null 2>&1 || true ;;
    esac
  done
fi

# --- gateway --------------------------------------------------------------
ensure_base_image_gateway() {
  if ! docker image inspect gateway:latest >/dev/null 2>&1 || [ "$BUILD_MODE" = always ]; then
    [ "$BUILD_MODE" = never ] && die "gateway:latest missing (BIOQ_BUILD=never)"
    log "building gateway image (make build-gateway)"
    (cd "$REPO_ROOT" && make build-gateway)
  fi
}
ensure_base_image_gateway
kind load docker-image gateway:latest --name "$CLUSTER" >/dev/null

# --- gateway DB (postgres | sqlite) ---------------------------------------
if [ "$DB_BACKEND" = postgres ]; then
  deploy_postgres
  DB_URL="postgresql+psycopg://$PG_USER:$PG_PASSWORD@bioq-postgres:5432/$PG_DB"
  # Gate the gateway's `alembic upgrade head` on postgres accepting connections.
  GW_INIT="      initContainers:
        - name: wait-postgres
          image: $PG_IMAGE
          imagePullPolicy: IfNotPresent
          command: [\"sh\", \"-c\", \"until pg_isready -h bioq-postgres -U $PG_USER; do echo waiting-for-postgres; sleep 2; done\"]
"
elif [ "$DB_BACKEND" = sqlite ]; then
  DB_URL="sqlite:////shared/gateway.db"
  GW_INIT=""
else
  die "BIOQ_DB_BACKEND must be 'postgres' or 'sqlite' (got: $DB_BACKEND)"
fi

# registry (services.yaml) for the selected services
REG=""
for svc in "${SERVICES[@]}"; do
  REG+="      $svc:
        url: http://placeholder
        function: $svc
        tier: warm
"
done

# --- in-cluster Keycloak + gateway OIDC wiring ----------------------------
GW_OIDC_ENV=""
if [ "$KEYCLOAK" = 1 ]; then
  deploy_keycloak
  # JWKS/token via cluster DNS (pod-reachable, backchannel-dynamic); issuer pinned
  # to the frontend host URL (what browser/bioq see). OIDC discovery for the SSO
  # code flow also goes via cluster DNS → authorize URL comes back as localhost.
  GW_OIDC_ENV="            - { name: GATEWAY_AUTH__JWT_JWKS_URL, value: ${KC_INCLUSTER}/protocol/openid-connect/certs }
            - { name: GATEWAY_AUTH__JWT_ISSUER, value: ${KC_ISSUER_FRONTEND} }
            - { name: GATEWAY_AUTH__JWT_AUDIENCE, value: gateway-server }
            - { name: GATEWAY_AUTH__OIDC_ISSUER, value: ${KC_INCLUSTER} }
            - { name: GATEWAY_AUTH__OIDC_CLIENT_ID, value: bioq-gateway }
            - { name: GATEWAY_AUTH__OIDC_CLIENT_SECRET, value: bioq-gateway-secret }
            - { name: GATEWAY_SESSION_SECRET, value: bioq-local-session-secret }
"
fi

log "deploying bioq gateway (openfaas + file storage, db=$DB_BACKEND)"
cat > "$MANIFEST_DIR/gateway.yaml" <<EOF
apiVersion: v1
kind: Namespace
metadata: { name: bioq }
---
apiVersion: v1
kind: ConfigMap
metadata: { name: bioq-registry, namespace: bioq }
data:
  services.yaml: |
    services:
$REG
---
apiVersion: apps/v1
kind: Deployment
metadata: { name: bioq-gateway, namespace: bioq, labels: { app: bioq-gateway } }
spec:
  replicas: 1
  selector: { matchLabels: { app: bioq-gateway } }
  template:
    metadata: { labels: { app: bioq-gateway } }
    spec:
      volumes:
        - { name: shared, hostPath: { path: /shared, type: DirectoryOrCreate } }
        - { name: registry, configMap: { name: bioq-registry } }
$GW_INIT      containers:
        - name: gateway
          image: gateway:latest
          imagePullPolicy: IfNotPresent
          ports: [ { containerPort: 9000 } ]
          env:
            - { name: GATEWAY_DISPATCH_BACKEND, value: openfaas }
            - { name: GATEWAY_OPENFAAS_GATEWAY_URL, value: http://gateway.openfaas.svc.cluster.local:8080 }
            - { name: GATEWAY_STORAGE_BACKEND, value: file }
            - { name: GATEWAY_FILE_BASE_DIR, value: /shared }
            - { name: GATEWAY_REGISTRY_PATH, value: /etc/bioq/services.yaml }
            - { name: GATEWAY_DB_URL, value: "$DB_URL" }
            - { name: GATEWAY_JOBS_BASE_DIR, value: /shared/gw_jobs }
            - { name: GATEWAY_AUTH__BYPASS_VPC, value: "false" }
$GW_OIDC_ENV          volumeMounts:
            - { name: shared, mountPath: /shared }
            - { name: registry, mountPath: /etc/bioq }
          readinessProbe: { httpGet: { path: /healthz, port: 9000 }, initialDelaySeconds: 5, periodSeconds: 5 }
---
apiVersion: v1
kind: Service
metadata: { name: bioq-gateway, namespace: bioq }
spec:
  selector: { app: bioq-gateway }
  ports: [ { name: http, port: 9000, targetPort: 9000 } ]
EOF
kubectl apply -f "$MANIFEST_DIR/gateway.yaml" >/dev/null
# ConfigMap change doesn't restart the pod; force a fresh rollout so the registry + image update.
kubectl -n bioq rollout restart deploy/bioq-gateway >/dev/null 2>&1 || true
kubectl -n bioq rollout status deploy/bioq-gateway --timeout=180s

# --- seed API key (idempotent) --------------------------------------------
log "seeding API key (account=$ACCOUNT)"
kubectl -n bioq exec deploy/bioq-gateway -- /opt/gateway/.venv/bin/python \
  /opt/gateway/scripts/seed_key.py --account-id "$ACCOUNT" --secret "$API_KEY" --key-id gk_local \
  >/dev/null 2>&1 || warn "key may already exist (ok)"

# --- port-forward ---------------------------------------------------------
PF_PID_FILE="$WORKDIR/port-forward.pid"
if [ -f "$PF_PID_FILE" ] && kill -0 "$(cat "$PF_PID_FILE")" 2>/dev/null; then
  kill "$(cat "$PF_PID_FILE")" 2>/dev/null || true
fi
# setsid + </dev/null fully detaches the forward so it survives this script exiting.
setsid kubectl -n bioq port-forward svc/bioq-gateway "$GATEWAY_PORT:9000" --address 127.0.0.1 \
  >"$WORKDIR/port-forward.log" 2>&1 </dev/null &
echo $! > "$PF_PID_FILE"
sleep 4

# Keycloak port-forward (browser + bioq reach the IdP at localhost:$KC_PORT).
if [ "$KEYCLOAK" = 1 ]; then
  KC_PF_PID_FILE="$WORKDIR/keycloak-port-forward.pid"
  if [ -f "$KC_PF_PID_FILE" ] && kill -0 "$(cat "$KC_PF_PID_FILE")" 2>/dev/null; then
    kill "$(cat "$KC_PF_PID_FILE")" 2>/dev/null || true
  fi
  setsid kubectl -n bioq port-forward svc/keycloak "$KC_PORT:8080" --address 127.0.0.1 \
    >"$WORKDIR/keycloak-port-forward.log" 2>&1 </dev/null &
  echo $! > "$KC_PF_PID_FILE"
  sleep 2
fi

# --- summary --------------------------------------------------------------
KC_INFO=""
if [ "$KEYCLOAK" = 1 ]; then
  KC_INFO="
  OIDC (Keycloak):
    console SSO : open http://127.0.0.1:$GATEWAY_PORT/admin/login -> \"Sign in with SSO\" (admin/admin)
    bioq login  : bioq --gateway-url http://127.0.0.1:$GATEWAY_PORT login --oidc --issuer $KC_ISSUER_FRONTEND --client-id bioq-cli
    make users  : make local-user ACCOUNT=alice PASSWORD=pw [ADMIN=1]
"
fi

log "waiting for functions to become ready..."
for svc in "${SERVICES[@]}"; do
  kubectl -n openfaas-fn rollout status "deploy/$svc" --timeout=120s || warn "[$svc] not ready yet"
done

cat <<EOF

$(printf '\033[1;32m[local-up] ready\033[0m')
  gateway URL : http://127.0.0.1:$GATEWAY_PORT
  API key     : $API_KEY   (header: X-API-Key)
  keycloak    : $([ "$KEYCLOAK" = 1 ] && echo "$KC_FRONTEND   (realm bioq; console admin/admin at /admin/master)" || echo "(disabled; BIOQ_KEYCLOAK=0)")
  db backend  : $DB_BACKEND$([ "$DB_BACKEND" = postgres ] && echo "   (svc bioq-postgres in ns bioq)")
  services    : ${SERVICES[*]}$([ "$GPU" = 1 ] && echo "   (gpu: nvidia.com/gpu x1)")
  weights dir : $MODELS_DIR -> /data/models   (put weights in <dir>/<svc>/)
  kubeconfig  : $KUBECONFIG   (export KUBECONFIG=... to use kubectl)

  smoke test:
    curl -s -H "X-API-Key: $API_KEY" -H "Host: public.example.com" \\
      http://127.0.0.1:$GATEWAY_PORT/v1/services

  run the gateway functional test (dockq-server):
    cd $REPO_ROOT/gateway && \\
    GATEWAY_BASE_URL=http://127.0.0.1:$GATEWAY_PORT GATEWAY_API_KEY=$API_KEY RUN_LOCAL_TESTS=1 \\
      uv run --with pytest --with pytest-asyncio python -m pytest tests/test_local_openfaas.py -v
${KC_INFO}
  tear down:  ./local-down.sh
EOF
