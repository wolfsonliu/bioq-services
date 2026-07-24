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

## 3. Seed an API key

VPC bypass is off in this stack, so requests need an API key:

```bash
docker compose -f deploy/compose/docker-compose.yml exec gateway \
    python scripts/seed_key.py --account-id local --key-id gk_local
# prints the secret — export it:
export BIOQ_KEY=<printed-secret>
```

## 4. Run a job

With the `bioq` CLI (point it at the local gateway):

```bash
export BIOQ_URL=http://localhost:9000
export BIOQ_API_KEY=$BIOQ_KEY
bioq services                                  # lists dockq-server
bioq run dockq-server score --file model=./a.pdb --file native=./b.pdb
bioq status <job_id>
bioq download <job_id> -o results.zip
```

Or with `curl` (job-centric flow):

```bash
JOB=$(uuidgen | tr -d - | cut -c1-20)
# stage an input through the gateway onto the shared volume:
curl -X PUT -H "x-api-key: $BIOQ_KEY" --data-binary @a.pdb \
    "http://localhost:9000/v1/files/users/local/$JOB/input/a.pdb"
# run (pass the staged file:// uri as the endpoint expects):
curl -X POST -H "x-api-key: $BIOQ_KEY" -H "x-bioagent-job-id: $JOB" \
    -H "content-type: application/json" \
    -d '{"model_uri":"file:///shared/users/local/'"$JOB"'/input/a.pdb"}' \
    "http://localhost:9000/v1/run/dockq-server/score"
curl -H "x-api-key: $BIOQ_KEY" "http://localhost:9000/v1/jobs/$JOB"
curl -L -H "x-api-key: $BIOQ_KEY" -o results.zip \
    "http://localhost:9000/v1/jobs/$JOB/download"
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
