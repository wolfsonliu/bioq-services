# Deploy — deploying to FC + console configuration

English | [中文](deploy.zh.md)

> ← Back to the [Adding a service cookbook overview](./index.md)

This page covers the common constraints for deploying to FC, the FC console configuration (async task
mode + OSS mount), and concurrency testing.

## Deploying to FC

See the steps in [services/genie3-server/README.md §阿里云函数计算部署](../../services/genie3-server/README.md);
the common constraints are:

| Item | Requirement |
|---|---|
| Startup | `/healthz` responds within ≤ 120 s (**do not** load model weights at import time) |
| Listen | `0.0.0.0:9000` (CAPort) |
| Keep-alive | uvicorn `--timeout-keep-alive 900` |
| Image size | GPU images ≤ 15 GB (usually 1.5-5 GB after externalizing weights) |
| Architecture | `--platform linux/amd64` |
| NAS mount (jobs) | `/fc → /data`, sharing job files across instances / services |
| NAS mount (weights) | `/fc → /data/models/<svc>`, **read-only**; weights must be uploaded before first deploy |

**First-deploy weight upload** (one-time):

```bash
# 1. download locally to the staging dir
./services/<svc>/scripts/fetch_weights.sh

# 2. rsync to NAS (path matches the settings.weights_dir default)
rsync -av services/<svc>/weights/ <NAS-mount>:/data/models/<svc>/

# 3. verify after deploy
curl https://fc-<svc>-XXX.cn-hangzhou-vpc.fcapp.run/healthz/detail
# expect: {"status":"ok", "weights_loaded": true, "weights_missing": {}}
```

Push to harbor + update the FC function image:

```bash
make push-<svc>                # build + tag + push (using the version in the VERSION file)
# then update the function image in the FC console to harbor.ruosheng.bio/aliyun_fc/<svc>:vX.Y.Z
```

## FC async-task-mode console configuration

After deploying each service image to FC, the console needs the following settings for the task
endpoint to truly run in async task mode:

| Config item | Recommended value | Notes |
|---|---|---|
| Async task mode | **Enable** | unlocks `X-Fc-Invocation-Type: Async` + `GetAsyncTask` + dedup |
| Max async task concurrency | per GPU quota (e.g. 5-10) | overflow auto-queues, no more 429 |
| Async task retry count | 0 | failures need manual triage — avoid burning GPU quota |
| Single-instance concurrency | 1 (default for GPU services) | the instance holds the GPU exclusively while the task endpoint blocks |
| Session Affinity | keep the HeaderField config | legacy submit/poll still needs it; task endpoint doesn't depend on it but doesn't conflict |
| Keepalive URL | **clear it** | task endpoint HTTP is active for the whole job — no external keepalive needed |
| PreStop hook | keep | instance-destroy fallback, marks running jobs interrupted |
| Function timeout | 86400s | GPU instances can run up to 24h |
| **OSS mount** | data-plane bucket `bioagent-inputs` → `/mnt/oss`, **read-write (RW)** | required for gateway-invoked services: direct input read (the gateway rewrites `oss://` inputs to `/mnt/oss/...`) + output pass-back (the output-sink mirrors the job dir to `/mnt/oss`). Missing mount → inputs need downstream OSS credentials and download falls back to downstream proxying |

How to enable: **function details page → async config → edit "task mode"**; the OSS mount is at
**function details page → config → storage → NAS/OSS mount**.

Post-deploy smoke verification:

```bash
# 1. sync invoke (legacy) should still return 200 immediately
curl -X POST https://fc-<svc>-XXX.cn-hangzhou-vpc.fcapp.run/api/<name> ...

# 2. async invoke of the task endpoint should be 202 Accepted (not 200)
curl -i -X POST https://fc-<svc>-XXX.cn-hangzhou-vpc.fcapp.run/api/tasks/<name> \
    -H "X-Fc-Invocation-Type: Async" \
    -H "X-Bioagent-Job-Id: smoke-001" \
    -F "..."
# expect the first line: HTTP/1.1 202 Accepted

# 3. sync GET of JobInfo (with affinity to reduce polling instances)
curl https://fc-<svc>-XXX.cn-hangzhou-vpc.fcapp.run/api/jobs/smoke-001 \
    -H "X-Bioagent-Session-Id: smoke-001"
```

If step 2 returns 200 instead of 202, async task mode isn't enabled in the console, or the
`X-Fc-Invocation-Type` header isn't being recognized by the gateway.

All the smoke commands above are covered in
[`tests/test_fc_task.py`](./testing.md#12b-servicessvctestsstest_fc_taskpy) —
after the console config changes, run `pytest -m fc services/<svc>/tests/test_fc_task.py` for a
complete regression.