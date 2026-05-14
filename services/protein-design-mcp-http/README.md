# protein-design-mcp-http

HTTP wrapper for the upstream [protein-design-mcp](https://github.com/jasonkim8652/protein-design-mcp) stdio MCP server. Exposes the same tools over two MCP HTTP transports (SSE + Streamable HTTP) so the service can be deployed on Alibaba Cloud Function Compute or any other HTTP host.

Upstream code under `opensource/protein-design-mcp/` is **not modified** — the wrapper imports the existing `Server` instance and bridges it to HTTP transports.

## Layout

```
services/protein-design-mcp-http/
├── Dockerfile         — self-contained build (inlines upstream Dockerfile stages)
├── http_server.py     — Starlette app: /healthz, /sse, /messages/, /mcp
└── README.md          — this file
```

## Endpoints

| Path                              | Method | Purpose                                                    |
|-----------------------------------|--------|------------------------------------------------------------|
| `/healthz`                        | GET    | Plaintext health probe (`ok`)                              |
| `/sse`                            | GET    | MCP 2024 SSE event stream                                  |
| `/messages/?session_id=<id>`      | POST   | SSE counterpart — client→server JSON-RPC                   |
| `/mcp`                            | ANY    | MCP 2025-03-26 Streamable HTTP transport (`stateless=True`)|

## Build

The Dockerfile is **self-contained** — it builds CUDA + PyTorch + RFdiffusion + ProteinMPNN + ColabFold + OpenFold + OpenMM + the upstream MCP server + the HTTP wrapper in a single pipeline. No prebuilt `protein-design-mcp:latest` is required.

Build context MUST be the **bioagent project root** so both `opensource/` and `services/` are reachable:

```bash
# From bioagent project root
docker build --platform linux/amd64 \
    -f services/protein-design-mcp-http/Dockerfile \
    -t protein-design-mcp-http:latest .
```

Expected:
- Full first build: 45–60+ minutes, ~14–18 GB final image (model weights ~3.7 GB are baked in — see below)
- Wrapper-only iteration (after first build): seconds, layer cache covers everything before the HTTP wrapper stage

### Baked-in model weights

The Dockerfile runs `opensource/protein-design-mcp/docker/download_models.py` at build time so weights ship inside the image and the container never fetches them at runtime:

| Model         | File                      | Size    | Source                                      |
|---------------|---------------------------|---------|---------------------------------------------|
| RFdiffusion   | `Complex_base_ckpt.pt`    | ~1.5 GB | `files.ipd.uw.edu` (UW)                     |
| ProteinMPNN   | `v_48_020.pt`             | ~150 MB | `github.com/dauparas/ProteinMPNN`           |
| ESMFold       | `esmfold_v1` torch hub    | ~2 GB   | `dl.fbaipublicfiles.com` (via `fair-esm`)   |

Build-time requirements: ~3 GB free RAM on the build host (ESMFold model is loaded on CPU once to populate the torch hub cache); reachable network for the three sources above. If a download flakes, `download_models.py` logs a warning and continues — the build won't fail, but the affected tool will error at runtime.

The script also creates symlinks under `/opt/RFdiffusion/models/` and `/opt/ProteinMPNN/vanilla_model_weights/` pointing at the baked weights, which is what the upstream tool handlers expect.

Prereq: `opensource/protein-design-mcp/` exists locally with `pyproject.toml`, `src/`, and `docker/` populated. The `opensource/` directory is git-ignored but Docker COPY reads from disk, so as long as the files are checked out it builds fine.

## Run

CPU mode (RFdiffusion disabled, the rest of the pipeline still works):

```bash
docker run --rm -p 9000:9000 -e DEVICE=cpu protein-design-mcp-http:latest
```

GPU mode with persistent model cache:

```bash
docker run --rm --gpus all -p 9000:9000 \
    -v $(pwd)/models:/models \
    -v $(pwd)/cache:/cache \
    protein-design-mcp-http:latest
```

Override port:

```bash
docker run --rm -p 8080:8080 -e PORT=8080 protein-design-mcp-http:latest
```

## Smoke test

```bash
# Health
curl http://localhost:9000/healthz
# → ok

# SSE — should return an event stream announcing the /messages/ endpoint
curl -N http://localhost:9000/sse

# Streamable HTTP initialize
curl -X POST http://localhost:9000/mcp \
     -H 'Content-Type: application/json' \
     -H 'Accept: application/json, text/event-stream' \
     -d '{"jsonrpc":"2.0","id":1,"method":"initialize",
          "params":{"protocolVersion":"2025-03-26","capabilities":{},
                    "clientInfo":{"name":"probe","version":"0.1"}}}'

# Streamable HTTP tools/list
curl -X POST http://localhost:9000/mcp \
     -H 'Content-Type: application/json' \
     -H 'Accept: application/json, text/event-stream' \
     -d '{"jsonrpc":"2.0","id":2,"method":"tools/list"}'
```

## Environment variables

Wrapper-specific:

| Var          | Default | Purpose                                       |
|--------------|---------|-----------------------------------------------|
| `PORT`       | `9000`  | uvicorn listen port                           |
| `LOG_LEVEL`  | `info`  | uvicorn log level (`debug`/`info`/`warning`)  |

Upstream (preserved from `opensource/protein-design-mcp/Dockerfile`):

| Var                       | Default              | Purpose                                  |
|---------------------------|----------------------|------------------------------------------|
| `DEVICE`                  | `auto`               | `auto` / `cuda` / `cpu`                  |
| `MODELS_DIR`              | `/models`            | Where model weights are cached           |
| `CACHE_DIR`               | `/cache`             | Runtime cache                            |
| `COLABFOLD_BACKEND`       | `api`                | `api` (default) or `local`               |
| `SKIP_MODEL_DOWNLOAD`     | `true`               | Weights are baked at build time; leave `true` |

## Deploying to Alibaba Cloud Function Compute

1. Push the image to Alibaba Container Registry (ACR), e.g. `registry.cn-hangzhou.aliyuncs.com/<ns>/protein-design-mcp-http:latest`.
2. Create a **Custom Container** function with HTTP trigger.
   - Listening port: `9000` (matches `EXPOSE 9000` / `ENV PORT`).
   - Health check path: `/healthz`.
   - For GPU tools (RFdiffusion etc.): pick a **GPU instance type**. CPU-only instances can still serve ESMFold / ProteinMPNN / OpenMM with `DEVICE=cpu`.
3. **Cold start is slow** — importing `protein_design_mcp.server` pulls in torch + CUDA bindings. Either enable **reserved instances** or **single-instance concurrency** to avoid 30+ s cold-start latency on every burst.
4. `stateless=True` is hard-coded in `http_server.py` for the Streamable HTTP transport, so requests can land on any instance.

## Notes

- The wrapper imports `from protein_design_mcp.server import server` at module load. This triggers `import torch` for the upstream device-detect block — heavy but unavoidable without modifying upstream.
- All tool handlers in the upstream `server.py` use lazy imports (inside the handler functions), so individual tool weights are downloaded only on first call to that tool.
- The Streamable HTTP session manager runs inside Starlette's lifespan context. If you swap to a non-Starlette ASGI host, replicate the `async with session_manager.run(): yield` pattern from `http_server.py`.
