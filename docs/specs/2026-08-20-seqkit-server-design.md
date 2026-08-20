# seqkit-server 设计

日期: 2026-08-20
状态: 已实现（v0.0.1）
适用: 把 SeqKit（FASTA/FASTQ 工具包）包装成 bioq 服务；同时作为论文实验
[E11 — Extensibility](../../../repos/bioq-paper/experiments/E11_extensibility/README.md)
的演示对象（"新增一个工具 = 一个容器 + 一条注册表条目"）
相关: [新增 service cookbook](../adding-a-new-service/index.md) ·
[CLI 批处理模式](../../decisions/2026-05-29-cli-batch-mode.md) ·
[FC 异步任务模式](../../decisions/2026-06-17-fc-async-task-mode.md)

## 概述

[SeqKit](https://github.com/shenwei356/seqkit)（Shen & Zou 2016, *PLOS ONE*
11(10):e0163962）是一个跨平台、超快的 FASTA/FASTQ 序列操作工具包，单静态 Go
二进制、CPU-only、无模型权重。它在 bioq 服务族中扮演**轻量序列操作**角色：
给 agent / pipeline 提供确定性、可校验的序列统计与变换能力（本服务只包
`stats` 与 `seq`（反向互补）两个子命令）。

与已有服务的边界：服务族里第一个**非模型类**工具——不占用 GPU、不需要 NAS
权重，用来证明 bioq 的契约对"一个静态二进制"这类极简工具同样成立（E11 的
论点）。与 mmseqs2/diamond（序列搜索）无功能重叠。

## 设计目标

1. **纯 argv 包装**：上游是单静态二进制，不 patch、不 import、不设 PYTHONPATH；
   wrapper 只构造命令行。
2. **两个确定性 endpoint**（`stats` / `revcomp`），输出可被独立实现逐字节校验
   （E11 的 oracle 即由此成立）。
3. **双模式对齐**：HTTP submit/poll + `/api/tasks/*` + CLI 批处理共享同一
   `tools.py` argv builder 与 `adapter.detect_outputs`。
4. **无权重**：`/healthz/detail` 探测二进制是否就位（而非 weights_loaded）。
5. **vendor 固定版本**：`scripts/vendor.sh` 下载 v2.13.0 release tarball，
   sha256 + md5 双校验，产物落 `upstream/`（已被根 `.gitignore`
   `services/*-server/upstream/*` 覆盖）。
6. **cookbook 全程无捷径**：本服务是 E11 的证据，必须走标准骨架
   （settings/models/tools/adapter/app/__main__/tests/Dockerfile/VERSION/README）。

## Endpoint 拓扑

| Endpoint | 说明 |
|---|---|
| `POST /api/stats` | 提交 `seqkit stats`（submit/poll 模式） |
| `POST /api/tasks/stats` | 同上，FC 异步任务模式（阻塞至完成） |
| `POST /api/revcomp` | 提交 `seqkit seq -r -p` 反向互补（submit/poll） |
| `POST /api/tasks/revcomp` | 同上，task 模式 |

**v0.0.1 明确不做**：seqkit 其余子命令（grep / split / subseq / mutate /
concat / faidx / amplicon 等）；FASTQ 质量过滤与转换（`-q`/`-b` 之外的质控）；
多文件批量输入；gzip 输出。

## 请求 Schema

文件输入走路由层 `File(...)` / `Form(...)`（`input_fasta` /
`input_fasta_uri`），不在 model 上。

**StatsRequest**

| 字段 | 类型 | 默认 | 约束 | 说明 |
|---|---|---|---|---|
| `all_stats` | bool | `true` | — | 对应 `seqkit stats --all`（quartile / N50 / GC 等全列）；false 时仅核心列 |

输出固定为 TSV（`--tabular` 恒开），保证下游可机读。

**RevcompRequest**

| 字段 | 类型 | 默认 | 约束 | 说明 |
|---|---|---|---|---|
| `seq_type` | enum | `auto` | auto\|dna\|rna\|protein | 对应 `-t`；非 auto 时显式传给 seqkit（auto 时 seqkit 对 complement 会打一条推荐 `-t` 的 WARN，不影响结果） |

## 输出

```
<jobs_base_dir>/<job_id>/
├── input/
│   └── input.fasta        # 上传/URI 落盘后的原始输入
├── output/
│   ├── stats.tsv          # stats endpoint：seqkit stats --tabular 结果
│   └── revcomp.fasta      # revcomp endpoint：反向互补序列（header 保留）
└── logs/
    └── run.log            # 子进程 stdout+stderr
```

`detect_outputs`：`output/stats.tsv` 或 `output/revcomp.fasta` 任一存在且非空。

## 实现要点

| 决策点 | 选择 |
|---|---|
| 包装 vs patch | **包装**——静态二进制，零源码接触 |
| 依赖栈 | 无算法依赖；venv 里只有服务框架 + FastAPI 栈 |
| argv 构造 | `tools.stats_argv` / `tools.revcomp_argv`，`-o` 直写 `output/` |
| `detect_outputs()` | 两个产物任一非空（endpoint 无关，label 区分） |
| `/healthz/detail` 探针 | `seqkit_bin` 存在且可执行 + `seqkit version` 可跑 |
| 并发 | `max_concurrent_jobs=2`（CPU 工具，seqkit 自身 `-j` 多线程） |

## 配置

`env_prefix = SEQKIT_`

| 字段 | 默认 | 说明 |
|---|---|---|
| `jobs_base_dir` | `/data/seqkit_jobs` | 任务目录根 |
| `bin` | `/opt/seqkit/bin/seqkit` | seqkit 二进制路径（测试可指 `/bin/true`） |
| `threads` | `4` | `-j/--threads` |
| `max_concurrent_jobs` | `2` | 并发任务上限 |

## 部署目标

- FC **CPU** 实例（无 GPU；2 vCPU / 4 GB 级别即可），镜像 ~200 MB 量级。
- 控制台开启异步任务模式（task endpoints 是 `bioq run` 的入口）。
- tier: warm；无 NAS 权重挂载；经 gateway 调用时按 cookbook 接 OSS mount
  （`oss_mount: true`，输入直读 + 结果回传）。

## 测试策略

| 层 | 文件 | 说明 |
|---|---|---|
| offline HTTP | `tests/test_app.py` | `SEQKIT_BIN=/bin/true`：health / manifest / 两端点 submit / 422 |
| offline CLI | `tests/test_cli.py` | endpoint 注册 + argv builder + create_cli e2e（mock runner） |
| FC sync | `tests/test_fc.py` | `@pytest.mark.fc`：healthz / manifest / stats+revcomp 端到端 + 产物内容断言 |
| FC async | `tests/test_fc_task.py` | `@pytest.mark.fc`：`/api/tasks/*` 原子执行 + 幂等 |

fixture：`tests/data/input.fasta`（4 条固定 DNA 序列，含回文对照；与 E11
实验的 oracle fixture 同源）。

## 风险 / 限制

- 二进制仅 linux/amd64（Makefile `PLATFORM=linux/amd64` 固定），其它架构需
  换 vendor URL。
- `seq_type=auto` 时 seqkit 对 complement 打 WARN（建议显式 `-t`）——日志噪音，
  结果不受影响；要干净日志就传 `seq_type=dna`。
- 单文件输入；超大 FASTA 走流式读取，内存友好，但 HTTP 上传仍受网关体积限制。
- 作为 E11 演示：单工具、单作者，外推到"全服务族扩展性"需结合其余 38+ 服务
  均走同一 cookbook 的事实（见实验 README 的 threats to validity）。

## Sources

- 上游: https://github.com/shenwei356/seqkit — pin **v2.13.0**
  （release tarball `seqkit_linux_amd64.tar.gz`，
  sha256 `7d686de448464fada1b1988e2e07d693bec68768312da62846bc0e2b502bfc46`，
  upstream md5 `872368c1e24706dbd1f931d26b38d7d1`）
- 论文: Shen W, Zou Y (2016) SeqKit: A Cross-Platform and Ultrafast Toolkit for
  FASTA/Q File Manipulation. *PLOS ONE* 11(10): e0163962.
- 参考实现: [plip-server](../../services/plip-server/)（CPU + 文件上传 +
  task endpoint 的最相近骨架）
- 实验: bioq-paper `experiments/E11_extensibility/`（设计 + 脚本 + 证据）
