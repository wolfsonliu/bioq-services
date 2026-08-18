# Conda 镜像映射集中化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 21 个 service Dockerfile 里复制粘贴的 conda 镜像映射收拢到 `deploy/conda/mirrors.condarc` 单点，并把全员从已下线的 TUNA 切到 PKU。

**Architecture:** 一个共享片段文件只承载 `default_channels` + `custom_channels`（换镜像唯一要改的部分）；各 Dockerfile 用 `COPY` + `cat >>` 把它拼接到各自自有的 `channels`/`channel_priority` 上。守护脚本 `scripts/check_conda_mirrors.py` 作为 TDD 的「测试」，编码三条不变量，迁移前 RED、迁移后 GREEN。

**Tech Stack:** Dockerfile / micromamba (`libmamba` 2.x) / condarc YAML / Python 3 标准库（守护脚本，无第三方依赖）。

**设计文档:** `docs/specs/2026-08-18-conda-mirror-consolidation-design.md`（先读它了解背景与方案对比）。

---

## 文件结构

- **Create** `deploy/conda/mirrors.condarc` — 唯一镜像映射来源（PKU），含 `default_channels` + `custom_channels`。
- **Create** `scripts/check_conda_mirrors.py` — TDD 守护脚本（标准库）。
- **Modify** 21 个 `services/*-server/Dockerfile` — 删内联 URL 块，改 `COPY` + `cat >>`。
- **Modify** `docs/specs/2026-08-18-conda-mirror-consolidation-design.md`、`docs/plans/本文件` — 文档（本计划产物）。
- **Modify** `docs/adding-a-new-service/dockerfile.md` — 同步「新增 conda 服务」模板（见 Task 7）。

受改的 21 个服务（写 `/root/.condarc` 者）：alphafold、bindflow、chembounce、deeprank-ab、
diffdock、diffdock-pp、diffusion-hopping、drughive、flowmol、iggm、immunebuilder、megalodon、
odesign、openadmet、openbpmd、pocketxmol、ppiflow、qligfep、rfdiffusion2、semlaflow、turbohopp。

---

## 统一变换（Canonical Edit，所有服务通用）

对每个服务，做三处编辑：

```dockerfile
# (1) 在「写 /root/.condarc 的 RUN」之前插入一行：
COPY deploy/conda/mirrors.condarc /tmp/mirrors.condarc

# (2) heredoc 内：只保留 channels / show_channel_urls / channel_priority（及描述它们的
#     YAML 注释）；删除 default_channels + custom_channels 整块 URL，以及描述映射的
#     YAML 注释（"only map channels that TUNA hosts / do NOT map pyg|nvidia" 类）。

# (3) 在 heredoc 结束标记 EOF 之后，追加一行独立 RUN（不可写成 `&& cat ...` 接在 EOF 后，
#     那是非法 shell：`syntax error near unexpected token '&&'`）：
    RUN cat /tmp/mirrors.condarc >> /root/.condarc
```

---

### Task 1: 共享镜像片段文件

**Files:**
- Create: `deploy/conda/mirrors.condarc`

- [ ] **Step 1: 写入完整内容**

```yaml
# Single source of truth for the conda mirror mapping used by every service
# Dockerfile. Each builder stage does:
#
#     COPY deploy/conda/mirrors.condarc /tmp/mirrors.condarc
#     RUN cat > /root/.condarc <<'EOF' ... EOF
#     RUN cat /tmp/mirrors.condarc >> /root/.condarc
#
# i.e. the keys below are appended to the service's own `channels:` /
# `show_channel_urls` / `channel_priority` keys, which stay in each Dockerfile
# because they genuinely differ per service (channel list, strict vs flexible).
# The keys never collide with those, so concatenation stays valid YAML.
#
# WHY PKU (not TUNA): TUNA (Tsinghua) dropped pkgs/r and pkgs/msys2 from its
# mirror (both answer 404 today); libmamba 2.x treats a 404 repodata.json as a
# fatal error and aborts env create. PKU still mirrors the full defaults set.
#
# `bioconda` is mapped unconditionally — harmless for services that never
# resolve it, and keeps this file the only thing a mirror switch touches.
#
# Intentionally NOT mapped (fall through to upstream conda.anaconda.org, which
# works fine but is slow); Chinese mirrors 404 on their `label/*` or return
# stale/incomplete metadata:
#   - nvidia / nvidia/label/cuda-*   (cuda-nvcc, cuda toolkit labels)
#   - pyg                            (torch-scatter / sparse / cluster)
#   - dglteam / dglteam/label/*      (dgl CPU/CUDA builds)
#   - conda.rosettacommons.org       (pyrosetta, proprietary, no mirror)
# Do NOT add those as custom_channels entries here.
#
# To switch mirrors (TUNA / PKU / Aliyun / ...), edit ONLY the URLs in this file.
default_channels:
  - https://mirrors.pku.edu.cn/anaconda/pkgs/main
  - https://mirrors.pku.edu.cn/anaconda/pkgs/r
  - https://mirrors.pku.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.pku.edu.cn/anaconda/cloud
  pytorch: https://mirrors.pku.edu.cn/anaconda/cloud
  bioconda: https://mirrors.pku.edu.cn/anaconda/cloud
```

- [ ] **Step 2: 校验文件可被 COPY 到构建上下文**

Run: `git status --short deploy/conda/mirrors.condarc`
Expected: 文件在仓库根下（路径 `deploy/conda/mirrors.condarc`），构建上下文是仓库根，`COPY deploy/conda/mirrors.condarc` 可解析。

- [ ] **Step 3: Commit**

```bash
git add deploy/conda/mirrors.condarc
git commit -m "feat(build): add shared conda mirror fragment (PKU)"
```

---

### Task 2: 守护脚本（TDD — 先写失败的测试）

**Files:**
- Create: `scripts/check_conda_mirrors.py`

- [ ] **Step 1: 写守护脚本**

```python
#!/usr/bin/env python3
"""Invariant guard for the shared conda mirror mapping.

Every service Dockerfile that writes /root/.condarc must source its mirror
mapping from deploy/conda/mirrors.condarc (PKU), and no TUNA mirror URL may
remain. Exit 0 iff all invariants hold; 1 otherwise (with a report).

Usage: python3 scripts/check_conda_mirrors.py
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "deploy" / "conda" / "mirrors.condarc"

PKU = "https://mirrors.pku.edu.cn/anaconda"
TUNA = "mirrors.tuna.tsinghua.edu.cn"


def conda_dockerfiles():
    return sorted(
        p for p in (ROOT / "services").glob("*/Dockerfile")
        if "cat > /root/.condarc" in p.read_text()
    )


def main() -> int:
    errors = []

    # Invariant 1: shared file exists, is PKU-only, has both mapping keys.
    if not SHARED.exists():
        errors.append(f"missing {SHARED}")
    else:
        text = SHARED.read_text()
        if TUNA in text:
            errors.append(f"{SHARED} still references TUNA")
        for key in ("default_channels", "custom_channels"):
            if key not in text:
                errors.append(f"{SHARED} missing `{key}`")
        for channel in ("pkgs/main", "pkgs/r", "pkgs/msys2", "cloud"):
            if f"{PKU}/{channel}" not in text:
                errors.append(f"{SHARED} missing {PKU}/{channel}")

    # Invariant 2 + 3: every conda-using Dockerfile is TUNA-free and consumes
    # the shared file via COPY + cat append.
    for df in conda_dockerfiles():
        text = df.read_text()
        if TUNA in text:
            errors.append(f"{df}: still references TUNA mirror")
        if "COPY deploy/conda/mirrors.condarc" not in text:
            errors.append(f"{df}: missing COPY of shared mirrors.condarc")
        if "cat /tmp/mirrors.condarc >> /root/.condarc" not in text:
            errors.append(f"{df}: missing cat-append of shared mirrors.condarc")

    if errors:
        print(f"FAILED ({len(errors)} issues):")
        for e in errors:
            print(f"  - {e}")
        return 1

    n = len(conda_dockerfiles())
    print(f"OK: {n} conda Dockerfiles use the shared PKU mirror mapping.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: 运行，确认失败（RED）**

Run: `python3 scripts/check_conda_mirrors.py`
Expected: exit 1，报告列出 ~20 个「still references TUNA」+ 1 个 rfdiffusion2「missing COPY」的条目。

- [ ] **Step 3: Commit**

```bash
git add scripts/check_conda_mirrors.py
git commit -m "test(build): add conda mirror invariant guard (RED)"
```

---

### Task 3: 迁移 rfdiffusion2-server（完整示例，供后续 20 个照抄）

**Files:**
- Modify: `services/rfdiffusion2-server/Dockerfile:63-88`

- [ ] **Step 1: 应用编辑**

把当前（上一轮临时内联 PKU）的区块：

```dockerfile
# conda: PKU Open Source mirror for the standard channels. TUNA (Tsinghua) has
# dropped pkgs/r and pkgs/msys2 from its mirror (both answer 404 today); libmamba
# 2.x treats a 404 repodata.json as a fatal error and aborts env create, so we
# mirror the full defaults set through PKU instead, which still hosts all three.
# Three upstream channels we leave alone (fall through to conda.anaconda.org):
#   - conda.rosettacommons.org    pyrosetta (proprietary, no Chinese mirror)
#   - nvidia/label/cuda-12.4.0    Chinese mirrors carry `cloud/nvidia` (main)
#                                 but NOT `label/*` — mapping it 404s the build.
#   - dglteam/label/th24_cu124    same limitation.
# ---------------------------------------------------------------------------
RUN mkdir -p /root/.config/pip \
    && printf '[global]\nindex-url = https://repo.huaweicloud.com/repository/pypi/simple/\nextra-index-url = https://pypi.org/simple\ntimeout = 120\nretries = 5\n' \
       > /root/.config/pip/pip.conf \
    && cat > /root/.condarc <<'EOF'
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.pku.edu.cn/anaconda/pkgs/main
  - https://mirrors.pku.edu.cn/anaconda/pkgs/r
  - https://mirrors.pku.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.pku.edu.cn/anaconda/cloud
  pytorch: https://mirrors.pku.edu.cn/anaconda/cloud
EOF
```

替换为：

```dockerfile
# conda: PKU Open Source mirror for the standard channels, sourced from the
# shared deploy/conda/mirrors.condarc (single source of truth — see
# docs/specs/2026-08-18-conda-mirror-consolidation-design.md). Three upstream
# channels we leave alone (fall through to conda.anaconda.org):
#   - conda.rosettacommons.org    pyrosetta (proprietary, no Chinese mirror)
#   - nvidia/label/cuda-12.4.0    Chinese mirrors carry `cloud/nvidia` (main)
#                                 but NOT `label/*` — mapping it 404s the build.
#   - dglteam/label/th24_cu124    same limitation.
# ---------------------------------------------------------------------------
COPY deploy/conda/mirrors.condarc /tmp/mirrors.condarc
RUN mkdir -p /root/.config/pip \
    && printf '[global]\nindex-url = https://repo.huaweicloud.com/repository/pypi/simple/\nextra-index-url = https://pypi.org/simple\ntimeout = 120\nretries = 5\n' \
       > /root/.config/pip/pip.conf \
    && cat > /root/.condarc <<'EOF'
channels:
  - defaults
show_channel_urls: true
EOF
RUN cat /tmp/mirrors.condarc >> /root/.condarc
```

- [ ] **Step 2: 运行守护脚本**

Run: `python3 scripts/check_conda_mirrors.py`
Expected: rfdiffusion2 的三条错误消失（其余服务仍 RED）。

- [ ] **Step 3: Commit**

```bash
git add services/rfdiffusion2-server/Dockerfile
git commit -m "refactor(rfdiffusion2-server): source conda mirror from shared file"
```

---

### Task 4: 迁移其余 20 个服务

> 每个服务都套用 Task 3 的三步：删内联 URL 块 → 插 `COPY` + `cat >>` → 运行守护脚本确认该服务错误消失 →
> 按服务提交。下面按「变换后 heredoc 保留内容」+「需删除的 URL 块形状」给出每个服务的精确做法。
> 所有服务共同的删除规则：**从 `default_channels:`（若无则 `custom_channels:`）起，直到 `EOF` 前的
> 所有 URL 行，连同描述映射的 YAML 注释一起删掉**；`channels` / `show_channel_urls` /
> `channel_priority` 及其注释保留。`COPY` 永远插在写 condarc 的 `RUN` 之前；`cat >>` 永远接在 `EOF` 后。

- [ ] **alphafold-server** — `services/alphafold-server/Dockerfile:67-78`
  保留：`channels: [defaults]` + `show_channel_urls: true`。删除 `default_channels`(main/r/msys2) +
  `custom_channels`(conda-forge, pytorch)。
- [ ] **bindflow-server** — `services/bindflow-server/Dockerfile:60-69`
  保留：`channels: [conda-forge, bioconda]` + `show_channel_urls: true` + `channel_priority: flexible`。
  删除 `custom_channels`(conda-forge, bioconda)。
- [ ] **chembounce-server** — `services/chembounce-server/Dockerfile:62-72`
  保留：`channels: [defaults]` + `show_channel_urls: true`。删除 `default_channels`(main, r；本服务无 msys2 行)
  + `custom_channels`(conda-forge, bioconda)。
- [ ] **deeprank-ab-server** — `services/deeprank-ab-server/Dockerfile:67-78`
  保留：`channels: [defaults]` + `show_channel_urls: true`。删除 `default_channels` + `custom_channels`(conda-forge, pytorch)。
- [ ] **diffdock-server** — `services/diffdock-server/Dockerfile:77-90`
  保留：`channels: [defaults]` + `show_channel_urls: true` + `channel_priority: strict`。
  删除 `default_channels` + `custom_channels`(conda-forge, bioconda, pytorch)。
- [ ] **diffdock-pp-server** — `services/diffdock-pp-server/Dockerfile:62-78`
  保留：`channels: [defaults]` + `show_channel_urls: true` + `channel_priority: strict`。
  删除 `default_channels` + 映射注释「custom_channels: only map channels that TUNA actually hosts /
  Do NOT map pyg|nvidia」+ `custom_channels`(conda-forge, bioconda, pytorch)。
- [ ] **diffusion-hopping-server** — `services/diffusion-hopping-server/Dockerfile:57-77`
  保留：`channels: [defaults]` + `show_channel_urls: true` + strict 注释 + `channel_priority: strict`。
  删除映射注释「custom_channels: only map channels that TUNA hosts / IMPORTANT: do NOT map pyg|nvidia」+
  `default_channels` + `custom_channels`(conda-forge, bioconda, pytorch)。
- [ ] **drughive-server** — `services/drughive-server/Dockerfile:67-85`
  保留：`channels: [defaults]` + `show_channel_urls: true` + strict 注释 + `channel_priority: strict`。
  删除映射注释「custom_channels: only map channels that TUNA hosts / do NOT map nvidia」+
  `default_channels` + `custom_channels`(conda-forge, bioconda, pytorch)。
- [ ] **flowmol-server** — `services/flowmol-server/Dockerfile:58-74`
  保留：`channels: [defaults]` + `show_channel_urls: true` + `channel_priority: strict`。
  删除映射注释「custom_channels: only map channels that TUNA hosts / do NOT map pyg|nvidia|dglteam」+
  `default_channels` + `custom_channels`(conda-forge, bioconda, pytorch)。
- [ ] **iggm-server** — `services/iggm-server/Dockerfile:59-71`
  保留：`channels: [defaults]` + `show_channel_urls: true` + `channel_priority: strict`。
  删除 `default_channels` + `custom_channels`(conda-forge, pytorch)。
- [ ] **immunebuilder-server** — `services/immunebuilder-server/Dockerfile:76-88`
  保留：`channels: [defaults]` + `show_channel_urls: true`。删除 `default_channels` + `custom_channels`(conda-forge, bioconda, pytorch)。
- [ ] **megalodon-server** — `services/megalodon-server/Dockerfile:73-84`
  保留：`channels: [defaults]` + `show_channel_urls: true` + `channel_priority: strict`。
  删除 `default_channels` + `custom_channels`(conda-forge)。
- [ ] **odesign-server** — `services/odesign-server/Dockerfile:71-82`
  保留：`channels: [defaults]` + `show_channel_urls: true`。删除 `default_channels` + `custom_channels`(conda-forge, pytorch)。
- [ ] **openadmet-server** — `services/openadmet-server/Dockerfile:78-89`
  保留：`channels: [conda-forge]` + `show_channel_urls: true` + flexible 注释 + `channel_priority: flexible`。
  删除 `custom_channels`(conda-forge)。
- [ ] **openbpmd-server** — `services/openbpmd-server/Dockerfile:55-62`
  保留：`channels: [conda-forge]` + `show_channel_urls: true` + `channel_priority: flexible`。
  删除 `custom_channels`(conda-forge)。
- [ ] **pocketxmol-server** — `services/pocketxmol-server/Dockerfile:69-81`
  保留：`channels: [defaults]` + `show_channel_urls: true` + `channel_priority: strict`。
  删除 `default_channels` + `custom_channels`(conda-forge, bioconda)。
- [ ] **ppiflow-server** — `services/ppiflow-server/Dockerfile:144-156`
  保留：`channels: [defaults]` + `show_channel_urls: true`。删除 `default_channels` + `custom_channels`(conda-forge, pytorch, bioconda)。
  另：上方 `:116-143` 大段「Redirect conda channels to TUNA mirror … TUNA is more reliable」注释
  需精简为一句「conda 镜像映射见共享文件 deploy/conda/mirrors.condarc；上游 nvidia/schrodinger/ostrokach/anaconda
  继续回落到 conda.anaconda.org」。此服务是**独立** `RUN cat > /root/.condarc`（无 mkdir/printf 链），
  `COPY` 插在该 `RUN` 前，`cat >>` 接在该 `EOF` 后。
- [ ] **qligfep-server** — `services/qligfep-server/Dockerfile:77-86`
  保留：`channels: [conda-forge]` + `show_channel_urls: true` + strict 注释 + `channel_priority: strict`。
  删除 `custom_channels`(conda-forge)。
- [ ] **semlaflow-server** — `services/semlaflow-server/Dockerfile:59-71`
  保留：`channels: [defaults]` + `show_channel_urls: true` + `channel_priority: strict`。
  删除 `default_channels` + `custom_channels`(conda-forge, pytorch)。
- [ ] **turbohopp-server** — `services/turbohopp-server/Dockerfile:60-73`
  保留：`channels: [defaults]` + `show_channel_urls: true` + `channel_priority: strict`。
  删除 `default_channels` + `custom_channels`(conda-forge, bioconda, pytorch)。

- [ ] **Step 1: 逐个应用上述编辑**（每个服务按 Task 3 三步处理，每个服务独立 commit，命名
  `refactor(<svc>): source conda mirror from shared file`）。
- [ ] **Step 2: 每个服务改完都跑一次守护脚本，确认该服务的错误消失。**

---

### Task 5: 全绿校验 + 收尾

- [ ] **Step 1: 运行守护脚本（GREEN）**

Run: `python3 scripts/check_conda_mirrors.py`
Expected: exit 0，输出 `OK: 21 conda Dockerfiles use the shared PKU mirror mapping.`

- [ ] **Step 2: 反向确认无 TUNA 残留**

Run: `grep -rn "mirrors.tuna.tsinghua.edu.cn" --include=Dockerfile services/ || echo "no TUNA mirrors"`
Expected: `no TUNA mirrors`

- [ ] **Step 3: 确认共享文件是本列表里唯一的 conda 镜像 URL 来源**

Run: `grep -rln "anaconda/cloud\|anaconda/pkgs" --include=Dockerfile services/`
Expected: 空（URL 已全部移入 `deploy/conda/mirrors.condarc`）。

- [ ] **Step 4: 抽查一个 defaults 服务、一个 conda-forge-only 服务构建到 `env create` 之前阶段**
  （可选，代价大；至少静态确认 YAML 拼接合法。完整镜像构建按需在 CI / 本地 `make build-*` 验证）。

- [ ] **Step 5: Commit 文档与模板**

```bash
git add docs/specs/2026-08-18-conda-mirror-consolidation-design.md \
        docs/plans/2026-08-18-conda-mirror-consolidation.md
git commit -m "docs: conda mirror consolidation design + plan"
```

---

### Task 6: 更新「新增服务」模板

**Files:**
- Modify: `docs/adding-a-new-service/dockerfile.md`

- [ ] **Step 1: 把文档中「内联 `.condarc` 写 TUNA mirror」的示例改为「`COPY deploy/conda/mirrors.condarc /tmp/mirrors.condarc` + heredoc 只保留 channels/priority + `cat /tmp/mirrors.condarc >> /root/.condarc`」，并指向本设计文档。**

- [ ] **Step 2: Commit**

```bash
git add docs/adding-a-new-service/dockerfile.md
git commit -m "docs: new-service conda template uses shared mirror file"
```

---

## Self-Review

- **Spec coverage：** 设计文档 §2 的三个目标 → Task 1（单点）、Task 4（切 PKU）、Task 2/5（守护脚本）均覆盖。
- **Placeholder scan：** 无 TBD/TODO；Task 4 逐服务给出保留/删除要素；共享文件与守护脚本为完整代码。
- **Type consistency：** 共享文件 key（`default_channels`/`custom_channels`）与守护脚本 Invariant 1 一致；
  `COPY deploy/conda/mirrors.condarc` / `cat /tmp/mirrors.condarc >> /root/.condarc` 两行字符串在
  模拟块、Task 3、Task 4、守护脚本 Invariant 2/3 中完全一致。