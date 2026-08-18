# Conda 镜像映射集中化设计

- **日期**：2026-08-18
- **状态**：草案（待评审）
- **范围**：`services/*/Dockerfile` 中所有写 `/root/.condarc` 的服务（共 21 个）的构建期 conda 镜像映射
- **关联计划**：`docs/plans/2026-08-18-conda-mirror-consolidation.md`

---

## 1. 背景与问题

### 1.1 直接故障

`make push-rfdiffusion2-server` 在 builder 阶段 `micromamba env create` 失败：

```
critical libmamba Multiple errors occurred:
    Transfer finalized, status: 404 [https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r/noarch/repodata.json]
    Subdir .../pkgs/main/noarch not loaded!
    Subdir .../pkgs/r/noarch not loaded!
    Subdir .../pkgs/msys2/noarch not loaded!
    Subdir conda-forge/noarch not loaded!
```

根因两层叠加：

1. **清华 TUNA 镜像已下线 `pkgs/r` 与 `pkgs/msys2` 两个 defaults 子频道**（实测整个频道 404；
   `pkgs/main`、`cloud/conda-forge`、`cloud/pytorch` 仍正常）。
2. **libmamba 2.x 把 `repodata.json` 的 404 视为致命错误**：一个子频道的 404 会中断整个频道加载
   （错误里其余 `not loaded!` 都是连锁症状，并非各自出错）。

### 1.2 结构性债务

镜像映射（`default_channels` + `custom_channels`）被逐字复制粘贴进了 **21 个** service 的
Dockerfile。一次换镜像 / 镜像站下线，就要改 21 处，且极易漏改（本次就是只修了一个、其余 16 个
仍在引用已下线的 TUNA）。这也是本次故障暴露出的真正债。

---

## 2. 目标

1. **单点治理**：镜像映射 URL 只存在一个文件里；切换镜像（TUNA / PKU / 阿里云 …）只改这一处。
2. **修复当前故障**：21 个服务全部脱离 TUNA，统一指向仍完整托管 defaults 频道的 PKU 镜像。
3. **机器可校验**：一个可跑的守护脚本，任何服务回退到 TUNA 或漏挂共享文件即失败（TDD 的「测试」）。

非目标：不统一 `channels:` 顶层列表、`channel_priority`、pip/apt 镜像；这些本就按服务不同。

---

## 3. 现状调研

全部 21 个写 `/root/.condarc` 的服务，其 condarc 形状差异如下（决定「只能抽共享映射、不能整份文件共享」）：

| 服务 | `channels:` | `channel_priority` | 原 `custom_channels` 键 |
|---|---|---|---|
| alphafold | defaults | （默认） | conda-forge, pytorch |
| bindflow | conda-forge, bioconda | flexible | conda-forge, bioconda |
| chembounce | defaults | （默认） | conda-forge, bioconda |
| deeprank-ab | defaults | （默认） | conda-forge, pytorch |
| diffdock | defaults | strict | conda-forge, bioconda, pytorch |
| diffdock-pp | defaults | strict | conda-forge, bioconda, pytorch |
| diffusion-hopping | defaults | strict | conda-forge, bioconda, pytorch |
| drughive | defaults | strict | conda-forge, bioconda, pytorch |
| flowmol | defaults | strict | conda-forge, bioconda, pytorch |
| iggm | defaults | strict | conda-forge, pytorch |
| immunebuilder | defaults | （默认） | conda-forge, bioconda, pytorch |
| megalodon | defaults | strict | conda-forge |
| odesign | defaults | （默认） | conda-forge, pytorch |
| openadmet | conda-forge | flexible | conda-forge |
| openbpmd | conda-forge | flexible | conda-forge |
| pocketxmol | defaults | strict | conda-forge, bioconda |
| ppiflow | defaults | （默认） | conda-forge, pytorch, bioconda |
| qligfep | conda-forge | strict | conda-forge |
| rfdiffusion2 | defaults | （默认） | conda-forge, pytorch |
| semlaflow | defaults | strict | conda-forge, pytorch |
| turbohopp | defaults | strict | conda-forge, bioconda, pytorch |

观察：

- `default_channels`（defaults 的展开）在所有 `defaults` 服务里几乎相同（chembounce 例外：漏了
  `msys2`，但无关紧要——linux 环境从不解析 msys2/windows 频道）。
- `custom_channels` 的键是 `{conda-forge, pytorch, bioconda}` 的**任意子集**，顺序不一。
- 惟一的真正公共标量是 **镜像 base URL**（`https://mirrors.<host>/anaconda`）。

---

## 4. 方案对比

| 方案 | 单一事实来源 | 缺点 |
|---|---|---|
| A. 共享 mirror 片段文件 + 各 Dockerfile `COPY` + `cat >>` | `deploy/conda/mirrors.condarc` 一个文件 | 每服务加一行 `COPY` + 一行 `cat >>` |
| B. build-arg `CONDA_MIRROR` + Makefile 传值 | Makefile 变量 | 17 份 Dockerfile 仍需 `ARG`+展开 heredoc；裸 `docker build` 还得带参数，易出错 |
| C. 整份 `~/.condarc` 复制粘贴 | 无（本来就是这个问题） | 21 处漂移 |

**采用方案 A**。理由：

- 直接把「因换镜像而变」的部分（URL）抽成单文件；各服务自有的 `channels` / `channel_priority` 仍在
  各自 Dockerfile，不改动求解行为。
- 用 **`COPY` 而非 bind-mount**：与仓库内 framework 的规则一致——镜像文件变更要进镜像、并正确失效
  builder 缓存。
- 裸 `docker build -f <svc>/Dockerfile .` 开箱即用，无需额外构建参数。

---

## 5. 详细设计

### 5.1 共享文件

`deploy/conda/mirrors.condarc`（构建上下文根 = 仓库根，`COPY` 路径合法）：

```yaml
default_channels:
  - https://mirrors.pku.edu.cn/anaconda/pkgs/main
  - https://mirrors.pku.edu.cn/anaconda/pkgs/r
  - https://mirrors.pku.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.pku.edu.cn/anaconda/cloud
  pytorch: https://mirrors.pku.edu.cn/anaconda/cloud
  bioconda: https://mirrors.pku.edu.cn/anaconda/cloud
```

要点：

- `default_channels` / `custom_channels` 两个 key 与各服务自身 heredoc 里的 `channels:` /
  `show_channel_urls` / `channel_priority` **不重叠**，`cat >>` 拼接后仍是合法单文档 YAML。
- `bioconda` 无条件提供：对不用它的服务只是多一条 name→URL 表项，无副作用，换取「文件即全集」。
- 文件头注释明确「**故意不映射** nvidia / pyg / dglteam / conda.rosettacommons.org」——这些回落到
  `conda.anaconda.org`（镜像站对它们的 `label/*` 404，或根本无镜像），把原先散在各服务 heredoc 的
  注释知识收拢到一处。

### 5.2 每个 service Dockerfile 的变换

对每个写 `cat > /root/.condarc <<'EOF'` 的 service：

1. 在写 condarc 的 `RUN` **之前**插入一行：
   ```dockerfile
   COPY deploy/conda/mirrors.condarc /tmp/mirrors.condarc
   ```
2. heredoc 里**只保留** `channels:` / `show_channel_urls:` / `channel_priority:` 以及描述这些键的
   YAML 注释；**删除** `default_channels:` / `custom_channels:` 整块 URL 映射，以及描述映射的
   YAML 注释（如「only map channels that TUNA hosts / do NOT map pyg/nvidia」——该知识已迁入共享文件头）。
3. 在 `EOF` 之后追加：
   ```dockerfile
       && cat /tmp/mirrors.condarc >> /root/.condarc
   ```

变换前后示意（rfdiffusion2）：

```dockerfile
# before
    && cat > /root/.condarc <<'EOF'
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  ...
custom_channels:
  conda-forge: ...
  pytorch: ...
EOF

# after
COPY deploy/conda/mirrors.condarc /tmp/mirrors.condarc     # 插到 RUN 前
RUN mkdir -p /root/.config/pip \
    && printf '...' > /root/.config/pip/pip.conf \
    && cat > /root/.condarc <<'EOF'
channels:
  - defaults
show_channel_urls: true
EOF
    && cat /tmp/mirrors.condarc >> /root/.condarc
```

---

## 6. 测试策略（TDD）

守护脚本 `scripts/check_conda_mirrors.py`（仅标准库，`python3 scripts/check_conda_mirrors.py` 运行，
退出码 0=通过 / 1=失败）编码三条不变量：

1. `deploy/conda/mirrors.condarc` 存在、含 `default_channels` + `custom_channels`、所有 URL 均为 PKU、
   无 TUNA 残留。
2. 每个写 `/root/.condarc` 的 service Dockerfile **不再包含** `mirrors.tuna.tsinghua.edu.cn`。
3. 每个这样的 Dockerfile 包含 `COPY deploy/conda/mirrors.condarc` 与
   `cat /tmp/mirrors.condarc >> /root/.condarc` 两行。

TDD 节奏：脚本先写（RED——当前 20 个服务仍引用 TUNA、rfdiffusion2 缺 COPY），逐服务迁移直至
GREEN，最后全绿提交。详细步骤见关联计划。

---

## 7. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| PKU 镜像某 channels 后端实际缺包 | 个别 solve 失败 | 已逐个子频道返回 200 + 真实 repodata JSON 验证；且 PKU 是官方长期镜像站，比 TUNA 当前状态更完整 |
| `cat >>` 拼接出非法 YAML | 构建期 condarc 解析失败 | 两段 key 不重叠；守护脚本 + 人工抽查 |
| 额外映射（bioconda/pytorch 全集）引入串扰 | 求解慢 / 误选 | 因 `channel_priority` 不变、多数走 strict；未引用的 custom_channel 只是未用表项，不参与求解 |
| 漏改/回退（新服务复制旧模板） | 故障复发 | 守护脚本列为 CI/本地门禁，模板更新到 `docs/adding-a-new-service/dockerfile.md` |

---

## 8. 迁移清单

见 `docs/plans/2026-08-18-conda-mirror-consolidation.md`（逐服务 TDD 任务）。