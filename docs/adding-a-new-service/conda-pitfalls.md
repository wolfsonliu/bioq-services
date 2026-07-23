# 包装 conda-based upstream 的常见陷阱

日期: 2026-07-14
适用: [新增 bioagent service cookbook](./index.md) —— conda/micromamba 骨架的踩坑参考
相关: [dockerfile](./dockerfile.md) · [总览](./index.md)

> ← 返回 [新增 service cookbook 总览](./index.md)

diffdock-pp-server v0.0.1 → v0.0.7 的 7 次 rebuild + diffdock-server 双坑回填出来的一组
陷阱。**新 conda 服务落地前请把这一页过一遍。** conda/micromamba Dockerfile 骨架本身见
[dockerfile](./dockerfile.md)。

##### 包装 conda-based upstream 的常见陷阱

diffdock-pp-server v0.0.1 → v0.0.7 的 7 次 rebuild 里踩到 8 个坑，每个都值得
上升到 guide 层——同样的错在下一个 conda-based service 上很可能复现。**新 conda 服务落地前请把这一节过一遍。**

**1. `ENV LANG=C.UTF-8` 必备**（v0.0.7 修复）

上游 `open(path)` 无 `encoding=` 的调用在 FC 容器里默认 ASCII，任何 UTF-8 字符就崩。
特别是包装 upstream config yaml 时，注释里但凡有一个 em dash / 中文 / 章节号，
第二次读 yaml 时 100% 崩 `UnicodeDecodeError: 'ascii' codec ...`。
**Dockerfile runtime 阶段两行 ENV 关掉整类问题**：

```dockerfile
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
```

也可以顺手把 bundled 的 config yaml / patch / txt 里的 UTF-8 字符改成 ASCII
作为 defense-in-depth——但根治靠 LANG。

**2. Bundle upstream config yaml 时 strip runtime-varying keys**（v0.0.4 修复）

很多 upstream 有形如 `configs/inference.yaml` 或 `single_pair_inference.yaml` 的
默认 config，我们通常想把它 vendor 进镜像（论文调优参数、模型架构 等固定值）。
**但很多上游的 `process_args()` 会把 yaml 值 `override` argparse 已解析的 CLI 值**
（DiffDock-PP、ODesign、genie3 都是这个模式）。

verbatim 复制 yaml 的**灾难性后果**：CLI 上传的 `--data_file <user_upload_csv>`
被 yaml 里 `data.data_file: datasets/some_sample/splits.csv` 静默覆盖 →
**每次请求都跑上游自带的 sample 数据，忽略用户上传**。测试可能仍然"成功完成推理"
（因为 sample 数据存在），只是输出跟输入完全不相关。DiffDock-PP v0.0.4 就是这么坑
了两个 build 才被 postprocess 阶段的 KeyError 揪出来。

**规矩**：bundle upstream yaml 前**只保留固定项**（模型架构、diffusion sigma、
temperature 调优等），把所有 wrapper 通过 CLI 控制的键**手动 strip**：

| 一般要 strip 的 key | 为什么 |
|---|---|
| `data.data_file`, `data.data_path`, `data.pose_file` | 用户上传/URI 输入路径，必须走 CLI |
| `num_samples`, `seed`, `batch_size` | 用户可选参数 |
| `mode` / `use_confidence_model` / `mirror_ligand` 等 flag | 用户可切 |
| `checkpoint_path` / `weights_dir` / `save_path` | 走 settings + CLI |
| 训练相关（`lr`, `epochs`, `patience`）| 推理服务不用；留着也没坏但污染 args |

**并配对：wrapper 里加防御 assertion**：

```python
# services/<svc>/inference.py
upstream_args = upstream_parse()   # 这里内部会 yaml.safe_load(config_file)
# 若哪天 yaml 又被手抖加回 data_file，服务在第一次推理开跑前就 fail-loud
assert str(upstream_args.data_file) == str(csv_path), \
    f"data_file drift: yaml overriding CLI ({upstream_args.data_file!r})"
```

**3. Upstream 有硬编码目录约定时，wrapper 必须准确 mirror**（v0.0.5 修复）

很多 dataset loader 在 `__init__` 里悄悄加子目录：

```python
class DB5Loader(Loader):
    def __init__(self, args):
        super().__init__(args)
        self.root = os.path.join(self.root, "structures")   # ← 隐蔽约定
```

DIPS loader 用 `.dill` 文件、DB5 用 `structures/` 子目录、ProteinMPNN 用 `pdbs/`
子目录…每个 upstream 都有自己的品味。wrapper 里 `prepare_dataset_layout()` 组
"1 pair 数据集"时**必须先 grep upstream 里所有 `os.path.join(self.root, ...)`
的用法**，把子目录约定完整复刻。

诊断：symptomatically 表现为 `FileNotFoundError: <某某>/xxx/xxx.pdb`，位置在
`parse_pdb` / `read_files` 附近。**注意**：如果同时踩了 §2 的 yaml override 坑，
这个坑可能被掩盖（因为 loader 走的是 upstream sample 数据的路径，那里天然有正
确子目录）；只有 §2 修完后 §3 才显形。

**4. 上游 top-level `import` 死代码要用 sys.modules stub**（v0.0.2 + v0.0.3）

很多 upstream `main_xxx.py` 顶层无条件 `import wandb`（训练用的） /
`from matplotlib import pyplot as plt`（画图用的）—— 推理路径根本用不到，但
`import` 语句还是要成功，否则模块加载即崩 `ModuleNotFoundError: 'wandb'`。

**选项对比**：

| 方案 | 增镜像大小 | 缺点 |
|---|---|---|
| 装真包（`conda install wandb`）| +50 MB（wandb 传递依赖）/ +80 MB (matplotlib+字体) | 全是死代码 |
| **sys.modules stub**（推荐）| 0 KB | 需要在 upstream import 前注入 |

在 wrapper 里 `from <upstream> import main` **之前**注入 stub：

```python
import sys, types
sys.modules.setdefault("wandb", types.ModuleType("wandb"))
_mpl = sys.modules.setdefault("matplotlib", types.ModuleType("matplotlib"))
_plt = sys.modules.setdefault("matplotlib.pyplot", types.ModuleType("matplotlib.pyplot"))
_mpl.pyplot = _plt   # `from matplotlib import pyplot as plt` 需要属性访问

from <upstream>.main import main   # 现在能过了
```

`from X import Y` 需要 `Y` 是 `X` 的属性——子模块 stub 必须手动挂到父模块上。
副作用：如果 someone 未来把 wandb_sweep flag 打开或加真实 plt 调用，会 fail-loud
`AttributeError` —— 我们希望的行为。

**5. Docker COPY 之后加"filesystem sanity + module-chain import" 双 smoke**
（v0.0.3 + v0.0.4 修复）

Conda 服务的 debug 循环是 build（20 min）→ push（3 min）→ FC 部署 → 跑第一个推理
（~10 min）→ 崩。这个 loop 太慢，Dockerfile 里加两级 smoke 把大部分 upstream
import 陷阱拦在 build 阶段：

```dockerfile
COPY services/<svc>/upstream /opt/<svc>

# 1) < 1s 文件系统 sanity：vendor.sh 忘了 re-run 之类
RUN set -e; \
    for f in /opt/<svc>/src/main.py \
             /opt/<svc>/src/model/factory.py \
             /opt/<svc>/src/notebooks/utils.py; do \
        [ -f "$f" ] || { echo "ERROR: $f missing; run scripts/vendor.sh"; exit 1; }; \
    done

# 2) 完整 upstream module-chain smoke（含 wrapper 的 stub 注入）
RUN /opt/conda/envs/<env>/bin/python -c "\
import sys, types; \
sys.path.insert(0, '/opt/<svc>/src'); \
sys.modules.setdefault('wandb', types.ModuleType('wandb')); \
mpl = sys.modules.setdefault('matplotlib', types.ModuleType('matplotlib')); \
plt = sys.modules.setdefault('matplotlib.pyplot', types.ModuleType('matplotlib.pyplot')); \
mpl.pyplot = plt; \
from <upstream>.main import main; \
print('upstream module chain OK')"
```

Smoke #1 catches missing vendor files instantly（<1s）。Smoke #2 caches
missing conda deps / dead-import blowups / dead-code path-relative imports.
两个 smoke 加起来给 Docker build 加 ~30 s，比"到 FC 上跑 10 min 才崩"划算太多。

**6. vendor.sh 改了 exclude 规则后必须 rm -rf upstream/ 重跑**

Docker COPY 只看 host 上 `upstream/` 现在的内容，不看 vendor.sh 脚本。
`vendor.sh` 里 `--exclude='src/notebooks/'` 改成 `--exclude='*.ipynb'` 后，
如果 host 上的 `upstream/src/notebooks/` 目录还是空的（老 vendor 留下的），
新 build 仍然会缺 `utils_notebooks.py`。**规矩**：改 vendor.sh 排除规则后，
先 `rm -rf services/<svc>/upstream/` 再重跑 `./scripts/vendor.sh`。

##### 陷阱 7：uv 装 CUDA 扩展包时的双重坑（VCS + build isolation）

装 `openfold` / `apex` / `SE3Transformer` 等**从 github 装的 CUDA 扩展包**
时，Dockerfile 会连吃两个坑（diffdock-server 2026-07-06 v0.0.1 build 首连败两轮）：

**坑 A：git CLI 缺失**（`uv pip install "<pkg> @ git+..."`）
```
× Failed to download and build `openfold @ git+https://...`
├─▶ Git operation failed
╰─▶ Git executable not found. Ensure that Git is installed and available.
```
→ apt-install `git` 到 builder（详见 [dockerfile.md](./dockerfile.md) 的 8.3 骨架内注释）。

**坑 B：setup.py 顶层 `import torch` × PEP 517 隔离 build env**
```
× Failed to build `openfold @ git+...`
├─▶ Call to `setuptools.build_meta:__legacy__.build_wheel` failed
╰─▶ ModuleNotFoundError: No module named 'torch'
```
uv 默认 PEP 517 隔离 build——build 环境只有 `[build-system].requires`
里声明的包，即使 target env 装了 torch 也不可见。openfold 的 setup.py
需要 `import torch` 才能拿到 CUDA arch / nvcc 路径。修法：**`--no-build-isolation`**
+ **确保 target conda env 已装齐 build-time deps**（`torch`, `setuptools`,
`wheel`, `ninja`, `pybind11`, `numpy`）：

```dockerfile
RUN micromamba create -n <env> ... \
        conda-forge::ninja \
        conda-forge::wheel \
        conda-forge::pybind11=2.11.1 \
        "setuptools=69.5.1"

RUN uv pip install --python /opt/conda/envs/<env>/bin/python \
        --no-build-isolation \
        "openfold @ git+https://github.com/aqlaboratory/openfold.git@<sha>" \
        "dllogger @ git+https://github.com/NVIDIA/dllogger.git"
```

**同种坑会在下列 upstream 上复现**：apex（NVIDIA 混精度）、
从源码装的 torch-scatter/sparse（有 wheel 时优先用 wheel）、
SE3Transformer、DGL from source、其他自写 CUDA 扩展。用了 wheel（`+cu117`
后缀那种）就绕开——`--no-build-isolation` 只在 setup.py 层出坑。

##### 参考实现（这些坑踩过并已修复）

- [diffdock-pp-server](../../services/diffdock-pp-server/)（**踩全 6 个坑**，
  推荐做新 conda 服务的 checklist 样板）
- [diffdock-server](../../services/diffdock-server/)（**踩全 7 个坑**，
  含新版 §7 uv git+torch 扩展双坑）
- [diffusion-hopping-server](../../services/diffusion-hopping-server/)（1、4）
- [deeprank-ab-server](../../services/deeprank-ab-server/)（runtime monkey-patch）

