# Common pitfalls when wrapping conda-based upstream

English | [中文](conda-pitfalls.zh.md)

> ← Back to the [Adding a service cookbook overview](./index.md)

Eight pitfalls hit across seven rebuilds of diffdock-pp-server v0.0.1 → v0.0.7 (plus two backfilled
on diffdock-server), each worth promoting to guide level — the same mistake is very likely to recur on
the next conda-based service. **Read this page before landing a new conda service.** The
conda/micromamba Dockerfile skeleton itself is in [dockerfile](./dockerfile.md).

**1. `ENV LANG=C.UTF-8` is required** (fixed in v0.0.7)

Upstream calls to `open(path)` without `encoding=` default to ASCII in an FC container, so any UTF-8
character crashes. In particular, when wrapping an upstream config yaml, a single em dash / Chinese
character / section number in a comment makes the second yaml read crash 100% with
`UnicodeDecodeError: 'ascii' codec ...`. **Two ENV lines in the Dockerfile runtime stage disable the
whole class of problems**:

```dockerfile
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
```

You can also convert the UTF-8 characters in bundled config yaml / patch / txt files to ASCII as
defense-in-depth — but the root fix is LANG.

**2. Strip runtime-varying keys when bundling an upstream config yaml** (fixed in v0.0.4)

Many upstreams have a default config like `configs/inference.yaml` or `single_pair_inference.yaml`
that we usually want to vendor into the image (paper-tuned params, model architecture, and other fixed
values). **But many upstream `process_args()` implementations `override` the already-parsed CLI values
with the yaml values** (DiffDock-PP, ODesign, and genie3 all follow this pattern).

The **disastrous consequence** of copying the yaml verbatim: the CLI-uploaded
`--data_file <user_upload_csv>` is silently overridden by `data.data_file:
datasets/some_sample/splits.csv` in the yaml →
**every request runs the upstream's bundled sample data, ignoring the user upload**. Tests may still
"complete inference successfully" (because the sample data exists) — the output just has no relation
to the input. DiffDock-PP v0.0.4 was burned by this for two builds before a KeyError in the
postprocess stage caught it.

**The rule**: before bundling an upstream yaml, **keep only the fixed items** (model architecture,
diffusion sigma, temperature tuning, etc.) and **manually strip** every key the wrapper controls via
CLI:

| Keys to generally strip | Why |
|---|---|
| `data.data_file`, `data.data_path`, `data.pose_file` | user upload/URI input paths — must go through CLI |
| `num_samples`, `seed`, `batch_size` | user-selectable params |
| flags like `mode` / `use_confidence_model` / `mirror_ligand` | user-switchable |
| `checkpoint_path` / `weights_dir` / `save_path` | go through settings + CLI |
| training-related (`lr`, `epochs`, `patience`) | unused by inference; harmless to keep but pollutes args |

**Pair it with**: add a defensive assertion in the wrapper:

```python
# services/<svc>/inference.py
upstream_args = upstream_parse()   # internally does yaml.safe_load(config_file)
# if data_file ever sneaks back into the yaml, the service fails loud before the first inference starts
assert str(upstream_args.data_file) == str(csv_path), \
    f"data_file drift: yaml overriding CLI ({upstream_args.data_file!r})"
```

**3. When upstream has hard-coded directory conventions, the wrapper must mirror them exactly**
(fixed in v0.0.5)

Many dataset loaders quietly append a subdirectory in `__init__`:

```python
class DB5Loader(Loader):
    def __init__(self, args):
        super().__init__(args)
        self.root = os.path.join(self.root, "structures")   # ← hidden convention
```

The DIPS loader uses `.dill` files, DB5 uses a `structures/` subdirectory, ProteinMPNN uses a `pdbs/`
subdirectory… every upstream has its own taste. When the wrapper assembles a "1-pair dataset" in
`prepare_dataset_layout()`, it **must first grep upstream for all `os.path.join(self.root, ...)`
usages** and replicate the subdirectory conventions fully.

Diagnosis: symptomatically it shows up as `FileNotFoundError: <something>/xxx/xxx.pdb` near
`parse_pdb` / `read_files`. **Note**: if you also hit the §2 yaml-override pitfall, this one can be
masked (because the loader walks the upstream sample-data path, which naturally has the correct
subdirectory); §3 only becomes visible after §2 is fixed.

**4. Stub upstream top-level dead-code `import`s via sys.modules** (v0.0.2 + v0.0.3)

Many upstream `main_xxx.py` files unconditionally `import wandb` (for training) /
`from matplotlib import pyplot as plt` (for plotting) at top level — the inference path never uses
them, but the `import` statement must still succeed or the module load crashes with
`ModuleNotFoundError: 'wandb'`.

**Option comparison**:

| Approach | Image-size increase | Downside |
|---|---|---|
| install the real package (`conda install wandb`) | +50 MB (wandb transitive deps) / +80 MB (matplotlib+fonts) | all dead code |
| **sys.modules stub** (recommended) | 0 KB | must be injected before the upstream import |

Inject the stubs in the wrapper **before** `from <upstream> import main`:

```python
import sys, types
sys.modules.setdefault("wandb", types.ModuleType("wandb"))
_mpl = sys.modules.setdefault("matplotlib", types.ModuleType("matplotlib"))
_plt = sys.modules.setdefault("matplotlib.pyplot", types.ModuleType("matplotlib.pyplot"))
_mpl.pyplot = _plt   # `from matplotlib import pyplot as plt` needs attribute access

from <upstream>.main import main   # now works
```

`from X import Y` requires `Y` to be an attribute of `X` — the submodule stub must be manually
attached to its parent module. Side effect: if someone later enables the wandb_sweep flag or adds a
real plt call, it fails loud with `AttributeError` — which is exactly the behavior we want.

**5. Add a dual "filesystem sanity + module-chain import" smoke after Docker COPY**
(fixed in v0.0.3 + v0.0.4)

The conda-service debug loop is build (20 min) → push (3 min) → FC deploy → run first inference
(~10 min) → crash. That loop is too slow, so add two smoke levels in the Dockerfile to catch most
upstream import traps at build time:

```dockerfile
COPY services/<svc>/upstream /opt/<svc>

# 1) < 1s filesystem sanity: vendor.sh forgotten to re-run, etc.
RUN set -e; \
    for f in /opt/<svc>/src/main.py \
             /opt/<svc>/src/model/factory.py \
             /opt/<svc>/src/notebooks/utils.py; do \
        [ -f "$f" ] || { echo "ERROR: $f missing; run scripts/vendor.sh"; exit 1; }; \
    done

# 2) full upstream module-chain smoke (including the wrapper's stub injection)
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

Smoke #1 catches missing vendor files instantly (<1s). Smoke #2 catches missing conda deps /
dead-import blowups / dead-code path-relative imports. The two smokes together add ~30 s to the
Docker build — far cheaper than the "run 10 min on FC only to crash" path.

**6. After changing vendor.sh exclude rules, `rm -rf upstream/` and re-run**

Docker COPY only sees the current contents of `upstream/` on the host, not the vendor.sh script.
After changing `--exclude='src/notebooks/'` to `--exclude='*.ipynb'` in `vendor.sh`, if the host's
`upstream/src/notebooks/` dir is still empty (left by the old vendor), the new build still misses
`utils_notebooks.py`. **Rule**: after changing vendor.sh exclude rules, `rm -rf
services/<svc>/upstream/` first, then re-run `./scripts/vendor.sh`.

##### Pitfall 7: the double trap when uv installs CUDA-extension packages (VCS + build isolation)

When installing **github-sourced CUDA-extension packages** such as `openfold` / `apex` /
`SE3Transformer`, the Dockerfile hits two traps back-to-back (diffdock-server 2026-07-06 v0.0.1 build
failed its first two rounds):

**Trap A: missing git CLI** (`uv pip install "<pkg> @ git+..."`)
```
× Failed to download and build `openfold @ git+https://...`
├─▶ Git operation failed
╰─▶ Git executable not found. Ensure that Git is installed and available.
```
→ apt-install `git` into the builder (see the 8.3 skeleton comments in
[dockerfile.md](./dockerfile.md)).

**Trap B: top-level `import torch` in setup.py × PEP 517 isolated build env**
```
× Failed to build `openfold @ git+...`
├─▶ Call to `setuptools.build_meta:__legacy__.build_wheel` failed
╰─▶ ModuleNotFoundError: No module named 'torch'
```
uv defaults to PEP 517 isolated builds — the build env only sees the packages declared in
`[build-system].requires`, so torch installed in the target env isn't visible. openfold's setup.py
needs `import torch` to obtain the CUDA arch / nvcc path. The fix: **`--no-build-isolation`** +
**make sure the target conda env already has the build-time deps** (`torch`, `setuptools`, `wheel`,
`ninja`, `pybind11`, `numpy`):

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

**The same trap recurs on these upstreams**: apex (NVIDIA mixed precision), torch-scatter/sparse
built from source (prefer the wheel when one exists), SE3Transformer, DGL from source, and other
hand-written CUDA extensions. Using a wheel (the `+cu117`-suffixed kind) sidesteps it —
`--no-build-isolation` only bites at the setup.py layer.

##### Reference implementations (these pitfalls were hit and fixed)

- [diffdock-pp-server](../../services/diffdock-pp-server/) (**hit all 6 pitfalls** — recommended as the
  checklist sample for new conda services)
- [diffdock-server](../../services/diffdock-server/) (**hit all 7 pitfalls**, including the new §7 uv
  git+torch-extension double trap)
- [diffusion-hopping-server](../../services/diffusion-hopping-server/) (1, 4)
- [deeprank-ab-server](../../services/deeprank-ab-server/) (runtime monkey-patch)