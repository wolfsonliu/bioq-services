"""Per-job config synthesis for megalodon-server.

Upstream configs reference statistics via OmegaConf interpolation
`${data.dataset_root}/processed/<file>` in two places:
  - interpolant.variables[*].custom_prior  (model-init discrete priors)
  - sample.node_distribution               (molecule-size sampling)

Our NAS layout is flat (`stats/<dataset>/<file>`, no `processed/`), and the
drugs flow-matching config uses a differently-named charges prior. So we
rewrite those paths per job, keeping each original *basename* (which makes
the drugs_fm `train_charges_prior.npy` vs others' `train_charges_prior_h.npy`
distinction handle itself).

Implemented with PyYAML (not OmegaConf) so it runs in the plain project venv
for offline tests. PyYAML resolves the upstream YAML anchors (`&x`/`*x`) to
concrete values on load; the only `${...}` interpolations are the two we
rewrite, so the dumped config is interpolation-free and the wrapper can load
it with OmegaConf without a dataset_root present.
"""

from __future__ import annotations

from pathlib import Path

import yaml


def build_config(
    *,
    src_config: Path,
    stats_dir: Path,
    out_path: Path,
) -> Path:
    """Read an upstream variant config, repoint its statistics paths at
    ``stats_dir``, disable wandb, and write it to ``out_path``.

    Returns ``out_path``.
    """
    cfg = yaml.safe_load(src_config.read_text())

    stats_dir = Path(stats_dir)

    # 1. Discrete-prior npy files (custom / data prior types).
    for var in cfg.get("interpolant", {}).get("variables", []) or []:
        cp = var.get("custom_prior")
        if cp:
            var["custom_prior"] = str(stats_dir / Path(str(cp)).name)

    # 2. Node (molecule-size) distribution pickle — may be null (drugs_fm).
    sample = cfg.get("sample")
    if isinstance(sample, dict):
        nd = sample.get("node_distribution")
        if nd:  # non-null, non-empty string
            sample["node_distribution"] = str(stats_dir / Path(str(nd)).name)

    # 3. Harmless: point dataset_root at stats_dir in case anything else reads
    #    it; and silence wandb (train-path only).
    if isinstance(cfg.get("data"), dict):
        cfg["data"]["dataset_root"] = str(stats_dir)
    if isinstance(cfg.get("wandb_params"), dict):
        cfg["wandb_params"]["mode"] = "disabled"

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out_path


__all__ = ["build_config"]
