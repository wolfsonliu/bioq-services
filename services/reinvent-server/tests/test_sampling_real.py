"""Opt-in real run: needs reinvent installed + a prior on disk.

Enable with: REINVENT_PRIOR_BASE=/path/to/priors uv run pytest -m slow \
    services/reinvent-server/tests/test_sampling_real.py
"""
import json
import os
import shutil
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow


def test_real_reinvent_sampling(tmp_path):
    prior_base = Path(os.environ.get("REINVENT_PRIOR_BASE", "/data/models/reinvent"))
    prior = prior_base / "reinvent_pubchem.prior"
    if not prior.exists():
        pytest.skip(f"no prior at {prior}")
    if shutil.which("reinvent") is None:
        pytest.skip("reinvent not installed in this env")

    from server import reinvent_cli
    work = tmp_path / "work"
    out = tmp_path / "output"
    out.mkdir(parents=True)
    work.mkdir(parents=True)
    params = {"generator": "reinvent", "model_file": ".reinvent", "num_smiles": 10,
              "unique_molecules": True, "randomize_smiles": True, "temperature": 1.0,
              "sample_strategy": "multinomial"}
    (work / "params.json").write_text(json.dumps(params))
    rc = reinvent_cli.run(
        run_type="sampling", params_json=work / "params.json",
        work_dir=work, output_dir=out, device="cpu",
        prior_base=prior_base, reinvent_bin=Path(shutil.which("reinvent")),
        files={},
    )
    assert rc == 0
    csv = out / "sampling.csv"
    assert csv.exists() and csv.stat().st_size > 0
