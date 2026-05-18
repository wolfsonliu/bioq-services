"""argv builder tests — lock in the Hydra override surface for each endpoint."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from server.models import (
    BinderRequest,
    CustomRequest,
    MotifRequest,
    SymmetryRequest,
    UnconditionalRequest,
)
from server.settings import RFdiffusionSettings
from server.tools import (
    OUTPUT_STEM,
    binder_argv,
    custom_argv,
    motif_argv,
    symmetry_argv,
    unconditional_argv,
)


@pytest.fixture
def settings(tmp_path: Path) -> RFdiffusionSettings:
    # Models dir exists but is empty by default; tests that need a checkpoint
    # write a placeholder file themselves.
    models = tmp_path / "models"
    models.mkdir()
    return RFdiffusionSettings(
        jobs_base_dir=tmp_path / "jobs",
        root=tmp_path / "rfdiffusion",
        models_dir=models,
        inference_script=tmp_path / "rfdiffusion" / "scripts" / "run_inference.py",
        python=tmp_path / "rfdiffusion" / ".venv" / "bin" / "python",
    )


def test_unconditional_basic(settings: RFdiffusionSettings, tmp_path: Path) -> None:
    job_dir = tmp_path / "j-uncond"
    argv = unconditional_argv(
        UnconditionalRequest(min_length=100, max_length=150, num_designs=4),
        job_dir=job_dir,
        settings=settings,
    )
    joined = " ".join(argv)
    assert "contigmap.contigs=[100-150]" in joined
    assert "inference.num_designs=4" in joined
    assert "diffuser.T=50" in joined  # default
    assert f"inference.output_prefix={job_dir / 'output' / OUTPUT_STEM}" in joined
    assert "inference.write_trajectory=false" in joined  # default off
    assert "inference.cyclic" not in joined


def test_unconditional_cyclic(settings: RFdiffusionSettings, tmp_path: Path) -> None:
    job_dir = tmp_path / "j-cyc"
    argv = unconditional_argv(
        UnconditionalRequest(min_length=12, max_length=18, cyclic=True),
        job_dir=job_dir,
        settings=settings,
    )
    assert "inference.cyclic=True" in argv
    assert "inference.cyc_chains=a" in argv


def test_unconditional_rejects_inverted_length(
    settings: RFdiffusionSettings, tmp_path: Path
) -> None:
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        unconditional_argv(
            UnconditionalRequest(min_length=200, max_length=150),
            job_dir=tmp_path / "j",
            settings=settings,
        )
    assert exc.value.status_code == 422


def test_motif_with_inpaint_seq(settings: RFdiffusionSettings, tmp_path: Path) -> None:
    job_dir = tmp_path / "j-motif"
    argv = motif_argv(
        MotifRequest(
            contigs="10-40/A163-181/10-40",
            length="55-55",
            inpaint_seq="A1/A30-40",
        ),
        input_pdb=tmp_path / "5TPN.pdb",
        job_dir=job_dir,
        settings=settings,
    )
    joined = " ".join(argv)
    assert f"inference.input_pdb={tmp_path / '5TPN.pdb'}" in joined
    assert "contigmap.contigs=[10-40/A163-181/10-40]" in joined
    assert "contigmap.length=55-55" in joined
    assert "contigmap.inpaint_seq=[A1/A30-40]" in joined


def test_binder_with_hotspots(settings: RFdiffusionSettings, tmp_path: Path) -> None:
    job_dir = tmp_path / "j-binder"
    argv = binder_argv(
        BinderRequest(contigs="A1-150/0 70-100", hotspots=" A59, A83 , A91 "),
        target_pdb=tmp_path / "target.pdb",
        job_dir=job_dir,
        settings=settings,
    )
    joined = " ".join(argv)
    assert "ppi.hotspot_res=[A59,A83,A91]" in joined
    # binder default is noise_scale=0.0 — applied to both axes.
    assert "denoiser.noise_scale_ca=0.0" in joined
    assert "denoiser.noise_scale_frame=0.0" in joined


def test_binder_without_hotspots(settings: RFdiffusionSettings, tmp_path: Path) -> None:
    argv = binder_argv(
        BinderRequest(contigs="B1-100/0 100-100"),
        target_pdb=tmp_path / "t.pdb",
        job_dir=tmp_path / "j",
        settings=settings,
    )
    assert all("ppi.hotspot_res" not in tok for tok in argv)


def test_symmetry_overrides(settings: RFdiffusionSettings, tmp_path: Path) -> None:
    argv = symmetry_argv(
        SymmetryRequest(
            symmetry="c6",
            total_length=480,
            guiding_potentials='["type:olig_contacts,weight_intra:1,weight_inter:0.1"]',
            guide_scale=2.0,
            guide_decay="quadratic",
            olig_intra_all=True,
            olig_inter_all=True,
        ),
        job_dir=tmp_path / "j",
        settings=settings,
    )
    assert argv[2:4] == ["--config-name", "symmetry"]
    joined = " ".join(argv)
    assert "inference.symmetry=c6" in joined
    assert "contigmap.contigs=[480-480]" in joined
    assert "potentials.guide_scale=2.0" in joined
    assert "potentials.guide_decay=quadratic" in joined
    assert "potentials.olig_intra_all=True" in argv
    assert "potentials.olig_inter_all=True" in argv


def test_custom_partial_diffusion(settings: RFdiffusionSettings, tmp_path: Path) -> None:
    argv = custom_argv(
        CustomRequest(
            contigs="79-79",
            extra_overrides=json.dumps({"diffuser.partial_T": 10}),
        ),
        input_pdb=tmp_path / "2KL8.pdb",
        job_dir=tmp_path / "j",
        settings=settings,
    )
    joined = " ".join(argv)
    assert "diffuser.partial_T=10" in joined
    assert "contigmap.contigs=[79-79]" in joined
    assert f"inference.input_pdb={tmp_path / '2KL8.pdb'}" in joined


def test_custom_invalid_json_returns_422(
    settings: RFdiffusionSettings, tmp_path: Path
) -> None:
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        custom_argv(
            CustomRequest(contigs="100-100", extra_overrides="{not json"),
            input_pdb=None,
            job_dir=tmp_path / "j",
            settings=settings,
        )
    assert exc.value.status_code == 422


def test_model_override_resolves_checkpoint(
    settings: RFdiffusionSettings, tmp_path: Path
) -> None:
    (settings.models_dir / "Complex_base_ckpt.pt").write_bytes(b"\x00")
    argv = unconditional_argv(
        UnconditionalRequest(min_length=100, max_length=100, model="complex_base"),
        job_dir=tmp_path / "j",
        settings=settings,
    )
    assert any(
        tok.startswith("inference.ckpt_override_path=")
        and tok.endswith("Complex_base_ckpt.pt")
        for tok in argv
    )


def test_unknown_model_returns_422(settings: RFdiffusionSettings, tmp_path: Path) -> None:
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        unconditional_argv(
            UnconditionalRequest(min_length=100, max_length=100, model="not_a_model"),
            job_dir=tmp_path / "j",
            settings=settings,
        )
    assert exc.value.status_code == 422


def test_missing_model_file_returns_422(
    settings: RFdiffusionSettings, tmp_path: Path
) -> None:
    """model=base but the .pt file isn't in the image — surface a clear 422."""
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        unconditional_argv(
            UnconditionalRequest(min_length=100, max_length=100, model="base"),
            job_dir=tmp_path / "j",
            settings=settings,
        )
    assert exc.value.status_code == 422
    assert "not present" in exc.value.detail.lower()
