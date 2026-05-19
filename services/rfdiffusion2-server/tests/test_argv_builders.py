"""argv builder tests — lock in the Hydra override surface for each endpoint."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from server.models import (
    ActiveSiteRequest,
    CustomRequest,
    SmallMoleculeBinderRequest,
)
from server.settings import RFdiffusion2Settings
from server.tools import (
    OUTPUT_STEM,
    active_site_argv,
    custom_argv,
    small_molecule_binder_argv,
)


@pytest.fixture
def settings(tmp_path: Path) -> RFdiffusion2Settings:
    # Models dir exists but is empty by default; tests that need a checkpoint
    # write a placeholder file themselves.
    models = tmp_path / "models"
    models.mkdir()
    return RFdiffusion2Settings(
        jobs_base_dir=tmp_path / "jobs",
        root=tmp_path / "rfdiffusion2",
        models_dir=models,
        inference_script=tmp_path / "rfdiffusion2" / "rf_diffusion" / "run_inference.py",
        python=tmp_path / "rfdiffusion2" / ".venv" / "bin" / "python",
        pythonpath=tmp_path / "rfdiffusion2",
    )


# ---------------------------------------------------------------------------
# active_site
# ---------------------------------------------------------------------------


def test_active_site_unindexed_atomic(
    settings: RFdiffusion2Settings, tmp_path: Path
) -> None:
    """Reproduces `open_source_demo.json::active_site_unindexed_atomic`."""
    job_dir = tmp_path / "j-as"
    argv = active_site_argv(
        ActiveSiteRequest(
            contigs="46,A106-106,59,A166-166,2,A169-169,23,A193-193,46",
            ligand="NAD,OXM",
            contig_atoms={
                "A106": "NE,CD,CZ",
                "A166": "OD1,CG",
                "A169": "NH2,CZ",
                "A193": "NE2,CD2,CE1",
            },
            contig_as_guidepost=True,
            num_designs=10,
        ),
        input_pdb=tmp_path / "M0584_1ldm.pdb",
        job_dir=job_dir,
        settings=settings,
    )
    joined = " ".join(argv)
    assert "--config-name=aa" in argv
    assert f"inference.input_pdb={tmp_path / 'M0584_1ldm.pdb'}" in joined
    assert "inference.ligand='NAD,OXM'" in argv
    assert "contigmap.contigs=['46,A106-106,59,A166-166,2,A169-169,23,A193-193,46']" in argv
    assert "inference.contig_as_guidepost=true" in argv
    assert any(
        tok.startswith("contigmap.contig_atoms={") and "A106: 'NE,CD,CZ'" in tok
        for tok in argv
    )
    assert "inference.num_designs=10" in argv
    assert f"inference.output_prefix={job_dir / 'output' / OUTPUT_STEM}" in joined


def test_active_site_indexed(settings: RFdiffusion2Settings, tmp_path: Path) -> None:
    """contig_as_guidepost=False → indexed atomic mode."""
    argv = active_site_argv(
        ActiveSiteRequest(
            contigs="20,A50-50,30",
            ligand="ATP",
            contig_atoms={"A50": "NE,CD"},
            contig_as_guidepost=False,
        ),
        input_pdb=tmp_path / "x.pdb",
        job_dir=tmp_path / "j",
        settings=settings,
    )
    assert "inference.contig_as_guidepost=false" in argv


def test_active_site_partial_ligand(
    settings: RFdiffusion2Settings, tmp_path: Path
) -> None:
    """partially_fixed_ligand renders as `++inference.partially_fixed_ligand={...}`."""
    argv = active_site_argv(
        ActiveSiteRequest(
            contigs="46,A106-106,46",
            ligand="NAD,OXM",
            contig_atoms={"A106": "NE,CD,CZ"},
            partially_fixed_ligand={
                "NAD": ["O7N", "C7N", "C3N"],
                "OXM": ["O3", "C2", "C1"],
            },
        ),
        input_pdb=tmp_path / "x.pdb",
        job_dir=tmp_path / "j",
        settings=settings,
    )
    matching = [tok for tok in argv if tok.startswith("++inference.partially_fixed_ligand=")]
    assert matching, f"no ++inference.partially_fixed_ligand= token in argv: {argv}"
    tok = matching[0]
    assert "NAD: [O7N,C7N,C3N]" in tok
    assert "OXM: [O3,C2,C1]" in tok


def test_active_site_some_indexed(
    settings: RFdiffusion2Settings, tmp_path: Path
) -> None:
    argv = active_site_argv(
        ActiveSiteRequest(
            contigs="46,A106-106,46",
            ligand="NAD",
            contig_atoms={"A106": "NE,CD,CZ"},
            only_guidepost_positions="A106",
        ),
        input_pdb=tmp_path / "x.pdb",
        job_dir=tmp_path / "j",
        settings=settings,
    )
    assert "inference.only_guidepost_positions='A106'" in argv


def test_active_site_rejects_bad_residue_key(
    settings: RFdiffusion2Settings, tmp_path: Path
) -> None:
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        active_site_argv(
            ActiveSiteRequest(
                contigs="46,A106-106,46",
                ligand="NAD",
                contig_atoms={"BAD KEY": "NE"},
            ),
            input_pdb=tmp_path / "x.pdb",
            job_dir=tmp_path / "j",
            settings=settings,
        )
    assert exc.value.status_code == 422


def test_active_site_rejects_quote_injection(
    settings: RFdiffusion2Settings, tmp_path: Path
) -> None:
    """contig strings must not contain quote chars (would break Hydra parsing)."""
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        active_site_argv(
            ActiveSiteRequest(
                contigs="46,'A106-106',46",
                ligand="NAD",
                contig_atoms={"A106": "NE"},
            ),
            input_pdb=tmp_path / "x.pdb",
            job_dir=tmp_path / "j",
            settings=settings,
        )
    assert exc.value.status_code == 422


def test_active_site_rejects_bad_atom_name(
    settings: RFdiffusion2Settings, tmp_path: Path
) -> None:
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        active_site_argv(
            ActiveSiteRequest(
                contigs="46,A106-106,46",
                ligand="NAD",
                contig_atoms={"A106": "NE;rm -rf /"},
            ),
            input_pdb=tmp_path / "x.pdb",
            job_dir=tmp_path / "j",
            settings=settings,
        )
    assert exc.value.status_code == 422


# ---------------------------------------------------------------------------
# small_molecule_binder
# ---------------------------------------------------------------------------


def test_smbinder_rasa_buried(settings: RFdiffusion2Settings, tmp_path: Path) -> None:
    """Reproduces `open_source_demo.json::small_molecule_binder_rasa_buried`."""
    job_dir = tmp_path / "j-sm"
    argv = small_molecule_binder_argv(
        SmallMoleculeBinderRequest(
            contigs="150",
            length="150-150",
            ligand="PH2",
            rasa_active=True,
            rasa_target=0.0,
        ),
        input_pdb=tmp_path / "trimmed.pdb",
        job_dir=job_dir,
        settings=settings,
    )
    joined = " ".join(argv)
    assert "--config-name=aa" in argv
    assert "contigmap.contigs=['150']" in argv
    assert "contigmap.length=150-150" in argv
    assert "inference.ligand=PH2" in argv
    assert "inference.conditions.relative_sasa_v2.active=True" in argv
    assert "inference.conditions.relative_sasa_v2.rasa=0.0" in argv
    assert f"inference.output_prefix={job_dir / 'output' / OUTPUT_STEM}" in joined


def test_smbinder_no_rasa(settings: RFdiffusion2Settings, tmp_path: Path) -> None:
    argv = small_molecule_binder_argv(
        SmallMoleculeBinderRequest(
            contigs="100", ligand="PH2", rasa_active=False
        ),
        input_pdb=tmp_path / "x.pdb",
        job_dir=tmp_path / "j",
        settings=settings,
    )
    assert all("relative_sasa_v2" not in tok for tok in argv)


# ---------------------------------------------------------------------------
# custom
# ---------------------------------------------------------------------------


def test_custom_extra_overrides(settings: RFdiffusion2Settings, tmp_path: Path) -> None:
    argv = custom_argv(
        CustomRequest(
            contigs="150",
            config_name="aa",
            extra_overrides=json.dumps(
                {"diffuser.T": 50, "inference.deterministic": True}
            ),
        ),
        input_pdb=None,
        job_dir=tmp_path / "j",
        settings=settings,
    )
    assert "--config-name=aa" in argv
    assert "diffuser.T=50" in argv
    assert "inference.deterministic=true" in argv


def test_custom_invalid_json_returns_422(
    settings: RFdiffusion2Settings, tmp_path: Path
) -> None:
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        custom_argv(
            CustomRequest(contigs="100", extra_overrides="{not json"),
            input_pdb=None,
            job_dir=tmp_path / "j",
            settings=settings,
        )
    assert exc.value.status_code == 422


def test_custom_input_pdb_required(
    settings: RFdiffusion2Settings, tmp_path: Path
) -> None:
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        custom_argv(
            CustomRequest(contigs="100", input_pdb_required=True),
            input_pdb=None,
            job_dir=tmp_path / "j",
            settings=settings,
        )
    assert exc.value.status_code == 422


def test_custom_with_ligand(settings: RFdiffusion2Settings, tmp_path: Path) -> None:
    argv = custom_argv(
        CustomRequest(contigs="100", ligand="NAD,OXM"),
        input_pdb=None,
        job_dir=tmp_path / "j",
        settings=settings,
    )
    assert "inference.ligand='NAD,OXM'" in argv


# ---------------------------------------------------------------------------
# model resolution
# ---------------------------------------------------------------------------


def test_model_override_resolves_checkpoint(
    settings: RFdiffusion2Settings, tmp_path: Path
) -> None:
    (settings.models_dir / "RFD_173.pt").write_bytes(b"\x00")
    argv = small_molecule_binder_argv(
        SmallMoleculeBinderRequest(
            contigs="100", ligand="PH2", model="rfd_173", rasa_active=False
        ),
        input_pdb=tmp_path / "x.pdb",
        job_dir=tmp_path / "j",
        settings=settings,
    )
    assert any(
        tok.startswith("inference.ckpt_path=") and tok.endswith("RFD_173.pt")
        for tok in argv
    )


def test_unknown_model_returns_422(
    settings: RFdiffusion2Settings, tmp_path: Path
) -> None:
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        small_molecule_binder_argv(
            SmallMoleculeBinderRequest(
                contigs="100", ligand="PH2", model="not_a_model"
            ),
            input_pdb=tmp_path / "x.pdb",
            job_dir=tmp_path / "j",
            settings=settings,
        )
    assert exc.value.status_code == 422


def test_missing_model_file_returns_422(
    settings: RFdiffusion2Settings, tmp_path: Path
) -> None:
    """model=rfd_140 but the .pt file isn't in the image — surface a clear 422."""
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        small_molecule_binder_argv(
            SmallMoleculeBinderRequest(
                contigs="100", ligand="PH2", model="rfd_140"
            ),
            input_pdb=tmp_path / "x.pdb",
            job_dir=tmp_path / "j",
            settings=settings,
        )
    assert exc.value.status_code == 422
    assert "not present" in exc.value.detail.lower()
