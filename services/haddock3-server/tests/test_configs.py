"""Unit tests for the pure config builders (no subprocess / no haddock import)."""

from __future__ import annotations

from server.configs import (
    build_protein_protein_cfg,
    finalize_general_cfg,
    write_cfg,
)


def test_protein_protein_full_pipeline():
    cfg = build_protein_protein_cfg(
        molecules=["/in/a.pdb", "/in/b.pdb"],
        run_dir="/jobs/x/output/run",
        ncores=8,
        sampling=200,
        do_flexref=True,
        do_emref=True,
        clustering=True,
        top_models=4,
        ambig_fname="/in/ambig.tbl",
        reference_fname="/in/ref.pdb",
    )
    assert 'run_dir = "/jobs/x/output/run"' in cfg
    assert 'mode = "local"' in cfg
    assert "ncores = 8" in cfg
    assert '"/in/a.pdb"' in cfg and '"/in/b.pdb"' in cfg
    for section in ("[topoaa]", "[rigidbody]", "[flexref]", "[emref]",
                    "[clustfcc]", "[seletopclusts]", "[caprieval]"):
        assert section in cfg
    assert "sampling = 200" in cfg
    assert "top_models = 4" in cfg
    assert 'ambig_fname = "/in/ambig.tbl"' in cfg
    assert 'reference_fname = "/in/ref.pdb"' in cfg


def test_protein_protein_minimal_omits_optional_stages():
    cfg = build_protein_protein_cfg(
        molecules=["/a.pdb", "/b.pdb"],
        run_dir="/r",
        ncores=4,
        sampling=50,
        do_flexref=False,
        do_emref=False,
        clustering=False,
        top_models=4,
    )
    assert "[rigidbody]" in cfg
    assert "[flexref]" not in cfg
    assert "[emref]" not in cfg
    assert "[clustfcc]" not in cfg
    assert "[seletopclusts]" not in cfg
    assert "[caprieval]" in cfg
    assert "ambig_fname" not in cfg
    assert "reference_fname" not in cfg


def test_finalize_general_prepends_header():
    body = "[topoaa]\n\n[rigidbody]\nsampling = 10\n\n[caprieval]\n"
    out = finalize_general_cfg(
        body, molecules=["a.pdb", "b.pdb"], run_dir="/jobs/y/output/run", ncores=16,
    )
    assert out.startswith('run_dir = "/jobs/y/output/run"')
    assert 'mode = "local"' in out
    assert "ncores = 16" in out
    assert '"a.pdb"' in out and '"b.pdb"' in out
    # caller body preserved
    assert "[topoaa]" in out and "sampling = 10" in out


def test_finalize_strips_caller_managed_keys():
    body = (
        'run_dir = "should-be-dropped"\n'
        'mode = "batch"\n'
        "ncores = 999\n"
        "molecules = [\n"
        '    "wrong.pdb"\n'
        "]\n"
        "\n"
        "[topoaa]\n"
    )
    out = finalize_general_cfg(
        body, molecules=["right.pdb"], run_dir="/correct/run", ncores=8,
    )
    assert "should-be-dropped" not in out
    assert "wrong.pdb" not in out
    assert 'mode = "batch"' not in out
    assert "ncores = 999" not in out
    assert '"right.pdb"' in out
    assert 'run_dir = "/correct/run"' in out
    assert out.count("[topoaa]") == 1


def test_write_cfg_roundtrip(tmp_path):
    p = write_cfg("hello\n", tmp_path / "sub" / "workflow.cfg")
    assert p.exists()
    assert p.read_text() == "hello\n"
