from __future__ import annotations

from server.oss_map import map_oss_inputs_to_mount


def test_maps_oss_bucket_uris_to_mount_path():
    body = {
        "structure_uri": "oss://bio-gateway/users/alice/j1/input/x.rst7",
        "parameters_uri": "oss://bio-gateway/users/alice/j1/input/x.prm7",
        "name": "run", "nreps": 2,
    }
    out = map_oss_inputs_to_mount(body, bucket="bio-gateway", mount="/mnt/oss")
    assert out["structure_uri"] == "/mnt/oss/users/alice/j1/input/x.rst7"
    assert out["parameters_uri"] == "/mnt/oss/users/alice/j1/input/x.prm7"
    assert out["name"] == "run" and out["nreps"] == 2  # non-URI values untouched


def test_leaves_other_buckets_and_schemes_untouched():
    body = {
        "a": "oss://other-bucket/k",       # different bucket
        "b": "https://example.com/x",      # different scheme
        "c": "plain-string",
    }
    out = map_oss_inputs_to_mount(body, bucket="bio-gateway", mount="/mnt/oss")
    assert out == body


def test_recurses_into_lists_and_dicts():
    body = {"ref_files": ["oss://bio-gateway/users/a/j/input/1.pdb",
                          "oss://bio-gateway/users/a/j/input/2.pdb"],
            "opts": {"tmpl": "oss://bio-gateway/users/a/j/input/t.pdb"}}
    out = map_oss_inputs_to_mount(body, bucket="bio-gateway", mount="/mnt/oss")
    assert out["ref_files"] == ["/mnt/oss/users/a/j/input/1.pdb",
                                "/mnt/oss/users/a/j/input/2.pdb"]
    assert out["opts"]["tmpl"] == "/mnt/oss/users/a/j/input/t.pdb"
