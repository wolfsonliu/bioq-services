from __future__ import annotations

from server.auth.api_key import hash_secret
from server.db.store import GatewayDB


def _db(tmp_path):
    db = GatewayDB(f"sqlite:///{tmp_path/'t.db'}")
    db.create_all()
    return db


def test_user_and_key_roundtrip(tmp_path):
    db = _db(tmp_path)
    db.create_user("alice")
    db.create_api_key("alice", secret="s3cr3t", key_id="gk_alice")
    row = db.find_api_key(hash_secret("s3cr3t"))
    assert row is not None
    assert row.key_id == "gk_alice"
    assert row.account_id == "alice"
    assert db.find_api_key(hash_secret("wrong")) is None


def test_job_lifecycle(tmp_path):
    db = _db(tmp_path)
    db.create_user("bob")
    db.create_job(job_id="j1", account_id="bob", svc="openbpmd",
                  endpoint="score", input_params={"nreps": 1},
                  output_prefix="users/bob/outputs/j1/")
    job = db.get_job("bob", "j1")
    assert job.account_id == "bob" and job.status == "pending"
    db.update_job("bob", "j1", status="completed", fc_task_id="bob-j1")
    assert db.get_job("bob", "j1").status == "completed"
    assert [j.job_id for j in db.list_jobs("bob")] == ["j1"]
    assert db.list_jobs("carol") == []
    # job_id is namespaced per account: carol reusing "j1" is a distinct row.
    assert db.get_job("carol", "j1") is None
