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


def test_revoke_api_key(tmp_path):
    db = _db(tmp_path)
    db.create_user("alice")
    db.create_api_key("alice", secret="s3cr3t", key_id="gk_a")
    assert db.find_api_key(hash_secret("s3cr3t")) is not None
    db.revoke_api_key("gk_a")
    assert db.find_api_key(hash_secret("s3cr3t")) is None   # only active keys found
    import pytest
    with pytest.raises(KeyError):
        db.revoke_api_key("nope")


def test_admin_queries(tmp_path):
    db = _db(tmp_path)
    db.create_user("alice")
    db.create_user("bob", role="admin")
    db.create_api_key("alice", secret="s1", key_id="k1")
    db.create_job(job_id="j1", account_id="alice", svc="rfdiffusion-server",
                  endpoint="generate", input_params={}, output_prefix=None)
    db.update_job("alice", "j1", status="completed")
    db.create_job(job_id="j2", account_id="bob", svc="boltz-server",
                  endpoint="predict", input_params={}, output_prefix=None)
    db.update_job("bob", "j2", status="running")

    assert {u.account_id for u in db.list_users()} == {"alice", "bob"}
    assert [k.key_id for k in db.list_api_keys("alice")] == ["k1"]
    assert db.list_api_keys("bob") == []
    assert db.count_jobs_by_status() == {"completed": 1, "running": 1}
    assert {j.job_id for j in db.list_all_jobs()} == {"j1", "j2"}
    assert [j.job_id for j in db.list_all_jobs(status="running")] == ["j2"]
    assert [j.job_id for j in db.list_all_jobs(svc="boltz-server")] == ["j2"]
    assert [j.job_id for j in db.list_all_jobs(account_id="alice")] == ["j1"]
    assert db.count_users() == 2
    assert db.count_jobs() == 2


def test_user_role_default_and_admin(tmp_path):
    db = _db(tmp_path)
    db.create_user("alice")
    assert db.get_user("alice").role == "user"      # 默认非 admin
    db.create_user("root", role="admin")
    assert db.get_user("root").role == "admin"
    db.set_role("alice", "admin")                    # 提升已存在账户
    assert db.get_user("alice").role == "admin"
    assert db.get_user("nobody") is None
