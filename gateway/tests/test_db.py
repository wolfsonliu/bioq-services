from __future__ import annotations

from server.db.store import GatewayDB


def _db(tmp_path):
    db = GatewayDB(f"sqlite:///{tmp_path/'t.db'}")
    db.create_all()
    return db


def test_user_roundtrip(tmp_path):
    db = _db(tmp_path)
    db.create_user("alice", display_name="Alice")
    u = db.get_user("alice")
    assert u is not None and u.account_id == "alice" and u.role == "user"
    assert db.get_user("nobody") is None


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


def test_upsert_user(tmp_path):
    db = _db(tmp_path)
    db.upsert_user("u1", display_name="U One", role="admin")   # 新建
    u = db.get_user("u1")
    assert u.role == "admin" and u.display_name == "U One"
    db.upsert_user("u1", display_name="U One", role="user")     # 降级同步
    assert db.get_user("u1").role == "user"
    db.upsert_user("u1")                                        # 无显式值不清空 display_name
    assert db.get_user("u1").display_name == "U One"


def test_admin_queries(tmp_path):
    db = _db(tmp_path)
    db.create_user("alice")
    db.create_user("bob", role="admin")
    db.create_job(job_id="j1", account_id="alice", svc="rfdiffusion-server",
                  endpoint="generate", input_params={}, output_prefix=None)
    db.update_job("alice", "j1", status="completed")
    db.create_job(job_id="j2", account_id="bob", svc="boltz-server",
                  endpoint="predict", input_params={}, output_prefix=None)
    db.update_job("bob", "j2", status="running")

    assert {u.account_id for u in db.list_users()} == {"alice", "bob"}
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
