"""GatewayDB — thin SQLAlchemy wrapper for users / api_keys / jobs.

MVP: SQLite, single process. Swap `db_url` to RDS/PostgreSQL when the
gateway needs HA/multi-instance (SQLite couples it to one instance).
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from server.auth.api_key import hash_secret

from .models import ApiKey, Base, Job, User


class GatewayDB:
    def __init__(self, db_url: str) -> None:
        is_sqlite = db_url.startswith("sqlite")
        connect_args = {"check_same_thread": False} if is_sqlite else {}
        # SQLite is single-file/in-process → no pool tuning. For a networked DB
        # (cloud PostgreSQL) size the pool to the anyio threadpool (sync handlers
        # each grab a connection) and defend against the cloud LB silently
        # dropping idle connections: pre_ping validates on checkout, recycle
        # retires connections before the server's idle timeout.
        pool_kwargs = {} if is_sqlite else dict(
            pool_size=20,
            max_overflow=40,
            pool_pre_ping=True,
            pool_recycle=1800,
        )
        self._engine = create_engine(
            db_url, connect_args=connect_args, future=True, **pool_kwargs
        )
        self._Session: sessionmaker[Session] = sessionmaker(
            bind=self._engine, future=True, expire_on_commit=False
        )

    def create_all(self) -> None:
        Base.metadata.create_all(self._engine)

    # ---- users / keys ----
    def create_user(self, account_id: str, display_name: str | None = None,
                    role: str = "user") -> None:
        with self._Session() as s, s.begin():
            if s.get(User, account_id) is None:
                s.add(User(account_id=account_id, display_name=display_name, role=role))

    def get_user(self, account_id: str) -> User | None:
        with self._Session() as s:
            return s.get(User, account_id)

    def set_role(self, account_id: str, role: str) -> None:
        with self._Session() as s, s.begin():
            u = s.get(User, account_id)
            if u is None:
                raise KeyError(account_id)
            u.role = role

    def create_api_key(self, account_id: str, *, secret: str, key_id: str) -> None:
        with self._Session() as s, s.begin():
            s.add(ApiKey(key_id=key_id, account_id=account_id, secret_hash=hash_secret(secret)))

    def find_api_key(self, secret_hash: str) -> ApiKey | None:
        with self._Session() as s:
            return s.scalars(
                select(ApiKey).where(ApiKey.secret_hash == secret_hash,
                                     ApiKey.status == "active")
            ).first()

    def touch_api_key(self, key_id: str) -> None:
        with self._Session() as s, s.begin():
            row = s.get(ApiKey, key_id)
            if row is not None:
                row.last_used_at = datetime.now(timezone.utc)

    # ---- jobs ----
    def create_job(self, *, job_id: str, account_id: str, svc: str, endpoint: str,
                   input_params: dict | None, output_prefix: str | None) -> None:
        with self._Session() as s, s.begin():
            s.add(Job(job_id=job_id, account_id=account_id, svc=svc, endpoint=endpoint,
                      input_params=input_params, output_prefix=output_prefix))

    def get_job(self, account_id: str, job_id: str) -> Job | None:
        with self._Session() as s:
            return s.get(Job, (account_id, job_id))

    def update_job(self, account_id: str, job_id: str, **fields) -> None:
        with self._Session() as s, s.begin():
            row = s.get(Job, (account_id, job_id))
            if row is None:
                raise KeyError((account_id, job_id))
            for k, v in fields.items():
                setattr(row, k, v)

    def list_jobs(self, account_id: str) -> list[Job]:
        with self._Session() as s:
            return list(s.scalars(
                select(Job).where(Job.account_id == account_id).order_by(Job.created_at)
            ))
