"""GatewayDB — thin SQLAlchemy wrapper for users / jobs.

MVP: SQLite, single process. Swap `db_url` to RDS/PostgreSQL when the
gateway needs HA/multi-instance (SQLite couples it to one instance).
"""

from __future__ import annotations

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from .models import Base, Job, User


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

    def upsert_user(self, account_id: str, *, display_name: str | None = None,
                    role: str = "user") -> None:
        """Create the user if new, else sync role / display_name (IdP is source of truth)."""
        with self._Session() as s, s.begin():
            u = s.get(User, account_id)
            if u is None:
                s.add(User(account_id=account_id, display_name=display_name, role=role))
            else:
                if u.role != role:
                    u.role = role
                if display_name and u.display_name != display_name:
                    u.display_name = display_name

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

    # ---- admin read-only aggregates ----
    def list_users(self) -> list[User]:
        with self._Session() as s:
            return list(s.scalars(select(User).order_by(User.created_at)))

    def count_users(self) -> int:
        with self._Session() as s:
            return s.scalar(select(func.count()).select_from(User)) or 0

    def list_all_jobs(self, *, status: str | None = None, svc: str | None = None,
                      account_id: str | None = None, limit: int = 200,
                      offset: int = 0) -> list[Job]:
        with self._Session() as s:
            q = select(Job)
            if status:
                q = q.where(Job.status == status)
            if svc:
                q = q.where(Job.svc == svc)
            if account_id:
                q = q.where(Job.account_id == account_id)
            q = q.order_by(Job.created_at.desc()).limit(limit).offset(offset)
            return list(s.scalars(q))

    def count_jobs(self) -> int:
        with self._Session() as s:
            return s.scalar(select(func.count()).select_from(Job)) or 0

    def count_jobs_by_status(self) -> dict[str, int]:
        with self._Session() as s:
            rows = s.execute(
                select(Job.status, func.count()).group_by(Job.status)
            ).all()
            return {status: n for status, n in rows}
