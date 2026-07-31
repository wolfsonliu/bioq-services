"""SQLAlchemy 2.0 models: User / ApiKey / Job (gateway user + job store)."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, ForeignKey, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    account_id: Mapped[str] = mapped_column(String, primary_key=True)
    display_name: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, default="active")
    role: Mapped[str] = mapped_column(String, default="user")   # user | admin
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class ApiKey(Base):
    __tablename__ = "api_keys"
    key_id: Mapped[str] = mapped_column(String, primary_key=True)
    account_id: Mapped[str] = mapped_column(ForeignKey("users.account_id"), index=True)
    secret_hash: Mapped[str] = mapped_column(String, index=True)
    status: Mapped[str] = mapped_column(String, default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class Job(Base):
    # Composite PK (account_id, job_id): job_id is namespaced per account, so
    # different accounts may reuse the same job_id string without colliding. The
    # globally-unique downstream identity (FC async-task-id / shared NAS dir) is
    # the derived `fc_task_id` = f"{account_id}-{job_id}", NOT job_id. See
    # engineering/decisions/2026-07-09-unified-service-access-cli.md §5.
    __tablename__ = "jobs"
    account_id: Mapped[str] = mapped_column(ForeignKey("users.account_id"), primary_key=True)
    job_id: Mapped[str] = mapped_column(String, primary_key=True)
    svc: Mapped[str] = mapped_column(String)
    endpoint: Mapped[str] = mapped_column(String)
    fc_task_id: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, default="pending")
    input_params: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    output_prefix: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
