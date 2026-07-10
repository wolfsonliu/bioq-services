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
    principal: Mapped[str] = mapped_column(String, primary_key=True)
    display_name: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class ApiKey(Base):
    __tablename__ = "api_keys"
    key_id: Mapped[str] = mapped_column(String, primary_key=True)
    principal: Mapped[str] = mapped_column(ForeignKey("users.principal"), index=True)
    secret_hash: Mapped[str] = mapped_column(String, index=True)
    status: Mapped[str] = mapped_column(String, default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class Job(Base):
    __tablename__ = "jobs"
    job_id: Mapped[str] = mapped_column(String, primary_key=True)
    principal: Mapped[str] = mapped_column(ForeignKey("users.principal"), index=True)
    svc: Mapped[str] = mapped_column(String)
    endpoint: Mapped[str] = mapped_column(String)
    fc_task_id: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, default="pending")
    input_params: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    output_prefix: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
