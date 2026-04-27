"""Pydantic models for API request/response schemas."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class StepName(str, Enum):
    RFDIFFUSION = "rfdiffusion"
    PROTEINMPNN = "proteinmpnn"
    RF2 = "rf2"


# --- RFdiffusion ---


class RFdiffusionRequest(BaseModel):
    num_designs: int = Field(default=10, ge=1, le=10000)
    design_loops: str = Field(default="H1:,H2:,H3:")
    hotspots: Optional[str] = Field(default=None, examples=["B146,B170,B177"])
    diffuser_t: int = Field(default=50, ge=1, le=200)
    final_step: int = Field(default=1, ge=1)
    deterministic: bool = False
    no_trajectory: bool = True


# --- ProteinMPNN ---


class ProteinMPNNRequest(BaseModel):
    loops: str = Field(default="H1,H2,H3")
    seqs_per_struct: int = Field(default=4, ge=1, le=100)
    temperature: float = Field(default=0.2, ge=0.01, le=2.0)
    omit_aas: str = Field(default="CX")
    deterministic: bool = False


# --- RF2 ---


class RF2Request(BaseModel):
    num_recycles: int = Field(default=10, ge=1, le=50)
    hotspot_show_prop: float = Field(default=0.1, ge=0.0, le=1.0)
    seed: Optional[int] = None


# --- Full pipeline ---


class PipelineRequest(BaseModel):
    rfdiffusion: RFdiffusionRequest = Field(default_factory=RFdiffusionRequest)
    proteinmpnn: ProteinMPNNRequest = Field(default_factory=ProteinMPNNRequest)
    rf2: RF2Request = Field(default_factory=RF2Request)


# --- Job ---


class JobInfo(BaseModel):
    job_id: str
    status: JobStatus
    step: Optional[StepName] = None
    message: Optional[str] = None
    progress: Optional[str] = None


class JobResult(BaseModel):
    job_id: str
    status: JobStatus
    message: Optional[str] = None
    output_files: list[str] = Field(default_factory=list)
