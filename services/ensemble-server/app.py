"""ensemble-server FastAPI app — multi-method aggregator.

Unlike most bioagent services, ensemble-server does NOT run subprocess
jobs.  It orchestrates remote FC calls via FCDispatcher, so we use
bioagent_service.ServiceSettings (for jobs_base_dir / NAS conventions)
but not the framework's JobRunner.

Routes are mounted from later phases (folding, jobs, manifest, healthz).
"""

from __future__ import annotations

import logging

from bioagent_service import read_version_file
from fastapi import FastAPI

from .settings import EnsembleSettings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

settings = EnsembleSettings()

app = FastAPI(
    title="Ensemble Server",
    version=read_version_file(__file__, default="0.0.1"),
)

# Routes mounted in later phases.
