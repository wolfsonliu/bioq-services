"""ensemble-server FastAPI app — multi-method aggregator.

Unlike most bioagent services, ensemble-server does NOT run subprocess
jobs.  It orchestrates remote FC calls via FCDispatcher, so we use
bioagent_service.ServiceSettings (for jobs_base_dir / NAS conventions)
but not the framework's JobRunner.
"""

from __future__ import annotations

import logging

from bioagent_service import read_version_file
from fastapi import FastAPI

from pipelines.framework.fc_dispatcher import FCDispatcher

from .adapters.folding.alphafold import AlphaFoldFoldingAdapter
from .adapters.folding.boltz import BoltzFoldingAdapter
from .adapters.folding.esmfold2 import ESMFold2FoldingAdapter
from .adapters.registry import registry
from .folding.aggregator import aggregate_folding
from .orchestrator.orchestrator import Orchestrator
from .orchestrator.store import EnsembleJobStore
from .routes import folding as folding_routes
from .routes import jobs as jobs_routes
from .routes import manifest as manifest_routes
from .settings import EnsembleSettings
from .task_kind import TaskKind

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


def _build_dispatcher(method_name: str, cfg) -> FCDispatcher:
    return FCDispatcher(
        region=cfg.region,
        function=cfg.function,
        access_key_id=settings.fc_access_key_id,
        access_key_secret=settings.fc_access_key_secret,
        http_base_url=cfg.http_base_url,
    )


# Register folding adapters (only those configured + enabled in settings.fc_methods).
_FOLDING_ADAPTER_CLASSES = (
    AlphaFoldFoldingAdapter,
    ESMFold2FoldingAdapter,
    BoltzFoldingAdapter,
)
for adapter_cls in _FOLDING_ADAPTER_CLASSES:
    cfg = settings.fc_methods.get(adapter_cls.name)
    if cfg is not None and cfg.enabled:
        dispatcher = _build_dispatcher(adapter_cls.name, cfg)
        registry.register(adapter_cls(dispatcher))
        logger.info("registered adapter: folding/%s", adapter_cls.name)
    else:
        logger.info(
            "skipped adapter folding/%s: no config or disabled",
            adapter_cls.name,
        )

# Wire orchestrator.  Ensure jobs_base_dir exists.
settings.jobs_base_dir.mkdir(parents=True, exist_ok=True)
store = EnsembleJobStore(settings.jobs_base_dir)
orchestrator = Orchestrator(
    registry=registry,
    store=store,
    aggregators={TaskKind.FOLDING: aggregate_folding},
)

# Stash on app.state for route handlers to consume.
app.state.settings = settings
app.state.registry = registry
app.state.store = store
app.state.orchestrator = orchestrator

# Mount routes.
app.include_router(manifest_routes.router)
app.include_router(folding_routes.router)
app.include_router(jobs_routes.router)
