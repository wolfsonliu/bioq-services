"""Orchestrator unit tests with fake adapters + mocked FCDispatcher.

Verifies the task-kind agnostic submit + refresh + aggregate flow without
touching real FC or any real adapter implementation.
"""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from server.adapters.base import MethodAdapter
from server.adapters.registry import MethodRegistry
from server.orchestrator.models import SubTaskStatus
from server.orchestrator.orchestrator import Orchestrator
from server.orchestrator.store import EnsembleJobStore
from server.task_kind import TaskKind

from pipelines.framework.dispatcher import DispatchHandle, TaskStatus


# Restrict anyio backend to asyncio (we don't need trio).
@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _FakeInput(BaseModel):
    name: str = "test"


class _FakeOptions(BaseModel):
    pass


class _FakeOutput(BaseModel):
    method: str
    payload: dict


class _FakeFoldingAdapter(MethodAdapter[_FakeInput, _FakeOutput]):
    task_kind = TaskKind.FOLDING
    method_options_schema = _FakeOptions

    def __init__(self, name: str, fc_mock: MagicMock) -> None:
        super().__init__(fc_mock)
        self.name = name

    def build_request(self, input, options):
        return "/api/tasks/fake", {"name": input.name}, {}

    def normalize_output(self, sub_task_id, downloaded_dir):
        return _FakeOutput(method=self.name, payload={"sub_task_id": sub_task_id})


def _make_fc_mock(function_name: str = "fake-server") -> MagicMock:
    m = MagicMock()
    m.function = function_name
    # submit returns a DispatchHandle
    m.submit.return_value = DispatchHandle(
        backend="fc",
        task_id="<set-per-call>",
        backend_ref={"invocation_id": "fake-inv-id", "function": function_name},
    )
    return m


def _make_orchestrator(
    aggregator=None,
    adapter_names: list[str] | None = None,
) -> tuple[Orchestrator, dict[str, MagicMock], EnsembleJobStore, Path]:
    """Build an orchestrator with N fake adapters and a temp store."""
    if adapter_names is None:
        adapter_names = ["fake_a", "fake_b"]
    td = tempfile.mkdtemp()
    store = EnsembleJobStore(Path(td))
    registry = MethodRegistry()
    fc_mocks: dict[str, MagicMock] = {}
    for n in adapter_names:
        fc_mock = _make_fc_mock(f"{n}-server")
        registry.register(_FakeFoldingAdapter(n, fc_mock))
        fc_mocks[n] = fc_mock
    aggregators = {TaskKind.FOLDING: aggregator} if aggregator else {}
    orch = Orchestrator(registry=registry, store=store, aggregators=aggregators)
    return orch, fc_mocks, store, Path(td)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_submit_creates_job_with_one_subtask_per_method():
    """Submitting 2 methods → EnsembleJob has 2 sub_tasks, all RUNNING."""
    orch, fc_mocks, store, _ = _make_orchestrator(adapter_names=["fake_a", "fake_b"])

    job = await orch.submit(
        task_kind=TaskKind.FOLDING,
        input=_FakeInput(name="hello"),
        methods=["fake_a", "fake_b"],
        method_options={},
        customer_id="customer1",
    )

    assert job.task_id.startswith("ens_fold_")
    assert set(job.sub_tasks.keys()) == {"fake_a", "fake_b"}
    for sub in job.sub_tasks.values():
        assert sub.status == SubTaskStatus.RUNNING
        assert sub.started_at is not None
        assert sub.fc_invocation_id == "fake-inv-id"
    assert store.get(job.task_id) is not None


@pytest.mark.anyio
async def test_refresh_aggregates_when_all_succeed(tmp_path):
    """After all FC sub-tasks succeed, refresh aggregates + marks completed_at."""

    def fake_aggregator(sub_tasks):
        return {
            "ranked_methods": sorted(
                s.method for s in sub_tasks if s.status == SubTaskStatus.SUCCEEDED
            )
        }

    orch, fc_mocks, store, _ = _make_orchestrator(
        aggregator=fake_aggregator,
        adapter_names=["fake_a", "fake_b"],
    )

    # Submit
    job = await orch.submit(
        task_kind=TaskKind.FOLDING,
        input=_FakeInput(name="hi"),
        methods=["fake_a", "fake_b"],
        method_options={},
        customer_id="c",
    )

    # Mock FCs to report SUCCEEDED + a fake fetch_result returning a zip path
    fake_zip_dir = tmp_path / "fake_zips"
    fake_zip_dir.mkdir()
    fake_zip = fake_zip_dir / "fake.zip"
    with zipfile.ZipFile(fake_zip, "w") as zf:
        zf.writestr("placeholder.txt", "ok")

    for _m, fc in fc_mocks.items():
        fc.get_status.return_value = TaskStatus.SUCCEEDED
        fc.fetch_result.return_value = fake_zip

    refreshed = await orch.refresh(job.task_id)
    assert refreshed is not None
    assert refreshed.completed_at is not None
    for sub in refreshed.sub_tasks.values():
        assert sub.status == SubTaskStatus.SUCCEEDED
        assert sub.completed_at is not None
        assert sub.output is not None
    assert refreshed.aggregated_output == {"ranked_methods": ["fake_a", "fake_b"]}


@pytest.mark.anyio
async def test_refresh_marks_failed_when_fc_failed(tmp_path):
    orch, fc_mocks, store, _ = _make_orchestrator(adapter_names=["fake_a"])
    job = await orch.submit(
        task_kind=TaskKind.FOLDING,
        input=_FakeInput(),
        methods=["fake_a"],
        method_options={},
        customer_id="c",
    )

    fc_mocks["fake_a"].get_status.return_value = TaskStatus.FAILED

    refreshed = await orch.refresh(job.task_id)
    assert refreshed is not None
    assert refreshed.sub_tasks["fake_a"].status == SubTaskStatus.FAILED
    assert refreshed.completed_at is not None
    # Aggregator NOT called if no successes
    assert refreshed.aggregated_output is None


@pytest.mark.anyio
async def test_partial_failure_still_aggregates(tmp_path):
    """One method succeeds, one fails → aggregator runs on the successful subset."""

    def fake_aggregator(sub_tasks):
        succeeded = [s for s in sub_tasks if s.status == SubTaskStatus.SUCCEEDED]
        return {"succeeded_count": len(succeeded)}

    orch, fc_mocks, store, _ = _make_orchestrator(
        aggregator=fake_aggregator,
        adapter_names=["fake_a", "fake_b"],
    )
    job = await orch.submit(
        task_kind=TaskKind.FOLDING,
        input=_FakeInput(),
        methods=["fake_a", "fake_b"],
        method_options={},
        customer_id="c",
    )

    fake_zip = tmp_path / "z.zip"
    with zipfile.ZipFile(fake_zip, "w") as zf:
        zf.writestr("ok.txt", "x")

    fc_mocks["fake_a"].get_status.return_value = TaskStatus.SUCCEEDED
    fc_mocks["fake_a"].fetch_result.return_value = fake_zip
    fc_mocks["fake_b"].get_status.return_value = TaskStatus.FAILED

    refreshed = await orch.refresh(job.task_id)
    assert refreshed is not None
    assert refreshed.sub_tasks["fake_a"].status == SubTaskStatus.SUCCEEDED
    assert refreshed.sub_tasks["fake_b"].status == SubTaskStatus.FAILED
    assert refreshed.aggregated_output == {"succeeded_count": 1}
    assert refreshed.completed_at is not None
