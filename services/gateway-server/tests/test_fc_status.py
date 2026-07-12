from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from server.fc_status import FcStatusClient


def _fake_client(fc_state: str):
    c = MagicMock()
    c.get_async_task.return_value = SimpleNamespace(body=SimpleNamespace(status=fc_state))
    return c


def test_disabled_without_creds():
    assert FcStatusClient().enabled is False
    assert FcStatusClient(access_key_id="a", access_key_secret="b").enabled is True
    assert FcStatusClient(client=_fake_client("Running")).enabled is True


def test_status_mapping():
    cases = {
        "Enqueued": "pending",
        "Running": "running",
        "Retrying": "running",
        "Succeeded": "completed",
        "Failed": "failed",
        "Stopped": "failed",
        "SomethingNew": "running",  # unknown -> running (safe default)
    }
    for fc_state, expected in cases.items():
        client = FcStatusClient(client=_fake_client(fc_state))
        assert client.get_status(function="fc_x", task_id="job-1") == expected


def test_get_status_calls_get_async_task():
    fake = _fake_client("Succeeded")
    client = FcStatusClient(client=fake)
    client.get_status(function="fc_proteinmpnn", task_id="abc", region="cn-hangzhou")
    kwargs = fake.get_async_task.call_args.kwargs
    assert kwargs["function_name"] == "fc_proteinmpnn"
    assert kwargs["task_id"] == "abc"
