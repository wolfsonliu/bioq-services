from pathlib import Path
from pydantic_settings import SettingsConfigDict


def test_settings_defaults():
    from server.settings import ReinventSettings

    class _Iso(ReinventSettings):
        model_config = SettingsConfigDict(env_prefix="REINVENT_TEST_", env_file=None, extra="ignore")

    s = _Iso()
    assert s.jobs_base_dir == Path("/data/reinvent_jobs")
    assert s.prior_base == Path("/data/models/reinvent")
    assert s.reinvent_bin == Path("/opt/reinvent-server/.venv/bin/reinvent")
    assert s.device == "cuda:0"
    assert s.task_endpoints_enabled is True
