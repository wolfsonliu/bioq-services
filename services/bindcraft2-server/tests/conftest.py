"""测试装置：把服务目录按镜像里的方式挂成 `server` 包。

Dockerfile 把 `services/bindcraft2-server/` 拷到 `/opt/bindcraft2/server/` 并以
`server.app:app` 导入；本地 pytest 用同样的别名，保证 import 路径与生产一致。
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest
from pydantic_settings import SettingsConfigDict

SERVICE_DIR = Path(__file__).resolve().parent.parent

if "server" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "server",
        SERVICE_DIR / "__init__.py",
        submodule_search_locations=[str(SERVICE_DIR)],
    )
    if spec is not None and spec.loader is not None:
        module = importlib.util.module_from_spec(spec)
        sys.modules["server"] = module
        spec.loader.exec_module(module)


# `fc` marker：opt-in 的 FC 线上测试。
from bioq_service.fc_testing import (  # noqa: E402
    register_fc_marker,
    skip_fc_tests_unless_enabled,
)
from server.settings import Bindcraft2Settings  # noqa: E402


def pytest_configure(config):
    register_fc_marker(config)


def pytest_collection_modifyitems(config, items):
    skip_fc_tests_unless_enabled(config, items)


# ---------------------------------------------------------------------------
# 离线 fixture：只把子进程换成 stub，其余全走真实框架
# （真实 JobRunner、真实 HTTP 路由、真实 job 目录与日志）。
# ---------------------------------------------------------------------------


class OfflineSettings(Bindcraft2Settings):
    """离线测试用：不读 .env，python 指向 stub，module 置空。"""

    model_config = SettingsConfigDict(
        env_prefix="BINDCRAFT2_TEST_", env_file=None, extra="ignore", case_sensitive=False
    )

    @classmethod
    def build(cls, tmp_path: Path, **overrides) -> "OfflineSettings":
        stub = tmp_path / "fake_bindcraft"
        shutil.copy2(SERVICE_DIR / "tests" / "data" / "fake_bindcraft.sh", stub)
        stub.chmod(0o755)
        # 子进程 cwd 是上游源码树（adapter.subprocess_cwd → settings.root）：生产里
        # /opt/bindcraft 必然存在，离线时必须先建出来，否则 Popen(cwd=...) 直接 ENOENT。
        (tmp_path / "root").mkdir(parents=True, exist_ok=True)
        params = dict(
            jobs_base_dir=tmp_path / "jobs",
            root=tmp_path / "root",
            # ProteinMPNN 随包在源码树里；生产默认值二者同为 /opt/bindcraft，
            # 离线时显式指向同一个 tmp 目录，否则探针会去查宿主的 /opt/bindcraft。
            shipped_weights_dir=tmp_path / "root",
            python=str(stub),
            module="",
            alphafold_params_dir=tmp_path / "af",
            compile_cache_dir=tmp_path / "cache",
            max_concurrent_jobs=1,
            gpu_probe_ttl_seconds=0,
        )
        params.update(overrides)
        return cls(**params)


@pytest.fixture
def offline_settings(tmp_path: Path) -> OfflineSettings:
    return OfflineSettings.build(tmp_path)
