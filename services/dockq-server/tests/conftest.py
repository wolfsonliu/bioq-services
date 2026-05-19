"""pytest hooks for dockq-server: register the `fc` marker + skip those by default."""

from bioagent_service.fc_testing import (
    register_fc_marker,
    skip_fc_tests_unless_enabled,
)


def pytest_configure(config):
    register_fc_marker(config)


def pytest_collection_modifyitems(config, items):
    skip_fc_tests_unless_enabled(config, items)
