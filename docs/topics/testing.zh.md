# 测试

[English](testing.md) | 中文

> **适用**：在任何一层（service / framework / gateway）跑或写测试时。
> **来源**：各 service `pyproject.toml` 的 `[dependency-groups]`、`framework/pyproject.toml` 的 extras、`gateway/pyproject.toml`。
> **刷新/删除条件**：测试命令变化时（留意 gateway 命令——gateway 无 dev group，需 `--with pytest`）。

测试按组件隔离。**改服务前先读该 service 的 `README.md` 与其测试用例。**

```bash
# 单个 service 的离线单测（在 service 目录内跑）
cd services/<svc>-server
uv run --group dev python -m pytest tests/ -q

# FC 集成测试（需已部署；默认 skip）
RUN_FC_TESTS=1 uv run --group dev python -m pytest -m fc tests/ -v

# framework 自身
cd framework && uv run --extra dev python -m pytest tests/ -q

# gateway（无 dev group → 注入 pytest）
cd gateway && uv run --with pytest --with pytest-asyncio python -m pytest tests/ -v

# lint 单个 service
uvx ruff check services/<svc>/

# 对着本地 kind 部署的 gateway 功能测试
make local-test
```

## 测试文件布局（per service）

| 文件 | 覆盖 |
|---|---|
| `test_app.py` | 离线 TestClient 单测（health / manifest / 一个端点） |
| `test_cli.py` | CLI 注册 / argv builder / `create_cli` |
| `test_fc.py` | FC sync submit/poll 集成（`@pytest.mark.fc`） |
| `test_fc_task.py` | `/api/tasks/<name>` 异步任务集成（`@pytest.mark.fc`） |

`conftest.py` 用 importlib 注册 `server` 模块并定义 `fc` marker。

## 注意

- 少数服务的测试读取 vendored 的 `upstream/`（git-ignore）。先跑该 service 的 `scripts/vendor.sh`，
  否则相关测试缺文件失败。
- FC 测试（`@pytest.mark.fc`）默认 skip；`RUN_FC_TESTS=1`（或 `-m fc`）对着已部署环境开启。
- 离线 service 测试 mock 掉子进程；只需轻量 `[dependency-groups] dev` 依赖，不需重型运行时栈。
- `make local-test` 目标是本地部署（`http://127.0.0.1:9000`）；它是 gateway 测试，不是 service 测试。
