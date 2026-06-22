"""HTTP-based dispatcher used to call downstream FC services.

Pure httpx — no AK/SK, no FC OpenAPI SDK, no `pipelines.framework` dependency.
The FC platform itself recognizes ``X-Fc-Invocation-Type: Async`` on the
HTTP trigger URL and returns 202, so submit/poll/fetch all work as plain
HTTP calls.  Trade-off vs the OpenAPI control plane: every poll wakes up
an FC instance.  Acceptable for Phase 1 ensemble fan-out (1-3 sub-tasks
per job); revisit if concurrency outgrows it.
"""

from .http import DispatchHandle, HTTPDispatcher, TaskStatus

__all__ = ["DispatchHandle", "HTTPDispatcher", "TaskStatus"]
