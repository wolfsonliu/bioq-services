"""把 campaign *目录* URI 解析成本地路径——零拷贝。

`bioq_service.uris.resolve_uri` 是文件级的，且 `shutil.copy2` 源文件。campaign
目录可达数百 MB（`3_Ranked/*.cif`），拷贝不可接受。本模块直接返回 NAS 上的原
路径；调用方依赖 NAS 是共享挂载。

安全性：`job://` 分支只允许落在该 job 的 `output/` 目录内，且 `job_id` 必须是
单个普通路径段（拒 `.` / `..`）。`file://` 与裸绝对路径分支**故意不做白名单**：
网关会把 `oss://` 输入重写成 `/mnt/oss/...` 的裸绝对路径，且本服务按设计仅限
组织内部使用（见设计文档的许可证边界），调用方已被信任。
"""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException

from .settings import Bindcraft2Settings

__all__ = ["OUTPUT_DIRNAME", "resolve_campaign_dir"]

# job 目录下存放上游产物的子目录名；`job://<id>` 与 `job://<id>/output` 同义。
OUTPUT_DIRNAME = "output"


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _normalize_subpath(sub: str) -> str:
    """去掉冗余的前导 `output` 段。

    `job://<id>`、`job://<id>/output`、`job://<id>/output/3_Ranked` 与
    `job://<id>/3_Ranked` 指向同一处。只认整段 `output`——`output2` 不受影响。
    """
    if sub == OUTPUT_DIRNAME:
        return ""
    prefix = OUTPUT_DIRNAME + "/"
    if sub.startswith(prefix):
        return sub[len(prefix) :]
    return sub


def resolve_campaign_dir(campaign_uri: str, settings: Bindcraft2Settings) -> Path:
    """`job://<id>[/<sub>]` / `file://<abs>` / 裸绝对路径 → 已存在的目录路径。

    不支持 `oss://` 与 `http(s)://`：它们是单对象语义，且网关把 `oss://` 输入
    重写成 `/mnt/oss/...` 的裸绝对路径，所以裸路径分支已覆盖网关场景。
    """
    uri = (campaign_uri or "").strip()
    if not uri:
        raise HTTPException(status_code=422, detail="campaign_uri is required.")

    if uri.startswith("job://"):
        body = uri[len("job://") :]
        job_id, _, sub = body.partition("/")
        # job_id 必须是单个普通路径段。在拼接之前就拒掉 `.` / `..`：否则
        # `job://..` 会让 root 落到 `<jobs_base_dir>/../output`，逃出 jobs 树。
        if not job_id or job_id in {".", ".."}:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid job URI; expected job://<job_id>[/<subdir>]: {uri}",
            )
        jobs_root = settings.jobs_base_dir.resolve()
        root = (jobs_root / job_id / OUTPUT_DIRNAME).resolve()
        # 双保险：即便 job_id 校验被绕过，也不允许 root 逃出 jobs_base_dir。
        if not _is_within(root, jobs_root):
            raise HTTPException(
                status_code=422,
                detail=f"job:// path escapes the jobs base directory: {uri}",
            )
        sub = _normalize_subpath(sub)
        path = (root / sub).resolve() if sub else root
        if not _is_within(path, root):
            raise HTTPException(
                status_code=422,
                detail=f"job:// path escapes the job output directory: {uri}",
            )
    elif uri.startswith("file://") or uri.startswith("/"):
        path = Path(uri[len("file://") :] if uri.startswith("file://") else uri)
    else:
        raise HTTPException(
            status_code=422,
            detail=(
                "Unsupported campaign URI scheme; expected job://, file:// or an "
                f"absolute path: {uri}"
            ),
        )

    if not path.is_dir():
        raise HTTPException(
            status_code=404, detail=f"Campaign directory not found: {path}"
        )
    return path
