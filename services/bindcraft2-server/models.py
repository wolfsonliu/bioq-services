"""各 endpoint 的 pydantic 请求模型。

文件输入（target 上传 / target_uri）走路由层 `File(...)` / `Form(...)`，不在
model 上——因此"三选一"校验是 `validate_target_selection()` 这个普通函数，
在路由里调用，而不是 `model_validator`。
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Optional

from fastapi import HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator

__all__ = [
    "PROPERTY_FIELDS",
    "DesignRequest",
    "FilterRequest",
    "RankRequest",
    "RankTable",
    "validate_target_selection",
]

# 长度解析失败的统一报错文案（两处 raise 共用，测试按它匹配）。
_BINDER_LENGTHS_MESSAGE = (
    "binder_lengths must be integers, e.g. '80,80' or '[60,100]'"
)

# 与上游 campaign JSON 顶层 key 一一对应的 9 个可选属性。
PROPERTY_FIELDS: tuple[str, ...] = (
    "forced_targeting",
    "humanize",
    "protease_stable",
    "disulfide_staple",
    "mixed_topology",
    "termini_together",
    "termini_accessible",
    "initial_guess",
    "bigbang",
)


class RankTable(str, Enum):
    """可选的源表（上游 `rank --table` / `filter --table`）。"""

    accepted = "accepted"
    candidates = "candidates"
    trajectories = "trajectories"


def _decode_list(value: Any) -> Any:
    """把 JSON 字符串或逗号分隔字符串解码成 list。

    HTTP 路径：`model_form_depends` 对复杂字段做 `json.loads`，拿到 list。
    CLI 路径：`cli._add_model_args` 对非 scalar 字段用 `type=str`，拿到原始字符串。
    两条路都要能走通，所以这里同时接受两种写法。
    """
    if value is None or isinstance(value, list):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            return json.loads(text)
        return [part.strip() for part in text.split(",") if part.strip()]
    return value


def validate_target_selection(
    target_name: Optional[str],
    has_upload: bool,
    target_uri: Optional[str],
) -> None:
    """`target_name` / `target` 上传 / `target_uri` 恰好提供一个。"""
    provided = [bool(target_name), bool(has_upload), bool(target_uri)]
    if sum(provided) != 1:
        raise HTTPException(
            status_code=422,
            detail=(
                "Exactly one of `target_name`, `target` (file upload) or `target_uri` "
                "is required."
            ),
        )


class DesignRequest(BaseModel):
    """`POST /api/design` 的参数；目标结构走路由层上传/URI。"""

    target_name: Optional[str] = Field(
        default=None,
        description="上游 shipped target 名（bindcraft design --list-targets）。",
    )
    target_chains: Optional[str] = Field(
        default=None, description="目标链，如 'A' 或 'A,B'。", examples=["A"]
    )
    hotspots: Optional[str] = Field(
        default=None,
        description="结合位点残基，编号取自输入结构。",
        examples=["54,56,66-70"],
    )
    coldspots: Optional[str] = Field(
        default=None, description="要求保持自由的区域。", examples=["90-95"]
    )
    modality: str = Field(
        default="binder",
        description="上游 modality 名，允许逗号组合（binder,VHH,ARP,scFv,Fab,...）。",
    )
    binder_lengths: Optional[list[int]] = Field(
        default=None,
        description="[80,80] 定长；[60,100] 范围。scaffold 模态不需要。",
    )
    number_of_final_designs: int = Field(
        default=10, ge=1, le=1000, description="收够 N 个 accepted 即停。"
    )
    max_trajectories: Optional[int] = Field(
        default=None,
        ge=1,
        description="尝试次数硬上限；未给则用服务端 default_max_trajectories。",
    )

    forced_targeting: bool = False
    humanize: bool = False
    protease_stable: bool = False
    disulfide_staple: bool = False
    mixed_topology: bool = False
    termini_together: bool = False
    termini_accessible: bool = False
    initial_guess: bool = False
    bigbang: bool = False

    core: Optional[str] = Field(
        default=None, description="上游 core profile，如 'benchmark'。"
    )
    campaign_name: Optional[str] = Field(
        default=None,
        pattern=r"^[A-Za-z0-9_-]{1,64}$",
        description="结果命名与元数据标签；未给则用 job_id。",
    )

    @field_validator("binder_lengths", mode="before")
    @classmethod
    def _decode_binder_lengths(cls, value: Any) -> Any:
        decoded = _decode_list(value)
        if decoded is None:
            return None
        lengths: list[int] = []
        for item in decoded:
            # bool 是 int 的子类，必须先拦掉；float 也必须拦——`int(80.5)` 会静默
            # 截断成 80，值变了却不报错，是最坏的一类错误。
            if isinstance(item, bool) or not isinstance(item, (int, str)):
                raise ValueError(_BINDER_LENGTHS_MESSAGE)
            text = str(item).strip()
            if not text.lstrip("+-").isdigit():
                raise ValueError(_BINDER_LENGTHS_MESSAGE)
            lengths.append(int(text))
        return lengths

    @model_validator(mode="after")
    def _check_binder_lengths(self) -> "DesignRequest":
        if self.binder_lengths is not None:
            if len(self.binder_lengths) not in (1, 2):
                raise ValueError("binder_lengths must have 1 or 2 entries")
            if any(value < 1 for value in self.binder_lengths):
                raise ValueError("binder_lengths entries must be >= 1")
            if (
                len(self.binder_lengths) == 2
                and self.binder_lengths[0] > self.binder_lengths[1]
            ):
                raise ValueError("binder_lengths must be ascending: [min, max]")
        return self

    @property
    def properties(self) -> list[str]:
        """已开启的属性名，顺序与 PROPERTY_FIELDS 一致。"""
        return [name for name in PROPERTY_FIELDS if getattr(self, name)]


class RankRequest(BaseModel):
    """`POST /api/rank` 的参数。"""

    campaign_uri: str = Field(
        description="源 campaign 目录：job://<id>、job://<id>/output、file:///abs 或裸绝对路径。"
    )
    on: list[str] = Field(default=["i_pDAE"], description="排序指标；多个用于 tie-break。")
    lowest_first: Optional[bool] = Field(
        default=None, description="为 None 时用上游自动方向判定。"
    )
    table: RankTable = RankTable.accepted
    top: Optional[int] = Field(default=None, ge=1, description="控制台显示行数上限。")

    @field_validator("on", mode="before")
    @classmethod
    def _decode_on(cls, value: Any) -> Any:
        return _decode_list(value)

    @model_validator(mode="after")
    def _check_on(self) -> "RankRequest":
        if not self.on:
            raise ValueError("on must contain at least one metric")
        return self


class FilterRequest(BaseModel):
    """`POST /api/filter` 的参数。"""

    campaign_uri: str = Field(
        description="源 campaign 目录：job://<id>、job://<id>/output、file:///abs 或裸绝对路径。"
    )
    where: Optional[list[str]] = Field(
        default=None,
        description="阈值表达式；空表示用 campaign 自身阈值重放。",
        examples=["i_pAE=0.45", "Interface_Residues>=7"],
    )
    table: RankTable = RankTable.candidates
    top: Optional[int] = Field(default=None, ge=1, description="控制台显示行数上限。")

    @field_validator("where", mode="before")
    @classmethod
    def _decode_where(cls, value: Any) -> Any:
        return _decode_list(value)
