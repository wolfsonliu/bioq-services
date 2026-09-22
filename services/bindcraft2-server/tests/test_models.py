"""DesignRequest / RankRequest / FilterRequest 的校验行为。"""

from __future__ import annotations

import pytest

from server.models import (
    DesignRequest,
    FilterRequest,
    RankRequest,
    RankTable,
    validate_target_selection,
)


def test_defaults_are_upstream_defaults():
    req = DesignRequest()
    assert req.modality == "binder"
    assert req.number_of_final_designs == 10
    assert req.max_trajectories is None
    assert req.properties == []


def test_properties_lists_only_enabled_flags():
    req = DesignRequest(humanize=True, bigbang=True)
    assert req.properties == ["humanize", "bigbang"]


def test_binder_lengths_accepts_list():
    assert DesignRequest(binder_lengths=[60, 100]).binder_lengths == [60, 100]


def test_binder_lengths_accepts_comma_string():
    # CLI 路径把 list 字段当 type=str 传入。
    assert DesignRequest(binder_lengths="60,100").binder_lengths == [60, 100]


def test_binder_lengths_accepts_json_string():
    # HTTP multipart 路径经 model_form_depends 的 json.loads 后应同时兼容。
    assert DesignRequest(binder_lengths="[60,100]").binder_lengths == [60, 100]


def test_binder_lengths_rejects_descending():
    with pytest.raises(ValueError, match="ascending"):
        DesignRequest(binder_lengths=[100, 60])


def test_binder_lengths_rejects_three_entries():
    with pytest.raises(ValueError, match="1 or 2"):
        DesignRequest(binder_lengths=[60, 70, 80])


def test_binder_lengths_rejects_non_positive():
    # 覆盖 `any(value < 1 ...)` 分支——少了这个测试，删掉该分支所有测试仍全绿
    # （Task 2 实现者用变异测试证明了这个缺口）。
    with pytest.raises(ValueError, match="must be >= 1"):
        DesignRequest(binder_lengths=[0])
    with pytest.raises(ValueError, match="must be >= 1"):
        DesignRequest(binder_lengths=[-5, 80])


def test_binder_lengths_rejects_empty_list():
    # 钉住 `is not None` 语义：改成真值判断会让 [] 被放行。
    with pytest.raises(ValueError, match="1 or 2"):
        DesignRequest(binder_lengths=[])


def test_binder_lengths_rejects_float_and_bool():
    # 钉住严格整数解析：`int(80.5)` 会静默截断成 80，必须报错而不是改值。
    with pytest.raises(ValueError, match="must be integers"):
        DesignRequest(binder_lengths=[80.5, 90.5])
    with pytest.raises(ValueError, match="must be integers"):
        DesignRequest(binder_lengths="[80.5,90.5]")
    with pytest.raises(ValueError, match="must be integers"):
        DesignRequest(binder_lengths=[True, True])


def test_max_trajectories_must_be_positive():
    with pytest.raises(ValueError):
        DesignRequest(max_trajectories=0)


def test_max_trajectories_is_bounded():
    # 上界兜住共享卡上的 GPU 成本：没有它 10**30 会被原样写进 campaign。
    assert DesignRequest(max_trajectories=100_000).max_trajectories == 100_000
    with pytest.raises(ValueError):
        DesignRequest(max_trajectories=100_001)


def test_top_must_be_positive():
    with pytest.raises(ValueError):
        RankRequest(campaign_uri="job://abc", top=0)
    with pytest.raises(ValueError):
        FilterRequest(campaign_uri="job://abc", top=0)


def test_binder_lengths_rejects_non_integer():
    with pytest.raises(ValueError, match="integers"):
        DesignRequest(binder_lengths="sixty,80")


def test_number_of_final_designs_bounds():
    with pytest.raises(ValueError):
        DesignRequest(number_of_final_designs=0)
    with pytest.raises(ValueError):
        DesignRequest(number_of_final_designs=1001)


def test_campaign_name_pattern():
    assert DesignRequest(campaign_name="pdl1_v2").campaign_name == "pdl1_v2"
    with pytest.raises(ValueError):
        DesignRequest(campaign_name="has spaces")


# --- FIX 1: target_name 与 target 级字段互斥 ---------------------------------


def test_target_name_rejects_explicit_target_chains():
    # shipped preset 自带 chains，target_chains 会被 build_campaign_json 静默丢弃。
    with pytest.raises(ValueError, match="target_chains"):
        DesignRequest(target_name="hPDL1", target_chains="A")


def test_target_name_rejects_explicit_hotspots():
    with pytest.raises(ValueError, match="hotspots"):
        DesignRequest(target_name="hPDL1", hotspots="54,56")


def test_target_name_rejects_explicit_coldspots():
    with pytest.raises(ValueError, match="coldspots"):
        DesignRequest(target_name="hPDL1", coldspots="90-95")


def test_target_name_alone_still_validates():
    # target_chains 有默认值：任何"字段非空/字段存在"式的检查都会在这里误报 422。
    req = DesignRequest(target_name="hPDL1")
    assert req.target_name == "hPDL1"
    assert req.target_chains is None


def test_target_name_tolerates_framework_injected_defaults():
    """HTTP 表单路径会把未发送的字段用默认值（None）显式传进构造函数。

    `model_form_depends` 的合成签名给每个字段都带 Form 默认值，FastAPI 会把这些
    默认值一并注入 kwargs，于是这三个字段总会出现在 `model_fields_set` 里——
    显式提供的判定必须再排除掉等于默认值的条目。
    """
    req = DesignRequest(
        target_name="hPDL1", target_chains=None, hotspots=None, coldspots=None
    )
    assert req.target_name == "hPDL1"


def test_upload_path_allows_target_fields():
    # 没有 target_name 时（target 上传 / target_uri 路径），这三个字段照常生效。
    req = DesignRequest(target_chains="A", hotspots="54,56", coldspots="90-95")
    assert req.target_chains == "A"
    assert req.hotspots == "54,56"
    assert req.coldspots == "90-95"


# --- FIX 2: modality 单值 / 逗号组合 / 数组 ----------------------------------


def test_modality_single_name_stays_string():
    assert DesignRequest(modality="binder").modality == "binder"


def test_modality_splits_comma_combination():
    assert DesignRequest(modality="binder,VHH").modality == ["binder", "VHH"]


def test_modality_accepts_list():
    assert DesignRequest(modality=["binder", "VHH"]).modality == ["binder", "VHH"]


def test_modality_strips_blank_parts():
    assert DesignRequest(modality="binder, VHH ,").modality == ["binder", "VHH"]


def test_rank_request_defaults():
    req = RankRequest(campaign_uri="job://abc")
    assert req.on == ["i_pDAE"]
    assert req.table is RankTable.accepted
    assert req.lowest_first is None


def test_rank_request_parses_comma_string_metrics():
    assert RankRequest(campaign_uri="job://abc", on="i_pTM,i_pDAE").on == ["i_pTM", "i_pDAE"]


def test_rank_request_rejects_empty_metrics():
    with pytest.raises(ValueError, match="at least one metric"):
        RankRequest(campaign_uri="job://abc", on=[])


def test_filter_request_parses_comma_string_where():
    req = FilterRequest(campaign_uri="job://abc", where="i_pAE=0.45,Interface_Residues>=7")
    assert req.where == ["i_pAE=0.45", "Interface_Residues>=7"]


def test_filter_request_defaults_to_candidates_table():
    assert FilterRequest(campaign_uri="job://abc").table is RankTable.candidates


def test_validate_target_selection_accepts_exactly_one():
    validate_target_selection("hPDL1", False, None)
    validate_target_selection(None, True, None)
    validate_target_selection(None, False, "job://abc/target.pdb")


def test_validate_target_selection_rejects_none():
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        validate_target_selection(None, False, None)
    assert exc.value.status_code == 422


def test_validate_target_selection_rejects_two():
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        validate_target_selection("hPDL1", True, None)
    assert exc.value.status_code == 422
