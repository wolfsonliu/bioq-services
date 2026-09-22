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
