"""Folding ensemble aggregator — produces cross-method ranking.

Default ranking metric: pLDDT of each method's rank-0 structure (when
available).  Methods missing pLDDT get score 0.0 and rank last.
"""

from __future__ import annotations

from typing import Any

from ..orchestrator.models import SubTaskRecord, SubTaskStatus


def aggregate_folding(sub_tasks: list[SubTaskRecord]) -> dict[str, Any]:
    """Cross-method ranking + ensemble score.

    Returns {ensemble_ranking, ensemble_score}.  ensemble_score is the max
    pLDDT across methods' rank-0 structures.  Failed methods are skipped.
    """
    candidates: list[dict[str, Any]] = []
    for sub in sub_tasks:
        if sub.status != SubTaskStatus.SUCCEEDED or not sub.output:
            continue
        structs = sub.output.get("structures", [])
        if not structs:
            continue
        score = structs[0].get("plddt")
        if score is None:
            score = 0.0
        candidates.append({
            "method": sub.method,
            "rank": 0,
            "score": float(score),
            "url": structs[0].get("url"),
        })

    candidates.sort(key=lambda c: c["score"], reverse=True)
    for i, c in enumerate(candidates):
        c["overall_rank"] = i

    return {
        "ensemble_ranking": candidates,
        "ensemble_score": max((c["score"] for c in candidates), default=None),
    }
