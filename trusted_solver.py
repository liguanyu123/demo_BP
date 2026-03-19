from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from packing_kernel import (
    canonicalize_placement,
    enumerate_candidate_positions,
    normalize_bin_size,
    normalize_items,
    placement_bbox_volume,
    placement_used_height,
)

# Search budget. These stay moderate because the geometry kernel already prunes
# candidate locations aggressively. Transport-stability filters further reduce
# branching, so we keep the solver practical for AFL iterations.
TOP_K_CANDIDATES_EARLY = 10
TOP_K_CANDIDATES_LATE = 6
EARLY_BEAM_WIDTH = 64
MID_BEAM_WIDTH = 32
LATE_BEAM_WIDTH = 12
EARLY_LEVELS = 8
MID_LEVELS = 18
MAX_TOTAL_STATES = 60000

HEAVY_TOP_K_CANDIDATES_EARLY = 16
HEAVY_TOP_K_CANDIDATES_LATE = 8
HEAVY_EARLY_BEAM_WIDTH = 128
HEAVY_MID_BEAM_WIDTH = 64
HEAVY_LATE_BEAM_WIDTH = 24
HEAVY_EARLY_LEVELS = 10
HEAVY_MID_LEVELS = 24
HEAVY_MAX_TOTAL_STATES = 180000
HEAVY_ITEM_THRESHOLD = 20

# Transport-stability heuristics.
BASE_BUILD_FRACTION = 0.40
MIN_BASE_BUILD_ITEMS = 4

ItemOrderFn = Callable[[Dict[int, Dict[str, Any]]], List[int]]
CandidateScoreFn = Callable[[int, Dict[str, Any], Dict[str, Any], Dict[str, Any], Sequence[int]], Any]


def _item_area(item: Dict[str, Any]) -> int:
    return item["l"] * item["w"]


def _item_volume(item: Dict[str, Any]) -> int:
    return item["l"] * item["w"] * item["h"]


def _stable_order_key(item_id: int, item: Dict[str, Any]) -> Tuple[Any, ...]:
    """Stable transport-oriented ordering.

    We do not permanently send fragile items to the very end. Instead, we bias
    toward large-footprint, lower-height boxes first so the solver constructs a
    stable base. Fragility is only a tie-breaker here; actual layer control is
    enforced later by candidate filtering.
    """
    return (
        -_item_area(item),
        item["h"],
        1 if item.get("fragile", False) else 0,
        -_item_volume(item),
        item_id,
    )


def default_order_items(items: Dict[int, Dict[str, Any]]) -> List[int]:
    items_n = normalize_items(items)
    ranked = sorted(items_n.items(), key=lambda kv: _stable_order_key(kv[0], kv[1]))
    return [item_id for item_id, _ in ranked]


def non_fragile_base_order_items(items: Dict[int, Dict[str, Any]]) -> List[int]:
    items_n = normalize_items(items)
    ranked = sorted(
        items_n.items(),
        key=lambda kv: (
            1 if kv[1].get("fragile", False) else 0,
            -_item_area(kv[1]),
            kv[1]["h"],
            -_item_volume(kv[1]),
            kv[0],
        ),
    )
    return [item_id for item_id, _ in ranked]


def low_height_footprint_order_items(items: Dict[int, Dict[str, Any]]) -> List[int]:
    items_n = normalize_items(items)
    ranked = sorted(
        items_n.items(),
        key=lambda kv: (
            kv[1]["h"],
            -_item_area(kv[1]),
            1 if kv[1].get("fragile", False) else 0,
            -_item_volume(kv[1]),
            kv[0],
        ),
    )
    return [item_id for item_id, _ in ranked]


def tall_first_order_items(items: Dict[int, Dict[str, Any]]) -> List[int]:
    items_n = normalize_items(items)
    ranked = sorted(
        items_n.items(),
        key=lambda kv: (
            -kv[1]["h"],
            -_item_area(kv[1]),
            1 if kv[1].get("fragile", False) else 0,
            -_item_volume(kv[1]),
            kv[0],
        ),
    )
    return [item_id for item_id, _ in ranked]


def volume_first_order_items(items: Dict[int, Dict[str, Any]]) -> List[int]:
    items_n = normalize_items(items)
    ranked = sorted(
        items_n.items(),
        key=lambda kv: (
            -_item_volume(kv[1]),
            -_item_area(kv[1]),
            kv[1]["h"],
            1 if kv[1].get("fragile", False) else 0,
            kv[0],
        ),
    )
    return [item_id for item_id, _ in ranked]


def fragile_last_order_items(items: Dict[int, Dict[str, Any]]) -> List[int]:
    items_n = normalize_items(items)
    ranked = sorted(
        items_n.items(),
        key=lambda kv: (
            1 if kv[1].get("fragile", False) else 0,
            -_item_area(kv[1]),
            kv[1]["h"],
            -_item_volume(kv[1]),
            kv[0],
        ),
    )
    return [item_id for item_id, _ in ranked]


def footprint_first_order_items(items: Dict[int, Dict[str, Any]]) -> List[int]:
    items_n = normalize_items(items)
    ranked = sorted(
        items_n.items(),
        key=lambda kv: (
            -_item_area(kv[1]),
            -(min(kv[1]["l"], kv[1]["w"])),
            kv[1]["h"],
            1 if kv[1].get("fragile", False) else 0,
            -_item_volume(kv[1]),
            kv[0],
        ),
    )
    return [item_id for item_id, _ in ranked]


def _transport_priority_terms(item: Dict[str, Any], candidate: Dict[str, Any]) -> Tuple[int, int, int, int]:
    footprint = _item_area(item)
    on_floor = candidate["z"] == 0
    fragile = bool(item.get("fragile", False))
    fragile_upper_penalty = 1 if fragile and not on_floor else 0
    floor_flag = 0 if on_floor else 1
    floor_footprint_bonus = -footprint if on_floor else 0
    floor_height_bonus = item["h"] if on_floor else 0
    return fragile_upper_penalty, floor_flag, floor_footprint_bonus, floor_height_bonus


def default_score_candidate(
    item_id: int,
    item: Dict[str, Any],
    candidate: Dict[str, Any],
    state: Dict[str, Any],
    bin_size: Sequence[int],
):
    fragile_penalty, floor_flag, floor_area_bonus, floor_height_bonus = _transport_priority_terms(item, candidate)
    return (
        fragile_penalty,
        floor_flag,
        floor_area_bonus,
        floor_height_bonus,
        candidate["z"],
        -candidate.get("support_area", 0.0),
        candidate["used_height_after"],
        candidate["bbox_volume_after"],
        candidate["y"] + candidate["x"],
        candidate["y"],
        candidate["x"],
        item_id,
    )


def support_greedy_score_candidate(
    item_id: int,
    item: Dict[str, Any],
    candidate: Dict[str, Any],
    state: Dict[str, Any],
    bin_size: Sequence[int],
):
    fragile_penalty, floor_flag, floor_area_bonus, floor_height_bonus = _transport_priority_terms(item, candidate)
    base_area = max(1, _item_area(item))
    support_ratio = candidate.get("support_area", 0.0) / float(base_area)
    return (
        fragile_penalty,
        floor_flag,
        floor_area_bonus,
        floor_height_bonus,
        candidate["z"],
        -support_ratio,
        -candidate.get("support_area", 0.0),
        candidate["used_height_after"],
        candidate["bbox_volume_after"],
        candidate["y"] + candidate["x"],
        candidate["y"],
        candidate["x"],
        item_id,
    )


def footprint_first_score_candidate(
    item_id: int,
    item: Dict[str, Any],
    candidate: Dict[str, Any],
    state: Dict[str, Any],
    bin_size: Sequence[int],
):
    fragile_penalty, floor_flag, floor_area_bonus, floor_height_bonus = _transport_priority_terms(item, candidate)
    return (
        fragile_penalty,
        floor_flag,
        floor_area_bonus,
        floor_height_bonus,
        candidate["bbox_volume_after"],
        candidate["used_height_after"],
        -candidate.get("support_area", 0.0),
        candidate["z"],
        candidate["x"] + candidate["y"],
        candidate["y"],
        candidate["x"],
        item_id,
    )


def height_compact_score_candidate(
    item_id: int,
    item: Dict[str, Any],
    candidate: Dict[str, Any],
    state: Dict[str, Any],
    bin_size: Sequence[int],
):
    fragile_penalty, floor_flag, floor_area_bonus, floor_height_bonus = _transport_priority_terms(item, candidate)
    return (
        fragile_penalty,
        floor_flag,
        floor_area_bonus,
        floor_height_bonus,
        candidate["used_height_after"],
        candidate["bbox_volume_after"],
        candidate["z"],
        -candidate.get("support_area", 0.0),
        candidate["y"],
        candidate["x"],
        item_id,
    )


def left_back_bottom_score_candidate(
    item_id: int,
    item: Dict[str, Any],
    candidate: Dict[str, Any],
    state: Dict[str, Any],
    bin_size: Sequence[int],
):
    fragile_penalty, floor_flag, floor_area_bonus, floor_height_bonus = _transport_priority_terms(item, candidate)
    return (
        fragile_penalty,
        floor_flag,
        floor_area_bonus,
        floor_height_bonus,
        candidate["x"] + candidate["y"] + candidate["z"],
        candidate["z"],
        -candidate.get("support_area", 0.0),
        candidate["bbox_volume_after"],
        item_id,
    )


def _validated_item_order(items: Dict[int, Dict[str, Any]], order_fn: Optional[ItemOrderFn]) -> List[int]:
    items_n = normalize_items(items)
    if order_fn is None:
        return default_order_items(items_n)
    ordered_ids = list(order_fn(items_n))
    expected = sorted(items_n.keys())
    if sorted(ordered_ids) != expected:
        raise ValueError(f"order_items() must return every item id exactly once. Expected {expected}, got {ordered_ids}")
    return ordered_ids


def _state_from_placement(items: Dict[int, Dict[str, Any]], placement: List[Dict[str, int]]) -> Dict[str, Any]:
    return {
        "placement": canonicalize_placement(placement),
        "used_height": placement_used_height(items, placement),
        "bbox_volume": placement_bbox_volume(items, placement),
        "placed_count": len(placement),
    }


def _safe_score(
    scorer: CandidateScoreFn,
    item_id: int,
    item: Dict[str, Any],
    candidate: Dict[str, Any],
    state: Dict[str, Any],
    bin_size: Sequence[int],
):
    try:
        score = scorer(item_id, item, candidate, state, bin_size)
    except Exception:
        score = default_score_candidate(item_id, item, candidate, state, bin_size)

    if isinstance(score, tuple):
        return score
    if isinstance(score, list):
        return tuple(score)
    return (score,)


def _base_build_count(item_count: int) -> int:
    if item_count <= 0:
        return 0
    return min(item_count, max(MIN_BASE_BUILD_ITEMS, int(math.ceil(item_count * BASE_BUILD_FRACTION))))


def _filter_candidates_for_transport_stability(
    candidates: List[Dict[str, Any]],
    item: Dict[str, Any],
    state: Dict[str, Any],
    total_items: int,
) -> List[Dict[str, Any]]:
    if not candidates:
        return candidates

    floor_candidates = [cand for cand in candidates if cand["z"] == 0]
    if not floor_candidates:
        return candidates

    if item.get("fragile", False):
        return floor_candidates

    if state.get("placed_count", 0) < _base_build_count(total_items):
        return floor_candidates

    return candidates


def _ranked_candidates(
    items: Dict[int, Dict[str, Any]],
    placement: List[Dict[str, int]],
    item_id: int,
    bin_size: Sequence[int],
    scorer: CandidateScoreFn,
) -> List[Dict[str, Any]]:
    item = items[item_id]
    raw = enumerate_candidate_positions(items, placement, item_id, bin_size)
    if not raw:
        return []

    state = _state_from_placement(items, placement)
    candidate_dicts = [cand.as_dict() for cand in raw]
    candidate_dicts = _filter_candidates_for_transport_stability(candidate_dicts, item, state, len(items))

    scored = []
    for d in candidate_dicts:
        score = _safe_score(scorer, item_id, item, d, state, bin_size)
        tie_break = (
            d["z"],
            -d.get("support_area", 0.0),
            d["used_height_after"],
            d["bbox_volume_after"],
            d["y"],
            d["x"],
        )
        scored.append((score + tie_break, d))
    scored.sort(key=lambda x: x[0])
    return [d for _, d in scored]


def _placement_quality(items: Dict[int, Dict[str, Any]], placement: List[Dict[str, int]]) -> Tuple[Any, ...]:
    used_h = placement_used_height(items, placement)
    bbox = placement_bbox_volume(items, placement)
    max_x = max((p["x"] + items[p["id"]]["l"] for p in placement), default=0)
    max_y = max((p["y"] + items[p["id"]]["w"] for p in placement), default=0)
    placed_volume = sum(_item_volume(items[p["id"]]) for p in placement)

    floor_area = 0
    non_fragile_floor_area = 0
    floor_count = 0
    stacked_fragile_count = 0
    for p in placement:
        item = items[p["id"]]
        if p["z"] == 0:
            area = _item_area(item)
            floor_area += area
            floor_count += 1
            if not item.get("fragile", False):
                non_fragile_floor_area += area
        elif item.get("fragile", False):
            stacked_fragile_count += 1

    return (
        -len(placement),
        stacked_fragile_count,
        -non_fragile_floor_area,
        -floor_area,
        -floor_count,
        used_h,
        bbox,
        max_y,
        max_x,
        -placed_volume,
    )


def _search_params_for_index(index: int, heavy: bool) -> Tuple[int, int, int]:
    if heavy:
        if index < HEAVY_EARLY_LEVELS:
            return HEAVY_TOP_K_CANDIDATES_EARLY, HEAVY_EARLY_BEAM_WIDTH, HEAVY_MAX_TOTAL_STATES
        if index < HEAVY_MID_LEVELS:
            return HEAVY_TOP_K_CANDIDATES_LATE, HEAVY_MID_BEAM_WIDTH, HEAVY_MAX_TOTAL_STATES
        return HEAVY_TOP_K_CANDIDATES_LATE, HEAVY_LATE_BEAM_WIDTH, HEAVY_MAX_TOTAL_STATES

    if index < EARLY_LEVELS:
        return TOP_K_CANDIDATES_EARLY, EARLY_BEAM_WIDTH, MAX_TOTAL_STATES
    if index < MID_LEVELS:
        return TOP_K_CANDIDATES_LATE, MID_BEAM_WIDTH, MAX_TOTAL_STATES
    return TOP_K_CANDIDATES_LATE, LATE_BEAM_WIDTH, MAX_TOTAL_STATES


def _greedy_with_policy(
    items_n: Dict[int, Dict[str, Any]],
    bin_size_n: Sequence[int],
    ordered_ids: List[int],
    scorer: CandidateScoreFn,
) -> List[Dict[str, int]]:
    placement: List[Dict[str, int]] = []
    for item_id in ordered_ids:
        ranked = _ranked_candidates(items_n, placement, item_id, bin_size_n, scorer)
        if not ranked:
            return []
        best = ranked[0]
        placement.append({"id": item_id, "x": best["x"], "y": best["y"], "z": best["z"]})
    return canonicalize_placement(placement)


def solve_with_policy(
    items: Dict[int, Dict[str, Any]],
    bin_size: Sequence[int],
    order_items_fn: Optional[ItemOrderFn] = None,
    score_candidate_fn: Optional[CandidateScoreFn] = None,
    heavy: bool = False,
) -> List[Dict[str, int]]:
    items_n = normalize_items(items)
    bin_size_n = normalize_bin_size(bin_size)
    ordered_ids = _validated_item_order(items_n, order_items_fn)
    scorer = score_candidate_fn or default_score_candidate

    greedy = _greedy_with_policy(items_n, bin_size_n, ordered_ids, scorer)
    if len(greedy) == len(ordered_ids):
        return greedy

    states: List[List[Dict[str, int]]] = [[]]
    total_generated = 0

    for index, item_id in enumerate(ordered_ids):
        candidate_width, beam_width, max_total_states = _search_params_for_index(index, heavy)
        next_states: List[Tuple[Tuple[Any, ...], List[Dict[str, int]]]] = []
        seen_signatures = set()

        for placement in states:
            ranked = _ranked_candidates(items_n, placement, item_id, bin_size_n, scorer)
            if not ranked:
                continue

            for cand in ranked[:candidate_width]:
                new_p = placement + [{"id": item_id, "x": cand["x"], "y": cand["y"], "z": cand["z"]}]
                sig = tuple(sorted((p["id"], p["x"], p["y"], p["z"]) for p in new_p))
                if sig in seen_signatures:
                    continue
                seen_signatures.add(sig)
                quality = _placement_quality(items_n, new_p)
                next_states.append((quality, new_p))
                total_generated += 1
                if total_generated >= max_total_states:
                    break
            if total_generated >= max_total_states:
                break

        if not next_states:
            return []

        next_states.sort(key=lambda x: x[0])
        states = [placement for _, placement in next_states[:beam_width]]

        if total_generated >= max_total_states:
            break

    if not states:
        return []

    best = min(states, key=lambda p: _placement_quality(items_n, p))
    if len(best) != len(ordered_ids):
        return []
    return canonicalize_placement(best)


def _policy_try_order() -> List[Tuple[Optional[ItemOrderFn], Optional[CandidateScoreFn], bool]]:
    return [
        (default_order_items, default_score_candidate, False),
        (default_order_items, support_greedy_score_candidate, False),
        (default_order_items, height_compact_score_candidate, False),
        (footprint_first_order_items, default_score_candidate, False),
        (footprint_first_order_items, footprint_first_score_candidate, False),
        (low_height_footprint_order_items, default_score_candidate, False),
        (non_fragile_base_order_items, support_greedy_score_candidate, False),
        (volume_first_order_items, support_greedy_score_candidate, False),
        (default_order_items, left_back_bottom_score_candidate, False),
    ]


def _heavy_policy_try_order() -> List[Tuple[Optional[ItemOrderFn], Optional[CandidateScoreFn], bool]]:
    return [
        (default_order_items, support_greedy_score_candidate, True),
        (footprint_first_order_items, support_greedy_score_candidate, True),
        (low_height_footprint_order_items, height_compact_score_candidate, True),
        (non_fragile_base_order_items, support_greedy_score_candidate, True),
        (volume_first_order_items, support_greedy_score_candidate, True),
        (fragile_last_order_items, support_greedy_score_candidate, True),
        (default_order_items, left_back_bottom_score_candidate, True),
    ]


def solve_packing(items: Dict[int, Dict[str, Any]], bin_size: Sequence[int]) -> List[Dict[str, int]]:
    items_n = normalize_items(items)
    bin_size_n = normalize_bin_size(bin_size)
    item_count = len(items_n)

    policies = _policy_try_order()
    if item_count >= HEAVY_ITEM_THRESHOLD:
        policies += _heavy_policy_try_order()

    for order_fn, score_fn, heavy in policies:
        solution = solve_with_policy(items_n, bin_size_n, order_fn, score_fn, heavy=heavy)
        if len(solution) == item_count:
            return solution

    return []


__all__ = [
    "default_order_items",
    "non_fragile_base_order_items",
    "low_height_footprint_order_items",
    "tall_first_order_items",
    "volume_first_order_items",
    "fragile_last_order_items",
    "footprint_first_order_items",
    "default_score_candidate",
    "support_greedy_score_candidate",
    "footprint_first_score_candidate",
    "height_compact_score_candidate",
    "left_back_bottom_score_candidate",
    "solve_with_policy",
    "solve_packing",
]
