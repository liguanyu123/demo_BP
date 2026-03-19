"""Deterministic geometry and physics kernel for strict 3D bin packing.

This module is intentionally hand-written and trusted. The LLM-generated code is not
allowed to reimplement these low-level legality checks. Instead, the model generates
higher-level item ordering and candidate scoring logic on top of this kernel.

Physical policy implemented here:
- axis-aligned boxes only
- boxes must stay inside the container
- no 3D overlap
- if z > 0, the full bottom footprint of a box must be supported by the union of
  coplanar top faces directly below it
- fragile boxes cannot be used as direct supports
- center-of-mass projection must lie inside the support union

The rules are conservative by design. This intentionally forbids overhangs and bridge-like
placements that are visually confusing and physically questionable in a static packing setting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

EPS = 1e-9


@dataclass(frozen=True)
class SupportInfo:
    item_id: int
    rect: Tuple[float, float, float, float]
    fragile: bool


@dataclass(frozen=True)
class Candidate:
    id: int
    x: int
    y: int
    z: int
    support_ratio: float
    support_area: float
    support_ids: Tuple[int, ...]
    used_height_after: int
    bbox_volume_after: int
    footprint_area: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "support_ratio": self.support_ratio,
            "support_area": self.support_area,
            "support_ids": list(self.support_ids),
            "used_height_after": self.used_height_after,
            "bbox_volume_after": self.bbox_volume_after,
            "footprint_area": self.footprint_area,
        }


def _to_int(value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError("Boolean value cannot be converted to int here")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if abs(value - round(value)) > EPS:
            raise ValueError(f"Expected an integer-like value, got {value!r}")
        return int(round(value))
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"Unsupported numeric value: {value!r}")



def normalize_items(items: Dict[Any, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    normalized: Dict[int, Dict[str, Any]] = {}
    for raw_id, raw_item in items.items():
        item_id = _to_int(raw_id)
        if not isinstance(raw_item, dict):
            raise TypeError(f"Item {item_id} must be a dict")
        required = {"l", "w", "h", "fragile"}
        missing = required - set(raw_item.keys())
        if missing:
            raise KeyError(f"Item {item_id} missing keys: {sorted(missing)}")
        item = {
            "l": _to_int(raw_item["l"]),
            "w": _to_int(raw_item["w"]),
            "h": _to_int(raw_item["h"]),
            "fragile": bool(raw_item["fragile"]),
        }
        if item["l"] <= 0 or item["w"] <= 0 or item["h"] <= 0:
            raise ValueError(f"Item {item_id} must have positive dimensions")
        normalized[item_id] = item
    return normalized



def normalize_bin_size(bin_size: Sequence[Any]) -> Tuple[int, int, int]:
    if len(bin_size) != 3:
        raise ValueError("bin_size must have length 3")
    L, W, H = (_to_int(v) for v in bin_size)
    if L <= 0 or W <= 0 or H <= 0:
        raise ValueError("bin_size entries must be positive")
    return (L, W, H)



def normalize_placement(placement: Sequence[Dict[str, Any]]) -> List[Dict[str, int]]:
    if not isinstance(placement, (list, tuple)):
        raise TypeError("placement must be a list")
    normalized: List[Dict[str, int]] = []
    for idx, p in enumerate(placement):
        if not isinstance(p, dict):
            raise TypeError(f"Placement entry #{idx} must be a dict")
        required = {"id", "x", "y", "z"}
        missing = required - set(p.keys())
        if missing:
            raise KeyError(f"Placement entry #{idx} missing keys: {sorted(missing)}")
        normalized.append(
            {
                "id": _to_int(p["id"]),
                "x": _to_int(p["x"]),
                "y": _to_int(p["y"]),
                "z": _to_int(p["z"]),
            }
        )
    return normalized



def canonicalize_placement(placement: Sequence[Dict[str, Any]]) -> List[Dict[str, int]]:
    normalized = normalize_placement(placement)
    return sorted(normalized, key=lambda p: (p["z"], p["y"], p["x"], p["id"]))



def item_bounds(item: Dict[str, int], pos: Dict[str, int]) -> Tuple[int, int, int, int, int, int]:
    return (
        pos["x"],
        pos["x"] + item["l"],
        pos["y"],
        pos["y"] + item["w"],
        pos["z"],
        pos["z"] + item["h"],
    )



def overlap_1d(a0: float, a1: float, b0: float, b1: float) -> bool:
    return not (a1 <= b0 + EPS or b1 <= a0 + EPS)



def overlap_3d(item_a: Dict[str, int], pos_a: Dict[str, int], item_b: Dict[str, int], pos_b: Dict[str, int]) -> bool:
    ax0, ax1, ay0, ay1, az0, az1 = item_bounds(item_a, pos_a)
    bx0, bx1, by0, by1, bz0, bz1 = item_bounds(item_b, pos_b)
    return (
        overlap_1d(ax0, ax1, bx0, bx1)
        and overlap_1d(ay0, ay1, by0, by1)
        and overlap_1d(az0, az1, bz0, bz1)
    )



def intersection_rect(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> Optional[Tuple[float, float, float, float]]:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 + EPS or y2 <= y1 + EPS:
        return None
    return (x1, y1, x2, y2)



def rect_area(rect: Tuple[float, float, float, float]) -> float:
    return max(0.0, rect[2] - rect[0]) * max(0.0, rect[3] - rect[1])



def union_area(
    target_rect: Tuple[float, float, float, float],
    rects: Sequence[Tuple[float, float, float, float]],
) -> float:
    if not rects:
        return 0.0
    tx1, ty1, tx2, ty2 = target_rect
    xs = {tx1, tx2}
    ys = {ty1, ty2}
    clipped: List[Tuple[float, float, float, float]] = []
    for rect in rects:
        inter = intersection_rect(target_rect, rect)
        if inter is None:
            continue
        clipped.append(inter)
        xs.add(inter[0])
        xs.add(inter[2])
        ys.add(inter[1])
        ys.add(inter[3])
    if not clipped:
        return 0.0
    xs_sorted = sorted(xs)
    ys_sorted = sorted(ys)
    area = 0.0
    for i in range(len(xs_sorted) - 1):
        for j in range(len(ys_sorted) - 1):
            cx1, cx2 = xs_sorted[i], xs_sorted[i + 1]
            cy1, cy2 = ys_sorted[j], ys_sorted[j + 1]
            if cx2 <= cx1 + EPS or cy2 <= cy1 + EPS:
                continue
            mx = (cx1 + cx2) / 2.0
            my = (cy1 + cy2) / 2.0
            covered = False
            for rx1, ry1, rx2, ry2 in clipped:
                if rx1 <= mx < rx2 and ry1 <= my < ry2:
                    covered = True
                    break
            if covered:
                area += (cx2 - cx1) * (cy2 - cy1)
    return area



def point_in_rect_union(point: Tuple[float, float], rects: Sequence[Tuple[float, float, float, float]]) -> bool:
    px, py = point
    for x1, y1, x2, y2 in rects:
        if x1 - EPS <= px <= x2 + EPS and y1 - EPS <= py <= y2 + EPS:
            return True
    return False



def footprint_rect(item: Dict[str, int], pos: Dict[str, int]) -> Tuple[float, float, float, float]:
    return (pos["x"], pos["y"], pos["x"] + item["l"], pos["y"] + item["w"])



def support_rectangles(
    items: Dict[int, Dict[str, int]],
    placement: Sequence[Dict[str, int]],
    item: Dict[str, int],
    pos: Dict[str, int],
) -> List[SupportInfo]:
    target = footprint_rect(item, pos)
    supports: List[SupportInfo] = []
    for base_pos in placement:
        base_item = items[base_pos["id"]]
        if base_pos["z"] + base_item["h"] != pos["z"]:
            continue
        base_rect = footprint_rect(base_item, base_pos)
        inter = intersection_rect(target, base_rect)
        if inter is None:
            continue
        supports.append(SupportInfo(item_id=base_pos["id"], rect=inter, fragile=base_item["fragile"]))
    return supports



def support_metrics(
    items: Dict[int, Dict[str, int]],
    placement: Sequence[Dict[str, int]],
    item: Dict[str, int],
    pos: Dict[str, int],
) -> Dict[str, Any]:
    target = footprint_rect(item, pos)
    total_area = rect_area(target)
    if pos["z"] == 0:
        return {
            "supported": True,
            "support_ratio": 1.0,
            "support_area": total_area,
            "support_ids": [],
            "reason": "supported_by_floor",
            "center_supported": True,
        }

    supports = support_rectangles(items, placement, item, pos)
    if not supports:
        return {
            "supported": False,
            "support_ratio": 0.0,
            "support_area": 0.0,
            "support_ids": [],
            "reason": "no_direct_support",
            "center_supported": False,
        }

    fragile_support_ids = [s.item_id for s in supports if s.fragile]
    if fragile_support_ids:
        return {
            "supported": False,
            "support_ratio": 0.0,
            "support_area": 0.0,
            "support_ids": sorted(set(s.item_id for s in supports)),
            "reason": f"supported_by_fragile_items:{sorted(set(fragile_support_ids))}",
            "center_supported": False,
        }

    rects = [s.rect for s in supports]
    area = union_area(target, rects)
    ratio = 0.0 if total_area <= EPS else area / total_area
    center = (pos["x"] + item["l"] / 2.0, pos["y"] + item["w"] / 2.0)
    center_supported = point_in_rect_union(center, rects)
    full_support = ratio >= 1.0 - EPS
    supported = full_support and center_supported
    if not supported:
        if not full_support:
            reason = f"partial_support:{ratio:.6f}"
        else:
            reason = "center_of_mass_outside_support"
    else:
        reason = "supported"
    return {
        "supported": supported,
        "support_ratio": ratio,
        "support_area": area,
        "support_ids": sorted(set(s.item_id for s in supports)),
        "reason": reason,
        "center_supported": center_supported,
    }



def can_place_strict(
    items: Dict[int, Dict[str, int]],
    placement: Sequence[Dict[str, int]],
    item_id: int,
    candidate_pos: Dict[str, int],
    bin_size: Sequence[int],
) -> Tuple[bool, str, Dict[str, Any]]:
    items_n = normalize_items(items)
    bin_size_n = normalize_bin_size(bin_size)
    placement_n = normalize_placement(placement)
    item = items_n[item_id]
    L, W, H = bin_size_n

    if candidate_pos["x"] < 0 or candidate_pos["y"] < 0 or candidate_pos["z"] < 0:
        return False, "negative_coordinate", {}
    if candidate_pos["x"] + item["l"] > L:
        return False, "exceeds_bin_length", {}
    if candidate_pos["y"] + item["w"] > W:
        return False, "exceeds_bin_width", {}
    if candidate_pos["z"] + item["h"] > H:
        return False, "exceeds_bin_height", {}

    for placed in placement_n:
        placed_item = items_n[placed["id"]]
        if overlap_3d(item, candidate_pos, placed_item, placed):
            return False, f"overlap_with_item_{placed['id']}", {}

    metrics = support_metrics(items_n, placement_n, item, candidate_pos)
    if not metrics["supported"]:
        return False, metrics["reason"], metrics

    return True, "ok", metrics



def placement_used_height(items: Dict[int, Dict[str, int]], placement: Sequence[Dict[str, int]]) -> int:
    if not placement:
        return 0
    items_n = normalize_items(items)
    placement_n = normalize_placement(placement)
    return max(p["z"] + items_n[p["id"]]["h"] for p in placement_n)



def placement_bbox_volume(items: Dict[int, Dict[str, int]], placement: Sequence[Dict[str, int]]) -> int:
    if not placement:
        return 0
    items_n = normalize_items(items)
    placement_n = normalize_placement(placement)
    max_x = max(p["x"] + items_n[p["id"]]["l"] for p in placement_n)
    max_y = max(p["y"] + items_n[p["id"]]["w"] for p in placement_n)
    max_z = max(p["z"] + items_n[p["id"]]["h"] for p in placement_n)
    return max_x * max_y * max_z



def enumerate_candidate_positions(
    items: Dict[int, Dict[str, int]],
    placement: Sequence[Dict[str, int]],
    item_id: int,
    bin_size: Sequence[int],
) -> List[Candidate]:
    items_n = normalize_items(items)
    placement_n = normalize_placement(placement)
    L, W, H = normalize_bin_size(bin_size)
    item = items_n[item_id]

    x_positions = {0}
    y_positions = {0}
    z_positions = {0}
    for placed in placement_n:
        placed_item = items_n[placed["id"]]
        x_positions.add(placed["x"])
        x_positions.add(placed["x"] + placed_item["l"])
        y_positions.add(placed["y"])
        y_positions.add(placed["y"] + placed_item["w"])
        z_positions.add(placed["z"] + placed_item["h"])

    candidates: List[Candidate] = []
    seen = set()
    for z in sorted(z_positions):
        if z + item["h"] > H:
            continue
        for x in sorted(x_positions):
            if x + item["l"] > L:
                continue
            for y in sorted(y_positions):
                if y + item["w"] > W:
                    continue
                key = (x, y, z)
                if key in seen:
                    continue
                seen.add(key)
                pos = {"id": item_id, "x": x, "y": y, "z": z}
                ok, _, metrics = can_place_strict(items_n, placement_n, item_id, pos, (L, W, H))
                if not ok:
                    continue
                hypothetical = list(placement_n) + [pos]
                candidates.append(
                    Candidate(
                        id=item_id,
                        x=x,
                        y=y,
                        z=z,
                        support_ratio=float(metrics["support_ratio"]),
                        support_area=float(metrics["support_area"]),
                        support_ids=tuple(metrics["support_ids"]),
                        used_height_after=placement_used_height(items_n, hypothetical),
                        bbox_volume_after=placement_bbox_volume(items_n, hypothetical),
                        footprint_area=item["l"] * item["w"],
                    )
                )
    return candidates



def packing_signature(items: Dict[int, Dict[str, Any]], bin_size: Sequence[int]) -> Dict[str, Any]:
    items_n = normalize_items(items)
    L, W, H = normalize_bin_size(bin_size)
    return {
        "problem_type": "3D-BBP-STRICT",
        "bin_size": [L, W, H],
        "num_items": len(items_n),
        "fragile_items": sum(1 for item in items_n.values() if item["fragile"]),
        "axis_aligned": True,
        "rotation_allowed": False,
        "strict_support": True,
        "fragile_not_load_bearing": True,
    }
