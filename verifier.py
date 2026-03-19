from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from packing_kernel import (
    canonicalize_placement,
    normalize_bin_size,
    normalize_items,
    normalize_placement,
    overlap_3d,
    support_metrics,
)


class BoxVerifier:
    """Strict verifier for physically valid 3D packings.

    The verifier is intentionally conservative:
    - full support is required for every stacked box,
    - fragile boxes may not directly support another box,
    - any invalid solution is rejected before visualization.
    """

    def __init__(self, bin_size: Sequence[int] = (1000, 1000, 1000)):
        self.bin_size = normalize_bin_size(bin_size)

    def verify(
        self,
        items: Dict[Any, Dict[str, Any]],
        placement: Sequence[Dict[str, Any]],
        return_report: bool = False,
    ):
        items_n = normalize_items(items)
        placement_n = normalize_placement(placement)
        L, W, H = self.bin_size

        report: Dict[str, Any] = {
            "bin_size": [L, W, H],
            "num_items": len(items_n),
            "num_placements": len(placement_n),
            "checks": {
                "structure": False,
                "completeness": False,
                "uniqueness": False,
                "bounds": False,
                "overlap": False,
                "support": False,
            },
            "support_details": [],
            "placement": canonicalize_placement(placement_n),
        }

        if len(placement_n) != len(items_n):
            msg = (
                f"Placement length {len(placement_n)} does not match item count {len(items_n)}."
            )
            return (False, msg, report) if return_report else (False, msg)
        report["checks"]["structure"] = True

        placed_ids = [p["id"] for p in placement_n]
        missing = sorted(set(items_n.keys()) - set(placed_ids))
        extras = sorted(set(placed_ids) - set(items_n.keys()))
        if extras:
            msg = f"Unknown item ids in placement: {extras}"
            return (False, msg, report) if return_report else (False, msg)
        if missing:
            msg = f"Missing placement for items: {missing}"
            return (False, msg, report) if return_report else (False, msg)
        report["checks"]["completeness"] = True

        if len(placed_ids) != len(set(placed_ids)):
            duplicates = []
            seen = set()
            for item_id in placed_ids:
                if item_id in seen and item_id not in duplicates:
                    duplicates.append(item_id)
                seen.add(item_id)
            msg = f"Duplicate placement for item ids: {duplicates}"
            return (False, msg, report) if return_report else (False, msg)
        report["checks"]["uniqueness"] = True

        for p in placement_n:
            item = items_n[p["id"]]
            if p["x"] < 0 or p["y"] < 0 or p["z"] < 0:
                msg = f"Item {p['id']} has negative coordinates"
                return (False, msg, report) if return_report else (False, msg)
            if p["x"] + item["l"] > L:
                msg = f"Item {p['id']} exceeds bin length boundary"
                return (False, msg, report) if return_report else (False, msg)
            if p["y"] + item["w"] > W:
                msg = f"Item {p['id']} exceeds bin width boundary"
                return (False, msg, report) if return_report else (False, msg)
            if p["z"] + item["h"] > H:
                msg = f"Item {p['id']} exceeds bin height boundary"
                return (False, msg, report) if return_report else (False, msg)
        report["checks"]["bounds"] = True

        for i, p1 in enumerate(placement_n):
            item1 = items_n[p1["id"]]
            for j in range(i + 1, len(placement_n)):
                p2 = placement_n[j]
                item2 = items_n[p2["id"]]
                if overlap_3d(item1, p1, item2, p2):
                    msg = f"Item {p1['id']} and {p2['id']} overlap in 3D space"
                    return (False, msg, report) if return_report else (False, msg)
        report["checks"]["overlap"] = True

        for p in canonicalize_placement(placement_n):
            item = items_n[p["id"]]
            metrics = support_metrics(items_n, placement_n, item, p)
            detail = {
                "id": p["id"],
                "x": p["x"],
                "y": p["y"],
                "z": p["z"],
                "support_ratio": metrics["support_ratio"],
                "support_area": metrics["support_area"],
                "support_ids": metrics["support_ids"],
                "center_supported": metrics["center_supported"],
                "reason": metrics["reason"],
            }
            report["support_details"].append(detail)
            if not metrics["supported"]:
                if metrics["reason"].startswith("partial_support"):
                    msg = (
                        f"Item {p['id']} is not fully supported. "
                        f"support_ratio={metrics['support_ratio']:.6f}"
                    )
                elif metrics["reason"].startswith("supported_by_fragile_items"):
                    msg = f"Item {p['id']} is supported by fragile item(s) {metrics['support_ids']}"
                elif metrics["reason"] == "no_direct_support":
                    msg = f"Item {p['id']} is floating without direct support"
                else:
                    msg = f"Item {p['id']} failed support check: {metrics['reason']}"
                return (False, msg, report) if return_report else (False, msg)
        report["checks"]["support"] = True

        msg = "Solution is valid under strict physical rules"
        return (True, msg, report) if return_report else (True, msg)
