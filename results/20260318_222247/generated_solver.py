def order_items(items):
    def rank(item_id):
        item = items[item_id]
        return (
            -(item["l"] * item["w"]),
            item["h"],
            1 if item.get("fragile", False) else 0,
            -(item["l"] * item["w"] * item["h"]),
            item_id,
        )
    return sorted(items.keys(), key=rank)

def score_candidate(item_id, item, candidate, state, bin_size):
    footprint = item["l"] * item["w"]
    on_floor = candidate["z"] == 0
    fragile = item.get("fragile", False)
    return (
        1 if fragile and not on_floor else 0,
        0 if on_floor else 1,
        -footprint if on_floor else 0,
        item["h"] if on_floor else 0,
        candidate["z"],
        -candidate["support_area"],
        candidate["used_height_after"],
        candidate["bbox_volume_after"],
        candidate["y"] + candidate["x"],
        candidate["y"],
        candidate["x"],
        item_id,
    )

from trusted_solver import solve_with_policy

def solve_packing(items, bin_size):
    return solve_with_policy(items, bin_size, order_items, score_candidate, heavy=(len(items) >= 20))
