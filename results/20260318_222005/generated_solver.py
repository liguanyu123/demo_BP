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
    fragile = item.get("fragile", False)
    z = candidate["z"]
    support_ratio = candidate["support_ratio"]
    used_height = candidate["used_height_after"]
    bbox_volume = candidate["bbox_volume_after"]
    support_area = candidate["support_area"]
    
    if fragile:
        if z == 0:
            return (0, -support_area, used_height, -bbox_volume)
        else:
            return (3, -support_ratio, used_height, -bbox_volume)
    else:
        if z == 0:
            return (1, -support_area, used_height, -bbox_volume)
        else:
            return (2, -support_ratio, used_height, -bbox_volume)

from trusted_solver import solve_with_policy

def solve_packing(items, bin_size):
    return solve_with_policy(items, bin_size, order_items, score_candidate, heavy=(len(items) >= 20))
