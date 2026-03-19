from trusted_solver import solve_with_policy


def order_items(items):
    def rank(entry):
        item_id, item = entry
        return (
            1 if item['fragile'] else 0,
            -(item['l'] * item['w']),
            -(item['l'] * item['w'] * item['h']),
            -item['h'],
            item_id,
        )
    return [item_id for item_id, _ in sorted(items.items(), key=rank)]


def score_candidate(item_id, item, candidate, state, bin_size):
    return (
        candidate['z'],
        candidate['used_height_after'],
        candidate['bbox_volume_after'],
        candidate['y'],
        candidate['x'],
        -candidate['support_area'],
        item_id,
    )


def solve_packing(items, bin_size):
    return solve_with_policy(items, bin_size, order_items, score_candidate)