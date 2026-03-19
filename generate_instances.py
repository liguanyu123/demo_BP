from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List, Sequence, Tuple

from packing_kernel import canonicalize_placement
from verifier import BoxVerifier


DEFAULT_BIN_SIZE = (1000, 1000, 1000)
MAX_STACK_SIZE = 3


def _choose_grid(num_items: int) -> Tuple[int, int, int]:
    floor_slots = max(1, (num_items + MAX_STACK_SIZE - 1) // MAX_STACK_SIZE)
    if floor_slots <= 4:
        cols = 2
    elif floor_slots <= 9:
        cols = 3
    else:
        cols = 4
    rows = (floor_slots + cols - 1) // cols
    return cols, rows, floor_slots


def _partition_axis(total: int, parts: int, min_size: int, rng: random.Random, fill_ratio: float) -> List[int]:
    fill_total = max(min_size * parts, min(total, int(round(total * fill_ratio))))
    base = fill_total // parts
    sizes = [base for _ in range(parts)]
    remainder = fill_total - base * parts
    for idx in range(remainder):
        sizes[idx % parts] += 1

    jitter_cap = max(6, min(28, total // max(1, parts * 5)))
    for idx in range(parts - 1):
        move = rng.randint(-jitter_cap, jitter_cap)
        new_left = sizes[idx] + move
        new_right = sizes[idx + 1] - move
        if new_left >= min_size and new_right >= min_size:
            sizes[idx] = new_left
            sizes[idx + 1] = new_right
    return sizes


def _stack_lengths(num_items: int, slots: int, rng: random.Random) -> List[int]:
    lengths = [1 for _ in range(slots)]
    extras = num_items - slots
    slot_order = list(range(slots))
    rng.shuffle(slot_order)
    while extras > 0:
        progressed = False
        for idx in slot_order:
            if extras <= 0:
                break
            if lengths[idx] < MAX_STACK_SIZE:
                lengths[idx] += 1
                extras -= 1
                progressed = True
        if not progressed:
            raise RuntimeError("Unable to distribute items into constructive stacks")
    rng.shuffle(lengths)
    return lengths


def _rand_int(rng: random.Random, low: int, high: int) -> int:
    low_i = int(round(low))
    high_i = int(round(high))
    if high_i < low_i:
        high_i = low_i
    return rng.randint(low_i, high_i)


def _sample_bottom_dims(cell_l: int, cell_w: int, rng: random.Random) -> Tuple[int, int]:
    l = _rand_int(rng, max(120, int(cell_l * 0.62)), max(150, int(cell_l * 0.90)))
    w = _rand_int(rng, max(120, int(cell_w * 0.62)), max(150, int(cell_w * 0.90)))
    return min(l, cell_l), min(w, cell_w)


def _sample_next_dims(prev_l: int, prev_w: int, rng: random.Random) -> Tuple[int, int]:
    l = _rand_int(rng, max(100, int(prev_l * 0.72)), max(100, int(prev_l * 0.98)))
    w = _rand_int(rng, max(100, int(prev_w * 0.72)), max(100, int(prev_w * 0.98)))
    return min(l, prev_l), min(w, prev_w)


def _sample_heights(stack_len: int, rng: random.Random, target_utilization: float) -> List[int]:
    profile = 0.78 + 0.45 * max(0.0, min(1.0, target_utilization))
    base_min = int(round(110 * profile))
    heights: List[int] = []
    remaining_cap = 1000
    for idx in range(stack_len):
        remaining_items = stack_len - idx - 1
        reserve = remaining_items * 120
        hi = min(260, remaining_cap - reserve)
        lo = min(max(110, base_min), hi)
        if remaining_items == 0:
            hi = min(300, remaining_cap)
            lo = min(lo, hi)
        h = _rand_int(rng, lo, max(lo, hi))
        heights.append(h)
        remaining_cap -= h
    return heights


def _stack_fragility(stack_len: int, fragile_ratio: float, rng: random.Random) -> List[bool]:
    if stack_len == 1:
        return [rng.random() < fragile_ratio]
    flags = [False for _ in range(stack_len)]
    top_fragile_prob = min(0.90, max(fragile_ratio, 0.40))
    flags[-1] = rng.random() < top_fragile_prob
    if stack_len == 3 and rng.random() < max(0.0, fragile_ratio - 0.45):
        flags[-2] = True
        flags[-1] = True
    return flags


def generate_constructive_instance(
    num_items: int,
    fragile_ratio: float = 0.35,
    seed: int | None = None,
    bin_size: Sequence[int] = DEFAULT_BIN_SIZE,
    target_utilization: float = 0.35,
    include_hidden_solution: bool = False,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    L, W, H = (int(v) for v in bin_size)
    cols, rows, slots = _choose_grid(num_items)

    col_sizes = _partition_axis(L, cols, min_size=180, rng=rng, fill_ratio=0.96)
    row_sizes = _partition_axis(W, rows, min_size=180, rng=rng, fill_ratio=0.96)
    x_origins = [0]
    for size in col_sizes[:-1]:
        x_origins.append(x_origins[-1] + size)
    y_origins = [0]
    for size in row_sizes[:-1]:
        y_origins.append(y_origins[-1] + size)

    cell_specs: List[Tuple[int, int, int, int]] = []
    for row_idx in range(rows):
        for col_idx in range(cols):
            if len(cell_specs) >= slots:
                break
            cell_specs.append((x_origins[col_idx], y_origins[row_idx], col_sizes[col_idx], row_sizes[row_idx]))
        if len(cell_specs) >= slots:
            break

    lengths = _stack_lengths(num_items, slots, rng)
    cell_order = list(range(slots))
    rng.shuffle(cell_order)

    items: Dict[int, Dict[str, Any]] = {}
    placement: List[Dict[str, int]] = []
    item_id = 1

    for cell_rank, cell_idx in enumerate(cell_order):
        x0, y0, cell_l, cell_w = cell_specs[cell_idx]
        stack_len = lengths[cell_rank]
        bottom_l, bottom_w = _sample_bottom_dims(cell_l, cell_w, rng)
        heights = _sample_heights(stack_len, rng, target_utilization)
        fragile_flags = _stack_fragility(stack_len, fragile_ratio, rng)

        dims: List[Tuple[int, int]] = [(bottom_l, bottom_w)]
        for _ in range(1, stack_len):
            dims.append(_sample_next_dims(dims[-1][0], dims[-1][1], rng))

        z = 0
        for level in range(stack_len):
            l_dim, w_dim = dims[level]
            item = {
                "l": int(l_dim),
                "w": int(w_dim),
                "h": int(heights[level]),
                "fragile": bool(fragile_flags[level]),
            }
            items[item_id] = item
            placement.append({"id": item_id, "x": int(x0), "y": int(y0), "z": int(z)})
            z += item["h"]
            item_id += 1

    placement = canonicalize_placement(placement)
    verifier = BoxVerifier(bin_size)
    ok, msg, _report = verifier.verify(items, placement, return_report=True)
    if not ok:
        raise RuntimeError(f"Constructive instance generation produced an invalid placement: {msg}")

    total_volume = sum(item["l"] * item["w"] * item["h"] for item in items.values())
    bin_volume = L * W * H
    result: Dict[str, Any] = {
        "bin_size": [L, W, H],
        "items": items,
        "meta": {
            "num_items": num_items,
            "fragile_ratio": fragile_ratio,
            "seed": seed,
            "target_utilization": target_utilization,
            "actual_utilization": total_volume / float(bin_volume),
            "generator": "constructive_grid_stacks",
            "guaranteed_feasible_layout_exists": True,
            "grid": {"cols": cols, "rows": rows, "slots": slots},
        },
    }
    if include_hidden_solution:
        result["hidden_reference_solution"] = placement
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate constructive 3D-BBP instances without relying on local solver search")
    parser.add_argument("--out", type=str, default="instances", help="Output directory")
    parser.add_argument("--fragile-ratio", type=float, default=0.35)
    parser.add_argument("--target-utilization", type=float, default=0.35)
    parser.add_argument("--bin-size", type=int, nargs=3, default=list(DEFAULT_BIN_SIZE))
    parser.add_argument("--per-size", type=int, default=3, help="Instances per item-count bucket")
    parser.add_argument("--sizes", type=int, nargs="+", default=[5, 10, 15, 20, 25, 30])
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--include-hidden-solution", action="store_true", help="Store the constructive witness placement inside the JSON (off by default)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    counter = int(args.seed_offset)
    for num_items in args.sizes:
        print(f"Generating {args.per_size} constructive instances with {num_items} items ...", flush=True)
        for bucket_idx in range(args.per_size):
            seed = counter
            inst = generate_constructive_instance(
                num_items=num_items,
                fragile_ratio=args.fragile_ratio,
                seed=seed,
                bin_size=tuple(args.bin_size),
                target_utilization=args.target_utilization,
                include_hidden_solution=args.include_hidden_solution,
            )
            filename = os.path.join(args.out, f"inst_{num_items}items_{bucket_idx}.json")
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(inst, f, indent=2)
            print(f"  wrote {filename} | util={inst['meta']['actual_utilization']:.3f} | seed={seed}", flush=True)
            counter += 1


if __name__ == "__main__":
    main()
