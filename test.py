from __future__ import annotations

import json

from generate_instances import generate_feasible_instance
from trusted_solver import solve_packing
from verifier import BoxVerifier


if __name__ == "__main__":
    instance = generate_feasible_instance(num_items=12, fragile_ratio=0.35, seed=7)
    placement = solve_packing(instance["items"], instance["bin_size"])
    verifier = BoxVerifier(instance["bin_size"])
    ok, msg, report = verifier.verify(instance["items"], placement, return_report=True)
    print(json.dumps(instance["meta"], indent=2))
    print(msg)
    print(f"Placed {len(placement)} items")
