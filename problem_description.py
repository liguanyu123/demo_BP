from __future__ import annotations

from typing import Any, Dict, List, Sequence

from packing_kernel import normalize_bin_size, normalize_items, packing_signature


PROBLEM_TYPE = "3D-BBP-STRICT"



def build_problem_description(items: Dict[Any, Dict[str, Any]], bin_size: Sequence[int]) -> Dict[str, Any]:
    items_n = normalize_items(items)
    bin_size_n = normalize_bin_size(bin_size)
    fragile_count = sum(1 for item in items_n.values() if item["fragile"])

    constraints: List[Dict[str, str]] = [
        {
            "name": "Boundary",
            "description": "Every box must lie fully inside the container bounds.",
        },
        {
            "name": "Non-overlap",
            "description": "No two axis-aligned boxes may overlap in 3D space.",
        },
        {
            "name": "Completeness",
            "description": "Every input item must be placed exactly once.",
        },
        {
            "name": "AxisAligned",
            "description": "Boxes are placed without rotation; use the given l, w, h as-is.",
        },
        {
            "name": "StrictSupport",
            "description": "Any box with z > 0 must have its entire bottom footprint supported by coplanar top faces directly beneath it.",
        },
        {
            "name": "NoFloating",
            "description": "Floating, hanging, bridging, or partially supported placements are invalid.",
        },
        {
            "name": "FragileNotLoadBearing",
            "description": "Fragile boxes may not directly support another box.",
        },
    ]

    return {
        "P": PROBLEM_TYPE,
        "S": (
            "Single-container 3D bin packing with strict physical feasibility. "
            "The solver must place all rectangular boxes into one fixed-size container "
            "while obeying conservative real-world static stability rules."
        ),
        "K": constraints,
        "X": [
            "bin_size",
            "items[item_id].l",
            "items[item_id].w",
            "items[item_id].h",
            "items[item_id].fragile",
        ],
        "Y": (
            "A complete placement list of dicts with keys id, x, y, z; each entry gives the "
            "lower-left-bottom coordinate of one placed box."
        ),
        "Z": (
            "Primary objective: find a complete physically feasible packing. Secondary objective: "
            "prefer transport-stable layouts with a denser ground layer, grounded fragile items when possible, "
            "lower used height, and smaller occupied bounding volume."
        ),
        "instance_summary": {
            "bin_size": list(bin_size_n),
            "num_items": len(items_n),
            "fragile_items": fragile_count,
            "non_fragile_items": len(items_n) - fragile_count,
        },
        "signature": packing_signature(items_n, bin_size_n),
    }



def description_as_text(description: Dict[str, Any]) -> str:
    problem_type = description.get("P", "3D-BBP-STRICT")
    summary = description.get("S", "")
    constraints = description.get("K", [])
    input_def = description.get("X", [])
    output_def = description.get("Y", "")
    objective = description.get("Z", "")
    instance_summary = description.get("instance_summary", {})
    signature = description.get("signature", "")

    lines = [
        f"Problem Type: {problem_type}",
        f"Summary: {summary}",
        f"Instance Summary: {instance_summary}",
        f"Signature: {signature}",
        "Constraints:",
    ]

    if constraints:
        for c in constraints:
            if isinstance(c, dict):
                name = c.get("name", "")
                desc = c.get("description", "")
                lines.append(f"- {name}: {desc}")
            else:
                lines.append(f"- {c}")
    else:
        lines.append("- No constraints provided.")

    lines.extend([
        f"Input: {input_def}",
        f"Output: {output_def}",
        f"Objective: {objective}",
    ])

    return "\n".join(lines)