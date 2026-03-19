from __future__ import annotations

from typing import Any, Dict


PROBLEM_DESCRIPTION_SYSTEM = (
    "You are a 3D bin packing expert. Be rigorous, conservative, and physically realistic. "
    "Prefer exact JSON outputs over prose. Never invent fields that are not present in the instance."
)


PROBLEM_DESCRIPTION_GENERATE = """
You are the Generation Agent (GA) for a 3D bin packing AFL workflow.

Given the instance summary below, write a JSON object with exactly these top-level keys:
P, S, K, X, Y, Z.

Requirements:
- P: a short problem type name.
- S: one concise description.
- K: an array of constraint objects, each with keys name and description.
- X: an array of required input elements.
- Y: a precise output description.
- Z: a precise objective description.
- Use conservative physics: no overlap, no boundary violation, no floating, no partial support,
  no direct support on fragile boxes.
- Do not introduce weights, friction coefficients, or rotations.
- The instance only contains bin_size and items with l, w, h, fragile.

Instance summary:
{instance_summary}
""".strip()


PROBLEM_DESCRIPTION_JUDGE = """
You are the Judgment Agent (JA) for a 3D bin packing AFL workflow.

You will receive:
1. The instance summary.
2. A candidate problem description JSON.

Return JSON with keys:
- ok: boolean
- issues: array of strings
- suggestions: array of strings

Judge whether the candidate description:
- matches the instance schema,
- does not invent missing fields,
- includes strict physical constraints,
- keeps outputs and objectives consistent with single-container 3D bin packing.
""".strip()


PROBLEM_DESCRIPTION_REVISE = """
You are the Revision Agent (RA) for a 3D bin packing AFL workflow.

Revise the candidate problem description JSON using the instance summary and judge feedback.
Return only the corrected JSON with keys P, S, K, X, Y, Z.
""".strip()


SOLVER_GENERATE_SYSTEM = (
    "You are a senior Python optimization engineer. Generate compact, executable Python code only. "
    "Do not include markdown fences. Obey the scaffold exactly."
)


SOLVER_GENERATE = """
You must generate Python code for the candidate-scoring policy of a strict 3D bin packing solver.

Return Python code only. Do not include markdown fences.

The code MUST define exactly this function:
- score_candidate(item_id, item, candidate, state, bin_size)

Hard requirements:
1. Do NOT define order_items.
2. Do NOT define solve_packing.
3. Do NOT import anything.
4. score_candidate(...) must return a tuple of sortable values.
5. The scaffold provides a fixed order_items(items) and solve_packing(items, bin_size).
6. Optimize for transport stability while preserving strict feasibility:
   - prefer grounded placements on z == 0,
   - if a fragile item can be grounded, strongly prefer the grounded candidate,
   - among grounded candidates, prefer larger footprint and lower height,
   - otherwise prefer stronger support, lower used height, and compact placement.
7. Never implement geometry checks yourself.
8. The final code must import nothing and contain only score_candidate.

Use the following item fields only:
- item["l"], item["w"], item["h"], item.get("fragile", False)

Use the following candidate fields only:
- candidate["x"], candidate["y"], candidate["z"]
- candidate["used_height_after"], candidate["bbox_volume_after"]
- candidate["support_area"], candidate["support_ratio"]

You may read only state["placed_count"] from state.

Problem description:
{problem_description}
""".strip()


SOLVER_JUDGE = """
You are the Judgment Agent (JA) for generated Python code.

Return JSON with keys:
- ok: boolean
- issues: array of strings
- suggestions: array of strings

Check whether the code:
- defines order_items, score_candidate, and solve_packing,
- imports only trusted_solver,
- avoids markdown fences,
- respects the required scaffold,
- does not invent unsupported item fields,
- relies on the trusted engine for legality.
""".strip()


SOLVER_REVISE = """
You are the Revision Agent (RA) for generated Python code.

Revise only the score_candidate(...) function according to:
- the problem description,
- the judge feedback,
- and, if present, error-analysis feedback.

Keep the fixed scaffold assumptions:
- order_items(items) is fixed by the trusted wrapper,
- solve_packing(items, bin_size) is fixed by the trusted wrapper,
- geometry legality stays inside trusted_solver / packing_kernel / verifier.

Return only corrected Python code for score_candidate(...). No markdown fences.
""".strip()


ERROR_ANALYSIS = """
You are the Error Analysis Agent (EAA) for an AFL workflow.

You will receive:
- the problem description,
- the generated Python code,
- the runtime error or verification error.

Return JSON with keys:
- diagnosis: short explanation of the root cause
- actionable_fixes: array of concrete code changes to make
- likely_stage: one of [order_items, score_candidate, solve_packing, import_contract, runtime, verification]
""".strip()


MOCK_PROBLEM_DESCRIPTION_JUDGE_OK = {
    "ok": True,
    "issues": [],
    "suggestions": [],
}

MOCK_SOLVER_JUDGE_OK = {
    "ok": True,
    "issues": [],
    "suggestions": [],
}

MOCK_ERROR_ANALYSIS = {
    "diagnosis": "Mock mode does not execute a real LLM. Use API mode for genuine error analysis.",
    "actionable_fixes": [],
    "likely_stage": "runtime",
}
