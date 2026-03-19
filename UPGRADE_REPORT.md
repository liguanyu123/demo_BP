# 3D-BBP AFL Transport-Stability Upgrade Report

## What was changed in this version

### 1. Verifier strength was preserved
`verifier.py` was intentionally left strict. No legality rule was relaxed.
The system still rejects:
- overlap,
- boundary violations,
- floating,
- partial support,
- fragile boxes directly supporting another box.

### 2. The optimization target moved from “legal only” to “legal and transport-stable”
The trusted solver was updated so that legal packings are now biased toward more stable transport layouts:
- build a denser ground layer first,
- reduce stacked fragile placements when floor placement is possible,
- favor larger-footprint, lower-height base boxes.

### 3. Floor-first candidate filtering was added to the trusted solver
`trusted_solver.py` now applies two hard solver-side heuristics before ranking candidates:
- **fragile item rule**: if a fragile item has any `z == 0` candidate, only floor candidates are considered;
- **base-build rule**: during the early 40% of the placement process, if the current item has any `z == 0` candidate, only floor candidates are considered.

These are solver heuristics, not verifier relaxations.

### 4. Candidate scoring now optimizes transport stability
The default scoring policy now heavily prefers:
- grounded placements (`z == 0`),
- grounded fragile placements,
- larger-footprint floor placements,
- lower-height floor placements,
- stronger support and compact layouts after floor preference is satisfied.

### 5. The fixed solver scaffold became stricter
To reduce brittle code-generation errors and keep the architecture stable:
- `order_items(items)` is now fixed by the trusted wrapper,
- `solve_packing(items, bin_size)` is now fixed by the trusted wrapper,
- the LLM only generates `score_candidate(...)`.

This keeps the AFL loop agentic while narrowing the mutation surface to the intended policy lever.

### 6. Problem-description objective was aligned with the new target
`problem_description.py` now states that the secondary objective is transport stability:
- denser ground layer,
- grounded fragile items when possible,
- lower used height,
- smaller occupied bounding volume.

### 7. New regression tests were added
A new test file, `tests_transport_stability.py`, checks that:
- fragile items are restricted to floor candidates when floor candidates exist,
- early base-building keeps non-fragile items on the floor when possible,
- the fixed order no longer permanently pushes every fragile item to the end,
- a simple end-to-end solve keeps a fragile item on the floor when a floor slot exists.

## Resulting workflow

1. generate instance
2. build problem description
3. LLM generates `score_candidate(...)`
4. trusted scaffold injects fixed `order_items(...)` and `solve_packing(...)`
5. execute generated code
6. verify strict physics
7. feed error back to LLM if needed
8. save verified result
9. visualize only if verification passes

## Important design choice

This repo still uses a conservative static-support model.
It intentionally does not model friction, deformability, or arbitrary real-world overhang stability.
The goal is reproducible legality plus better transport stability, not permissive physical simulation.
