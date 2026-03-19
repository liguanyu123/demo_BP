# AFL-style 3D-BBP Demo (strict physics, transport-stable solver)

This project keeps the AFL closed loop for 3D bin packing:

1. build a structured problem description,
2. ask the LLM to generate a policy module,
3. execute the generated code,
4. verify the result with a strict physics verifier,
5. feed errors back to the LLM for repair,
6. save the verified solution and render the views.

## What is different in this version

- The verifier remains strict; no physics rule is relaxed.
- The solver is now biased toward **transport-stable layouts** rather than merely legal layouts.
- The trusted solver tries to build a denser ground layer before stacking upward.
- Fragile items are pushed to the floor whenever a valid floor candidate exists.
- The LLM no longer needs to generate `order_items(...)`; the scaffold fixes item ordering and `solve_packing(...)`.
- The LLM now only customizes `score_candidate(...)`, which reduces brittle code-generation errors while keeping the AFL loop agentic.

## Transport-stability policy

The trusted solver now follows these additional preferences while preserving strict legality:

- during the early 40% of the placement process, if a floor (`z == 0`) candidate exists, only floor candidates are considered;
- if a fragile item has any floor candidate, only floor candidates are considered for that item;
- the fixed ordering favors larger footprint boxes and lower-height boxes for base construction, with non-fragile items preferred as a tie-breaker instead of a permanent hard rule;
- candidate scoring heavily favors grounded placements, grounded fragile placements, larger-footprint floor placements, and lower-height floor placements.

These are **solver heuristics only**. The verifier is unchanged and still rejects overlap, boundary violations, floating, partial support, and fragile load-bearing.

## Core trust boundary

The LLM is **not** allowed to invent low-level physics checks.
Instead, it generates only the candidate-scoring policy:

- `score_candidate(item_id, item, candidate, state, bin_size)`

The trusted scaffold fixes:

- `order_items(items)`
- `solve_packing(items, bin_size)`
- all geometry legality in `packing_kernel.py`
- all final legality checks in `verifier.py`

This keeps the AFL loop agentic, but preserves deterministic legality checks.

## Files

- `afl_workflow.py`
  - main controller for the GA / JA / RA / EAA loop
  - fixed `order_items(...)` / `solve_packing(...)` scaffold
  - the LLM revises only `score_candidate(...)`

- `trusted_solver.py`
  - trusted search engine used by the generated policy module
  - transport-stability candidate filtering
  - floor-first base construction bias
  - grounded-fragile preference

- `packing_kernel.py`
  - deterministic geometry kernel
  - conservative support model
  - pruned candidate generation for faster 20-30 item search

- `verifier.py`
  - strict legality verifier
  - rejects overlap, boundary violations, floating, partial support, fragile support
  - unchanged in strength relative to the strict physics version

- `generate_instances.py`
  - constructive random instance generator
  - default output has **no hidden solution field**
  - use `--include-hidden-solution` only if you explicitly want a witness layout for debugging

- `visualize_results.py`
  - validates before rendering
  - 3D orthographic view + top/front/side projections
  - explicit origin marker and axis direction markers

## Recommended API configuration

Default recommendation for this repo:

```bash
export DEEPSEEK_API_KEY=YOUR_KEY
export DEEPSEEK_CHAT_MODEL=deepseek-chat
export DEEPSEEK_REASONER_MODEL=deepseek-reasoner
```

Use `--provider deepseek` for real online solving.

Notes:

- `mock_reference` is only for local dry-run testing.
- `mock_reference` does **not** call any external API.
- If you want token consumption, you must run with `--provider deepseek` (or `auto` with `DEEPSEEK_API_KEY` set).

## Run

### 1) Generate fresh instances (5/10/15/20/25/30 items)

```bash
python -u generate_instances.py \
  --out instances \
  --sizes 5 10 15 20 25 30 \
  --per-size 3 \
  --target-utilization 0.32
```

### 2) Local dry-run without API

```bash
python afl_workflow.py --provider mock_reference --instance instances/inst_25items_0.json
```

### 3) Real DeepSeek run

```bash
export DEEPSEEK_API_KEY=YOUR_KEY
export DEEPSEEK_CHAT_MODEL=deepseek-chat
export DEEPSEEK_REASONER_MODEL=deepseek-reasoner
python afl_workflow.py --provider deepseek --instance instances/inst_25items_0.json
```

### 4) Validate a saved solution

```bash
python validate_solution.py results/<run_id>/solution.json
```

### 5) Render validated views

```bash
python visualize_results.py --instance <run_id>
```

## Tests

```bash
python tests_physics.py
python tests_transport_stability.py
```

## Expected behavior

- 5/10/15/20 item constructive instances should be easy.
- 25/30 item constructive instances should still be solvable in the current trusted-search scaffold.
- The generated layouts should now use a denser first layer and keep fragile items on the floor whenever a valid floor slot exists.
- If the online model fails, the run should end as `FAILED` instead of silently pulling an answer from the JSON.
