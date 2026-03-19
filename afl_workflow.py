from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import os
import re
import subprocess
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

from llm_client import LLMClient
from problem_description import build_problem_description, description_as_text
from prompt_library import (
    ERROR_ANALYSIS,
    PROBLEM_DESCRIPTION_GENERATE,
    PROBLEM_DESCRIPTION_JUDGE,
    PROBLEM_DESCRIPTION_REVISE,
    PROBLEM_DESCRIPTION_SYSTEM,
    SOLVER_GENERATE_SYSTEM,
    SOLVER_JUDGE,
    SOLVER_REVISE,
)
from verifier import BoxVerifier


DEFAULT_ITEMS = {
    1: {"l": 220, "w": 220, "h": 180, "fragile": False},
    2: {"l": 180, "w": 160, "h": 140, "fragile": False},
    3: {"l": 210, "w": 190, "h": 120, "fragile": False},
    4: {"l": 150, "w": 150, "h": 110, "fragile": True},
    5: {"l": 160, "w": 140, "h": 100, "fragile": True},
}
DEFAULT_BIN_SIZE = (1000, 1000, 1000)


SAFE_ORDER_ITEMS = """
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
""".strip()


DEFAULT_SCORE_CANDIDATE = """
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
""".strip()


SAFE_SOLVE_PACKING = """
from trusted_solver import solve_with_policy

def solve_packing(items, bin_size):
    return solve_with_policy(items, bin_size, order_items, score_candidate, heavy=(len(items) >= 20))
""".strip()


GEN_POLICY_PROMPT = """
You are designing only the candidate-scoring policy for a strict 3D bin packing solver.

Return Python code only. Do not include markdown fences.
Return exactly one function definition with this signature:
- def score_candidate(item_id, item, candidate, state, bin_size):

Hard rules:
1. Do not define order_items.
2. Do not define solve_packing.
3. Do not import anything.
4. score_candidate(...) must return a tuple of sortable values.
5. The scaffold will inject a fixed order_items(items) and solve_packing(items, bin_size).
6. You may use only these item fields: item["l"], item["w"], item["h"], item.get("fragile", False)
7. You may use only these candidate fields:
   candidate["x"], candidate["y"], candidate["z"],
   candidate["used_height_after"], candidate["bbox_volume_after"],
   candidate["support_area"], candidate["support_ratio"]
8. You may read only state["placed_count"] from state.
9. Never implement overlap / support / boundary logic yourself.
10. Optimize for transport stability while preserving feasibility:
    - strongly prefer z == 0 when available,
    - strongly prefer fragile items on z == 0 when they can be grounded,
    - among z == 0 candidates prefer larger footprint and lower height,
    - otherwise prefer stronger support, lower used height, and compact placement.
11. The code must be valid Python.

Problem description:
{problem_description}
""".strip()


REVISE_POLICY_PROMPT = """
You are revising only the candidate-scoring policy for a strict 3D bin packing solver.

Return Python code only. Do not include markdown fences.
Return exactly one function definition with this signature:
- def score_candidate(item_id, item, candidate, state, bin_size):

Hard rules:
1. Do not define order_items.
2. Do not define solve_packing.
3. Do not import anything.
4. score_candidate(...) must return a tuple of sortable values.
5. The scaffold will inject a fixed order_items(items) and solve_packing(items, bin_size).
6. You may use only these item fields: item["l"], item["w"], item["h"], item.get("fragile", False)
7. You may use only these candidate fields:
   candidate["x"], candidate["y"], candidate["z"],
   candidate["used_height_after"], candidate["bbox_volume_after"],
   candidate["support_area"], candidate["support_ratio"]
8. You may read only state["placed_count"] from state.
9. Keep the transport-stability priorities intact: grounded placements first, fragile-on-floor when possible, larger-footprint and lower-height floor placements, then support/compactness.
10. The code must be valid Python.

Current error feedback:
{error_feedback}

Judge feedback:
{judge_feedback}

Problem description:
{problem_description}
""".strip()


class AFLWorkflow:
    def __init__(
        self,
        items: Dict[Any, Dict[str, Any]],
        bin_size: Sequence[int],
        save_dir: str = "./results",
        provider: str = "auto",
        model: Optional[str] = None,
        max_iterations: int = 4,
        trusted_final_attempt: bool = False,
    ):
        self.items = items
        self.bin_size = tuple(int(v) for v in bin_size)
        self.verifier = BoxVerifier(self.bin_size)
        self.llm = LLMClient(provider=provider, model=model)
        self.max_iterations = max_iterations
        self.trusted_final_attempt = trusted_final_attempt

        self.experiment_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = save_dir
        self.exp_dir = os.path.join(save_dir, self.experiment_id)
        os.makedirs(self.exp_dir, exist_ok=True)
        self.log_content: List[Dict[str, Any]] = []
        print(f"Experiment directory: {self.exp_dir}")
        print(f"LLM provider: {self.llm.provider} | model: {self.llm.model}")

    def log(self, step: str, payload: Dict[str, Any]) -> None:
        self.log_content.append({"step": step, "payload": payload})

    def _current_llm_meta(self) -> Dict[str, Any]:
        return self.llm.get_last_meta()

    @staticmethod
    def _merge_problem_description(baseline: Dict[str, Any], candidate: Any) -> Dict[str, Any]:
        if not isinstance(candidate, dict):
            return dict(baseline)
        merged = dict(baseline)
        for key in ["P", "S", "K", "X", "Y", "Z"]:
            if key in candidate:
                merged[key] = candidate[key]
        return merged

    def _generate_problem_description(self) -> Dict[str, Any]:
        baseline = build_problem_description(self.items, self.bin_size)

        if self.llm.is_mock:
            judge = {"ok": True, "issues": [], "suggestions": []}
            self.log("problem_description_mock", {"description": baseline, "judge": judge, "llm_meta": {}})
            return baseline

        prompt = PROBLEM_DESCRIPTION_GENERATE.format(
            instance_summary=json.dumps(baseline["instance_summary"], ensure_ascii=False, indent=2)
        )
        try:
            candidate = self.llm.generate_json(
                prompt,
                PROBLEM_DESCRIPTION_SYSTEM,
                role_hint="problem_description_generate",
                context={"items": self.items, "bin_size": self.bin_size},
                max_tokens=2200,
                temperature=0.1,
                use_reasoner=False,
            )
        except Exception as exc:
            self.log("problem_description_generate_failed", {"prompt": prompt, "error": str(exc), "llm_meta": self._current_llm_meta()})
            return baseline
        candidate = self._merge_problem_description(baseline, candidate)
        self.log("problem_description_generate", {"prompt": prompt, "response": candidate, "llm_meta": self._current_llm_meta()})

        for attempt in range(self.max_iterations):
            judge_prompt = (
                f"Instance summary:\n{json.dumps(baseline['instance_summary'], ensure_ascii=False, indent=2)}\n\n"
                f"Candidate description:\n{json.dumps(candidate, ensure_ascii=False, indent=2)}"
            )
            judge = self.llm.generate_json(
                judge_prompt,
                PROBLEM_DESCRIPTION_JUDGE,
                role_hint="problem_description_judge",
                context={"candidate_description": candidate},
                max_tokens=2200,
                temperature=0.1,
                use_reasoner=False,
            )
            self.log("problem_description_judge", {"attempt": attempt + 1, "response": judge, "llm_meta": self._current_llm_meta()})
            if judge.get("ok"):
                return candidate

            revise_prompt = (
                f"Instance summary:\n{json.dumps(baseline['instance_summary'], ensure_ascii=False, indent=2)}\n\n"
                f"Candidate description:\n{json.dumps(candidate, ensure_ascii=False, indent=2)}\n\n"
                f"Judge feedback:\n{json.dumps(judge, ensure_ascii=False, indent=2)}"
            )
            try:
                revised = self.llm.generate_json(
                    revise_prompt,
                    PROBLEM_DESCRIPTION_REVISE,
                    role_hint="problem_description_revise",
                    context={"candidate_description": candidate},
                    max_tokens=2200,
                    temperature=0.1,
                    use_reasoner=False,
                )
                candidate = self._merge_problem_description(baseline, revised)
                self.log("problem_description_revise", {"attempt": attempt + 1, "response": candidate, "llm_meta": self._current_llm_meta()})
            except Exception as exc:
                self.log("problem_description_revise_failed", {"attempt": attempt + 1, "error": str(exc), "llm_meta": self._current_llm_meta()})
                return baseline

        return baseline

    @staticmethod
    def _extract_code(text: str) -> str:
        cleaned = text.strip()
        fenced_match = re.search(r"```(?:python)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
        if fenced_match:
            cleaned = fenced_match.group(1).strip()
        elif cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:python)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        return cleaned.strip()

    def _order_function_fallback(self) -> str:
        return SAFE_ORDER_ITEMS

    def _score_function_fallback(self) -> str:
        return DEFAULT_SCORE_CANDIDATE

    def _assemble_solver_code(self, order_code: str, score_code: str) -> str:
        return f"{order_code.strip()}\n\n{score_code.strip()}\n\n{SAFE_SOLVE_PACKING}\n"

    def _make_default_solver_code(self) -> str:
        return self._assemble_solver_code(self._order_function_fallback(), self._score_function_fallback())

    def _validate_order_function(self, code: str) -> str:
        code = self._extract_code(code).strip()
        if "def order_items" not in code:
            return self._order_function_fallback()
        lowered = code.lower()
        if any(fragment in lowered for fragment in ["import ", "from ", "open(", "exec(", "eval(", "__import__"]):
            return self._order_function_fallback()
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return self._order_function_fallback()
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
            return self._order_function_fallback()
        func = tree.body[0]
        if func.name != "order_items":
            return self._order_function_fallback()
        arg_names = [arg.arg for arg in func.args.args]
        if arg_names != ["items"]:
            return self._order_function_fallback()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                return self._order_function_fallback()
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"open", "exec", "eval", "compile", "globals", "locals"}:
                return self._order_function_fallback()
        return code

    def _validate_or_fallback_score(self, code: str) -> str:
        code = self._extract_code(code).strip()
        if "def score_candidate" not in code:
            return self._score_function_fallback()

        lowered = code.lower()
        banned_fragments = [
            "import ",
            "from ",
            "open(",
            "exec(",
            "eval(",
            "def solve_packing",
            "__import__",
        ]
        if any(fragment in lowered for fragment in banned_fragments):
            return self._score_function_fallback()

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return self._score_function_fallback()

        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
            return self._score_function_fallback()

        func = tree.body[0]
        if func.name != "score_candidate":
            return self._score_function_fallback()

        arg_names = [arg.arg for arg in func.args.args]
        if arg_names != ["item_id", "item", "candidate", "state", "bin_size"]:
            return self._score_function_fallback()

        allowed_candidate_keys = {
            "x",
            "y",
            "z",
            "used_height_after",
            "bbox_volume_after",
            "support_area",
            "support_ratio",
        }
        allowed_item_keys = {"l", "w", "h"}
        allowed_item_get_keys = {"fragile"}
        allowed_state_keys = {"placed_count"}

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                return self._score_function_fallback()
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in {"open", "exec", "eval", "compile", "globals", "locals"}:
                    return self._score_function_fallback()
                if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                    if node.func.value.id == "state":
                        if node.func.attr != "get":
                            return self._score_function_fallback()
                        if not node.args or not isinstance(node.args[0], ast.Constant) or node.args[0].value not in allowed_state_keys:
                            return self._score_function_fallback()
                    if node.func.value.id == "candidate" and node.func.attr != "get":
                        return self._score_function_fallback()
                    if node.func.value.id == "item":
                        if node.func.attr != "get":
                            return self._score_function_fallback()
                        if not node.args or not isinstance(node.args[0], ast.Constant) or node.args[0].value not in allowed_item_get_keys:
                            return self._score_function_fallback()
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
                key = None
                sl = node.slice
                if isinstance(sl, ast.Constant):
                    key = sl.value
                elif hasattr(ast, "Index") and isinstance(sl, ast.Index) and isinstance(sl.value, ast.Constant):
                    key = sl.value.value
                if node.value.id == "state" and key not in allowed_state_keys:
                    return self._score_function_fallback()
                if node.value.id == "candidate" and key not in allowed_candidate_keys:
                    return self._score_function_fallback()
                if node.value.id == "item" and key not in allowed_item_keys:
                    return self._score_function_fallback()

        return code

    def _force_solver_shape(self, code: str) -> str:
        extracted = self._extract_code(code)
        score_match = re.search(
            r"def\s+score_candidate\s*\(\s*item_id\s*,\s*item\s*,\s*candidate\s*,\s*state\s*,\s*bin_size\s*\)\s*:\s*.*?(?=\ndef\s|\Z)",
            extracted,
            flags=re.DOTALL,
        )
        order_code = self._order_function_fallback()
        score_code = self._validate_or_fallback_score(score_match.group(0).strip()) if score_match else self._score_function_fallback()
        return self._assemble_solver_code(order_code, score_code)

    def _generate_solver_code(self, description: Dict[str, Any]) -> str:
        problem_text = description_as_text(description)
        prompt = GEN_POLICY_PROMPT.format(problem_description=problem_text)
        response = self.llm.generate(
            prompt,
            SOLVER_GENERATE_SYSTEM,
            role_hint="solver_generate",
            context={"items": self.items, "bin_size": self.bin_size},
            max_tokens=2600,
            temperature=0.2,
            use_reasoner=len(self.items) >= 20,
        ).strip()
        full_code = self._force_solver_shape(response)
        self.log("solver_generate", {"prompt": prompt, "response": full_code, "llm_meta": self._current_llm_meta()})
        return full_code

    def _judge_solver_code(self, code: str, description: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"Problem description:\n{description_as_text(description)}\n\nCandidate code:\n{code}"
        result = self.llm.generate_json(
            prompt,
            SOLVER_JUDGE,
            role_hint="solver_judge",
            context={"candidate_code": code},
            max_tokens=1600,
            temperature=0.1,
            use_reasoner=False,
        )
        self.log("solver_judge", {"response": result, "llm_meta": self._current_llm_meta()})
        return result

    def _revise_solver_code(
        self,
        code: str,
        description: Dict[str, Any],
        judge_feedback: Optional[Dict[str, Any]] = None,
        error_feedback: Optional[Dict[str, Any]] = None,
        use_reasoner: bool = False,
    ) -> str:
        prompt = REVISE_POLICY_PROMPT.format(
            problem_description=description_as_text(description),
            judge_feedback=json.dumps(judge_feedback or {}, ensure_ascii=False, indent=2),
            error_feedback=json.dumps(error_feedback or {}, ensure_ascii=False, indent=2),
        )
        response = self.llm.generate(
            prompt,
            SOLVER_REVISE,
            role_hint="solver_revise",
            context={"items": self.items, "bin_size": self.bin_size},
            max_tokens=2600,
            temperature=0.1 if use_reasoner else 0.15,
            use_reasoner=use_reasoner,
        ).strip()
        revised = self._force_solver_shape(response)
        self.log("solver_revise", {"use_reasoner": use_reasoner, "prompt": prompt, "response": revised, "llm_meta": self._current_llm_meta()})
        return revised

    def _analyze_error(self, code: str, description: Dict[str, Any], error_msg: str, use_reasoner: bool) -> Dict[str, Any]:
        prompt = f"Problem description:\n{description_as_text(description)}\n\nCode:\n{code}\n\nError:\n{error_msg}"
        result = self.llm.generate_json(
            prompt,
            ERROR_ANALYSIS,
            role_hint="error_analysis",
            context={"error_msg": error_msg},
            max_tokens=1800,
            temperature=0.1,
            use_reasoner=use_reasoner,
        )
        self.log("error_analysis", {"response": result, "llm_meta": self._current_llm_meta()})
        return result

    def _should_use_reasoner(
        self,
        iteration: int,
        error_msg: str = "",
        judge_feedback: Optional[Dict[str, Any]] = None,
        consecutive_failures: int = 0,
    ) -> bool:
        if len(self.items) >= 20:
            return True
        if iteration >= 3 or consecutive_failures >= 2:
            return True
        combined = error_msg
        if judge_feedback:
            combined += "\n" + json.dumps(judge_feedback, ensure_ascii=False)
        keywords = (
            "NameError",
            "KeyError",
            "JSONDecodeError",
            "AttributeError",
            "IndentationError",
            "SyntaxError",
            "TypeError",
            "Placement length 0",
        )
        return any(token in combined for token in keywords)

    @staticmethod
    def _is_empty_placement_error(error_msg: str) -> bool:
        lowered = error_msg.lower()
        return "placement length 0 does not match item count" in lowered or "returned none" in lowered

    def _execute_solver(self, code: str) -> Tuple[List[Dict[str, int]], Dict[str, Any]]:
        code = self._force_solver_shape(code)
        namespace: Dict[str, Any] = {}
        exec(code, namespace, namespace)
        solve_func = namespace.get("solve_packing")
        if solve_func is None:
            raise RuntimeError("solve_packing function not found in generated code")
        placement = solve_func(self.items, self.bin_size)
        if placement is None:
            raise RuntimeError("solve_packing returned None")
        if not isinstance(placement, list):
            raise RuntimeError(f"solve_packing must return a list, got {type(placement)}")
        if len(placement) != len(self.items):
            raise RuntimeError(f"Placement length {len(placement)} does not match item count {len(self.items)}")
        ok, msg, report = self.verifier.verify(self.items, placement, return_report=True)
        if not ok:
            raise RuntimeError(msg)
        return placement, report

    def _usage_totals(self) -> Dict[str, int]:
        totals: Dict[str, int] = {}
        for entry in self.log_content:
            llm_meta = entry.get("payload", {}).get("llm_meta") or {}
            usage = llm_meta.get("usage") or {}
            for key, value in usage.items():
                if isinstance(value, int):
                    totals[key] = totals.get(key, 0) + value
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, int):
                            merged_key = f"{key}.{sub_key}"
                            totals[merged_key] = totals.get(merged_key, 0) + sub_value
        return totals

    def save_artifacts(
        self,
        code_content: str,
        placement: List[Dict[str, Any]],
        verification_report: Dict[str, Any],
        problem_description: Dict[str, Any],
        status: str,
        last_error: str = "",
    ) -> None:
        with open(os.path.join(self.exp_dir, "generated_solver.py"), "w", encoding="utf-8") as f:
            f.write(code_content)

        with open(os.path.join(self.exp_dir, "solution.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "placement": placement,
                    "items": self.items,
                    "bin_size": self.bin_size,
                    "verification_report": verification_report,
                },
                f,
                indent=2,
            )

        with open(os.path.join(self.exp_dir, "instance.json"), "w", encoding="utf-8") as f:
            json.dump({"items": self.items, "bin_size": self.bin_size}, f, indent=2)

        with open(os.path.join(self.exp_dir, "problem_description.json"), "w", encoding="utf-8") as f:
            json.dump(problem_description, f, indent=2, ensure_ascii=False)

        with open(os.path.join(self.exp_dir, "verification_report.json"), "w", encoding="utf-8") as f:
            json.dump(verification_report, f, indent=2)

        with open(os.path.join(self.exp_dir, "interaction_log.json"), "w", encoding="utf-8") as f:
            json.dump(self.log_content, f, indent=2, ensure_ascii=False)

        summary = {
            "experiment_id": self.experiment_id,
            "status": status,
            "iterations": self.max_iterations,
            "llm_provider": self.llm.provider,
            "llm_model": self.llm.model,
            "llm_usage_totals": self._usage_totals(),
            "last_error": last_error,
            "final_placement": placement,
        }
        with open(os.path.join(self.exp_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        try:
            subprocess.run(
                [
                    "python3",
                    "visualize_results.py",
                    "--instance",
                    self.experiment_id,
                    "--results",
                    self.save_dir,
                    "--output",
                    os.path.join(os.path.dirname(self.exp_dir), "visualizations"),
                ],
                cwd=os.path.dirname(__file__),
                check=True,
            )
        except Exception as exc:
            print(f"Visualization generation failed: {exc}")

    def run(self) -> bool:
        print("=== AFL-style 3D Bin Packing Workflow (strict physics) ===")
        problem_description = self._generate_problem_description()
        code = self._generate_solver_code(problem_description)
        judge = self._judge_solver_code(code, problem_description)
        if not judge.get("ok"):
            code = self._revise_solver_code(code, problem_description, judge_feedback=judge, use_reasoner=len(self.items) >= 20)
            judge = self._judge_solver_code(code, problem_description)

        last_error = ""
        consecutive_failures = 0

        for iteration in range(1, self.max_iterations + 1):
            print(f"[Iteration {iteration}] executing generated solver ...")
            try:
                placement, verification_report = self._execute_solver(code)
                print("[SUCCESS] verified packing found")
                self.save_artifacts(code, placement, verification_report, problem_description, status="SUCCESS", last_error=last_error)
                return True
            except Exception:
                consecutive_failures += 1
                last_error = traceback.format_exc()
                print(f"[ERROR] {last_error}")
                self.log("solver_execution_error", {"iteration": iteration, "error": last_error, "code": code})

                use_reasoner = self._should_use_reasoner(
                    iteration=iteration,
                    error_msg=last_error,
                    judge_feedback=judge,
                    consecutive_failures=consecutive_failures,
                )
                error_feedback = self._analyze_error(code, problem_description, last_error, use_reasoner=use_reasoner)
                code = self._revise_solver_code(
                    code,
                    problem_description,
                    judge_feedback=judge,
                    error_feedback=error_feedback,
                    use_reasoner=use_reasoner,
                )
                judge = self._judge_solver_code(code, problem_description)
                if not judge.get("ok"):
                    code = self._revise_solver_code(
                        code,
                        problem_description,
                        judge_feedback=judge,
                        error_feedback=error_feedback,
                        use_reasoner=True,
                    )

        if self.trusted_final_attempt:
            from trusted_solver import solve_packing as trusted_solve_packing

            print("[FINAL_ATTEMPT] running trusted solver because --trusted-final-attempt is enabled")
            placement = trusted_solve_packing(self.items, self.bin_size)
            if placement:
                ok, msg, verification_report = self.verifier.verify(self.items, placement, return_report=True)
                if ok:
                    self.save_artifacts(code, placement, verification_report, problem_description, status="TRUSTED_FINAL_ATTEMPT_SUCCESS", last_error=last_error)
                    return True
                self.log("trusted_final_attempt_failed", {"message": msg, "report": verification_report})

        empty_report = {"checks": {}, "support_details": [], "placement": []}
        self.save_artifacts(code, [], empty_report, problem_description, status="FAILED", last_error=last_error)
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=str, default=None, help="Path to instance JSON file")
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument(
        "--provider",
        type=str,
        default="auto",
        choices=["auto", "openai", "deepseek", "mock_reference"],
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--max-iterations", type=int, default=4)
    parser.add_argument("--trusted-final-attempt", action="store_true", help="Developer-only final local solver attempt; off by default")
    args = parser.parse_args()

    if args.instance and os.path.exists(args.instance):
        with open(args.instance, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = data["items"]
        bin_size = tuple(data["bin_size"])
        print(f"Loaded instance from: {args.instance}")
    else:
        items = DEFAULT_ITEMS
        bin_size = DEFAULT_BIN_SIZE
        print("Using default built-in instance")

    workflow = AFLWorkflow(
        items,
        bin_size,
        save_dir=args.save_dir,
        provider=args.provider,
        model=args.model,
        max_iterations=args.max_iterations,
        trusted_final_attempt=args.trusted_final_attempt,
    )
    ok = workflow.run()
    raise SystemExit(0 if ok else 1)
