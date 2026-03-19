from __future__ import annotations

import argparse
import json
import os
import re
import traceback
import datetime as dt
from typing import Any, Dict, List, Optional

from llm_client import LLMClient
from prompt_library import (
    PROBLEM_DESCRIPTION_GENERATE,
    PROBLEM_DESCRIPTION_JUDGE,
    PROBLEM_DESCRIPTION_REVISE,
    SOLVER_GENERATE_SYSTEM,
    FUNCTION_GENERATE,
    FUNCTION_JUDGE,
    FUNCTION_REVISE,
    ERROR_ANALYSIS,
)

# 必须严格按顺序生成的 7 个核心函数
TARGET_FUNCTIONS = [
    "read_bp",
    "initial",
    "validate",
    "cost",
    "destroy",
    "insert",
    "main"
]

class AFLWorkflow:
    def __init__(
        self,
        instance_path: str,
        save_dir: str = "./results",
        provider: str = "auto",
        model: Optional[str] = None,
        max_iterations: int = 10,
    ):
        self.instance_path = instance_path
        self.max_iterations = max_iterations
        self.llm = LLMClient(provider=provider, model=model)
        
        # 记录生成的代码字典 { "函数名": "具体代码" }
        self.generated_functions: Dict[str, str] = {}
        self.problem_description: Dict[str, Any] = {}

        # 实验目录记录
        self.experiment_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join(save_dir, self.experiment_id)
        os.makedirs(self.exp_dir, exist_ok=True)
        self.log_content: List[Dict[str, Any]] = []

        # 原始实例数据
        self.raw_instance = self._load_instance(instance_path)

        print(f"初始化 AFL 真实闭环工作流 | 目录: {self.exp_dir} | 模型: {self.llm.model}")

    def _load_instance(self, path: str) -> Dict[str, Any]:
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        raise FileNotFoundError(f"找不到实例文件: {path}，必须提供真实的 JSON 数据！")

    def log(self, step: str, payload: Dict[str, Any]) -> None:
        self.log_content.append({"step": step, "payload": payload})

    @staticmethod
    def _extract_code(text: str) -> str:
        """剥离 LLM 喜欢乱加的 Markdown 标签，提取纯代码。使用 {3} 避开前端渲染器截断 Bug"""
        cleaned = text.strip()
        # 用 `{3}` 替代连续写三个反引号，彻底阻断渲染器发癫
        fenced_match = re.search(r"`{3}(?:python)?\n?(.*?)\n?`{3}", cleaned, flags=re.DOTALL | re.IGNORECASE)
        if fenced_match:
            return fenced_match.group(1).strip()
        
        if re.search(r"^`{3}", cleaned):
            cleaned = re.sub(r"^`{3}(?:python)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*`{3}$", "", cleaned)
        return cleaned.strip()

    def _get_context_code(self, current_target: str) -> str:
        """组装当前已生成的上下文代码"""
        context = []
        for func in TARGET_FUNCTIONS:
            if func == current_target:
                break
            if func in self.generated_functions:
                context.append(self.generated_functions[func])
        return "\n\n".join(context)

    # ==========================================
    # 阶段一：问题描述闭环
    # ==========================================
    def phase1_problem_description(self) -> None:
        print("\n=== [阶段 1] 让智能体吃透问题本质，形成规则宪法 ===")
        # 直接把实例喂给 GA
        instance_summary = json.dumps(self.raw_instance, ensure_ascii=False, indent=2)
        
        prompt = PROBLEM_DESCRIPTION_GENERATE.format(instance_summary=instance_summary)
        candidate = self.llm.generate_json(prompt, "严格按要求输出 JSON")
        self.log("problem_description_generate", {"candidate": candidate})

        for attempt in range(3): # 强制修正循环
            judge_prompt = PROBLEM_DESCRIPTION_JUDGE.format(candidate_description=json.dumps(candidate, ensure_ascii=False))
            judge_result = self.llm.generate_json(judge_prompt, "你是评判智能体。")
            self.log("problem_description_judge", {"attempt": attempt, "result": judge_result})

            if judge_result.get("ok"):
                print(">> JA 审计通过：问题描述规则已锁定。")
                self.problem_description = candidate
                return

            print(f">> JA 审计驳回，RA 正在进行第 {attempt+1} 次修正...")
            revise_prompt = PROBLEM_DESCRIPTION_REVISE.format(
                judge_feedback=json.dumps(judge_result, ensure_ascii=False),
                candidate_description=json.dumps(candidate, ensure_ascii=False)
            )
            candidate = self.llm.generate_json(revise_prompt, "你是修改智能体。")
            self.log("problem_description_revise", {"attempt": attempt, "candidate": candidate})

        print(">> 警告：达到最大修正次数，强行采用当前规则。")
        self.problem_description = candidate

    # ==========================================
    # 阶段二：代码生成闭环 (逐函数生成)
    # ==========================================
    def phase2_generate_code(self) -> None:
        print("\n=== [阶段 2] 逐函数精编，内置内生校验 ===")
        desc_text = json.dumps(self.problem_description, ensure_ascii=False, indent=2)

        for target in TARGET_FUNCTIONS:
            print(f"正在生成函数: {target} ...")
            context = self._get_context_code(target)
            
            prompt = FUNCTION_GENERATE.format(
                target_function=target,
                context_code=context if context else "# (尚无上下文，这是第一个函数)",
                problem_description=desc_text
            )
            code_text = self.llm.generate(prompt, SOLVER_GENERATE_SYSTEM)
            code_text = self._extract_code(code_text)
            self.log("function_generate", {"function": target, "code": code_text})

            # JA 进行局部审计
            for attempt in range(3):
                judge_prompt = FUNCTION_JUDGE.format(target_function=target, candidate_code=code_text)
                judge_result = self.llm.generate_json(judge_prompt, "你是代码评判智能体。")
                self.log("function_judge", {"function": target, "attempt": attempt, "result": judge_result})

                if judge_result.get("ok"):
                    print(f"  -> JA 审计通过: {target}")
                    break
                
                print(f"  -> JA 驳回 {target}，RA 进行修正 (尝试 {attempt+1})")
                revise_prompt = FUNCTION_REVISE.format(
                    target_function=target,
                    feedback=json.dumps(judge_result, ensure_ascii=False),
                    candidate_code=code_text
                )
                code_text = self.llm.generate(revise_prompt, SOLVER_GENERATE_SYSTEM)
                code_text = self._extract_code(code_text)
                self.log("function_revise", {"function": target, "attempt": attempt, "code": code_text})

            # 无论如何，把函数存起来供下一个函数做上下文
            self.generated_functions[target] = code_text

    # ==========================================
    # 阶段三：运行试错与单点修复闭环
    # ==========================================
    def phase3_execute_and_fix(self) -> bool:
        print("\n=== [阶段 3] 组装执行，EAA 精准诊断，RA 单点修复 ===")
        desc_text = json.dumps(self.problem_description, ensure_ascii=False, indent=2)

        for iteration in range(1, self.max_iterations + 1):
            full_code = "\n\n".join(self.generated_functions[func] for func in TARGET_FUNCTIONS)
            print(f"\n[Iteration {iteration}] 尝试运行完整代码...")
            
            try:
                # 建立隔离环境执行代码
                namespace: Dict[str, Any] = {}
                exec(full_code, namespace, namespace)
                
                if "main" not in namespace:
                    raise RuntimeError("生成的代码中找不到 main 函数！")
                
                # 运行主流程，直接传入原始数据
                result = namespace["main"](self.raw_instance)
                print(">> 执行成功！得到可行解。")
                self._save_artifacts(full_code, result, "SUCCESS", "")
                return True
                
            except Exception as e:
                error_msg = traceback.format_exc()
                print(f">> 运行崩溃或校验失败！捕获异常，移交 EAA...")
                self.log("execution_error", {"iteration": iteration, "error": error_msg})

                # EAA 诊断
                eaa_prompt = ERROR_ANALYSIS.format(
                    problem_description=desc_text,
                    error_msg=error_msg
                )
                eaa_result = self.llm.generate_json(eaa_prompt, "你是错误分析智能体。")
                self.log("error_analysis", {"result": eaa_result})
                
                target_func = eaa_result.get("target_function")
                if target_func not in self.generated_functions:
                    print(f"!! EAA 指出的肇事函数 {target_func} 不在目标列表中，强制指向 'main'")
                    target_func = "main"

                print(f">> EAA 诊断肇事函数为: {target_func}。原因: {eaa_result.get('diagnosis')}")
                
                # RA 仅针对目标函数进行单点修复
                print(f">> RA 正在定点修复 {target_func} ...")
                revise_prompt = FUNCTION_REVISE.format(
                    target_function=target_func,
                    feedback=json.dumps(eaa_result, ensure_ascii=False),
                    candidate_code=self.generated_functions[target_func]
                )
                fixed_code = self.llm.generate(revise_prompt, SOLVER_GENERATE_SYSTEM)
                self.generated_functions[target_func] = self._extract_code(fixed_code)
                self.log("function_single_point_fix", {"function": target_func, "code": self.generated_functions[target_func]})

        print("\n[FAILED] 达到最大迭代次数，无法修复所有错误。")
        full_code = "\n\n".join(self.generated_functions[func] for func in TARGET_FUNCTIONS)
        self._save_artifacts(full_code, None, "FAILED", error_msg)
        return False

    def _save_artifacts(self, code: str, result: Any, status: str, error: str):
        with open(os.path.join(self.exp_dir, "generated_solver.py"), "w", encoding="utf-8") as f:
            f.write(code)
        with open(os.path.join(self.exp_dir, "problem_description.json"), "w", encoding="utf-8") as f:
            json.dump(self.problem_description, f, indent=2, ensure_ascii=False)
        with open(os.path.join(self.exp_dir, "interaction_log.json"), "w", encoding="utf-8") as f:
            json.dump(self.log_content, f, indent=2, ensure_ascii=False)
        if result:
            with open(os.path.join(self.exp_dir, "solution.json"), "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        summary = {
            "status": status,
            "experiment_id": self.experiment_id,
            "last_error": error
        }
        with open(os.path.join(self.exp_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    def run(self):
        self.phase1_problem_description()
        self.phase2_generate_code()
        success = self.phase3_execute_and_fix()
        return success

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=str, required=True, help="必须提供真实的 instance.json 路径")
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--provider", type=str, default="auto")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--max-iterations", type=int, default=10)
    args = parser.parse_args()

    workflow = AFLWorkflow(
        instance_path=args.instance,
        save_dir=args.save_dir,
        provider=args.provider,
        model=args.model,
        max_iterations=args.max_iterations,
    )
    ok = workflow.run()
    raise SystemExit(0 if ok else 1)
