from __future__ import annotations

# ==========================================
# 阶段一：问题描述闭环 (Problem Description)
# ==========================================

PROBLEM_DESCRIPTION_SYSTEM = (
    "你是一个顶级的 3D 物理与装箱优化专家。你的任务是从原始实例中提取无遗漏、无偏差的问题规则宪法。"
    "禁止捏造实例中不存在的字段，必须严格输出 JSON 格式。"
)

PROBLEM_DESCRIPTION_GENERATE = """
你是生成智能体 (GA)。请根据以下 3D-BPP 测试实例的摘要，提取并输出一个严格的 JSON 对象，必须包含以下顶级键：P, S, K, X, Y, Z。

核心物理与业务约束提示（必须体现在 K 约束集合中）：
1. 姿态自由度：物品可任意三维旋转（共6种正交姿态），必须满足物理稳定约束（无重叠、不超界）。允许竖放/侧放以适配窄缝。
2. 支撑约束：所有悬空物品（z>0）必须有下方物品支撑，且非易碎品支撑面积比必须 ≥ 80%。
3. 易碎品红线：易碎品绝对不得作为任何物品的支撑面！仅可放置在顶层（上方无任何物品），不得被其他物品压在下方。
4. 填充策略：
   - 底层/中层必须 100% 使用非易碎品构建稳定支撑。
   - 易碎品仅用于填充顶层剩余空隙。

严格按以下格式输出：
- P: 问题类型名称 (如 "3D-BPP-Autonomous")
- S: 一句话精炼总结
- K: 约束对象数组 (每个对象含 name 和 description)
- X: 必须的输入元素列表 (如物品长宽高、易碎属性、箱体尺寸)
- Y: 明确的输出格式描述 (包含物品的 x,y,z 坐标及旋转后的实际 l,w,h)
- Z: 优化目标 (易碎品安全放置，最大化空间利用率，构建稳定底层)

实例摘要：
{instance_summary}
""".strip()

PROBLEM_DESCRIPTION_JUDGE = """
你是评判智能体 (JA)。
请以原始实例和业务红线为唯一依据，严格审查 GA 生成的问题描述 JSON。

必须核对以下红线：
1. 是否包含三维旋转许可？
2. 是否明确了易碎品“绝对不可作为支撑面且只能放在顶层”的红线？
3. 是否要求非易碎品支撑面积 ≥ 80%？

返回 JSON 格式：
- ok: boolean (是否完美无瑕)
- issues: 字符串数组 (发现的漏洞或缺失项)
- suggestions: 字符串数组 (修改建议)

候选描述：
{candidate_description}
""".strip()

PROBLEM_DESCRIPTION_REVISE = """
你是修改智能体 (RA)。
请根据 JA 的驳回意见，针对性地重写并修复候选的问题描述 JSON。
返回修复后的完整 JSON，必须包含 P, S, K, X, Y, Z。

JA 反馈：
{judge_feedback}

当前候选描述：
{candidate_description}
""".strip()


# ==========================================
# 阶段二：代码生成闭环 (Function-by-Function Generation)
# ==========================================

SOLVER_GENERATE_SYSTEM = (
    "你是一个高级 Python 算法工程师。你将参与编写一个完全自主的 3D BPP 求解器。"
    "你必须直接输出可执行的 Python 代码，不要使用 Markdown 代码块 (```python)，不要写解释性的废话。"
)

# 逐函数生成的核心 Prompt
FUNCTION_GENERATE = """
你是生成智能体 (GA)。当前正在构建 3D BPP 求解器。
你需要编写当前指定的单个函数：`{target_function}`。

求解器整体包含以下环节：read_bp -> initial -> validate -> cost -> destroy -> insert -> main。
目前已生成的上下文代码如下（你不可修改这些，只能基于它们编写当前函数）：
{context_code}

当前任务：
请编写 `{target_function}` 函数。
业务规则：
{problem_description}

特别要求：
如果当前是 `validate` 函数，你必须自己手写 3D 碰撞检测、超界检测、80%支撑面积检测以及易碎品顶层检测逻辑，绝对不可调用外部黑盒模块！
如果当前是 `insert` 函数，你必须考虑物品的 6 种三维旋转姿态。

仅返回 `{target_function}` 的 Python 代码定义。
""".strip()

FUNCTION_JUDGE = """
你是评判智能体 (JA)。负责审计刚刚生成的 `{target_function}` 函数。

请检查：
1. 语法是否正确？
2. 逻辑是否严丝合缝地遵守了问题描述中的所有物理约束（尤其是三维旋转和易碎品红线）？
3. 是否正确衔接了上下文中已有的代码？

返回 JSON：
- ok: boolean
- issues: 发现的问题数组
- suggestions: 修改建议数组

新生成的代码：
{candidate_code}
""".strip()

FUNCTION_REVISE = """
你是修改智能体 (RA)。你需要修复有缺陷的 `{target_function}` 函数。
请仅输出修复后的该函数 Python 代码，不要动上下文代码，不要 Markdown 标记。

错误/评判反馈：
{feedback}

待修复的代码：
{candidate_code}
""".strip()

# ==========================================
# 阶段三：运行纠错闭环 (Error Analysis & Fixing)
# ==========================================

ERROR_ANALYSIS = """
你是错误分析智能体 (EAA)。代码在运行时崩溃或 validate 失败。
请进行“外科手术式诊断”，不要直接改代码，仅指出根因。

问题描述：
{problem_description}

出错环节/报错信息：
{error_msg}

返回 JSON 格式：
- diagnosis: 根本原因简述 (例如：insert 插入时未更新旋转后的尺寸导致 validate 重叠报错)
- target_function: 导致该错误的最可能的罪魁祸首函数名 (如 'insert' 或 'validate')
- actionable_fixes: 具体修改建议数组
""".strip()
