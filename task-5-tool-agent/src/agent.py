"""手写 ReAct 循环：Thought / Action / Action Input / Observation，不用任何 agent 框架。

四个实测踩出来的关键点：

1. **生成时拿 "Observation:" 当停止串**。0.5B 会顺着 few-shot 的格式把 Observation
   自己编出来（实测真的见它写过 "Observation: 2500"），不掐掉的话 agent 自问自答，
   真工具一次都不会被调用。
2. **工具异常一律 catch 成 Observation 文本再喂回去**（M3）。单次工具失败不该让整个
   run 崩掉；模型看到「工具 xxx 执行失败：...」才有机会改参数重试。
3. **Action 解析要容错**。模型偶尔把 Action Input 写成非严格 JSON（单引号、裸表达式、
   带 markdown 围栏），解析阶梯是：严格 JSON → 首个花括号块 → ast.literal_eval →
   单参数工具直接当裸值。全失败就把「格式错误」当 Observation 塞回去让它重写。
4. **每一步全文都进 trace**，报告要贴 Thought / Action / Observation 原文。

success 的判定（自检不认这个字段，只按 final_answer 的关键词判分）：只有真的产出了
非空的 Final Answer 才算 True，步数耗尽没收敛算 False。
"""
from __future__ import annotations

import ast
import json
import os
import re

from .llm import ChatModel
from .tools import calculator, file_search, python_sandbox, wiki

# 停止串：模型一写到这里就停，交给循环去执行工具、真拿 Observation 回来
STOP_STRINGS = ("\nObservation:", "Observation:")

MAX_OBSERVATION_CHARS = 1200     # 工具输出回灌前截断，别让一次大 dump 撑爆上下文
FORMAT_HINT = ("错误：没有解析出 Action 或 Final Answer。请严格按格式输出，"
               "要么给 'Action: 工具名' 加 'Action Input: {...}'，"
               "要么给 'Final Answer: 最终答案'。")
NUDGE_HINT = ("错误：你还没有成功调用过任何工具，目前手上没有任何真实结果，"
              "不要凭印象作答。请修正 Action Input（把工具需要的参数写全）后重试，"
              "拿到 Observation 再给 Final Answer。")
COMPUTE_HINT = ("错误：题目要求给出具体数字，但你一次都没有成功调用过 calculator 或 python_sandbox，"
                "答案里的数字不可信。请先用工具算一遍，再给 Final Answer。")

# 收尾前自检用的线索：题目里出现这些词，就要求必须真算过数字
_COMPUTE_HINTS = ("计算", "多少年", "多少岁", "几岁", "小数点后", "几位小数", "求和", "的和", "位数")
_COMPUTE_TOOLS = ("calculator", "python_sandbox")

_SYSTEM_TEMPLATE = """你是一个会用工具解决问题的助手。每一步只输出下面两种格式之一。

格式一（需要调工具）：
Thought: 你为什么要这么做
Action: 工具名
Action Input: {"参数名": "参数值"}

格式二（信息够了，给答案）：
Thought: 我已经知道答案了
Final Answer: 最终答案

规则：
1. Action 必须是下面列出的工具之一；Action Input 必须是合法 JSON，而且要把该工具的参数写全（file_search 需要 pattern 和 dir 两个）。
2. 一次只调一个工具，拿到 Observation 再决定下一步。
3. 不要自己编造 Observation，Observation 来自工具的真实返回。
4. 工具报错时看清错误信息，改参数重试或换工具，不要因为一次失败就直接放弃。
5. Final Answer 要直接回答问题，并把工具返回里的关键信息抄进去：
   问结果和位数就写「13579，共 5 位」，不要只写「位数是 5」；
   问文件路径就把完整路径写出来（如 data/agent-fixtures/todo_note.md），不要只说「找到了几个文件」；
   问文件里写了什么就把那句话原文抄出来；
   问某个东西的测试结果就把对象和结果一起写出来（如 level: True，world: False）。
   答案里的数字必须来自 Observation 里工具的真实返回，不要照抄示例里的数字。
6. 题目里出现「计算」「多少年」「多少岁」「小数点后几位」时，必须用 calculator 真算一遍，不要心算；
   需要小数精度时直接抄 calculator 给出的 6 位小数结果。

选工具的判断：
- 算术、数学函数、算数字 → calculator
- 要跑一段代码、写函数、测试若干输入的结果 → python_sandbox
- 找文件、读文件内容 → file_search
- 查百科知识（人物、年份、概念）→ wiki

可用工具：
{tool_lines}

示例一（要小数精度：用 calculator，并抄它给出的 6 位小数）：
Question: 计算 sqrt(2) 的小数点后 6 位结果
Thought: 要求精度，用 calculator
Action: calculator
Action Input: {"expression": "sqrt(2)"}
Observation: 1.414214（保留 6 位小数；完整值 1.4142135623730951）
Thought: 我已经知道答案了
Final Answer: sqrt(2) ≈ 1.414214（小数点后 6 位）

示例二（工具报错后自我纠错）：
Question: 在 data 目录下找包含 REPORT 的文件
Thought: 先按内容检索
Action: file_search
Action Input: {"pattern": "REPORT", "dir": "data/不存在的目录"}
Observation: 工具 file_search 执行失败：ValueError: 目录或文件不存在：data/不存在的目录。请检查参数是否正确，改参数重试或换一个工具。
Thought: 目录写错了，换成 data 重试
Action: file_search
Action Input: {"pattern": "REPORT", "dir": "data"}
Observation: 在 data 下按「REPORT」找到 1 个文件：…
Thought: 我已经知道答案了
Final Answer: 找到了 1 个文件

示例三（要读某个文件里的内容）：
Question: 读取 docs/notes.md，告诉我它的第一段写了什么
Thought: 要读文件内容，用 file_search；pattern 给文件名，dir 给文件所在目录
Action: file_search
Action Input: {"pattern": "notes.md", "dir": "docs"}
Observation: 在 docs 下按「notes.md」找到 1 个文件（按文件名或内容匹配）：
- docs/notes.md（120 字节，文件名命中）
  片段：这里是笔记文件的第一段。
Thought: 我已经知道答案了
Final Answer: 它的第一段是：这里是笔记文件的第一段。

示例四（跑代码：要让 print 带上对象名，答案里照抄 True/False）：
Question: 用 Python 写个函数判断单词是不是回文，并测试 radar 和 python
Thought: 用 python_sandbox，写完函数要调用它、print 时必须带上单词名
Action: python_sandbox
Action Input: {"code": "def is_pal(w):\n    return w == w[::-1]\nfor w in ['radar', 'python']:\n    print(w, is_pal(w))"}
Observation: radar True
python False
Thought: 我已经知道答案了
Final Answer: radar: True，python: False

示例五（需要两步：先查资料，再用 calculator 算）：
Question: 查维基百科 GPT 是哪一年发布的，并算到 2026 年过了多少年
Thought: 先查发布年份
Action: wiki
Action Input: {"query": "GPT"}
Observation: 维基百科「GPT」：生成式预训练变换器（GPT）由 OpenAI 于 2018 年提出…
Thought: 拿到年份了，再用 calculator 算 2026 - 2018
Action: calculator
Action Input: {"expression": "2026 - 2018"}
Observation: 8（整数，共 1 位）
Thought: 我已经知道答案了
Final Answer: GPT 于 2018 年提出，到 2026 年过了 8 年（2026 - 2018 = 8）。"""

_FINAL_RE = re.compile(r"Final\s*Answer\s*[:：]\s*(.*)\Z", re.S | re.I)
_ACTION_RE = re.compile(r"^\s*Action\s*(?!Input)\s*[:：]\s*([^\n]+)$", re.M | re.I)
_ACTION_INPUT_RE = re.compile(r"^\s*Action\s*Input\s*[:：]\s*(.*)\Z", re.M | re.S | re.I)
_THOUGHT_RE = re.compile(
    r"Thought\s*[:：]\s*(.*?)(?=\n\s*(?:Action|Final\s*Answer)\s*[:：]|\Z)", re.S | re.I)


def _tool_lines(tools: dict) -> str:
    """工具列表从各模块的 TOOL_SCHEMA 动态拼——改 schema 不用同步改 prompt。"""
    lines = []
    for name, module in tools.items():
        function = module.TOOL_SCHEMA["function"]
        params = function["parameters"]["properties"]
        args = "、".join(f'{key}: {value.get("type", "string")}'
                        for key, value in params.items())
        lines.append(f"- {name}({args})：{function['description']}")
    return "\n".join(lines)


def _first_json_object(text: str):
    """从文本里截出第一个花括号配对完整的片段。"""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    return None


def _load_json_like(payload: str):
    """把模型给的 Action Input 尽量解析成 dict，不行就原样返回字符串。"""
    text = re.sub(r"^```[a-zA-Z]*|```\s*$", "", payload.strip()).strip()
    for candidate in (text, _first_json_object(text)):
        if not candidate:
            continue
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    try:
        # literal_eval 除了 ValueError/SyntaxError，还会对被改写过的字面量抛 TypeError
        # （如 {{"a": 1}} 被当成 set 里的 dict）——这是尽力而为的兜底解析，任何失败
        # 都只意味着"再退一级"，绝不能让整个 run 崩掉。
        value = ast.literal_eval(text)
        if isinstance(value, dict):
            return value
    except Exception:
        pass

    if text.startswith("{"):
        # 半截 JSON 的最后一道抢救：硬抠 "key": "value" 对。小模型偶尔漏掉收尾的引号
        # 或括号（实测出现过 {"query": "Transformer (机器学习模型)}），整条作废太浪费。
        pairs = {}
        for key, value in re.findall(r'"([^"]+)"\s*:\s*"([^"}]*)"?', text):
            pairs.setdefault(key, value)
        if pairs:
            return pairs

    return text or None      # 裸值（如 `2 + 3 * 4`），由调用方按单参数工具兜住


def _parse_action(raw: str):
    """返回 (action, action_input, error)。没有 Action 时 action 为 None。"""
    action_match = _ACTION_RE.search(raw)
    if not action_match:
        return None, None, None
    action = action_match.group(1).strip().strip("`\"'")
    input_match = _ACTION_INPUT_RE.search(raw)
    if not input_match:
        return action, None, ("错误：有 Action 但没有 Action Input。"
                              '请补一行 \'Action Input: {"参数名": "值"}\'。')
    payload = _load_json_like(input_match.group(1))
    if payload is None:
        return action, None, ("错误：Action Input 是空的。"
                              '请按格式给出 \'Action Input: {"参数名": "值"}\'。')
    return action, payload, None


def _extract_thought(raw: str) -> str:
    match = _THOUGHT_RE.search(raw)
    return match.group(1).strip() if match else ""


def _extract_final(raw: str) -> str:
    match = _FINAL_RE.search(raw)
    return match.group(1).strip()[:500] if match else ""


def _looks_like_error(observation) -> bool:
    """判断 tool 返回的是真结果，还是需要重试的错误/提示。"""
    text = str(observation or "")
    return (text.startswith(("错误：", "需要重试："))
            or " 执行失败：" in text[:40])


class ReActAgent:
    """手写 ReAct：解析 Action → 调工具 → 把结果当 Observation 喂回去。"""

    def __init__(self, model_dir=None, max_steps: int = 5, verbose=None):
        self.tools = {module.TOOL_SCHEMA["function"]["name"]: module
                      for module in (calculator, python_sandbox, file_search, wiki)}
        self.max_steps = max_steps
        self.verbose = bool(os.environ.get("AGENT_VERBOSE")) if verbose is None else verbose
        self.system_prompt = _SYSTEM_TEMPLATE.replace("{tool_lines}", _tool_lines(self.tools))
        self.llm = ChatModel(model_dir)
        self.inject_error = None      # 预留钩子：想让某一步的工具"故意失败"时设成一个可调用对象

    # ---- 内部步骤 -------------------------------------------------------

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    def _finish_blocker(self, task, has_result, used_compute):
        """收尾前的自检：不满足就返回一段打回去的 Observation，满足则返回 None。

        小模型最常见的两种"假收尾"：一次工具都没调成就凭印象编答案；以及题目明明要算
        数字（多少年、小数点后几位），它却心算。两种都拦下来让它补一轮。
        """
        if not has_result:
            return NUDGE_HINT
        if not used_compute and any(word in task for word in _COMPUTE_HINTS):
            return COMPUTE_HINT
        return None

    def _wrap_bare_input(self, action, text):
        """模型把 Action Input 写成裸值时，按该工具唯一的必填参数兜住。"""
        module = self.tools.get(action)
        if module is None:
            return None
        parameters = module.TOOL_SCHEMA["function"]["parameters"]
        required = parameters.get("required") or list(parameters["properties"])
        if len(required) != 1:
            return None
        # 小模型偶尔把整个 JSON 又套一层当参数值（"code": "{\"code\": ...}"），
        # 先试着拆出里层，免得把一段 JSON 当 Python 代码丢给沙箱。
        try:
            inner = json.loads(text)
            if isinstance(inner, dict) and required[0] in inner:
                return inner
        except Exception:
            pass
        return {required[0]: text}

    def _unwrap_double_encoded(self, action, action_input):
        """拆掉"把整个 JSON 当成参数值再套一层"的写法：{"code": "{\\"code\\": ...}"}。

        小模型偶尔会这么写，直接喂给沙箱就成了"把一段 JSON 当 Python 代码执行"，
        报出来的是 SyntaxError，模型根本看不懂自己错在哪。
        """
        module = self.tools.get(action)
        if module is None or not isinstance(action_input, dict):
            return action_input
        required = module.TOOL_SCHEMA["function"]["parameters"].get("required", [])
        for _ in range(2):                      # 最多拆两层，防止无限套娃
            if len(required) != 1:
                break
            value = action_input.get(required[0])
            if not isinstance(value, str):
                break
            try:
                inner = json.loads(value)
            except Exception:
                break
            if isinstance(inner, dict) and required[0] in inner:
                action_input = inner
            else:
                break
        return action_input

    def _call_tool(self, action, action_input) -> str:
        module = self.tools.get(action)
        if module is None:
            return (f"错误：没有名为「{action}」的工具。"
                    f"可用工具：{'、'.join(self.tools)}。请从里面选一个。")
        try:
            output = str(module.run(dict(action_input)))
        except KeyError as exc:
            # 参数写漏是小模型最常见的错法，直接把「该怎么补」写进 Observation，
            # 比只回一句 KeyError: 'dir' 有用得多。
            missing = exc.args[0] if exc.args else "?"
            required = module.TOOL_SCHEMA["function"]["parameters"].get("required", [])
            example = json.dumps({name: "" for name in required}, ensure_ascii=False)
            return (f"错误：{action} 缺少必需参数 {missing}。这个工具需要参数 {required}，"
                    f"请按 'Action Input: {example}' 的形式把参数补全后重试。")
        except Exception as exc:        # M3：工具异常变成 Observation，不让整个 run 崩
            return (f"工具 {action} 执行失败：{type(exc).__name__}: {exc}。"
                    f"请检查参数是否正确，改参数重试或换一个工具。"
                    f"（注意：目前还没有拿到任何真实结果，不要凭印象作答。）")
        if len(output) > MAX_OBSERVATION_CHARS:
            output = output[:MAX_OBSERVATION_CHARS] + "…（已截断）"
        return output

    # ---- 主循环 ---------------------------------------------------------

    def run(self, task: str) -> dict:
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": task}]
        steps = []
        final_answer = ""
        fallback_final = ""
        last_signature = None
        last_observation = ""
        last_failed = False
        has_result = False           # 是否拿到过至少一次真实的工具返回
        used_compute = False         # 是否成功调用过计算类工具

        for index in range(self.max_steps):
            raw = self.llm.chat(messages, stop=STOP_STRINGS)
            thought = _extract_thought(raw)
            action, action_input, error = _parse_action(raw)
            final = _extract_final(raw)
            if final:
                fallback_final = final

            step = {"step": index + 1, "thought": thought, "action": action,
                    "action_input": action_input, "raw": raw, "observation": None}
            steps.append(step)

            if action is None:
                if final:
                    blocker = (self._finish_blocker(task, has_result, used_compute)
                               if index + 1 < self.max_steps else None)
                    if blocker is None:
                        final_answer = final
                        self._log(f"[step {index + 1}] Final Answer: {final}")
                        break
                    observation = blocker
                    self._log(f"[step {index + 1}] 收尾自检未过，要求补一轮")
                else:
                    observation = error or FORMAT_HINT
            else:
                if isinstance(action_input, str):
                    action_input = self._wrap_bare_input(action, action_input)
                    step["action_input"] = action_input
                if action_input is None:
                    observation = error or ("错误：Action Input 不是 JSON 对象。"
                                            '请按 \'Action Input: {"参数名": "值"}\' 重写。')
                else:
                    action_input = self._unwrap_double_encoded(action, action_input)
                    step["action_input"] = action_input
                    signature = (action, json.dumps(action_input, sort_keys=True,
                                                    ensure_ascii=False))
                    if signature == last_signature and not last_failed:
                        # 上一次同样的调用是成功的，再调一次只会拿到同样的结果：模型卡住了
                        observation = (f"错误：上一步已经调用过同样的 {action}，"
                                       f"结果还是：{last_observation}。请换参数或直接给 Final Answer。")
                    else:
                        last_signature = signature
                        # 钩子存在就用钩子的返回；钩子返回 None 表示"这一步不注入"，
                        # 照常去调真工具（inject_error 为 None 时同一条路径）
                        observation = (self.inject_error(index, action, action_input)
                                       if self.inject_error is not None else None)
                        if observation is None:
                            observation = self._call_tool(action, action_input)
                        if not _looks_like_error(observation):
                            has_result = True
                            used_compute = used_compute or action in _COMPUTE_TOOLS
                        # 上一次调用成功才拦"重复调用"；失败时同样的参数再试一次是对的
                        # （工具报错往往是一次性的，实测模型确实会用原参数重试并成功）
                        last_failed = _looks_like_error(observation)

            step["observation"] = observation
            last_observation = observation
            self._log(f"[step {index + 1}] Action: {action} {action_input}\n"
                      f"  Observation: {observation[:200]}")
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"Observation: {observation}"})
        else:
            final_answer = fallback_final or f"（{self.max_steps} 步内没有得出结论）"
            self._log(f"[warn] 步数耗尽，final_answer={final_answer[:80]}")

        return {"task": task, "steps": steps, "final_answer": final_answer,
                "success": bool(final_answer) and not final_answer.startswith("（"),
                "n_steps": len(steps)}
