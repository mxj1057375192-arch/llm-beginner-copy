"""任务六 M3 + M4：主 agent loop 与 trace。

循环形状：while not done: model → tool(MCP) → observation → loop。

两个刻意的设计选择：
1. **不用模型的 JSON tool-calling，改用 ReAct 文本协议**。1.5B 级模型吐嵌套 JSON
   很不稳（尤其 content 里带换行要转义），而 "Action: / Action Input:" 这种扁平格式
   好解析也好纠错——解析失败时把格式要求当成 observation 喂回去，模型能自己改。
2. **工具执行走 MCP client**，不是直接 import 函数。agent 起一个 mcp_server 子进程，
   用 stdio 说 JSON-RPC，这样工具层是真实可替换的（换成别人写的 MCP server 也能跑）。

停机条件是显式的（M4 要求）：测试全绿 / 模型输出 Final Answer / 一轮都没调成工具，
三者之一即停；max_steps 只是兜底，不当作正常停机理由。
"""
from __future__ import annotations

import asyncio
import difflib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from .llm import ChatModel
from .skill_loader import SkillLoader

TASK_ROOT = Path(__file__).resolve().parents[1]
MCP_SERVER = Path(__file__).resolve().parent / "mcp_server.py"
SKILLS_DIR = Path(__file__).resolve().parent / "skills"

MAX_STEPS = 12
OBS_MAX_CHARS = 1500
MAX_PARSE_FAILS = 3                      # 连续解析失败这么多次就认输，别空转
SNAPSHOT_MAX_FILES = 300

SYSTEM_PROMPT = """你是一个修复代码 bug 的编程助手。你只能通过工具操作仓库，不许凭空猜测文件内容。

每一轮严格按下面三行输出（不要自己写 Observation，那是系统填的）：
Thought: 一句话说明你下一步要做什么
Action: 工具名
Action Input: JSON 参数

可用工具：
read_file   {"path": "相对路径", "start_line": 1, "max_lines": 200}
edit_file   {"path": "相对路径", "old_str": "原文片段", "new_str": "替换成"}
write_file  {"path": "相对路径", "content": "完整文件内容"}
run_tests   {}
git_diff    {}

输出示例（另一个仓库的例子，只示范格式）。**注意每一轮你只输出一组，然后停下**，
系统会执行工具、把结果作为 Observation 发给你，再轮到你输出下一组：

第 1 轮你输出的内容：
Thought: 先读一下 utils.py 里 mul 的实现
Action: read_file
Action Input: {"path": "utils.py"}

（系统执行 read_file，把文件内容作为 Observation 发给你）

第 2 轮你输出的内容：
Thought: mul 里把乘号写成了除号，改回来
Action: edit_file
Action Input: {"path": "utils.py", "old_str": "    return a / b", "new_str": "    return a * b"}

（系统执行 edit_file）

第 3 轮你输出的内容：
Thought: 跑测试确认改对了
Action: run_tests
Action Input: {}

规则（逐条遵守）：
1. **一轮只输出一组 Thought/Action/Action Input，然后立刻停下**，
   不要在一条回复里把后面几轮也写出来（你没执行工具，不可能知道结果）。
2. **没有调用过 run_tests 并看到它通过之前，绝对不要输出 Final Answer。**
3. 不要说"已修改""已修复"这类话——只有真的调用过 edit_file / write_file 才算改过。
4. 改已有代码优先用 edit_file，old_str 必须与文件里的原文逐字一致（含缩进）。
5. 第一轮应该是 read_file，先把相关文件读出来。
6. 当 run_tests 显示全部通过后，输出一行 `Final Answer: 你改了什么` 结束。
"""

_ACTION_RE = re.compile(r"Action\s*[:：]\s*([A-Za-z_][A-Za-z0-9_]*)")
# 只吃到下一个 Thought/Action 之前：模型时不时会把后面几轮一起写出来，
# 不截断的话 param 解析会把下一轮的 JSON 也吞进来。
_INPUT_RE = re.compile(r"Action\s*Input\s*[:：]\s*(.*?)(?=\n\s*(?:Thought|Action|Final)\s*[:：]|\Z)",
                       re.S)
_FINAL_RE = re.compile(r"Final\s*Answer\s*[:：]\s*(.*)", re.S)
_FENCE_RE = re.compile(r"```[a-zA-Z]*\n?|```")


# --------------------------------------------------------------- 解析

def _loads_lenient(text: str):
    """尽力把模型给的参数解析成 dict，返回 (obj, error)。

    小模型常见的毛病都在这兜住：套 ```json 围栏、JSON 后面跟废话、单引号。
    """
    raw = _FENCE_RE.sub("", text or "").strip()
    if not raw:
        return None, "Action Input 是空的"
    try:
        obj = json.loads(raw)
        return (obj, None) if isinstance(obj, dict) else (None, "参数必须是 JSON 对象")
    except Exception as exc:
        first = raw.find("{")
        if first != -1:                              # 花括号配对，截出第一个完整对象
            depth = 0
            for i, ch in enumerate(raw[first:], first):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(raw[first:i + 1])
                            if isinstance(obj, dict):
                                return obj, None
                        except Exception as exc2:
                            return None, str(exc2)
                        break
        return None, str(exc)


def parse_response(text: str) -> dict:
    """把一轮模型输出解析成 {kind: final|action|invalid, ...}。"""
    if not text or not text.strip():
        return {"kind": "invalid", "reason": "模型返回了空内容"}
    final = _FINAL_RE.search(text)
    action = _ACTION_RE.search(text)
    # 谁先出现谁作数：模型常把整条轨迹（多个 Action + Final Answer）一口气写出来，
    # 只要第一个 Action 在 Final Answer 前面，就该先执行那个 Action。
    if final and (action is None or final.start() < action.start()):
        return {"kind": "final", "answer": final.group(1).strip()[:500]}
    if not action:
        return {"kind": "invalid", "reason": "没有找到 `Action:` 行"}
    name = action.group(1).strip()
    input_match = _INPUT_RE.search(text)
    if not input_match:
        return {"kind": "invalid", "reason": f"工具 {name} 缺少 `Action Input:` 行"}
    args, err = _loads_lenient(input_match.group(1))
    if args is None:
        return {"kind": "invalid", "reason": f"Action Input 不是合法 JSON：{err}"}
    return {"kind": "action", "tool": name, "arguments": args,
            "thought": _extract_thought(text)}


def _trim_to_first_round(text: str) -> str:
    """只保留到第一个 Action Input 的 JSON 末尾，丢掉模型多写的后续轮次。"""
    match = _INPUT_RE.search(text or "")
    return (text[:match.end()].rstrip() if match else (text or ""))


def _extract_thought(text: str) -> str:
    m = re.search(r"Thought\s*[:：]\s*(.*?)(?=\n\s*(?:Action|Final)\s*[:：]|\Z)",
                  text, re.S)
    return (m.group(1).strip()[:300] if m else "")


# --------------------------------------------------------------- patch

def _snapshot(repo: Path) -> dict:
    """给 repo 里的文本文件拍个快照，供没有 git 时算 patch。"""
    shots = {}
    for path in sorted(repo.rglob("*.py"))[:SNAPSHOT_MAX_FILES]:
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        try:
            shots[str(path.relative_to(repo))] = path.read_text(
                encoding="utf-8", errors="replace")
        except OSError:
            continue
    return shots


def _diff_snapshot(before: dict, repo: Path) -> str:
    chunks = []
    for rel, old in before.items():
        try:
            new = (repo / rel).read_text(encoding="utf-8", errors="replace")
        except OSError:
            new = ""
        if new != old:
            chunks.append("".join(difflib.unified_diff(
                old.splitlines(True), new.splitlines(True),
                fromfile=f"a/{rel}", tofile=f"b/{rel}")))
    return "\n".join(chunks)


def _collect_patch(repo: Path, before: dict) -> str:
    """优先用 git diff（能覆盖新建文件之外的改动），失败就退回快照比对。"""
    if (repo / ".git").exists():
        try:
            proc = subprocess.run(["git", "diff"], cwd=str(repo), capture_output=True,
                                  text=True, encoding="utf-8", errors="replace",
                                  timeout=30, shell=False)
            if proc.returncode == 0 and proc.stdout.strip():
                return proc.stdout
        except Exception:
            pass
    return _diff_snapshot(before, repo)


def _pytest_passed(repo: Path) -> bool:
    try:
        proc = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=str(repo),
                              capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=60, shell=False)
        return proc.returncode == 0
    except Exception:
        return False


def _truncate(text: str, limit: int = OBS_MAX_CHARS) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit] + f"\n...(已截断，共 {len(text)} 字)"


# --------------------------------------------------------------- agent

class CodingAgent:
    """读懂 issue → 改代码 → 跑测试直到全绿。`run()` 返回 trace dict。"""

    def __init__(self, model: ChatModel | None = None, max_steps: int = MAX_STEPS,
                 skills_dir: str | None = None, use_skills: bool = True,
                 verbose: bool = False):
        # 自检里是 CodingAgent() 无参构造，所以模型端点只能从环境变量/默认值来。
        self.model = model or ChatModel()
        self.max_steps = max_steps
        self.use_skills = use_skills
        self.verbose = verbose
        self.skills = SkillLoader(str(skills_dir or SKILLS_DIR))

    # ---- 对外入口（同步，自检直接调）

    def run(self, repo_path: str, issue: str) -> dict:
        return asyncio.run(self._run(repo_path, issue))

    # ---- 内部

    def _build_system(self, issue: str) -> tuple[str, list]:
        """命中的 Skill 才把正文塞进 system prompt（渐进式披露）。"""
        system = SYSTEM_PROMPT
        used = []
        if self.use_skills:
            for meta in self.skills.match(issue, top_k=1):
                try:
                    system += f"\n\n## 参考流程：{meta['name']}\n{self.skills.load(meta['name'])}"
                    used.append(meta["name"])
                except KeyError:
                    continue
        return system, used

    async def _run(self, repo_path: str, issue: str) -> dict:
        repo = Path(repo_path).resolve()
        started = time.time()
        trace = {
            "repo_path": str(repo), "issue": issue,
            "steps": [], "patch": "", "tests_passed": False,
            "done_reason": None, "skills_used": [], "model_calls": 0,
            "total_tokens": 0, "elapsed_s": 0.0,
        }
        if not repo.is_dir():
            trace["done_reason"] = f"repo 不存在：{repo}"
            return trace

        before = _snapshot(repo)
        system, trace["skills_used"] = self._build_system(issue)

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        env = dict(os.environ)
        env["AGENT_REPO_ROOT"] = str(repo)          # 工具层的安全边界就是它
        params = StdioServerParameters(command=sys.executable,
                                       args=[str(MCP_SERVER)], env=env)

        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": f"仓库里有一个待修的问题：\n{issue}"}]

        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                parse_fails = 0
                final_rejects = 0
                last_run_tests_passed = None      # 最近一次 run_tests 的结果
                for index in range(self.max_steps):
                    payload = None                # 本轮工具的原始返回值（结构化）
                    reply = self.model.chat(messages, stop=["\nObservation"])
                    parsed = parse_response(reply)

                    if parsed["kind"] == "final":
                        # 小模型很爱在第一步就编一句"已修复"逃逸，所以对 Final Answer
                        # 加两道守卫：一次工具没调过、或上次 run_tests 还是红的，都不认。
                        if not trace["steps"] and final_rejects < 2:
                            final_rejects += 1
                            nudge = ("你还没有读过任何文件，也没有调用过任何工具，"
                                     "不能凭空断定改好了。请先调用 read_file。")
                        elif last_run_tests_passed is False and final_rejects < 2:
                            final_rejects += 1
                            nudge = ("上一次 run_tests 仍然是失败的，不能就此收工。"
                                     "请根据失败输出继续修改，直到 run_tests 通过。")
                        else:
                            trace["done_reason"] = f"模型声明完成：{parsed['answer'][:120]}"
                            break
                        observation = nudge
                        parsed = {"kind": "invalid", "reason": nudge, "thought": ""}
                        parse_fails = 0               # 这是守卫不是模型的锅，别计入
                    elif parsed["kind"] == "invalid":
                        parse_fails += 1
                        observation = (
                            f"格式错误：{parsed['reason']}。请严格按 "
                            "`Thought: ...` / `Action: 工具名` / `Action Input: {...}` 三行输出。")
                        if parse_fails >= MAX_PARSE_FAILS:
                            trace["steps"].append({"index": index, "thought": "",
                                                   "tool_call": None,
                                                   "observation": observation,
                                                   "error": "连续解析失败"})
                            trace["done_reason"] = "连续解析失败，提前停机"
                            break
                    else:
                        parse_fails = 0
                        observation, payload = await self._dispatch(
                            session, parsed["tool"], parsed["arguments"])
                        observation = _truncate(observation)
                        if parsed["tool"] == "run_tests" and payload is not None:
                            last_run_tests_passed = payload.get("passed")

                    if self.verbose:                 # 实时看它每步在干嘛
                        call = parsed.get("tool") or "—"
                        print(f"[step {index}] {call} :: {parsed.get('thought', '')[:60]}"
                              f"\n    obs: {observation[:180]}", file=sys.stderr, flush=True)

                    trace["steps"].append({
                        "index": index,
                        "thought": parsed.get("thought", ""),
                        "tool_call": (None if parsed["kind"] == "invalid"
                                      else {"name": parsed["tool"],
                                            "arguments": parsed["arguments"]}),
                        "observation": observation[:OBS_MAX_CHARS],
                        "raw": reply[:600],
                    })
                    # 只把第一轮写回历史：模型多写的那几轮还没执行，留着会污染后续推理
                    messages.append({"role": "assistant",
                                     "content": _trim_to_first_round(reply)})
                    messages.append({"role": "user", "content": f"Observation: {observation}"})

                    # 显式停机信号之一：测试真的全绿了，不必等模型自己喊停
                    if payload is not None and payload.get("passed") is True:
                        trace["tests_passed"] = True
                        trace["done_reason"] = "run_tests 全绿"
                        break
                else:
                    trace["done_reason"] = f"达到步数上限 {self.max_steps}（兜底，非正常停机）"

        # 收尾以真实 pytest 为准，不看模型的自述
        trace["tests_passed"] = _pytest_passed(repo)
        trace["patch"] = _collect_patch(repo, before)
        trace["model_calls"] = getattr(self.model, "calls", 0)
        trace["total_tokens"] = getattr(self.model, "total_tokens", 0)
        trace["elapsed_s"] = round(time.time() - started, 1)
        return trace

    async def _dispatch(self, session, tool: str, arguments: dict) -> tuple[str, dict | None]:
        """把一次工具调用发给 MCP server，返回 (给模型看的文本, 解析后的结果)。

        返回原始 dict 是为了让停机判断基于结构化字段（如 run_tests 的 passed），
        而不是去 grep 文本。
        """
        try:
            result = await session.call_tool(tool, arguments)
        except Exception as e:
            return json.dumps({"ok": False, "error": f"MCP 调用失败：{e}"},
                              ensure_ascii=False), None
        text = "".join(getattr(block, "text", "") for block in (result.content or []))
        text = text or "(工具没有返回内容)"
        try:
            payload = json.loads(text)
        except Exception:
            payload = None
        return text, (payload if isinstance(payload, dict) else None)
