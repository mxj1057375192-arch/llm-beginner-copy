"""任务六 M1：手写 MCP server，暴露 5 个原子工具。

两条路都要通：
1. `python src/mcp_server.py` —— 以 stdio 起一个标准 MCP server，供 MCP client 调用
2. `from src.mcp_server import list_tools` —— 模块顶层导出工具清单，供自检枚举

工具表 `TOOL_SPECS` 是唯一事实来源：`list_tools()` 直接返回它，下面 `@_mcp.tool()`
的包装函数按同样的参数签名注册（签名要和 specs 里的 properties 保持一致）。
不用 `MCPServer` 自己的 schema 生成，是为了 `list_tools()` 在没装 mcp 的环境里
也能同步返回——自检就是直接 import 它的。

安全约束（README 逐条要求）：
- 路径先 resolve 再校验落在 repo 内，挡掉 `..` 越界与绝对路径
- subprocess 一律 list 形式 + 限定 cwd + 超时，不拼 shell 串
- git 只允许 diff / apply，禁用 reset --hard / clean -fd 这类会丢改动的命令
- 任何异常都转成结构化 error 返回，不让 server crash

目标仓库根由环境变量 `AGENT_REPO_ROOT` 指定（agent 起 server 子进程时注入），
不放进工具 schema —— 模型不必知道也不需要编造这个路径。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

try:
    from mcp.server.mcpserver import MCPServer
except ImportError:                    # 没装 mcp 时仍可 import list_tools()
    MCPServer = None                   # type: ignore[assignment]

PYTEST_TIMEOUT = 60
GIT_TIMEOUT = 30
READ_MAX_CHARS = 20000


# ---------------------------------------------------------------- 基础设施

def _ok(**kw):
    return {"ok": True, **kw}


def _err(msg):
    return {"ok": False, "error": str(msg)}


def _repo_root() -> Path:
    root = os.environ.get("AGENT_REPO_ROOT")
    if root:
        return Path(root).resolve()
    return (Path(__file__).resolve().parents[1] / "data" / "toy-repo").resolve()


def _resolve_in_repo(rel: str) -> Path:
    """把 rel 解析成 repo 内的绝对路径，越界就抛 ValueError。

    resolve() 会展开 `..` 和符号链接，所以必须在 resolve *之后* 再比对前缀，
    否则 `../../secret` 会绕过检查。
    """
    root = _repo_root()
    if not root.is_dir():
        raise ValueError(f"repo 目录不存在：{root}")
    candidate = Path(rel)
    if candidate.is_absolute():
        raise ValueError(f"只接受相对路径，收到绝对路径：{rel}")
    target = (root / candidate).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"路径越界，拒绝访问 repo 外的文件：{rel}")
    return target


def _run(cmd: list, cwd: Path, timeout: int) -> dict:
    """跑子进程，统一收口超时与异常。绝不用 shell=True。"""
    try:
        proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout,
                              shell=False)
    except subprocess.TimeoutExpired:
        return _err(f"命令超时（>{timeout}s）：{' '.join(cmd)}")
    except FileNotFoundError as e:
        return _err(f"命令不存在：{e}")
    return _ok(returncode=proc.returncode,
               stdout=(proc.stdout or "")[-4000:],
               stderr=(proc.stderr or "")[-2000:])


# ---------------------------------------------------------------- 工具实现

def op_read_file(path: str, start_line: int = 1, max_lines: int = 400) -> dict:
    try:
        target = _resolve_in_repo(path)
        if not target.is_file():
            return _err(f"不是文件或不存在：{path}")
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(1, int(start_line))
        chunk = lines[start - 1:start - 1 + int(max_lines)]
        body = "\n".join(f"{start + i:>4} | {line}" for i, line in enumerate(chunk))
        return _ok(path=path, total_lines=len(lines), content=body[:READ_MAX_CHARS])
    except Exception as e:
        return _err(e)


def op_write_file(path: str, content: str) -> dict:
    try:
        target = _resolve_in_repo(path)
        if target.is_dir():
            return _err(f"目标是目录，不能写入：{path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="\n")
        return _ok(path=path, bytes=len(content.encode("utf-8")))
    except Exception as e:
        return _err(e)


def op_edit_file(path: str, old_str: str, new_str: str, count: int = 1) -> dict:
    """把文件里的一段原文替换成新内容。

    old_str 必须唯一命中——模型很容易只给一行 `return a - b` 这种到处都有的片段，
    真让它替换就可能改错地方，所以命中不唯一时直接拒绝，让它加长上下文再来。
    """
    try:
        target = _resolve_in_repo(path)
        if not target.is_file():
            return _err(f"文件不存在：{path}")
        text = target.read_text(encoding="utf-8", errors="replace")
        hits = text.count(old_str)
        if hits == 0:
            return _err(f"在 {path} 里找不到 old_str；先用 read_file 核对原文（注意缩进和空格）")
        if hits > 1 and count != 0:
            return _err(f"old_str 在 {path} 里出现了 {hits} 次，不唯一；"
                        "请连同上下文一起给，让它只匹配一处")
        target.write_text(text.replace(old_str, new_str, count),
                          encoding="utf-8", newline="\n")
        return _ok(path=path, replaced=min(count, hits) if count else hits)
    except Exception as e:
        return _err(e)


def op_run_tests(timeout: int = PYTEST_TIMEOUT) -> dict:
    try:
        root = _repo_root()
        if not root.is_dir():
            return _err(f"repo 目录不存在：{root}")
        result = _run([sys.executable, "-m", "pytest", "-q"], root,
                      min(int(timeout), PYTEST_TIMEOUT))
        if result.get("ok"):
            result["passed"] = result["returncode"] == 0
        return result
    except Exception as e:
        return _err(e)


def op_git_diff() -> dict:
    try:
        root = _repo_root()
        if not (root / ".git").exists():
            return _err("不是 git 仓库（toy-repo 未配 git 身份时属正常，可跳过）")
        result = _run(["git", "diff"], root, GIT_TIMEOUT)
        if result.get("ok") and not result["stdout"].strip():
            result["stdout"] = "(工作区无改动)"
        return result
    except Exception as e:
        return _err(e)


def op_git_apply(patch: str) -> dict:
    """先用 `git apply --check` 预检，失败了就不落盘（不会留下半截改动）。"""
    try:
        root = _repo_root()
        if not (root / ".git").exists():
            return _err("不是 git 仓库，无法 apply")
        check = subprocess.run(["git", "apply", "--check", "-"], cwd=str(root),
                               input=patch, capture_output=True, text=True,
                               encoding="utf-8", errors="replace",
                               timeout=GIT_TIMEOUT, shell=False)
        if check.returncode != 0:
            return _err(f"patch 预检失败，未落盘：{(check.stderr or '')[-500:]}")
        applied = subprocess.run(["git", "apply", "-"], cwd=str(root), input=patch,
                                 capture_output=True, text=True, encoding="utf-8",
                                 errors="replace", timeout=GIT_TIMEOUT, shell=False)
        if applied.returncode != 0:
            return _err(f"apply 失败：{(applied.stderr or '')[-500:]}")
        return _ok(applied=True)
    except Exception as e:
        return _err(e)


# ------------------------------------------------------- 工具表（唯一事实来源）

TOOL_SPECS = [
    {
        "name": "read_file",
        "description": "读取仓库内某个文件的带行号内容。想看某段代码时先用它。",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "相对仓库根的文件路径"},
                "start_line": {"type": "integer", "description": "起始行，默认 1"},
                "max_lines": {"type": "integer", "description": "最多读多少行，默认 400"},
            },
            "required": ["path"],
        },
        "fn": op_read_file,
    },
    {
        "name": "write_file",
        "description": "把完整内容写入仓库内某个文件（覆盖原文件）。改 bug 用这个。",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "相对仓库根的文件路径"},
                "content": {"type": "string", "description": "写入的完整文件内容"},
            },
            "required": ["path", "content"],
        },
        "fn": op_write_file,
    },
    {
        "name": "edit_file",
        "description": "把文件里的一段原文替换成新内容（old_str 必须唯一命中）。改已有代码优先用它。",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "相对仓库根的文件路径"},
                "old_str": {"type": "string", "description": "要被替换的原文片段，须逐字一致"},
                "new_str": {"type": "string", "description": "替换成的内容"},
            },
            "required": ["path", "old_str", "new_str"],
        },
        "fn": op_edit_file,
    },
    {
        "name": "run_tests",
        "description": "在仓库里执行 python -m pytest -q，返回是否全绿与失败输出。",
        "input_schema": {
            "type": "object",
            "properties": {"timeout": {"type": "integer", "description": "超时秒数，默认 60"}},
            "required": [],
        },
        "fn": op_run_tests,
    },
    {
        "name": "git_diff",
        "description": "查看当前工作区相对 HEAD 的改动，用来确认自己改了什么。",
        "input_schema": {"type": "object", "properties": {}, "required": []},
        "fn": op_git_diff,
    },
    {
        "name": "git_apply",
        "description": "应用一个 unified diff 格式的补丁；预检失败不会落盘。",
        "input_schema": {
            "type": "object",
            "properties": {"patch": {"type": "string", "description": "unified diff 文本"}},
            "required": ["patch"],
        },
        "fn": op_git_apply,
    },
]

_BY_NAME = {spec["name"]: spec for spec in TOOL_SPECS}


def list_tools() -> list[dict]:
    """自检入口：同步返回工具清单（name / description / input_schema）。"""
    return [{"name": s["name"], "description": s["description"],
             "input_schema": s["input_schema"]} for s in TOOL_SPECS]


def call_tool(name: str, arguments: dict) -> dict:
    """统一调用口：异常一律转成结构化 error，绝不让 server crash。"""
    spec = _BY_NAME.get(name)
    if spec is None:
        return _err(f"未知工具：{name}（可用：{', '.join(_BY_NAME)}）")
    try:
        return spec["fn"](**(arguments or {}))
    except TypeError as e:                       # 参数名错/缺参
        return _err(f"参数不匹配：{e}")
    except Exception as e:
        return _err(f"{type(e).__name__}: {e}")


# ------------------------------------------------------------ MCP 注册

_mcp = MCPServer("coding-agent-tools") if MCPServer else None


def _dump(result: dict) -> str:
    return json.dumps(result, ensure_ascii=False)


if _mcp is not None:                            # pragma: no cover - 依赖 mcp 包

    @_mcp.tool()
    def read_file(path: str, start_line: int = 1, max_lines: int = 400) -> str:
        """读取仓库内某个文件的带行号内容。

        Args:
            path: 相对仓库根的文件路径。
            start_line: 起始行号，默认 1。
            max_lines: 最多读取的行数，默认 400。
        """
        return _dump(call_tool("read_file", {"path": path, "start_line": start_line,
                                             "max_lines": max_lines}))

    @_mcp.tool()
    def write_file(path: str, content: str) -> str:
        """把完整内容写入仓库内某个文件（覆盖原文件）。

        Args:
            path: 相对仓库根的文件路径。
            content: 要写入的完整文件内容。
        """
        return _dump(call_tool("write_file", {"path": path, "content": content}))

    @_mcp.tool()
    def edit_file(path: str, old_str: str, new_str: str) -> str:
        """把文件里的一段原文替换成新内容（old_str 必须唯一命中）。

        Args:
            path: 相对仓库根的文件路径。
            old_str: 要被替换的原文片段，必须与文件内容逐字一致。
            new_str: 替换成的内容。
        """
        return _dump(call_tool("edit_file", {"path": path, "old_str": old_str,
                                             "new_str": new_str}))

    @_mcp.tool()
    def run_tests(timeout: int = PYTEST_TIMEOUT) -> str:
        """在仓库里执行 python -m pytest -q，返回是否全绿与失败输出。"""
        return _dump(call_tool("run_tests", {"timeout": timeout}))

    @_mcp.tool()
    def git_diff() -> str:
        """查看当前工作区相对 HEAD 的改动。"""
        return _dump(call_tool("git_diff", {}))

    @_mcp.tool()
    def git_apply(patch: str) -> str:
        """应用一个 unified diff 补丁；预检失败不会落盘。

        Args:
            patch: unified diff 文本。
        """
        return _dump(call_tool("git_apply", {"patch": patch}))


def main() -> int:
    """stdio 服务入口。stdout 被 JSON-RPC 独占，调试输出必须走 stderr。"""
    if _mcp is None:
        print("[mcp_server] 未安装 mcp 包，无法起服务：pip install mcp", file=sys.stderr)
        return 1
    print(f"[mcp_server] stdio 就绪，repo={_repo_root()}，"
          f"工具 {len(TOOL_SPECS)} 个：{', '.join(s['name'] for s in TOOL_SPECS)}",
          file=sys.stderr)
    _mcp.run(transport="stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
