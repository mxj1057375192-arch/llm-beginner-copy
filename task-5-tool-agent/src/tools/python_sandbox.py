"""受限执行的 Python 沙箱：白名单 builtins / import、行数预算、stdout 捕获。

⚠️ 只是教学级防护，别拿去跑真正不可信的代码：
   - 白名单能挡掉 import os 和 open()，但挡不住 () .__class__.__bases__ 这类反射
     路径——它们不需要任何 builtins 就能拿到任意模块；
   - sys.settrace 的行数预算能掐断死循环，但挡不住一次分配几个 G 的内存。
真要隔离得子上子进程 + resource 限制 / RestrictedPython / 容器。

行数预算的做法：sys.settrace 注册一个回调，每执行一行代码计数一次，超预算就抛
异常。Windows 上没有 signal.SIGALRM，线程也没法从外部杀掉，这个法子是少数能
在没有子进程的情况下真正掐断 `while True: pass` 的手段。
"""
from __future__ import annotations

import ast
import builtins
import contextlib
import io
import sys
import time

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "python_sandbox",
        "description": "在受限环境里执行一段 Python 代码并返回它的 print 输出。"
                       "凡是「用 Python 算/写个函数测一下」的活儿都用它。"
                       "注意：结果要用 print(...) 打印；print() 本身没有返回值，"
                       "不要把 print(...) 当表达式继续参与运算。",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": '要执行的 Python 代码，用 print(...) 打印结果，例如 '
                                   '"print(sum(range(10)))"',
                },
            },
            "required": ["code"],
        },
    },
}

_ALLOWED_MODULES = {
    "math", "statistics", "itertools", "functools", "random", "decimal",
    "fractions", "collections", "heapq", "bisect", "re", "string", "json",
    "textwrap", "unicodedata", "operator",
}
_ALLOWED_BUILTIN_NAMES = (
    "abs all any bin bool bytearray bytes chr complex dict divmod enumerate "
    "filter float format frozenset hasattr hex int isinstance iter len list map "
    "max min next object oct ord pow print range repr reversed round set slice "
    "sorted str sum tuple type zip __build_class__ __name__ "
    "Exception ValueError TypeError ZeroDivisionError KeyError IndexError "
    "StopIteration RuntimeError ArithmeticError AttributeError AssertionError "
    "NotImplementedError OverflowError"
).split()

_MAX_LINES = 200_000          # 计数上限，够跑正常的评测代码，又能掐断死循环
_MAX_SECONDS = 8.0
_MAX_OUTPUT = 4000


class _BudgetExceeded(RuntimeError):
    """超出执行预算（行数或墙钟）时抛出，用来中断沙箱里的代码。"""


class _Budget:
    """sys.settrace 回调：数执行行数，超预算抛异常。"""

    def __init__(self, max_lines: int = _MAX_LINES, max_seconds: float = _MAX_SECONDS):
        self.lines = 0
        self.deadline = time.monotonic() + max_seconds

    def __call__(self, frame, event, arg):
        self.lines += 1
        if self.lines > _MAX_LINES:
            raise _BudgetExceeded(f"代码执行超过 {_MAX_LINES} 行，已中断（疑似死循环）")
        if self.lines % 2048 == 0 and time.monotonic() > self.deadline:
            raise _BudgetExceeded(f"代码执行超过 {_MAX_SECONDS:.0f} 秒，已中断")
        return self


def _safe_import(name, *args, **kwargs):
    if name.split(".")[0] not in _ALLOWED_MODULES:
        raise ImportError(f"沙箱不允许 import {name}（只放行 {sorted(_ALLOWED_MODULES)}）")
    return builtins.__import__(name, *args, **kwargs)


_SAFE_BUILTINS = {n: getattr(builtins, n) for n in _ALLOWED_BUILTIN_NAMES
                  if hasattr(builtins, n)}
_SAFE_BUILTINS["__import__"] = _safe_import


def run(args: dict) -> str:
    code = str(args["code"])
    tree = ast.parse(code, mode="exec")           # SyntaxError 直接抛给 agent

    # 最后一行如果是表达式，按 REPL 的习惯把它的值也打印出来——模型经常只写
    # `sum(range(10))` 而不写 print，不该因此报「没有输出」。
    tail = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        tail = ast.Expression(tree.body[-1].value)
        tree.body = tree.body[:-1]

    scope = {"__name__": "__main__", "__builtins__": _SAFE_BUILTINS}
    buffer = io.StringIO()
    budget = _Budget()
    sys.settrace(budget)
    try:
        with contextlib.redirect_stdout(buffer):
            exec(compile(tree, "<python_sandbox>", "exec"), scope)
            if tail is not None:
                value = eval(compile(tail, "<python_sandbox>", "eval"), scope)
                if value is not None:
                    print(repr(value))
    finally:
        sys.settrace(None)

    output = buffer.getvalue().strip()
    if not output:
        # 小模型最常见的错法：只写一个 def 就交差，既没调用也没 print。这里把"该怎么补"
        # 直接写在返回值里（并以"需要重试："开头，agent 会据此认定还没拿到真实结果）。
        return ("需要重试：代码执行完了但没有任何输出。只写函数定义不会打印任何东西——"
                "请在代码末尾把要算的东西 print 出来（如 print(f(100))），"
                "并让每条结果带上它的对象名（如 print('level', f('level'))）。")
    if len(output) > _MAX_OUTPUT:
        output = output[:_MAX_OUTPUT] + "\n…（输出过长已截断）"
    return output


if __name__ == "__main__":
    print(run({"code": "print(sum(range(10)))"}))
    print(run({"code": "print(sum(n for n in range(2, 100) if all(n % d for d in range(2, int(n ** 0.5) + 1))))"}))
