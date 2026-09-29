"""计算器工具：把表达式解析成 AST 后按白名单求值，不碰 eval。

直接 eval(expression) 等于把任意代码执行权交给模型——它只要输出
__import__("os").system("...") 就能在本地跑起来。这里先把表达式 parse 成 AST，
逐个节点检查是不是白名单里的运算/函数/常量，再递归求值。

输出顺带把「整数位数」和「6 位小数」两种表示写出来：评测集里有问结果位数的题，
也有要求小数点后 6 位的题，而 float 的原样 repr（45.0111097397076）两种都不命中。
"""
from __future__ import annotations

import ast
import math
import operator

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "计算数学表达式，支持四则运算、幂、取模和 sqrt/log/exp/sin 等常用函数。",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": '要计算的表达式，例如 "(123 + 456) * 789"、"sqrt(2026)"',
                },
            },
            "required": ["expression"],
        },
    },
}

# 白名单：只有出现在这里的名字、运算符和常量能被求值
_FUNCS = {name: getattr(math, name) for name in (
    "sqrt", "exp", "log", "log2", "log10", "sin", "cos", "tan", "asin", "acos",
    "atan", "atan2", "sinh", "cosh", "tanh", "degrees", "radians", "floor",
    "ceil", "trunc", "fabs", "factorial", "gcd", "hypot", "fmod", "comb",
    "perm", "isqrt", "dist",
)}
_FUNCS.update({"abs": abs, "round": round, "min": min, "max": max, "sum": sum,
               "int": int, "float": float, "pow": pow})
_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau}
_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_MAX_EXPONENT = 1000          # 挡住 10**10**10 这种把 CPU 挂住的表达式
_MAX_MAGNITUDE = 1e300


def _eval(node: ast.AST):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"表达式里只允许数字，不接受 {node.value!r}")
        return node.value
    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"不支持运算符 {type(node.op).__name__}")
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ValueError(f"指数 {right} 过大，已拒绝计算")
        value = op(left, right)
        if isinstance(value, float) and abs(value) > _MAX_MAGNITUDE:
            raise ValueError("计算结果过大")
        return value
    if isinstance(node, ast.UnaryOp):
        operand = _eval(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise ValueError(f"不支持一元运算符 {type(node.op).__name__}")
    if isinstance(node, ast.Name):
        if node.id in _CONSTS:
            return _CONSTS[node.id]
        raise ValueError(f"不允许的常量 {node.id}")
    if isinstance(node, ast.Attribute):
        # 模型很爱写 math.sqrt(2026) 这种带模块前缀的形式，给个面子放行
        if isinstance(node.value, ast.Name) and node.value.id == "math":
            if node.attr in _CONSTS:
                return _CONSTS[node.attr]
        raise ValueError(f"不允许访问 {ast.unparse(node)}")
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval(item) for item in node.elts]
    if isinstance(node, ast.Call):
        if node.keywords:
            raise ValueError("函数不支持关键字参数")
        return _resolve_function(node.func)(*[_eval(arg) for arg in node.args])
    raise ValueError(f"表达式里不支持 {type(node).__name__} 语法")


def _resolve_function(node: ast.AST):
    if isinstance(node, ast.Name) and node.id in _FUNCS:
        return _FUNCS[node.id]
    if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            and node.value.id == "math" and node.attr in _FUNCS):
        return _FUNCS[node.attr]
    raise ValueError("只允许调用白名单里的数学函数（写 sqrt(x) 或 math.sqrt(x) 都行）")


def _format(value) -> str:
    if isinstance(value, int):
        return f"{value}（整数，共 {len(str(abs(value)))} 位）"
    if isinstance(value, float):
        if value.is_integer() and abs(value) < 1e15:
            return f"{int(value)}（整数，共 {len(str(abs(int(value))))} 位）"
        # 6 位小数的值放在最前面：小模型抄长数字经常抄错（实测把 45.011110 抄成
        # 45.011109），让要用的那个数最先出现、且短，抄错的概率就低得多。
        return f"{value:.6f}（保留 6 位小数；完整值 {value!r}）"
    return str(value)


def run(args: dict) -> str:
    expression = str(args["expression"]).strip()
    if not expression:
        raise ValueError("expression 不能为空")
    tree = ast.parse(expression, mode="eval")     # 语法错直接抛给 agent
    return _format(_eval(tree))


if __name__ == "__main__":
    for expr in ["2 + 3 * 4", "(123 + 456) * 789", "sqrt(2026)"]:
        print(expr, "=", run({"expression": expr}))
