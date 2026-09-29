"""任务五的四个工具：calculator / python_sandbox / file_search / wiki。

每个模块导出 TOOL_SCHEMA（OpenAI function calling 格式）与 run(args) -> str；
自检 eval/run.py 按模块名导入，并以固定参数键调用 run：
calculator → {"expression"}、python_sandbox → {"code"}、
file_search → {"pattern", "dir"}、wiki → {"query"}。
"""
from . import calculator, file_search, python_sandbox, wiki

__all__ = ["calculator", "file_search", "python_sandbox", "wiki"]
