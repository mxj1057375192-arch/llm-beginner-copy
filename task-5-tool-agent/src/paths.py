"""任务五的路径常量，供工具与 agent 共用，避免每个文件各写一份相对路径。"""
from __future__ import annotations

import os
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = TASK_ROOT / "data"
FIXTURE_DIR = DATA_DIR / "agent-fixtures"     # file_search 的检索夹具
TASKS_JSON = DATA_DIR / "tasks.json"          # 10 题评测集
MODELS_DIR = TASK_ROOT / "models"

# file_search 只允许在这个根内检索：dir 参数 resolve 之后必须落在里面，
# 否则传 "../../" 就能读到工作区外的文件。
FILE_SEARCH_ROOTS = (TASK_ROOT,)

# agent 用的本地模型，按 AGENT_MODEL 环境变量 → 本任务 models/ → 任务四已下载的
# 0.5B 顺序找。本机没有 GPU，0.5B 是唯一能在 CPU 上跑完整 10 题的量级。
_MODEL_CANDIDATES = (
    MODELS_DIR / "Qwen2.5-0.5B-Instruct",
    MODELS_DIR / "Qwen2.5-1.5B-Instruct",
    Path(r"D:\新生任务\llm-beginner\task-4-rag\models\Qwen2.5-0.5B-Instruct"),
)


def resolve_model_dir() -> Path:
    """找一个能用的本地模型目录。"""
    env = os.environ.get("AGENT_MODEL")
    if env:
        return Path(env)
    for path in _MODEL_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "找不到本地 Qwen2.5-Instruct 权重；用 AGENT_MODEL 环境变量指定模型目录")
