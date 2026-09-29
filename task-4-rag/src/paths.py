"""任务四的路径常量，供各模块共用，避免每个文件各写一份相对路径。"""
from __future__ import annotations

from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = TASK_ROOT / "data"
MODELS_DIR = TASK_ROOT / "models"
INDEX_DIR = TASK_ROOT / "index"          # 建好的 FAISS 索引 + chunk 落盘处

KB_PDF = DATA_DIR / "kb.pdf"             # 知识库：《神经网络与深度学习（第二版）》
GOLD_QA = DATA_DIR / "gold_qa.jsonl"     # 30 条评测题

EMBED_MODEL = MODELS_DIR / "bge-small-zh-v1.5"
RERANK_MODEL = MODELS_DIR / "bge-reranker-base"
GEN_MODEL = MODELS_DIR / "Qwen2.5-0.5B-Instruct"
