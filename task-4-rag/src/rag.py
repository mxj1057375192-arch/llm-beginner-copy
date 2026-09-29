"""端到端 RAG：召回 -> 精排 -> 生成（M4）。

`answer(query)` 返回 `{"answer": str, "sources": [dict]}`。

链路是两阶段检索的标准形态：向量召回 RETRIEVE_K 条（要远大于最终用量，rerank 才有
得挑）-> reranker 精排留 RERANK_TOP_N 条 -> 去重截断后拼进 prompt。召回阶段 k 给太小，
rerank 救不回来。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List

from .generator import build_prompt, get_generator
from .paths import RERANK_MODEL
from .reranker import get_reranker
from .retriever import get_retriever

RETRIEVE_K = 20           # 向量召回条数
RERANK_TOP_N = 4          # 精排后送进生成条数
MAX_CHUNK_CHARS = 900     # 单个 chunk 截断长度
MAX_CONTEXT_CHARS = 3200  # 上下文总长度上限


def _dedupe(results: List[dict]) -> List[dict]:
    """去掉近重复片段，保留原始顺序。

    切分时留了 overlap，相邻块会大量互相包含；只判「完全相同」是不够的——
    实测常见 4 条上下文里两条是同一段内容，等于白占了槽位。这里把互为子串的也算重复。
    """
    kept, keys = [], []
    for r in results:
        key = re.sub(r"\s+", "", r.get("text", ""))
        if not key or any(key in k or k in key for k in keys):
            continue
        kept.append(r)
        keys.append(key)
    return kept


def _fit_context(results: List[dict]) -> List[dict]:
    """截断到 prompt 预算内：单块限长 + 总量限长。"""
    kept, total = [], 0
    for r in results:
        text = r["text"][:MAX_CHUNK_CHARS]
        if kept and total + len(text) > MAX_CONTEXT_CHARS:
            break
        kept.append({**r, "text": text})
        total += len(text)
    return kept


def _rerank(query: str, hits: List[dict]) -> List[dict]:
    if not hits:
        return hits
    if not Path(RERANK_MODEL).exists():
        # reranker 没下好时不硬报错：退回纯向量召回，但要让人看见这一步被跳过了
        print(f"[rag] 未找到 {RERANK_MODEL.name}，跳过精排，直接取向量召回 top-{RERANK_TOP_N}")
        return hits[:RERANK_TOP_N]
    return get_reranker().rerank(query, hits, top_n=RERANK_TOP_N)


def answer(query: str) -> dict:
    """检索 + 生成，返回 {"answer": 答案文本, "sources": 用到的片段}。"""
    hits = get_retriever().retrieve(query, k=RETRIEVE_K)
    hits = _dedupe(hits)          # 先删近重复，rerank 才有多样化的候选可挑
    hits = _rerank(query, hits)
    contexts = _fit_context(hits)
    text = get_generator().generate(build_prompt(query, contexts))
    return {"answer": text, "sources": contexts}


if __name__ == "__main__":
    import json
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    question = " ".join(sys.argv[1:]) or "什么是注意力机制？"
    result = answer(question)

    print(f"\n问题：{question}\n")
    print(f"回答：\n{result['answer']}\n")
    print("--- 依据的片段 ---")
    for i, s in enumerate(result["sources"], 1):
        print(f"[{i}] cosine={s.get('score')}  rerank={s.get('rerank_score')}  {s.get('source')}")
        print(f"    {json.dumps(s['text'][:100], ensure_ascii=False)}...")
