"""两阶段检索的第二阶段：bge-reranker 精排。

它吃的是 `[query, doc]` **文本对**，输出一个相关性分数（交叉编码器，query 和 doc
一起过模型，比双塔的向量内积准得多，代价是不能预计算、必须在线算）。别把它当成
第二个 embedding 模型用。

用法上是「召回多、精排少」：向量召回 20 条左右，rerank 后只留 top 3-5 送进生成，
召回数给太小则 rerank 救不回来。
"""
from __future__ import annotations

from typing import List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .paths import RERANK_MODEL


class Reranker:
    """bge-reranker-base 交叉编码器。"""

    def __init__(self, model_dir=None, device: str = "cpu",
                 batch_size: int = 16, max_length: int = 512):
        self.model_dir = str(model_dir or RERANK_MODEL)
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model = self.model.to(device).eval()

    @torch.no_grad()
    def score(self, query: str, docs: List[str]) -> List[float]:
        """给每个 doc 打一个 [0,1] 的相关性分数。"""
        if not docs:
            return []
        scores = []
        for i in range(0, len(docs), self.batch_size):
            batch = docs[i:i + self.batch_size]
            enc = self.tokenizer([query] * len(batch), batch, padding=True, truncation=True,
                                 max_length=self.max_length,
                                 return_tensors="pt").to(self.device)
            logits = self.model(**enc).logits.view(-1)
            scores.extend(torch.sigmoid(logits).cpu().tolist())
        return scores

    def rerank(self, query: str, docs: List[dict], top_n: int = None) -> List[dict]:
        """按 rerank 分数重排；每项附 `rerank_score`。top_n 为 None 时全返回。"""
        if not docs:
            return []
        scores = self.score(query, [d["text"] for d in docs])
        ranked = [{**d, "rerank_score": round(s, 4)} for d, s in zip(docs, scores)]
        ranked.sort(key=lambda d: d["rerank_score"], reverse=True)
        return ranked[:top_n] if top_n else ranked


_DEFAULT: Reranker = None


def get_reranker(**kwargs) -> Reranker:
    """进程内复用的默认 Reranker。"""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Reranker(**kwargs)
    return _DEFAULT


if __name__ == "__main__":
    import sys

    from .retriever import get_retriever

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    query = " ".join(sys.argv[1:]) or "什么是注意力机制？"

    hits = get_retriever().retrieve(query, k=10)
    print("--- 只用向量召回 ---")
    for i, r in enumerate(hits[:3], 1):
        print(f"[{i}] {r['score']}  {r['text'][:60]}...")

    print("\n--- 加 reranker 之后 ---")
    for i, r in enumerate(get_reranker().rerank(query, hits, top_n=3), 1):
        print(f"[{i}] rerank={r['rerank_score']}  向量={r['score']}  {r['text'][:60]}...")
