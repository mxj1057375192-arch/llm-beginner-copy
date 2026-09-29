"""向量召回（M3）。

`Retriever()` 不接参数——自检里就是直接实例化它，要求构造时就自己把索引加载好
（索引先跑 `python -m src.indexer` 建出来）。`retrieve(query, k)` 返回的每个 dict
至少含 `text` / `score` / `source`。
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from .indexer import BGEEmbedder, load_index
from .paths import INDEX_DIR


class Retriever:
    """FAISS 向量召回：query 编码 -> top-k 内积检索。"""

    def __init__(self, index_dir=None, model_dir=None, device: str = "cpu"):
        self.index_dir = Path(index_dir or INDEX_DIR)
        self.index, self.chunks, self.meta = load_index(self.index_dir)
        self.embedder = BGEEmbedder(model_dir or self.meta.get("model_dir"), device=device)
        print(f"[retriever] 载入 {len(self.chunks)} 个 chunk（{self.index_dir}）")

    def retrieve(self, query: str, k: int = 10) -> List[dict]:
        if not query or k <= 0 or not self.chunks:
            return []
        k = min(k, len(self.chunks))

        query_vec = self.embedder.encode([query], is_query=True)   # 查询侧加检索前缀
        scores, ids = self.index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            chunk = self.chunks[int(idx)]
            source = chunk.get("source", "data/kb.pdf")
            page = chunk.get("page")
            results.append({
                "text": chunk["text"],
                "score": round(float(score), 4),          # 已归一化，内积即 cosine
                "source": f"{source}#p{page}",
                "page": page,
                "chunk_id": int(idx),
            })
        return results


_DEFAULT: Retriever = None


def get_retriever(**kwargs) -> Retriever:
    """进程内复用的默认 Retriever：模型只加载一次，end-to-end 时不必反复初始化。"""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Retriever(**kwargs)
    return _DEFAULT


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    query = " ".join(sys.argv[1:]) or "什么是注意力机制？"
    for i, r in enumerate(get_retriever().retrieve(query, k=5), 1):
        print(f"\n[{i}] score={r['score']}  {r['source']}")
        print(f"    {r['text'][:150]}...")
