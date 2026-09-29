"""BGE 句向量 + FAISS 索引（M2）。

两个最容易踩的点：

1. BGE 的句向量取 **CLS 位**（`last_hidden_state[:, 0]`）再 L2 归一化，不是 mean pooling。
   归一化之后 FAISS 的 `IndexFlatIP`（内积）才等价于 cosine；忘了归一化，检索分数全乱。
2. 查询侧要加检索指令前缀 `QUERY_INSTRUCTION`，文档侧不加。加错或漏加都掉召回。

索引选型：全书切完只有几千个 chunk，`IndexFlatIP` 精确检索已经是毫秒级，
不必上 IVF / HNSW 这类近似索引——近似会丢召回，而这个任务卡的正是召回。
"""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .chunker import chunk_pdf
from .paths import EMBED_MODEL, INDEX_DIR, KB_PDF

# BGE 检索指令：只加在 query 侧
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


class BGEEmbedder:
    """bge-small-zh-v1.5 句向量编码器（CLS 池化 + L2 归一化）。"""

    def __init__(self, model_dir=None, device: str = "cpu",
                 batch_size: int = 32, max_length: int = 512):
        self.model_dir = str(model_dir or EMBED_MODEL)
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModel.from_pretrained(self.model_dir).to(device).eval()

    @torch.no_grad()
    def encode(self, texts, is_query: bool = False, show_progress: bool = False) -> np.ndarray:
        """返回 (n, dim) 的 float32 已归一化句向量；is_query=True 时自动加检索前缀。"""
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.zeros((0, 1), dtype="float32")
        if is_query:
            texts = [QUERY_INSTRUCTION + t for t in texts]

        batches = []
        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="embedding", unit="batch")
        for i in iterator:
            batch = self.tokenizer(texts[i:i + self.batch_size], padding=True, truncation=True,
                                   max_length=self.max_length,
                                   return_tensors="pt").to(self.device)
            hidden = self.model(**batch).last_hidden_state
            vecs = torch.nn.functional.normalize(hidden[:, 0], p=2, dim=1)   # BGE 取 CLS 位
            batches.append(vecs.cpu().numpy().astype("float32"))
        return np.concatenate(batches, axis=0)


def build_index(pdf_path=None, out_dir=None, chunk_size: int = 512, overlap: int = 64,
                model_dir=None, limit: int = None) -> dict:
    """从 PDF 建索引并落盘：faiss.index + chunks.jsonl + meta.json。"""
    pdf_path = Path(pdf_path or KB_PDF)
    out_dir = Path(out_dir or INDEX_DIR)
    if not pdf_path.exists():
        raise FileNotFoundError(f"找不到知识库 {pdf_path}；先跑 python data/download.py")

    chunks = chunk_pdf(pdf_path, chunk_size=chunk_size, overlap=overlap)
    if limit:
        chunks = chunks[:limit]
    print(f"[indexer] {pdf_path.name} -> {len(chunks)} 个 chunk "
          f"(chunk_size={chunk_size}, overlap={overlap})")

    embedder = BGEEmbedder(model_dir)
    embs = embedder.encode([c["text"] for c in chunks], show_progress=True)

    index = faiss.IndexFlatIP(embs.shape[1])   # 向量已归一化，内积即 cosine
    index.add(embs)

    out_dir.mkdir(parents=True, exist_ok=True)
    # 注意：不要用 faiss.write_index —— 它内部是 C++ 的窄字符文件 IO，在 Windows 上
    # 遇到非 ASCII 路径（本项目路径含中文「新生任务」）会 fopen 失败报
    # "could not open ... for writing"。改成拿 serialize_index 的字节流、由 Python 写文件。
    (out_dir / "faiss.index").write_bytes(faiss.serialize_index(index).tobytes())
    with (out_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    meta = {
        "pdf": str(pdf_path),
        "n_chunks": len(chunks),
        "dim": int(embs.shape[1]),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "model_dir": embedder.model_dir,
        "index_type": "IndexFlatIP (cosine)",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    print(f"[indexer] 索引已写入 {out_dir}")
    return meta


def load_index(index_dir=None):
    """读回 (faiss 索引, chunk 列表, meta)。"""
    index_dir = Path(index_dir or INDEX_DIR)
    index_file = index_dir / "faiss.index"
    if not index_file.exists():
        raise FileNotFoundError(f"找不到索引 {index_file}；先跑 python -m src.indexer")

    # 与写入同理：faiss.read_index 也走 C++ 文件 IO，非 ASCII 路径会失败
    raw = np.frombuffer(index_file.read_bytes(), dtype="uint8").copy()
    index = faiss.deserialize_index(raw)
    chunks = [json.loads(line) for line in
              (index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    meta_file = index_dir / "meta.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
    return index, chunks, meta


if __name__ == "__main__":
    import argparse
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="从 data/kb.pdf 建 BGE + FAISS 索引")
    ap.add_argument("--pdf", default=None)
    ap.add_argument("--out", default=None, help="索引输出目录，默认 index/")
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="只索引前 N 个 chunk（调试用）")
    args = ap.parse_args()

    build_index(args.pdf, args.out, args.chunk_size, args.overlap, limit=args.limit)
