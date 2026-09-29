"""任务四实现：手写 RAG 流水线（不依赖 LlamaIndex / LangChain 的高层封装）。

阅读顺序建议（配套讲解见 ../docs/walkthrough.md）：

    1. chunker.py    PDF 抽取 + 字符级切分        —— M1
    2. indexer.py    BGE 句向量 + FAISS 索引      —— M2
    3. retriever.py  向量召回 top-k               —— M3
    4. reranker.py   bge-reranker 两阶段精排（加分项 S2 的核心）
    5. generator.py  拼 prompt + Qwen 生成
    6. rag.py        answer() 串起端到端          —— M4
    7. paths.py      路径常量（非任务要求，便利模块）
"""
