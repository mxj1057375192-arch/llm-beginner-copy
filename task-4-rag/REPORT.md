# 任务四实验报告：RAG 文档问答

## 一、自检结果

```
[通过] chunking_sanity        {"chunks": 17, "avg_len": 241.9, "expected_avg_range": [128.0, 307.2]}
[通过] nndl_gold_recall_at_10 {"n": 30, "recall_at_1": 0.4, "recall_at_3": 0.6, "recall_at_5": 0.8,
                               "recall_at_10": 0.867, "mrr": 0.547}
[通过] rag_end_to_end         返回非空 answer + 非空 sources
```

完整结果见 [eval/result.json](eval/result.json)。必做项 M1–M4 全部通过，Recall@10 = **0.867**（通过线 0.6）。

## 二、实现

| 文件 | 内容 |
|---|---|
| [src/chunker.py](src/chunker.py) | PDF 抽取 + 字符级切分（M1） |
| [src/indexer.py](src/indexer.py) | BGE 句向量 + FAISS 索引（M2） |
| [src/retriever.py](src/retriever.py) | 向量召回 top-k（M3） |
| [src/reranker.py](src/reranker.py) | bge-reranker 两阶段精排 |
| [src/generator.py](src/generator.py) | prompt 拼接 + Qwen 生成 |
| [src/rag.py](src/rag.py) | `answer()` 串起端到端（M4） |
| [src/sweep_chunking.py](src/sweep_chunking.py) | chunk 策略预筛工具 |

关键实现点：

1. **切分**：目标块长取 `chunk_size - overlap` 而非 `chunk_size`。因为最后还要把上一块的尾巴 `overlap` 个字符补到本块开头，留出余量后块长才刚好落在 `chunk_size` 附近；若按 `chunk_size` 装再补重叠，平均长度会稳定超标。实测 256/32 下 17 个 chunk、平均 241.9 字符，落在 (128, 307.2) 内。
2. **PDF 跨页处理**：页间用 `\n` 拼接并记录每页的起始字符偏移。这样跨页的句子在全文里仍然连续，跨页的 gold anchor 不会被切断，同时每个 chunk 还能报出页码。页眉页脚只删「页首/页尾各两行内的孤立纯数字行」，避免误删正文数字；公式表格一律原样保留——宁可留垃圾也不能删掉正文导致 anchor 命中不了。
3. **BGE 编码**：取 **CLS 位**（`last_hidden_state[:, 0]`）而非 mean pooling，再 L2 归一化（实测范数正好 1.0）。**查询侧加检索前缀**「为这个句子生成表示以用于检索相关文章：」，文档侧不加。
4. **索引选型**：`IndexFlatIP` 精确检索而非 IVF/HNSW。全书只有 1411 个 chunk，精确检索已是毫秒级，而近似索引会丢召回——这个任务卡的正是召回。
5. **两阶段检索**：先去掉互为子串的近重复片段，再由 reranker 精排。顺序很重要——先删重复，rerank 才有多样化的候选可挑。
6. **生成约束**：prompt 写死三条要求（只用资料回答 / 资料不足就直说 / 结尾标引用编号），用贪心解码保证可复现。

## 三、索引配置与环境

| | |
|---|---|
| 知识库 | `data/kb.pdf`（7.23 MB），pypdf 抽出 606,652 字符 |
| 切分 | chunk_size=512, overlap=64 → **1411 个 chunk**，平均 420 字符 |
| 向量 | bge-small-zh-v1.5，512 维，已 L2 归一化 |
| 索引 | FAISS `IndexFlatIP`（内积 ≡ cosine） |
| 建索引耗时 | 1 分 44 秒（1411 chunk） |
| 生成模型 | Qwen2.5-0.5B-Instruct |

环境：i5-1135G7（4 核 8 线程）+ 16 GB 内存，**纯 CPU**，无 CUDA。官方建议的生成模型是 Qwen2.5-7B-Instruct，本机无 GPU，用 0.5B 跑通链路；代码里换模型只需改 `paths.py` 的 `GEN_MODEL` 或设环境变量 `RAG_GEN_MODEL`。

## 四、实验观察（约 450 字）

**切分不是瓶颈，这是算出来的而不是试出来的。** 建索引前我先用 [src/sweep_chunking.py](src/sweep_chunking.py) 在纯文本层面筛了 6 组 chunk_size/overlap（128/16 到 1024/128），算「多少条题的 anchor 完整落在某个 chunk 里」——也就是自检的命中口径。结果**六组全是 1.000**，且 30/30 题的 anchor 都出现在 PDF 抽文本里。这说明切分策略对召回上限没有影响，4 条未命中的题（`nndl-transformer-attention`、`nndl-inductive-bias`、`nndl-flashattention-gqa`、`nndl-graphsage-gat`）纯粹是**排序问题**：anchor 都在某个 chunk 里，只是向量召回没把它排进前 10。因此我没有走「改参数 → 重建索引 → 重跑评测」的试错循环，直接按检索质量考虑定了 512/64（块内有足够上下文，1411 个 chunk 的竞争面也比 6499 个小）。这个方法省掉了好几轮索引重建。

**端到端抽查抓到一个干净的因果链。** 以 `nndl-adamw-weight-decay` 为例：「AdamW 为什么要把权重衰减从 Adam 的梯度更新中解耦出来？」——anchor 在召回 20 条里排第 2，去重后保留，rerank 后确实进了送进 prompt 的 top-4，**但 0.5B 模型仍然回答「根据提供的资料无法回答」**。证据就在上下文里，模型取不出来。rerank 分数的断层印证了精排本身工作良好：前四名 0.9991 / 0.9989 / 0.9955 / 0.9733，第五名直接掉到 0.5475。

**结论是瓶颈已经转移。** 检索侧 Recall@10 = 0.867、MRR = 0.547，且证据链每一环都验证过；这类失败靠调检索参数修不好，只能换更大的生成模型——这正是任务规格指定 7B 而非小模型的原因。另外小模型在资料不足时会反复复述同一句话，prompt 里「不知道就说不知道」的约束在 0.5B 上执行得不稳定。

## 五、踩到的坑

1. **下载源速度差 10 倍**：hf-mirror 实测只有 **~110KB/s**，ModelScope **~1MB/s**。第一次用 hf-mirror 下 bge-small 时 1GB 级的权重下到一半就断了，白等一轮。改用 ModelScope 后三个模型（bge-small / bge-reranker / Qwen0.5B，合计约 2.2 GB）才顺利下完。
2. **faiss 的文件 IO 不能用于非 ASCII 路径**：`faiss.write_index` / `read_index` 走 C++ 窄字符文件 IO，本项目路径含中文「新生任务」，索引明明目录存在却报 `could not open ... for writing: No such file or directory`。改用 `faiss.serialize_index(idx).tobytes()` 配合 Python 自己读写文件，读回用 `faiss.deserialize_index`。
3. **pypdf 的 XObject 警告**：抽文本时报 `Exceeded 5000 form XObject invocations`，部分页的表单内容会被跳过。实测不影响 anchor 命中（30/30 都能找到），但这是个沉默的丢文本来源，值得留意。
4. **CPU 编码比预期快**：bge-small 编码 1411 个 chunk 只要 1 分 44 秒，所以索引重建成本很低，chunk 策略扫描（S1）可以放心做多组。
