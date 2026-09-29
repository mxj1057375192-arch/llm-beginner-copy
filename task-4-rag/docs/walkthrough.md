# 任务四实现讲解

配套代码在 `../src/`。自检结果见 `../eval/result.json`。

## 整体链路

```
data/kb.pdf
   │  chunker.py    抽文本 → 按句子边界切分
   ▼
index/chunks.jsonl  (1411 个 chunk, 每个约 420 字符)
   │  indexer.py    BGE 编码 → L2 归一化 → FAISS IndexFlatIP
   ▼
index/faiss.index
   │  retriever.py  query 加检索前缀 → top-20 召回
   ▼
   │  reranker.py   bge-reranker 交叉编码器精排 → top-4
   ▼
   │  generator.py  拼 prompt → Qwen 生成
   ▼
rag.py::answer()  →  {"answer": ..., "sources": [...]}
```

## 逐文件要点

### `paths.py` — 路径常量

非任务要求，纯粹为了避免六个模块各写一份相对路径。`TASK_ROOT` 用
`Path(__file__).resolve().parents[1]` 定位，所以在哪个目录下跑都不受影响。

### `chunker.py` — PDF 抽取 + 切分（M1）

**自检口径**：`chunk_size` / `overlap` 按**字符**算，要求 chunk 数 > 10 且平均长度落在
`(chunk_size*0.5, chunk_size*1.2)`。所以关键设计是：

- 目标块长取 `chunk_size - overlap`，**不是** `chunk_size`。因为最后还要把上一块的尾巴
  `overlap` 个字符补到本块开头，留出余量后块长才刚好落在 `chunk_size` 附近。
  若按 `chunk_size` 装再补重叠，平均长度会稳定超标到 1.1 倍以上。
- 实测：256/32 下 17 个 chunk、平均 241.9 字符，落在 (128, 307.2) 内。

**PDF 的两个处理**：

1. 页间用 `\n` 拼接并记下每页的起始偏移（`load_pdf_text`）。这样跨页的句子在全文里
   仍然连续，跨页的 gold anchor 不会被切断；同时每个 chunk 还能报出页码。
2. 页眉页脚常被抽成孤立的纯数字行，在**页首/页尾各两行内**删掉——只删这个范围，
   避免误删正文里的数字。

清洗刻意做得很轻：公式、表格抽出来再乱也照原样保留。宁可留垃圾，也不要删掉正文
导致 anchor 命中不了。

### `indexer.py` — BGE 编码 + FAISS 索引（M2）

三个容易踩的点：

1. **BGE 取 CLS 位**（`last_hidden_state[:, 0]`）而非 mean pooling，再 L2 归一化。
   归一化之后 `IndexFlatIP` 的内积才等价于 cosine。实测输出 L2 范数正好 1.0。
2. **查询侧加检索前缀**（`QUERY_INSTRUCTION`），文档侧不加。加错或漏加都掉召回。
3. **Windows 非 ASCII 路径**：faiss 自带的 `write_index` / `read_index` 走 C++ 窄字符
   文件 IO，本项目路径含中文「新生任务」时会报
   `could not open ... for writing`。改成 `faiss.serialize_index` 拿字节流、
   由 Python 自己读写文件（见 `indexer.py` 里的注释）。

索引选 `IndexFlatIP` 精确检索而非 IVF/HNSW：全书只有 1411 个 chunk，精确检索已是
毫秒级，而近似索引会丢召回——这个任务卡的正是召回。

### `retriever.py` — 向量召回（M3）

`Retriever()` 无参构造，构造时就加载索引和编码器（自检就是直接实例化它）。
`retrieve(query, k)` 返回 `[{text, score, source, page, chunk_id}]`。

`get_retriever()` 是进程内单例，避免端到端时反复加载模型。

### `reranker.py` — 两阶段检索第二阶段

bge-reranker-base 是**交叉编码器**：吃 `[query, doc]` 文本对，query 和 doc 一起过模型
出一个相关性分数。比双塔的向量内积准得多，代价是不能预计算、必须在线算。

所以用法一定是「召回多、精排少」：召回 20 条 → 精排留 4 条。召回数给太小，rerank 救不回来。

### `generator.py` — 拼 prompt + 生成

prompt 里写死了三条要求：只能用资料回答、资料不足就直说、结尾标引用编号。
**这条是为了不让流畅的答案掩盖检索失败**——如果允许模型用预训练知识硬答，
检索明明没召回对，答案看起来还是对的，指标就失去意义了。

生成用贪心解码（`do_sample=False`），要的是可复现而不是文采。
模型路径默认 0.5B，换 7B 只需改 `paths.py` 的 `GEN_MODEL` 或设环境变量 `RAG_GEN_MODEL`。

### `rag.py` — 端到端（M4）

`answer(query)` 返回 `{"answer": str, "sources": [dict]}`。

两个细节：

- `_dedupe` 做去重 + 总长度截断。召回片段之间有 overlap，原样全拼既费词元又稀释关键信息。
- reranker 模型缺失时**降级**为纯向量召回并打印提示，而不是直接报错——
  这样 M3 的验证不被可选组件的下载状态阻塞。

### `sweep_chunking.py` — chunk 策略扫描（加分项 S1 的工具）

不建索引、不跑模型，只在文本层面算「多少条 gold 题的 anchor 完整落在某个 chunk 里」，
也就是自检的命中口径。**先用它挑参数再建索引**，能省掉几轮「重建索引 → 重跑评测」的来回。

实测结果（`python -m src.sweep_chunking`）：

```
anchor 出现在 PDF 全文里的题数（覆盖率上限）：30/30

chunk_size  overlap   chunks   avg_len  命中  覆盖率
       128       16     6499      92.5   30/30  1.000
       256       32     2954     201.6   30/30  1.000
       256       64     3490     201.8   30/30  1.000
       512       64     1411     419.7   30/30  1.000
       512      128     1659     419.4   30/30  1.000
      1024      128      692     854.1   30/30  1.000
```

所有组合的文本层面覆盖率都是 1.000，说明**切分不是瓶颈**——30 条题的 anchor 都能
完整落在某个 chunk 里。于是参数选择转向检索质量考虑，最终选 512/64：
块内有足够上下文，且 1411 个 chunk 的竞争面比 6499 个小，更利于 top-10 命中。

## 自检结果

```
[通过] chunking_sanity        chunks=17  avg_len=241.9
[通过] nndl_gold_recall_at_10 Recall@1=0.4  @3=0.6  @5=0.8  @10=0.867  MRR=0.547
[通过] rag_end_to_end         返回非空 answer + 非空 sources
```

Recall@10 = 0.867 > 0.6 通过。4 条未命中的题（`nndl-transformer-attention`、
`nndl-inductive-bias`、`nndl-flashattention-gqa`、`nndl-graphsage-gat`）**不是切分问题**
——扫描已经证明它们的 anchor 都在某个 chunk 里，是纯排序问题：这些题的问法和正文措辞
差异较大，向量召回没能把正确 chunk 排进前 10。

### 端到端语义抽查：检索成功但生成失败

DoD 要求 M4「手动验证语义」，抽查时抓到一个值得记下来的现象。

以 `nndl-adamw-weight-decay`（"AdamW 为什么要把权重衰减从 Adam 的梯度更新中解耦出来？"）
为例，逐步追踪证据链：

```
anchor「在Adam更新之外，单独对参数施加权重衰减」
  召回 20 条里排第 2          ← 检索成功
  去重后保留                  ← 上下文组装成功
  rerank top-4 里在上下文中   ← 精排成功，证据确实送进了 prompt
  但 0.5B 模型答「根据提供的资料无法回答」  ← 生成失败
```

rerank 分数的断层也很清楚：0.9991 / 0.9989 / 0.9955 / 0.9733，第 5 名直接掉到 0.5475。
说明精排判别力足够，**瓶颈已经完全转移到生成模型**：证据就在上下文里，0.5B 仍然取不出来。

这类失败靠调检索参数是修不好的，只能换更大的生成模型。这也是任务规格指定
Qwen2.5-**7B**-Instruct 而不是小模型的原因；本机无 GPU，用 0.5B 只是为了让整条链路
先跑通，端到端的答案质量不能拿它来代表。

另一个观察：小模型在资料不足时会反复复述同一句话（"主要解决问题是……"），
prompt 里的「不知道就说不知道」约束在 0.5B 上执行得不稳定，7B 上会好很多。

## 环境相关的两个坑（本机实测）

1. **下载源**：hf-mirror 实测只有 ~110KB/s，ModelScope ~1MB/s，差约 10 倍。
   模型最终从 ModelScope 拉取，URL 形如
   `https://www.modelscope.cn/api/v1/models/{org}/{name}/repo?Revision=master&FilePath={file}`。
2. **faiss 非 ASCII 路径**：见 `indexer.py` 注释。

另外对本地 Qwen 服务要留意：官方文档给的生成模型是 Qwen2.5-7B-Instruct，
本机（4 核 CPU、无 CUDA）用的是 Qwen2.5-0.5B-Instruct 跑通链路。
换 7B 只需改模型路径，但 CPU 上约 2-4 token/s，端到端 30 题评测会明显变慢。
