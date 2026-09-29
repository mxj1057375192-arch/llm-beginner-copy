# 任务四操作手册（写给第一次接触 RAG 的人）

这份文档只讲**怎么把它跑起来、怎么看结果、出错了怎么办**。
每一步在原理上为什么这么设计，见 [walkthrough.md](walkthrough.md)。

---

## 〇、先搞懂这件事在做什么

用大白话讲：**让电脑拿着一本书回答问题，而且答案必须来自这本书，不能瞎编。**

流程是这样的：

```
你的 PDF 书
  ↓ ① 切碎        切成一段段小文本（叫 chunk）
  ↓ ② 变成数字     每段算成一个向量（一串数字，代表这段话的意思）
  ↓ ③ 存进索引     一个专门用来"快速找相似"的数据库
                                                  ← 以上三步叫「建索引」，只做一次
  ─────────────────────────────────────────────
  ↓ ④ 你提问       问题也变成向量，去索引里找最像的 20 段   ← 叫「召回」
  ↓ ⑤ 精挑细选     20 段再挑出最相关的 4 段                ← 叫「精排 rerank」
  ↓ ⑥ 交给大模型   "只准根据这 4 段回答"                   ← 叫「生成」
  → 答案 + 出处
```

「RAG」就是这套流程的名字（Retrieval-Augmented Generation，检索增强生成）。

### 几个你一定会看到的名词

| 名词 | 大白话解释 |
|---|---|
| **chunk** | 把长文档切碎后的一小段。本任务是 512 个字符一段 |
| **embedding / 向量** | 把一段文字变成一串数字（这里是 512 个数字）。意思相近的文字，算出来的数字也相近 |
| **FAISS** | 一个专门用来存储向量、并快速找出「最像的几个」的工具库 |
| **召回 recall** | 从全部 1411 段里先捞出 20 段候选，这一步宁可多捞，后面再精挑 |
| **rerank 精排** | 把召回的 20 段逐个和问题放在一起仔细比对，重新排序 |
| **Recall@10** | 正确答案在前 10 名里的比例。本任务要求 > 0.6，我们做到 0.867 |
| **MRR** | 正确答案排得越靠前分越高，第一名得 1 分、第二名 0.5 分、第三名 0.33 分……取平均 |

---

## 一、环境（只需要配一次）

### 为什么命令里要写一大串 python 路径？

因为这台电脑上默认的 `python` 是 Anaconda 里的 3.9.12，**它没装 torch**，
一跑就会报 `No module named 'torch'`。本任务要用的是另一个环境：

```
d:/新生任务/.venv/Scripts/python.exe
```

所以你会在下面反复看到 `"$PY"` 这个写法——它就是把那个长路径存起来，少打几次字。

### 在终端里先执行这两行

```bash
cd "d:/新生任务/llm-beginner/task-4-rag"
PY="d:/新生任务/.venv/Scripts/python.exe"
```

> ⚠️ 每次**新开一个终端窗口**都要重新执行这两行，因为路径设置不会保留。

### 检查环境对不对

```bash
"$PY" -c "import torch, faiss, pypdf, transformers; print('环境正常')"
```

看到 `环境正常` 就对了。如果报 `No module named 'faiss'`，运行：

```bash
"$PY" -m pip install faiss-cpu pypdf tqdm
```

---

## 二、数据和模型（只需要下一次）

需要三样东西，**已经下好了**，这节只是让你知道它们在哪、万一删了怎么补：

| 东西 | 位置 | 干什么用的 | 大小 |
|---|---|---|---|
| NNDL 电子书 | `data/kb.pdf` | 知识库，要拿它回答问题 | 7.2 MB |
| bge-small-zh-v1.5 | `models/bge-small-zh-v1.5/` | 把文字变成向量（②③④步） | 162 MB |
| bge-reranker-base | `models/bge-reranker-base/` | 精排（第⑤步） | 1.1 GB |
| Qwen2.5-0.5B-Instruct | `models/Qwen2.5-0.5B-Instruct/` | 生成答案（第⑥步） | 954 MB |

### 万一需要重新下载

**注意**：这台机器上 `hf-mirror.com` 只有 ~110KB/s，而 ModelScope 有 ~1MB/s，**快 10 倍**，
所以走 ModelScope。下载命令（示例，换成你要的文件名和模型名）：

```bash
curl -L -o models/bge-small-zh-v1.5/model.safetensors \
  "https://www.modelscope.cn/api/v1/models/BAAI/bge-small-zh-v1.5/repo?Revision=master&FilePath=model.safetensors"
```

地址的规律是：
`https://www.modelscope.cn/api/v1/models/{组织名}/{模型名}/repo?Revision=master&FilePath={文件名}`

---

## 三、日常操作：四件事

### ① 建索引（改了切分参数才需要重做）

```bash
"$PY" -m src.indexer --chunk-size 512 --overlap 64
```

**这一步在干嘛**：读 `data/kb.pdf` → 抽文字 → 切碎 → 每段算成向量 → 存进 `index/` 文件夹。

**跑多久**：约 2 分钟（会显示进度条 `embedding: 100%`）。

**怎么算成功**：最后打印 `[indexer] 索引已写入 ...`，并且 `index/` 下多出三个文件：
`faiss.index`（向量索引）、`chunks.jsonl`（每段文本）、`meta.json`（配置记录）。

> `--chunk-size` 改了就**必须重新建索引**，因为切法变了，向量全变了。

### ② 跑自检（最重要的命令）

```bash
"$PY" eval/run.py
```

**这一步在干嘛**：用仓库给你的 30 道标准题考你的系统，自动打分。

**跑多久**：约 3-5 分钟（要加载模型、跑 30 道题、还要生成一次答案）。

**怎么算成功**：看到三个 `[通过]`：

```
[通过] chunking_sanity
[通过] nndl_gold_recall_at_10     ← 重点看这个，里面有 Recall@10 和 MRR
[通过] rag_end_to_end
```

**三种状态的含义**：
- `[通过]` — 这项达标了
- `[跳过]` — 前置条件没就绪（比如索引还没建），**不是错误**
- `[失败]` — 没达标，下面会跟 `error` 说明原因

结果同时会写进 `eval/result.json`，交作业时要附上这个文件。

> ⚠️ **必须在仓库里跑**，不能把 task-4-rag 文件夹单独拷到别处——
> 自检脚本依赖仓库根目录的 `_eval_harness.py`。

### ③ 问它一个问题（体验完整流程）

```bash
"$PY" -m src.rag "什么是注意力机制？"
```

**跑多久**：约 1-2 分钟（要把四个模型都加载一遍）。

**会看到**：
- `回答：` 后面是模型生成的答案
- `依据的片段：` 后面是它参考的原文和出处（页码）

### ④ 做切分实验（可选的加分项）

```bash
"$PY" -m src.sweep_chunking
```

**这一步在干嘛**：在**不建索引、不跑模型**的前提下，算 6 组不同的切分参数下
「有多少道题的答案片段能完整落进某个 chunk」。几秒钟就出结果。

这张表是本任务最有价值的实验，因为它能告诉你：**召回失败的锅是不是切分的**。

---

## 四、每个文件是干嘛的，我想改该动哪里

```
task-4-rag/
├── data/
│   ├── kb.pdf              知识库（不要动）
│   └── gold_qa.jsonl       30 道标准考题（不要动）
├── models/                 三个模型（不要动）
├── index/                  建索引的产物（可以删，删了重新建）
├── eval/
│   ├── run.py              自检脚本（不要动）
│   └── result.json         自检结果（交作业要附）
├── src/                    ← 你要看的、要改的都在这里
│   ├── chunker.py          ① 抽 PDF 文字 + 切碎
│   ├── indexer.py          ②③ 变向量 + 建索引
│   ├── retriever.py        ④ 召回 20 段
│   ├── reranker.py         ⑤ 精排挑出 4 段
│   ├── generator.py        ⑥ 拼提示词 + 调大模型
│   ├── rag.py              把 ④⑤⑥ 串起来
│   ├── sweep_chunking.py   切分参数扫描实验
│   └── paths.py            路径配置 ← 想换模型改这里
└── docs/
    ├── walkthrough.md      代码原理讲解
    └── HOWTO.md            本文件
```

### 最常改的三个地方

**换生成模型**（比如换成 7B）：改 [src/paths.py](src/paths.py) 里的 `GEN_MODEL`，
或者临时用环境变量：

```bash
RAG_GEN_MODEL="d:/某个路径/Qwen2.5-7B-Instruct" "$PY" -m src.rag "你的问题"
```

**改召回数量 / 精排数量**：改 [src/rag.py](src/rag.py) 开头的三个常量：

```python
RETRIEVE_K = 20           # ④ 先捞几段（要远大于下面这个数，不然精排没得挑）
RERANK_TOP_N = 4          # ⑤ 挑几段给大模型看
MAX_CONTEXT_CHARS = 3200  # 拼进提示词的总字数上限
```

**改切分参数**：就是命令行的 `--chunk-size` / `--overlap`，改完要**重新建索引**。

---

## 五、常见报错对照表

| 报错信息 | 原因 | 怎么办 |
|---|---|---|
| `No module named 'torch'` | 用了系统默认的 python | 用 `"$PY"` 而不是 `python` |
| `FileNotFoundError: 找不到索引 ...；先跑 python -m src.indexer` | 索引还没建 | 先跑 `"$PY" -m src.indexer` |
| `could not open ... for writing` | 老版本代码用了 `faiss.write_index` | 现版本已修（改用 serialize_index），若复现见 walkthrough.md |
| `No such file or directory: 'data/kb.pdf'` | 书还没下 | 见上面「二、数据和模型」 |
| `Could not load library ... torchvision` | Anaconda 的旧包泄漏，**无害警告** | 忽略即可 |
| `Exceeded 5000 form XObject invocations` | pypdf 抽 PDF 的**无害警告** | 忽略，已验证不影响答题 |
| `载入 1411 个 chunk` 后卡住很久 | 在加载模型，正常 | 等，第一次加载要 20-30 秒 |
| 下载速度只有 100KB/s | 用了 hf-mirror | 换 ModelScope，快 10 倍 |

---

## 六、想看结果，看哪里

| 想看什么 | 看哪里 |
|---|---|
| 系统答得准不准（**最重要**） | `eval/result.json` 里的 `recall_at_10` 和 `mrr` |
| 哪几道题没答对、排第几 | `result.json` 里 `nndl_gold_recall_at_10` 的 `details` 数组 |
| 切分参数影响大不大 | 跑 `"$PY" -m src.sweep_chunking` 看那张表 |
| 某个具体问题的答案和出处 | 跑 `"$PY" -m src.rag "问题"` |
| 为什么代码这么写 | [walkthrough.md](walkthrough.md) |
| 完整实验报告 | [REPORT.md](../REPORT.md) |

---

## 七、想继续深入可以做的事

1. **换 7B 生成模型**：本机无 GPU，用 0.5B 只是为了让链路先跑通。报告里已经验证了
   「检索成功但 0.5B 答不出来」的案例——换大模型是提升答案质量最直接的一步。
2. **做 chunk_size 扫描实验**：用 `sweep_chunking.py` 挑参数 → 建索引 → 跑自检，
   对比 Recall 变化，这就是加分项 S1。
3. **对比加不加 reranker**：把 `models/bge-reranker-base` 临时改名，`rag.py` 会自动
   降级为纯向量召回并打印提示，对比前后答案质量——这就是加分项 S2。
