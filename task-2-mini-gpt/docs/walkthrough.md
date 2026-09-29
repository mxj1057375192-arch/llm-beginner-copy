# 任务二 · 从零实现 mini-GPT —— 手把手读懂版（小白向）

> 这份文档是给第一次做这个任务的你写的：**先讲清楚每一块在干什么、为什么这么干，再给可复制的命令**。
> 代码全部带中文注释，配合本文一起看效果最好。
> 真正的"实验观察报告"在 [`report.md`](report.md)，自检结果在 `eval/result.json`。

---

## 0. 一句话总览：我们在造什么？

任务要求做一个"最小可用"的语言模型底座。你可以把它想成**一个只会接话茬的机器**：

```
输入：床前明月光            （prompt，提示词）
输出：床前明月光，疑是地上霜。举头望明月，低头思故乡。   （它一个 token 一个 token 地"接"下去）
```

它没有"理解"能力，只是在做一件事：**看前文，猜下一个 token 是什么**（next-token prediction）。
把这件事做准了，句子看起来就通顺；这个能力足够强、语料足够多时，就"涌现"出写作、翻译、推理等能力。

任务的 5 个必做项（DoD）和代码的对应关系：

| 要求 | 通俗解释 | 代码位置 |
|---|---|---|
| **M1** 手写 BPE 分词器 | 把文字切成"词块"，并给每个块编号（模型只认数字） | `src/tokenizer.py` |
| **M2** decoder-only 模型 + RoPE | 模型主体；位置信息不用查表，而是"旋转"进 Q/K | `src/model.py` `src/rope.py` |
| **M3** KV cache | 生成加速：把算过的 K/V 存起来，不重复算 | `src/attention.py` |
| **M4** 预训练 | 在唐诗语料上训练，让困惑度降到阈值以下 | `src/train.py` |
| **M5** 四种采样 | 同一个模型、不同的"抽词规则"，生成风格不同 | `src/sampling.py` `src/generate.py` |

---

## 1. 先认识数据

```bash
cd llm-beginner/task-2-mini-gpt
python data/download.py                 # 默认唐诗档（约 49KB），CPU 即可
```

脚本会生成：

- `data/train.txt`（约 44KB，训练用）
- `data/dev.txt`（约 4.9KB，验证用，**永远不参与训练**）
- `data/dataset_info.json`（记录数据集名和困惑度阈值：唐诗是 **50**）

看一眼数据长什么样：

```bash
python -c "print(open('data/train.txt',encoding='utf-8').read()[:200])"
```

你会看到一首首唐诗，每首之间空行分隔。注意：**数据里没有任何标签**——语言模型的标签就是文本自己（把每个位置的下一个字符当答案），所以叫"自监督"。

> 小知识：`dev.txt` 是 `train.txt` 后面的 10%。因为我们**按位置**切分（不是随机打乱），训练时模型没见过 dev 里的诗，这样算出来的困惑度才有意义。

---

## 2. M1：BPE 分词器（`src/tokenizer.py`）

### 2.1 为什么非要"分词"？

神经网络只能算数字。所以要把 `床前明月光` 变成 `[1234, 567, ...]` 这样的 id 列表。
最朴素的两种做法都有问题：

| 做法 | 问题 |
|---|---|
| 按"字"切 | 常用汉字几千个，词表一开始就很大；遇到语料里没有的字只能记成 `<unk>`（信息丢失） |
| 按"字节"切 | 任何字都能表示（UTF-8 字节 0~255），但序列变得很长，一个汉字要 3 个 token |

**BPE（Byte Pair Encoding）是折中方案**：先按字节起步（保证无 `<unk>`），再把**高频相邻块**反复合并成更大的块。

训练过程（代码里 `train_bpe`）就是重复一件事：

```
初始：  明 月          （其实是 <明>的三个字节, <月>的三个字节）
统计：  "明"+"月" 这一对出现得最频繁
合并：  把 "明月" 记成一个新 token，词表 +1
重复到词表达到目标大小（本项目 2048）
```

合并过程的产物就是一张**有序的 merge 表**——顺序即优先级。编码时按这个顺序贪心合并：

```python
ids = tok.encode("床前明月光")
[repr(tok.id_to_token[i]) for i in ids]
# ['å', 'º', 'ĬåīįæĺİæľĪåħī', ...]   ← 前面几项是"还没合并的字节"，后面 'Ĭ...' 是合并好的"床前明月光"
```

（那些 `å`、`º` 是字节的可打印代理字符，GPT-2 也这么干，目的是让词表能安全地写进 JSON。）

### 2.2 两个关键设计（也是任务 README 里的"常见坑"）

1. **字节级 → 永远不会丢字**。`encode` 先转 UTF-8 字节，`decode` 再把字节拼回去解码，所以中文、emoji、生僻字都能原样还原。自检 `tokenizer_roundtrip` 就是查这个。
2. **预分词**：先用正则把文本切成"词"（连续汉字/字母算一个词，标点各自独立），merge 只在词内部发生。这样不会把"光"和下一句的"床"粘成一个 token。

### 2.3 跑起来

```bash
python src/tokenizer_train.py --mode byte --vocab_size 2048   # 本次采用
python src/tokenizer_train.py --mode char --vocab_size 3500   # 对照实验用
# 产出：ckpt/tokenizer.json   以及压缩率、roundtrip 自检打印
```

`ckpt/tokenizer.json` 里存的就是 `{"special_tokens": [...], "merges": [["a","b"], ...]}`，
`BPETokenizer.from_pretrained(path)` 能把它还原成完全一样的编码器（自检就是这么加载的）。

### 2.4 实测：字节级 vs 字符级（一个反直觉的结论）

我两种都实现了，在**完全相同的数据和超参**下对比：

| 词表方案 | 词表大小 | 压缩率 | 单字 token 占比 | dev 困惑度 |
|---|---|---|---|---|
| byte 字节级 | 2048 | 0.97 字符/token | 12.4%（含字节碎片） | **379** |
| char 字符级 | 3206 | **1.11 字符/token** | 100% | 1038 |

压缩率更高、token 更"干净"的字符级反而差了近 3 倍——因为 dev 里有 6.7% 的字训练集没见过：
字节级可以"用已知字节拼出"生字，字符级只能从 3206 个候选里瞎猜。
**所以本项目最终采用字节级**，这也印证了 README 把"字节级 fallback"列为必检项的原因。

---

## 3. M2：模型主体 + RoPE

### 3.1 整体结构（`src/model.py`）

```
token id (B, T)
  ↓  查表 token_emb            → (B, T, d_model)     把数字变成 256 维向量
  ↓  N 个 Block（每个含两段残差）
  │     x = x + Attention(RMSNorm(x))     ← 让每个位置"看"前面的位置，融合信息
  │     x = x + SwiGLU(RMSNorm(x))        ← 逐位置的前馈网络，做非线性变换
  ↓  RMSNorm（收尾归一化）
  ↓  lm_head → (B, T, vocab)   每个位置输出"下一个 token 的概率分布"
```

- **d_model = 256**：每个 token 用 256 个数表示；**N = 4 层**：叠 4 次 Block。
- **Pre-LN**（先归一化再进子层）比 Post-LN 训练更稳，现代 LLM 都这么干。
- **weight tying**：输入查表和输出投影共用同一张矩阵（省参数，小模型通常还有增益）。

### 3.2 注意力：一句话讲明白

> 每个位置发出一个**问题 Q**（我在找什么），每个位置提供一个**名片 K**（我是什么）和**内容 V**（我的信息）。
> Q 和所有 K 做点积得到"匹配分数" → softmax 归一成权重 → 用权重把 V 加权求和。

三个必须记牢的细节：

1. **缩放**：分数要除以 `sqrt(head_dim)`，否则维度一大点积数值爆炸，softmax 饱和、梯度消失。
2. **causal mask**：只能看自己和左边（不能偷看未来），被挡的位置填 `-inf`（填 0 就没挡住！）。
3. **多头**：把 256 维切成 4 个 64 维，各自算注意力再拼回来——让模型同时关注不同的模式（有的头盯韵脚，有的头盯句式）。

### 3.3 RoPE（`src/rope.py`）：为什么不用"位置 1、2、3…"的表？

传统做法是给每个位置准备一个可学习的向量，加到输入上。问题是：**表长被训练时的最大长度写死**，序列变长就只能外推失败。

RoPE 换个思路：**不改输入，而是把 Q、K 向量按位置旋转一个角度**。

```
把 d 维向量两两配对成 d/2 个二维平面上的点
位置 m 的第 i 对，旋转角度 m × θᵢ，其中 θᵢ = 10000^(-2i/d)
```

旋转有个漂亮性质：`旋转后的 q_m · 旋转后的 k_n` 只依赖 **m−n（相对距离）**。
所以注意力天然是"看相对距离"的，比绝对位置更符合语言直觉，外推也更稳。

代码里能验证这一点（`src/unit_check.py` 的 `RoPE_内积只依赖相对位置`）：
距离都是 3 的两对位置，内积几乎相同；距离不同的则明显不同。

> ⚠️ RoPE 只作用在 **Q 和 K** 上，**不要**作用在 V 上（V 是纯内容）。
> ⚠️ 实现用的是"前后折半"配对（`(x[i], x[i+d/2])`），训练和推理必须一致。

---

## 4. M3：KV cache（`src/attention.py`）

### 4.1 问题在哪

生成是**一步一步**的：有了"床前明月光"，猜出"疑"，再拿"床前明月光疑"猜"是"……
如果每一步都把整段话重新送进模型（`use_cache=False`），那么：

- 第 1 步算 1 个词的 QKV；
- 第 2 步把前 2 个词都算一遍（第 1 个词白算了）；
- ……到第 T 步，前面 T−1 个词又全算一遍。

总计算量约 **O(T³)**，越到后面越慢。

### 4.2 cache 怎么省

每层的 K、V 算完就**存下来**（形状 `(B, 头数, 已生成长度, head_dim)`）：

- 新的一步只把**新词元**（1 个）送进模型；
- 新词元的 Q/K/V 算出来后，K、V **拼到历史后面**（在序列长度那一维拼！）；
- Q 只有 1 个，去和"历史 + 自己"的全部 K 打分，再取 V 加权。

总计算量降到约 **O(T²)**，单步耗时几乎与历史长度无关。代价是显存里多存了 K/V。

### 4.3 最容易错的一点：位置要接着历史算

RoPE 是"按位置旋转"的。增量解码时新词元的绝对位置 = **历史长度**，
所以代码里必须传 `position_offset = past_len`（见 `CausalSelfAttention.forward`）。
**忘了这一步，角度就错，logits 和全量前向对不上** —— 自检 `kv_cache_equivalence` 正是抓这个。

自检期望：开 cache 与不开 cache 的 logits 误差 < 1e-4。我们的实测结果是 **2.4e-07**（浮点误差量级）。

---

## 5. M4：预训练（`src/train.py`）

### 5.1 训练循环的全部内容

```python
x, y = get_batch()          # 随机取一段 token 流：x = ids[i:i+T]，y = ids[i+1:i+T+1]（右移一位）
logits = model(x)           # 前向
loss = cross_entropy(logits, y)   # 每个位置都在做"猜下一个 token"
loss.backward()             # 反向传播
clip_grad_norm_(...)        # 梯度裁剪，防止偶发 spike
optimizer.step()            # AdamW 更新
```

外加三件工程上必须做的事：

1. **warmup + cosine 学习率**：开头 lr 从 0 线性升到峰值（防止早期大梯度炸掉），之后按余弦退火。
2. **区分 train / dev**：dev 只用来看困惑度，绝不参与梯度。
3. **保存最优 + 早停**：语料小容易过拟合，dev 困惑度变差就保留之前最好的那次。

### 5.2 困惑度（perplexity, PPL）怎么读

```
PPL = exp(平均交叉熵)
```

可以理解成"模型在每个位置平均在多少个候选里犹豫"。**PPL 越低越好**，理论上：
- PPL = 2048（我们的词表大小）≈ 完全瞎猜；
- PPL = 50 ≈ 每个位置在 50 个候选里犹豫，已经能生成通顺的唐诗句子。

自检阈值：唐诗 **< 50**。

### 5.3 跑起来

```bash
# 正式训练（本机 CPU 约 5 分钟，最优模型在 step 200 附近）
python -u src/train.py --d_model 128 --n_layers 4 --n_heads 4 \
    --block_size 128 --dropout 0.25 --steps 700 --lr 5e-4 --warmup 200 \
    --eval_interval 100 --log_every 100

# 只想先跑通流程（几十秒）
python -u src/train.py --preset tiny --steps 100 --eval_interval 50 --out_dir ckpt_smoke
```

产出：`ckpt/best.pt`（最优模型 + 配置）、`ckpt/loss_curve.png`（训练曲线）、`ckpt/train_log.json`。

> **为什么打印了两个困惑度？**
> `dev ppl（随机窗口）`：训练时的口径，每段都有完整前文，数值偏低；
> `自检口径 ppl`：完全模拟 `eval/run.py`——把 dev 文本从开头按 `block_size` 硬切，
> 每块开头几个 token 缺上文，天然更难，所以数值更高。
> **看后者才能知道离阈值还有多远。**

> ⚠️ **实战结论（重要）**：在 49KB 唐诗 quick-start 档上，dev 困惑度**到不了 50**。
> 实测最优 375，而且 step 200 之后开始过拟合（验证困惑度回升）。
> 原因：这个语料的 unigram 基线就有 602（即"只看字频瞎猜"的水平），dev 里还有 6.7% 的字训练集没见过。
> 详细分析见 [`report.md`](report.md) §4。想真正达标需要换更大语料（TinyStories / SkyPile）。

---

## 6. M5：四种采样策略（`src/sampling.py`）

模型每步输出的是**整个词表的概率分布**，怎么从中挑一个 token，就决定了生成风格：

| 策略 | 做法 | 效果 |
|---|---|---|
| **greedy** | 永远取概率最大的 | 确定、稳，但容易重复、死板 |
| **temperature** | logits 先除以 T 再 softmax；T<1 更尖（保守），T>1 更平（多样） | 调控随机性 |
| **top-k** | 只在概率最高的 k 个里抽 | 砍掉长尾胡言乱语 |
| **top-p** | 按概率从大到小累加，累计过 p 就截断，只在这"核"里抽 | 比 top-k 自适应：分布尖时留得少，平时留得多 |

**顺序不能乱**：温度缩放 → top-k → top-p → 重新归一化 → 采样。
被过滤的候选要填 `-inf`（不是 0），否则它们还会分到概率；`temperature=0` 要特判成 greedy，否则 0/0 出 NaN。

```bash
python src/generate.py                             # 默认几组 prompt × 全部策略
python src/generate.py --prompt "春" --max_new_tokens 60
# 产出：docs/generation_samples.md（报告里直接引用）
```

---

## 7. 怎么跑自检、怎么看结果

```bash
# 一条命令跑全部（推荐）
python run_all.py

# 或者分开跑
python src/unit_check.py    # 自己写的单元自测（RoPE/KV cache/causal/采样/分词），秒级出结果
python eval/run.py          # 官方自检，结果写入 eval/result.json
```

自检的三项与状态含义（**本次实测结果**）：

| 测试 | 通过标准 | 对应 | 实测 |
|---|---|---|---|
| `tokenizer_roundtrip` | encode→decode 还原中文 | M1 | ✅ 通过 |
| `kv_cache_equivalence` | 开/关 cache 的 logits 误差 < 1e-4 | M3 | ✅ 通过（1.9e-06） |
| `perplexity_on_dev` | dev 困惑度 < 阈值（唐诗 50） | M4 | ❌ 375.32（未达标，分析见 report.md §4） |

- **[通过]** 满足契约；**[跳过]** 前置条件没就绪（如缺 ckpt），不是错误；**[失败]** 看 `error` 字段修。

---

## 8. 加分项（可选，做不动可以先跳过）

```bash
python src/benchmark_kv_cache.py            # S3：cache 开/关速度对比 -> docs/kv_cache_benchmark.md
python src/experiments.py --sweep param     # S1：参数量扫描   -> docs/experiments.md
python src/experiments.py --sweep posemb    # S2：RoPE vs 绝对位置编码
```

- **S1** 参数量 vs 困惑度：模型变大通常会变好，但数据只有 44KB 时会很快饱和/过拟合。
- **S2** 位置编码对比：`--pos_emb learned` 换用可学习绝对位置编码（代码里预留了开关）。
  关键看**训练用 128 上下文、测试喂更长序列**时的表现差异——绝对位置表没见过的位置只能瞎猜，RoPE 靠旋转可以外推。
- **S3** KV cache 加速比：序列越长差距越大（实测见 `docs/kv_cache_benchmark.md`）。
- **S4** TinyStories：`python data/download.py --dataset tinystories`（约 100MB，CPU 可训但慢），
  10M 参数模型就能学会简单叙事，能直观看到"涌现"。

---

## 9. 常见报错速查

| 现象 | 原因与处理 |
|---|---|
| `ckpt/tokenizer.json 不存在` → 自检跳过 | 先跑 `python src/tokenizer_train.py` |
| `ckpt/best.pt 不存在` → 自检跳过 | 先跑 `python src/train.py` |
| 自检 `perplexity_on_dev` 失败 | 训练不够/模型太小，加大 `--steps`、提高 `--d_model` 或加层数重训 |
| `kv_cache_equivalence` 失败 | 检查 cache 是否在序列维拼接、RoPE 的 `position_offset` 是否等于历史长度 |
| 控制台中文乱码 | `eval/run.py` 已自动处理；自己写脚本可设 `$env:PYTHONIOENCODING="utf-8"` |
| 训练 loss 偶发飙高 | 确认 `--grad_clip` 生效（默认 1.0） |
| 生成重复、绕圈 | 用 top-k/top-p，或加 `repetition_penalty`（`generate(..., repetition_penalty=1.2)`） |

---

## 10. 文件清单（我实现了什么）

```
task-2-mini-gpt/
├── run_all.py                    一条命令跑全部自检
├── src/
│   ├── tokenizer.py              M1  byte-level BPE（训练/编码/解码/存取）
│   ├── tokenizer_train.py        训练词表 -> ckpt/tokenizer.json
│   ├── rope.py                   M2  RoPE（预计算 cos/sin 表，支持位置偏移）
│   ├── attention.py              M2+M3 causal 多头注意力 + KV cache
│   ├── block.py                  RMSNorm / SwiGLU / Pre-LN Block
│   ├── model.py                  M2  MiniGPT（forward/generate/load_for_eval）
│   ├── sampling.py               M5  greedy / top-k / top-p / temperature
│   ├── train.py                  M4  预训练循环 + 困惑度监控 + 曲线
│   ├── generate.py               M5  生成样例对比 -> docs/generation_samples.md
│   ├── unit_check.py             自测：RoPE/KV cache/causal/采样/分词
│   ├── benchmark_kv_cache.py     可选加分项 S3：推理加速对比
│   └── experiments.py            可选加分项 S1/S2 扫描实验
├── ckpt/                         tokenizer.json / best.pt / 曲线 / 训练日志
├── docs/
│   ├── walkthrough.md            本文（手把手教程）
│   ├── report.md                 实验观察报告（200-500 字 + 数据）
│   ├── generation_samples.md     不同采样策略的生成样例
│   ├── kv_cache_benchmark.md     （可选）S3 结果
│   └── experiments.md            （可选）S1/S2 结果
└── eval/result.json              官方自检结果
```

---

## 11. 想从头复现一遍？三条命令

```bash
cd llm-beginner/task-2-mini-gpt
python data/download.py                                            # 1. 准备唐诗数据
python src/tokenizer_train.py --mode byte --vocab_size 2048         # 2. 训 BPE 词表（3 秒）
python -u src/train.py --d_model 128 --n_layers 4 --n_heads 4 \
       --block_size 128 --dropout 0.25 --steps 700 --lr 5e-4 \
       --warmup 200 --eval_interval 100                            # 3. 训练（约 5 分钟，CPU）
python run_all.py                                                  # 4. 自检 + 单元自测
python src/generate.py                                             # 5. 看四种采样策略的生成
```

（Windows 下如果中文输出乱码，先执行 `$env:PYTHONIOENCODING="utf-8"`。）
