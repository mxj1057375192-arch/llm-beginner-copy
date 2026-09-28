> 这份文档是给「第一次做这个任务」的人看的。它不替代 `README.md`（那是任务书），
> 而是把**每一步到底在干什么、为什么这么干、结果怎么看**讲清楚。
> 建议对照 `src/` 下的代码一边读一边跑。

---

# 任务一 · 手写 Transformer：从零到通过自检

## 0. 先搞清楚任务要什么

任务书里最重要的是 **Definition of Done 的 5 项必做**（M1–M5）和 **3 项自检**：

| 必做 | 内容 | 怎么证明 |
|---|---|---|
| M1 | 手写 `scaled_dot_product_attention` | 自检 `attention_correctness` 误差 < 1e-5 |
| M2 | 手写 `MultiHeadAttention` + `TransformerBlock` | 前向不报错、形状对 |
| M3 | 在 ChnSentiCorp 上训练分类器 | dev 准确率 ≥ 0.80（参考 ~0.85） |
| M4 | 加 causal mask 跑 toy 语言模型 | 自检 `causal_mask` 通过 |
| M5 | ≥ 3 张注意力热图 | `figures/` 下有图，且能说清模型在看什么 |

交付物一共 4 件：

```
ckpt/best.pt                训练好的模型
figures/*.png               ≥ 3 张注意力热图
eval/result.json            自检结果
报告文字                     200-500 字实验观察
```

**注意**：`eval/run.py` 是按固定接口名去 import 你的代码的，接口对不上就报
`ImportError`。所以第一件事不是写代码，是先看清 `README.md` 的「实现约定」表。

---

## 1. 目录里每个文件是谁写的、干什么用的

先建立这张地图，后面就不会迷路：

```
task-1-transformer/
├── README.md              任务书（要求、接口约定、评分标准）
├── requirements.txt        依赖清单
├── data/
│   ├── download.py         官方下载脚本（把 ChnSentiCorp 存成 parquet）
│   ├── download_offline.py 备用下载脚本（官方脚本在新版 datasets 上会失败，见 §2）
│   ├── train.parquet       9600 条训练数据
│   ├── validation.parquet  1200 条验证数据
│   └── test.parquet        1200 条测试数据
├── src/                    ★ 这里是要自己写的实现
│   ├── attention.py        M1 缩放点积注意力 + M2 多头注意力
│   ├── block.py            M2 encoder block（attention + FFN + residual + LayerNorm）
│   ├── model.py            M3 分类器 + 自检用的 load_for_eval 工厂
│   ├── tokenizer.py        字符级分词器
│   └── block_lm.py         M4 的 decoder-only block（多一个 causal mask）
├── train.py                M3 训练脚本
├── visualize.py            M5 画注意力热图
├── toy_lm.py               M4 toy 语言模型（唐诗 next-token 预测）
├── eval/
│   ├── run.py              ★ 自检脚本（不用改，按接口调用你的代码）
│   └── tutor_prompt.md     贴给大模型做 code review 的提示词
├── ckpt/                   训练产物（best.pt / 各消融模型）
└── figures/                图片产物（热图、loss 曲线）
```

**别去改 `eval/run.py`**。它是「考官」，你改它等于改考卷。
（任务书允许改，但改了要自己保证仍然检查同样的事情。新手不要动。）

---

## 2. 环境：本次实际踩到的坑（重要）

### 坑 1：`torch` import 直接崩

现象：

```
OSError: [WinError 1114] 动态链接库(DLL)初始化例程失败。
Error loading "...\torch\lib\c10.dll" or one of its dependencies.
```

这不是 `pip install` 没装好，而是 **MSVC 运行库版本冲突**。本机上：

- `D:\Anaconda\msvcp140.dll` 是 **14.27**（很老）
- Windows 自带的 `C:\Windows\System32\msvcp140.dll` 是 **14.50**
- PyTorch 的官方 wheel 是用更新的 MSVC 编的，DLL 初始化时加载到老版本就失败

只要 Python 解释器所在目录（或其 DLL 搜索路径）里有一份老的 `msvcp140.dll`，
就一定会崩。**验证方法**：先手动加载系统的新运行库，再加载 `c10.dll`：

```python
import ctypes, os
sys32 = r"C:\Windows\System32"
for n in ("msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll",
          "vcruntime140.dll", "vcruntime140_1.dll"):
    ctypes.WinDLL(os.path.join(sys32, n))      # 先把新的加载进来
ctypes.WinDLL(r"...\torch\lib\c10.dll")        # 这次就能成功
```

**解决办法**：不要用带老运行库的解释器建 venv。本机可用的干净解释器是
`D:\Anaconda\envs\py310\python.exe`（它的 `msvcp140.dll` 是 14.51，够新）：

```powershell
& "D:\Anaconda\envs\py310\python.exe" -m venv --system-site-packages D:\新生任务\.venv
& "D:\新生任务\.venv\Scripts\python.exe" -m pip install "numpy<2" pandas pyarrow matplotlib datasets
& "D:\新生任务\.venv\Scripts\python.exe" -c "import torch; print(torch.__version__)"
```

两个细节：

- `numpy<2`：`py310` 里那份 torch 2.1.0 是用 NumPy 1.x 编的，NumPy 2.x 会报
  `Failed to initialize NumPy: _ARRAY_API not found`（不致命，但会一直刷警告）。
- 本机没有 NVIDIA 显卡，所以装的是 `+cpu` 版本，训练会慢一些但完全能跑。

> 新手提示：**环境报错先看错误最后一行**（`Error loading ...`），
> 再用「最小复现」去定位。上面那个 6 行的 ctypes 脚本就是最小复现。

### 坑 2：官方 `data/download.py` 在新版 `datasets` 上失败

现象：

```
RuntimeError: Dataset scripts are no longer supported, but found ChnSentiCorp.py
```

`seamew/ChnSentiCorp` 是一个「带加载脚本」的老式数据集，而 `datasets>=4.0`
已经删掉了脚本式数据集的支持。

**两条路**：降到 `datasets<4`，或者直接下仓库里现成的 `.arrow` 文件自己转 parquet。
我加了 `data/download_offline.py` 走第二条路（不动原有文件）：

```powershell
python data/download_offline.py
# train: 9600 条 -> train.parquet
# validation: 1200 条 -> validation.parquet
# test: 1200 条 -> test.parquet
```

脚本里还顺手做了**样本数校验**（对比 `dataset_info.json` 里官方声明的数量），
这样如果下载被截断，会立刻 assert 失败而不是悄悄训出一个差模型。

### 坑 3：国内网络

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"     # HF 走镜像
$env:HTTP_PROXY  = "http://127.0.0.1:7890"     # pip 走本地代理（本机代理端口）
$env:HTTPS_PROXY = "http://127.0.0.1:7890"
```

---

## 3. M1：缩放点积注意力（`src/attention.py`）

### 公式

```
Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) V
```

### 怎么用大白话理解

把注意力想成**查字典**：

- `Q`（query，查询）：我手上这个词「想找什么」
- `K`（key，键）：每个词「能提供什么」的标签
- `Q·Kᵀ`：查询和每个标签的**匹配分**——分高 = 这两个词相关
- `softmax`：把匹配分变成一组和为 1 的权重（可解释成「关注度百分比」）
- `V`（value，值）：真正的**内容**
- 权重 × 内容再求和：**按关注度把相关内容加权平均回来**

### 代码里 4 行对应 4 步

```python
d_k = Q.size(-1)
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)   # 1) 打分
if mask is not None:
    scores = scores.masked_fill(mask, float("-inf"))             # 2) 屏蔽
attn = torch.softmax(scores, dim=-1)                             # 3) 归一化
out = torch.matmul(attn, V)                                      # 4) 加权求和
```

### 三个新手最容易写错的地方

**(1) 缩放因子是 `√d_k` 不是 `√d_model`。**
每个头只负责 `d_k = d_model / n_heads` 维。不缩放的话，点积的方差随维度线性增长，
分数会很大，`softmax` 被推到饱和区（一个位置接近 1，其余接近 0），
梯度几乎为 0，训练就废了。

**(2) mask 要填 `-inf`，不能乘 0。**
乘 0 只是让那个位置的**分数**变 0，`softmax` 之后它仍然会分到概率
（`e⁰ = 1` 并不小），信息就泄漏了。填 `-inf` 后 `e^{-inf} = 0`，才是真的屏蔽。
`eval/run.py` 的 `causal_mask` 这一项专门抓这个错。

**(3) `softmax` 的维度是最后一维（key 方向）。**
`attn[i][j]` 的含义是「第 i 个 query 分给第 j 个 key 的注意力」，
所以每一**行**加起来等于 1。

### 自检怎么跑

```powershell
python eval/run.py
```

```
[通过] attention_correctness: {"test": "attention_correctness", "pass": true, "max_abs_diff": 1.19e-07}
```

`max_abs_diff` 是和 PyTorch 官方 `F.scaled_dot_product_attention` 比出来的最大误差，
要求 `< 1e-5`。如果你写错缩放因子，这个数会变成 0.1 量级，一眼就能看出来。

---

## 4. M2：多头注意力 + Transformer block

### 为什么要多头

一个注意力头只能学**一种**对齐模式。多头 = 把 `d_model` 切成 `n_heads` 份
（每份 `d_k` 维），并行做 `n_heads` 次注意力，最后拼回来。
好处是有的头可能学会「盯情感词」，有的头学会「盯否定词」，模型容量更大。

### reshape 的顺序（最容易写错的地方）

```python
# 输入 x: (B, T, d_model)
Q = self.q_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)   # -> (B, H, T, d_k)
...
out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)          # 拼回去
```

记法：**先 view 把最后一维劈成 (H, d_k)，再 transpose 把 H 挪到 T 前面**。
`transpose` 之后内存不连续，必须 `.contiguous()` 才能 `view`，否则报
`RuntimeError: view size is not compatible with input tensor's size and stride`。

一个可以自己验证的形状断言：`(3, 7, 32)` 输入、4 个头 ->
输出 `(3, 7, 32)`、缓存的注意力权重 `(3, 4, 7, 7)`。7×7 就是 T×T 的注意力矩阵。

### TransformerBlock = attention + FFN + 两个 residual + 两个 LayerNorm

```python
x = x + Attn(LN(x))        # Pre-LN
x = x + FFN(LN(x))
```

- **残差 `x + ...`**：给梯度开一条「高速公路」，反向传播时梯度可以原封不动传回去，
  深层网络才训得动。
- **LayerNorm `LN(x)`**：把每个词元向量重新标准化（均值 0、方差 1），稳定训练。
- **Pre-LN vs Post-LN**：原始论文用 Post-LN（`x = LN(x + Attn(x))`），
  现代实现多用 Pre-LN。Pre-LN 的梯度路径更直接，**不用 warmup 也容易训稳**，
  所以这个任务选了 Pre-LN。

`block.py` 里留了 `use_residual` / `use_layernorm` 两个开关，是给加分项 S2
（拆掉看看还能不能收敛）用的，正常训练保持 `True`。

---

## 5. M3：训练文本分类器

### 数据长什么样

```
text:  "选择珠江花园的原因就是方便，有电动扶梯直接到达海边，周围餐馆..."
label: 1        # 0 = negative, 1 = positive
```

`text` 是中文句子，`label` 是二分类情感标签。**分词用最笨的办法**：
一个汉字 = 一个词元，词表从训练集统计出来（出现次数 < 2 的字归入 `<unk>`）。
任务一考的是 Transformer，不是分词，所以这里刻意不引入任何预训练模型。

### 模型前向的流水线（`src/model.py`）

```
ids (B, T)
 └─ 词嵌入 + 位置编码（乘 √d_model 对齐量级）      -> (B, T, d_model)
 └─ 算 padding mask：True 的位置是 PAD
 └─ N 层 TransformerBlock（每层都吃 mask）         -> (B, T, d_model)
 └─ masked mean pooling（只对真实词元求平均）      -> (B, d_model)
 └─ Linear 分类头                                  -> (B, num_classes) = logits
```

三个关键点：

**(1) padding mask**。一个 batch 里句子长短不一，短的要用 `<pad>` 补齐。
如果不加 mask，PAD 也会被当成正常词元参与注意力，污染表示。
mask 形状 `(B, 1, 1, T)`，靠广播自动变成 `(B, H, T_q, T_k)`。

**(2) masked mean pooling**。求平均时必须排除 PAD，否则句子越短被稀释得越厉害：

```python
h = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
denom = (~pad_mask).sum(dim=1, keepdim=True).clamp(min=1)
pooled = h.sum(dim=1) / denom
```

**(3) `load_for_eval` 工厂**。自检脚本只拿到一个 `ckpt` 路径，
所以 checkpoint 里必须同时存 **权重 + 词表 + 超参**，
这样自检不用知道你的模型结构就能复现：

```python
ckpt = torch.load(path, map_location="cpu")
model = TransformerClassifier(**ckpt["config"])
model.load_state_dict(ckpt["state_dict"])
tokenizer = CharTokenizer(ckpt["vocab"])
```

### 训练脚本里的工程细节（`train.py`）

| 技巧 | 为什么 |
|---|---|
| 按长度分桶组 batch | 一个 batch 内序列要补到等长，长短混在一起会产生大量无用的 PAD 计算 |
| AdamW + `betas=(0.9, 0.98)` | Transformer 的常规配置，`beta2` 小一点更稳 |
| 线性 warmup + 余弦衰减 | 前期学习率小，避免一开始就把参数带飞；后期衰减，收敛更精细 |
| 梯度裁剪 `clip_grad_norm_(1.0)` | Transformer 偶发梯度爆炸，裁一下更稳 |
| early stopping by dev acc | 用验证集准确率挑最好的那一轮存下来（`best.pt`） |

### 跑训练

```powershell
# 建议起点（任务书给的）
python train.py --d-model 128 --n-heads 4 --n-layers 4 --epochs 8 --tag d128_h4_l4

# 消融实验：换配置 + 换 tag，结果会自动追加到 figures/experiments.json
python train.py --d-model 128 --n-heads 8 --n-layers 4 --tag abl_h8 --out ckpt/abl_h8.pt
python train.py --d-model 128 --n-heads 4 --n-layers 4 --no-residual  --tag abl_nores
python train.py --d-model 128 --n-heads 4 --n-layers 4 --no-layernorm --tag abl_noln
```

每一轮会打印：

```
[3/8] train loss 0.1876 acc 0.9281 | dev loss 0.2455 acc 0.8917 | lr 2.79e-04 | 45.3s  <- best
```

怎么看：

- `train acc` 高、`dev acc` 低并且差距越来越大 -> **过拟合**（加大 dropout / 减小模型 / early stop）
- 两个都不涨、`dev loss` 还在震荡 -> **学习率太大** 或 **没 warmup**
- `train loss` 一直不动 -> **学习率太小** 或 **代码有 bug**（比如 mask 全 True 了）

---

## 6. M5：注意力热图怎么看

```powershell
python visualize.py
```

产出 4 张图：

| 文件 | 内容 |
|---|---|
| `figures/attn_positive.png` | 正面样本，最后一层多头平均 |
| `figures/attn_negative.png` | 负面样本 |
| `figures/attn_long.png` | 最长的样本（截到 60 个字，否则太密） |
| `figures/attn_layers.png` | 同一句话在每一层的注意力对比 |

**读图方法**（`visualize.py` 里已经标好了轴）：

- 横轴 = **被关注的词元**（key），纵轴 = **当前词元**（query）
- 第 i 行第 j 列的颜色深浅 = 「第 i 个词在多大程度上看了第 j 个词」
- 每一行加起来（对 j 求和）等于 1，因为 softmax 是按行归一化的

**要看什么**：

1. **对角线亮不亮** —— 亮说明模型大量在「看自己」，这是 self-attention 的常见模式
2. **关键词列亮不亮** —— 比如「不错」「失望」这些情感词所在的列是否被很多行关注，
   如果亮，说明模型确实学到了「根据情感词判断极性」
3. **层与层的变化** —— 浅层往往关注局部（相邻字），深层更关注全局语义，
   同时注意力也更「分散」
4. **句子级证据** —— 情感分类的注意力还有一条捷径：模型可能主要盯着
   「句首 + 句尾」或者干脆盯 PAD 之外的所有位置（也就是近似平均池化）。
   如果看到某一行几乎均匀分布，说明那个位置没有特别关注谁。

> 一句提醒：注意力权重 ≠ 重要性解释。注意力只是模型内部的一个中间量，
> 它**相关**于模型的判断依据，但不构成严格的因果解释。

---

## 7. M4：causal mask 与 toy 语言模型

### causal mask 是什么

语言模型在预测第 i 个字时，只能看到前 i 个字，绝对不能偷看后面的答案。
做法就是屏蔽注意力矩阵的**上三角**：

```python
causal = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)   # True = 屏蔽
```

`diagonal=1` 表示从主对角线**上方一条**开始，主对角线本身（看自己）保留。

### 自检怎么验

`eval/run.py` 的 `causal_mask` 用的是**行为验证**，比看代码更可靠：

1. 算一遍输出 `out`
2. 把最后一个位置的 `V` 改成 999（相当于「未来词元的答案变了」）
3. 再算一遍 `out2`
4. 比较**过去位置**的输出：`out[:, :, :-1]` 必须和 `out2` 完全一样

本次结果 `leaked_diff = 0.0`，说明未来信息没有泄漏。

### 真跑一个 toy 语言模型

```powershell
python toy_lm.py --steps 600     # 4 核 CPU 约 4 分钟；默认 1500 步约 8-12 分钟
```

这个脚本用**和分类器完全同一套** attention / block 代码（`src/block_lm.py`），
只把 mask 换成因果 mask，在仓库自带的唐诗语料上做字符级 next-token 预测：

```
[因果性验证] 改动最后 5 个词元后：
    过去 19 个位置的输出最大变化 = 0.000e+00   <- 应该约等于 0
    被改动位置的输出最大变化       = 2.670e+00   <- 应该明显 > 0
```

两行都要看：第一行是「没泄漏」，第二行是「模型确实用了上下文」。
如果两行都是 0，说明 mask 把**所有**位置都屏蔽了（比如写成全 `-inf`），
输出退化成常数 —— 这同样是个 bug。

---

## 8. 交给学长核实前，自查清单

```powershell
# 1. 数据在
ls data/*.parquet

# 2. 自检三项全绿
python eval/run.py
cat eval/result.json

# 3. 模型在
ls ckpt/best.pt

# 4. 图在（>= 3 张热图）
ls figures/*.png

# 5. 消融结果在
cat figures/experiments.json
```

`eval/result.json` 应该是这样（三项 `pass` 分别是 `true/true/true`）：

```json
[
  {"test": "attention_correctness", "pass": true, "max_abs_diff": 1.19e-07},
  {"test": "causal_mask", "pass": true, "leaked_diff": 0.0},
  {"test": "classifier_accuracy", "pass": true, "accuracy": 0.89, "baseline_reference": 0.85}
]
```

如果某一项是 `pass: null`，那是**[跳过]**不是失败 —— 表示前置条件没就绪
（比如 `ckpt/best.pt` 还没训出来），补齐后重跑即可。

---

## 9. 常见报错对照表

| 报错 | 原因 | 解决 |
|---|---|---|
| `Error loading "...\c10.dll"` | MSVC 运行库版本冲突 | 换干净解释器建 venv（见 §2 坑 1）|
| `Dataset scripts are no longer supported` | `datasets>=4` 移除了脚本式数据集 | 用 `data/download_offline.py` |
| `view size is not compatible with ... stride` | `transpose` 后没 `.contiguous()` | reshape 前加 `.contiguous()` |
| `attention_correctness` 误差 0.1 量级 | 缩放因子写成 `√d_model` 了 | 改成 `√d_k` |
| `causal_mask` 泄漏不为 0 | mask 用了乘 0 而不是 `-inf` | 用 `masked_fill(mask, -inf)` |
| `classifier_accuracy` 卡在 0.5 | mask 方向反了 / pooling 没排 PAD | 打印 `attn_mask` 看一眼 True 的位置 |
| `ImportError: cannot import name 'load_for_eval'` | 没按接口约定导出工厂函数 | 见 `src/model.py` 末尾 |

---

## 10. 想更进一步（加分项）

- **S1 消融**：head 数（1/2/4/8）、层数（1/2/4）、宽度（64/128）各跑一组，
  看准确率怎么变。经验规律：head 数的收益很快饱和，层数/宽度更值钱。
- **S2 拆组件**：`--no-residual`（去掉残差）通常直接训不动；
  `--no-layernorm` 往往还能跑但更慢更抖。这正好反过来印证 §5 里说的道理。
- **S3 冲 0.88+**：加大 `max_len`、多层堆叠、加 dropout 调参、训练更久。
- **S4 换 RoPE**：把可学习的绝对位置编码换成正弦编码或旋转位置编码（RoPE），
  对比长句表现。RoPE 是任务二的内容，提前预热正好。
