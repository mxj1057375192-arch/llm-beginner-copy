# 任务三 · 指令微调与偏好对齐 —— 手把手读懂版（小白向）

> 这份文档是给第一次做这个任务的你写的：**先讲清楚每一块在干什么、为什么这么干，再给可复制的命令**。
> 代码全部带中文注释，配合本文一起看效果最好。
> 实验数据与观察在 [`../REPORT.md`](../REPORT.md)，自检结果在 `eval/result.json`。

---

## 0. 先用大白话讲：这个任务到底在干什么

### 0.1 一个「只会接话茬」的模型

Qwen2.5-0.5B 是一个**基座模型（base model）**。它读了海量文本，学会的唯一本领是：

```
输入：今天天气真好，我想
输出：出去走走。阳光洒在身上，暖洋洋的……      ← 它只是在"接着往下写"
```

你问它问题，它不会回答，而是**把你的问题当成文章开头继续写下去**：

```
输入：什么是深度学习？
输出：什么是机器学习？什么是神经网络？什么是……    ← 它在"续写"，不是在"回答"
```

### 0.2 我们要把它改造成「会对话的助手」

需要两个阶段，这也是任务的两条主线：

```
① SFT（监督微调）  —— 教它"对话的格式和套路"
       给它看几十万条「人问 → 助手答」的样例，让它学会：
       看到用户消息 → 该输出一段回答 → 该在哪里停下来

② DPO（偏好对齐）  —— 教它"什么样的回答更好"
       同一个问题给它看两份回答（一份好、一份差），
       告诉它"好的那份你应该更倾向于说"
```

打个比方：

| 阶段 | 类比 |
|---|---|
| base | 一个博览群书、但只会自顾自往下讲的人 |
| SFT 之后 | 他学会了"别人问你，你要先回答，说完要停" |
| DPO 之后 | 他还能分辨"这两种说法哪种更妥当" |

### 0.3 三个模型的关系（全文最重要的一张图）

```
                    Qwen2.5-0.5B（base）
                          │
                    ① 注入 LoRA
                          │
                          ▼
                   ┌─────────────┐
   MOSS 对话数据 ──▶│  SFT 训练   │──▶ ckpt/sft/lora.pt   ← 「SFT 模型」
                   └─────────────┘
                          │
                    ② 在 SFT 之上继续
                          │
                          ▼
                   ┌─────────────┐
  偏好数据(好/差) ─▶│  DPO 训练   │──▶ ckpt/dpo/lora.pt   ← 「DPO 模型」
                   └─────────────┘
```

注意：**SFT 和 DPO 训练出来的都只是 LoRA 权重（2.2 MB），不是整个模型（988 MB）**。
用的时候是「基座 + LoRA 权重」拼起来。为什么能这样？见第 4 节。

### 0.4 必做项（DoD）与代码的对应关系

| 要求 | 通俗解释 | 代码位置 |
|---|---|---|
| **M1** 手写 LoRA | 只训练一小撮参数，别动原模型 | `src/lora.py` |
| **M2** chat template + loss masking | 按 Qwen 格式拼对话；只在"助手的回答"上算 loss | `src/chat.py` |
| **M3** SFT | 用 MOSS 对话数据训练，产出 `ckpt/sft/` | `src/train_sft.py` |
| **M4** DPO + 对比 | 在 SFT 之上做偏好对齐，并对比三个模型 | `src/train_dpo.py` `src/compare.py` |

---

## 1. 完全照着敲：从零到跑通

> 前提：在 `llm-beginner/task-3-sft-dpo` 目录下，用 `D:\新生任务\.venv` 这个虚拟环境。

```bash
cd D:/新生任务/llm-beginner/task-3-sft-dpo
PY="D:/新生任务/.venv/Scripts/python.exe"          # 后面都用 $PY 代替

# ── 第 1 步：环境（只需做一次）────────────────────────
$PY -m pip install -r requirements.txt
$PY -m pip install transformers accelerate          # 任务三必需

# ── 第 2 步：下载基座模型（约 988 MB）──────────────────
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
  $PY data/download.py

# ── 第 3 步：准备数据（几 MB，不用下整个 2.68 GB）──────
$PY data/fetch_zip_head.py 800                      # MOSS 对话
curl -sL -o data/dpo/dpo_zh.json \
  "https://hf-mirror.com/datasets/hiyouga/DPO-En-Zh-20k/resolve/main/dpo_zh.json"

# ── 第 4 步：跑自检，看 M1/M2 过没过 ─────────────────
$PY eval/run.py

# ── 第 5 步：SFT 训练（约 11 分钟）───────────────────
$PY src/train_sft.py --limit 800 --steps 500

# ── 第 6 步：DPO 训练（约 18 分钟）───────────────────
$PY src/train_dpo.py --limit 400 --steps 300

# ── 第 7 步：三方对比，看效果 ────────────────────────
$PY src/compare.py

# ── 第 8 步：再跑一次完整自检，确认三项全过 ──────────
$PY eval/run.py
```

跑完你会得到：

```
ckpt/sft/lora.pt        SFT 的 LoRA 权重（2.2 MB）
ckpt/dpo/lora.pt        DPO 的 LoRA 权重（2.2 MB）
ckpt/compare.md         base / SFT / DPO 三方输出对比
eval/result.json        自检结果
```

**本机实测总耗时约 1 小时**（含下载）。其中下载占了大头，训练本身只要半小时。

> 每一步卡住的话，直接跳到第 9 节「出问题了怎么办」。

---

## 2. 环境准备（这里坑最多，先说清楚）

### 2.1 Hugging Face 下载：两个必设的环境变量

```bash
export HF_ENDPOINT=https://hf-mirror.com     # 国内镜像，直连 huggingface.co 不通
export HF_HUB_DISABLE_XET=1                  # 关键！否则 401 报错
```

**第二个变量为什么要加？** Hugging Face 新推出了叫 Xet 的传输协议，但 hf-mirror 镜像**不代理**它。
不关掉的话，下载会跳过镜像去连 `cas-server.xethub.hf.co`，然后报：

```
RuntimeError: CAS Client Error: HTTP status client error (401 Unauthorized)
```

### 2.2 虚拟环境的一个隐藏坑：`--system-site-packages`

这台机器的 `D:\新生任务\.venv` 里，`pyvenv.cfg` 写着：

```ini
include-system-site-packages = true
```

意思是：**这个 venv 能看见 Anaconda 里装的包**。平时没事，但会撞上这个：

Anaconda 的 py310 里装着 `torchaudio 2.1`（为旧版 torch 编译的）。而升到 torch 2.14 之后，
transformers 5 在加载模型时会执行 `import torchaudio`，旧包加载自己的原生库失败，于是**整个自检直接挂掉**：

```
OSError: Could not load this library: ...\torchaudio\lib\libtorchaudio.pyd
```

解决办法不是卸 Anaconda 的包（那是另一个环境，别动），而是往 venv 里装一个新版把它盖住：

```bash
$PY -m pip install --ignore-installed --no-deps torchaudio
```

torchaudio 现在已经是被废弃的库，2.11 版是个 **0.3 MB 的空壳包、没有任何依赖**，
所以装它既不会降级 torch，也不需要真的用到它——只是让 `import` 能过去。

### 2.3 torch 版本

任务要求 `torch >= 2.7`。装的时候注意：**PyPI 上 Windows 的 torch 轮子是 124 MB 的 CPU 版**
（不是 2.5 GB 的 CUDA 版），所以升级成本可以接受：

```bash
$PY -m pip install --upgrade "torch>=2.7"
```

---

## 3. 数据长什么样

### 3.1 MOSS-003-sft：给 SFT 用

官方数据集是一堆对话。**原始格式长这样**（一行一条，jsonl）：

```json
{
  "conversation_id": 1,
  "num_turns": 5,
  "category": "Brainstorming",
  "chat": {
    "turn_1": {
      "Human": "<|Human|>: 如何保障工作中遵循正确的安全准则？<eoh>\n",
      "Inner Thoughts": "<|Inner Thoughts|>: None<eot>\n",
      "Commands": "<|Commands|>: None<eoc>\n",
      "Tool Responses": "<|Results|>: None<eor>\n",
      "MOSS": "<|MOSS|>: 为了保障……以下是一些建议：……<eom>\n"
    },
    "turn_2": { ... }
  }
}
```

三个要注意的地方：

1. **它有自己的控制符**：`<|Human|>:` 开头、`<eoh>` 结尾。这些是 MOSS 自己的格式，
   我们用的是 Qwen 的格式，所以**必须剥掉**（[src/data.py](../src/data.py) 的 `clean_human` / `clean_moss`）。
2. **它有很多轮**：实测 800 条样本里，绝大多数是 4–8 轮。整段喂进去序列会非常长，
   CPU 根本跑不动。所以我们**只取前 2 轮**（`--max-turns 2`）。
3. `Inner Thoughts` / `Commands` / `Tool Responses` 在 no-tools 子集里全是 None，忽略即可。

### 3.2 为什么不用下载整个 2.68 GB？

MOSS 的压缩包有 **2.68 GB**，但我们只需要几百条对话。所以用了个小技巧
（见 [data/fetch_zip_head.py](../data/fetch_zip_head.py)）：

```
① 用 HTTP Range 请求读文件的"最后 64 KB" → 里面有压缩包的目录，能查到数据从第几字节开始
② 再 Range 请求那个位置的开头几 MB → 边解压边读，读够 800 行就停
```

结果：**2.68 GB → 几 MB**，省了 20 多分钟。这个技巧对任何"只想取开头一部分"的大压缩包都适用。

### 3.3 DPO-En-Zh-20k：给 DPO 用

```json
{
  "conversations": [{"from": "human", "value": "请写一份关于……"}],
  "chosen":   {"from": "gpt", "value": "（更好的回答）"},
  "rejected": {"from": "gpt", "value": "（更差的回答）"}
}
```

这就是「同一个问题 + 一份好答案 + 一份差答案」。DPO 训练就是在学怎么区分这两者。

---

## 4. M1：手写 LoRA（`src/lora.py`）

### 4.1 为什么不直接微调整个模型？

一个 0.5B 的模型有 **4.9 亿个参数**。全量微调的代价：

| 问题 | 说明 |
|---|---|
| 显存/内存 | 不光要存参数，还要存每个参数的梯度 + 优化器状态（AdamW 要存 2 份），直接翻 3 倍 |
| 存不下 | 每训一版就要存一份 ~2 GB 的完整权重 |
| 容易遗忘 | 改得太狠，模型原来会的东西可能就忘了（灾难性遗忘） |

### 4.2 LoRA 的想法：别重印整本书，旁边贴张勘误表

LoRA 的假设是：**微调带来的改变，其实是很"低秩"的**——通俗说就是"没那么多花样"，
不需要一个完整的 896×896 矩阵来描述，用一个「896×8 的细长矩阵」乘上「8×896 的细长矩阵」就够了。

```
原来：  y = W₀ · x                    W₀ 是 896×896，冻结不动
现在：  y = W₀ · x + (alpha/r) · B · A · x
                    └────────┬────────┘
                      只训练这个部分
        A 是 896×8，B 是 8×896      ← 一共才 14,336 个数，而不是 802,816
```

这叫"低秩分解"：用两个小矩阵相乘来近似一个大矩阵。`r`（这里是 8）叫**秩**，越小参数越少。

### 4.3 三个必须做对的细节（任务 README 里点名的"坑"）

```python
# ① 形状别搞反
self.lora_A = nn.Parameter(torch.empty(in_features, r))   # in × r
self.lora_B = nn.Parameter(torch.zeros(r, out_features))  # r × out

# ② 初始化：A 随机，B 必须是零！
nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
# B 保持全零 → 训练开始时 B·A = 0 → 模型输出和注入前【一模一样】
# 如果 B 也随机初始化，一注入就把模型的输出带偏了

# ③ 缩放必须写成 alpha / r，而不是直接用 alpha
self.scaling = alpha / r
# 这样将来把 r 从 8 改成 16 时，等效学习率不会跟着漂
```

### 4.4 最容易漏的一步：冻结

这是自检 `lora_param_count`（要求可训参数 < 5%）**能不能过的关键**：

```python
# 先全部冻结，再只放开 LoRA 的 A、B
for p in model.parameters():
    p.requires_grad = False
for m in model.modules():
    if isinstance(m, LoRALinear):
        m.lora_A.requires_grad = True
        m.lora_B.requires_grad = True
```

如果只冻结了被替换的那两层，剩下的 `k_proj` / `o_proj` / `gate_proj` / `up_proj` / `down_proj`
加起来有 **3 亿多参数**都是可训的，占比轻松超过 5%，自检直接挂。

**实测结果：可训参数 540,672 / 总数 494,573,440 = 0.109%** ✅

### 4.5 `merge_lora` 是干嘛的？

训练完，LoRA 权重是**单独存在旁边**的。推理时有两种选择：

- **不合并**：每次前向都多算一次 `B(Ax)`，慢一点，但灵活（可以切换不同 LoRA）
- **合并**：把 `(alpha/r)·B·A` 直接加进原来的 `W₀`，变成一个普通矩阵，推理没有任何额外开销

`merge_lora` 做的就是第二件事。注意**合并后的前向结果必须和合并前一致**——
自检不测这个，但是 AI 助教会查。实测最大误差 6e-5（fp32 的正常舍入误差）：

```
LoRA 分支数: 合并前 48 → 合并后 0
最大绝对误差: 6.03e-05
前向一致: True
```

---

## 5. M2：chat template 与 loss masking（`src/chat.py`）

### 5.1 什么是 chat template？

模型是"认 token 不认角色"的。多轮对话必须被拍平成**一个字符串**，用特殊标记区分谁在说话。
Qwen 的格式长这样：

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
你好！很高兴见到你。<|im_end|>
```

拆开看：

| 片段 | 作用 |
|---|---|
| `<|im_start|>` | 一条消息开始 |
| `system` / `user` / `assistant` | 这条消息是谁说的 |
| `<|im_end|>` | 这条消息结束 |

**`<|im_end|>` 特别重要**——模型就是靠学会输出它来"停下来"的。base 模型不会停，
所以它会一直往下写（第 8 节对比里 base 的复读就是这个原因）。

> 我们手写的 `format_messages` 和 Qwen 官方的 `tokenizer.apply_chat_template` 做了**逐字比对**，
> 三种用例（有 system / 无 system / 单轮）全部一致。

### 5.2 什么是 loss masking？为什么需要？

先说 loss 是什么：训练时模型每预测一个 token，都会和"正确答案"比一下，算出一个误差（loss），
然后朝着减小误差的方向调参数。

问题来了。一段对话文本长这样：

```
<|im_start|>system
You are a helpful assistant.<|im_end|>        ← 这是我们自己写的模板，不是模型该学的
<|im_start|>user
你好<|im_end|>                                 ← 这是用户说的，模型不该学"怎么当用户"
<|im_start|>assistant
你好！很高兴见到你。<|im_end|>                  ← 只有这段是模型该学的
```

如果全都算 loss，模型会学到一堆不该学的东西：学着复读 `<|im_start|>`、
学着模仿用户的提问口吻。**所以要把不该算的地方"屏蔽"掉。**

PyTorch 里的做法：`labels` 里对应位置填 **-100**，交叉熵损失函数会自动忽略 -100 的位置。

```python
labels = [ -100, -100, ...,  -100, 1234, 5678, ..., -100 ]
#          system 部分全屏蔽    ↑ 只有 assistant 的回答保留真实 token id
```

### 5.3 怎么实现的？

难点是：**怎么知道 `input_ids` 里哪些位置属于 assistant 的回答？**

用 tokenizer 的 `offset_mapping`——它会告诉你每个 token 对应原文的**第几个字符到第几个字符**：

```python
enc = tok(text, return_offsets_mapping=True)
# enc["offset_mapping"][i] = (起始字符位置, 结束字符位置)
```

于是流程是：

```
① 拼出完整文本时，顺手记下每段 assistant 回答的字符区间 (起始, 结束)
② 分词时拿到每个 token 的字符区间
③ 两者的区间有重叠 → 这个 token 属于回答 → 保留
   否则 → 填 -100
```

**实测结果：mock 多轮对话的 mask 比例 0.643**，且解码回来后恰好是两轮回答的内容：

```
参与 loss 的文本: '你好！很高兴见到你。深度学习是机器学习的一个分支，使用多层神经网络。'
                  └── 第一轮回答 ──┘ └────────── 第二轮回答 ──────────┘
```

可以看到**多轮的每个 assistant turn 都参与了训练**——如果只算最后一轮，就是任务 README 里点名的坑。

---

## 6. M3：SFT 训练（`src/train_sft.py`）

### 6.1 训练循环在干嘛

```python
for input_ids, labels, attn in loader:
    out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
    loss = out.loss          # 模型自己算好的交叉熵（已忽略 -100）
    loss.backward()          # 反向传播：算出每个 LoRA 参数该往哪调
    opt.step()               # 更新参数
    opt.zero_grad()          # 清空梯度，准备下一步
```

就这四行，重复 500 次。

几个参数的作用：

| 参数 | 我们用的值 | 为什么 |
|---|---|---|
| `--limit` | 800 | 拿多少条对话来训 |
| `--max-length` | 128 | 超过 128 个 token 就截断。**这个值直接决定训练速度** |
| `--batch-size` | 1 | 一次喂几条。CPU 内存有限，就 1 条 |
| `--lr` | 2e-4 | 学习率，每次调参数的幅度 |
| `--rank` | 8 | LoRA 的秩 r，越大参数越多、能力越强 |

### 6.2 怎么判断训练有没有效果？

看 loss 有没有下降。本次实测：

```
前 50 步均值: 1.276
末 50 步均值: 1.185
```

**降得不多是正常的**，原因有二：

1. batch size = 1，每一步只看了 1 条样本，loss 抖得很厉害（单步可能在 0.7–1.7 之间跳）
2. 语言模型的 loss 本来就不太会降到接近 0——文本本身就有不确定性，
   "你好！很高兴见到你。"后面接什么都有可能，1.2 左右是正常水平

**更可靠的判断方式不是看 loss，而是直接看输出**（第 8 节）。

---

## 7. M4：DPO 训练（`src/train_dpo.py`）

### 7.1 DPO 想解决什么问题

SFT 之后模型会对话了，但它不知道**什么样的回答更好**。传统做法是 RLHF：
另外训一个"打分模型"给回答打分，再用强化学习让主模型去刷高分——又复杂又难训。

DPO 的贡献是：**证明可以跳过打分模型，直接用一个简单的损失函数达到类似效果。**

### 7.2 损失函数逐项拆解

```
L = -log σ( β · [ (log π/π_ref)(chosen) - (log π/π_ref)(rejected) ] )
```

一个个看：

| 符号 | 含义 | 通俗解释 |
|---|---|---|
| `chosen` / `rejected` | 好回答 / 差回答 | 数据里给的 |
| `log π(y)` | 当前模型说这句话的对数概率 | 把所有 token 的概率加起来 |
| `log π_ref(y)` | **参考模型**说这句话的对数概率 | 参考模型 = SFT 权重的冻结副本 |
| `π/π_ref` | 两者相除（取对数就是相减） | 「相对于参考模型，现在的模型有多想说这句」 |
| 方括号里的差 | chosen 的比值 − rejected 的比值 | 「模型更喜欢好回答多少」 |
| `β` | 0.1 | 控制偏离参考模型的惩罚强度 |
| `-log σ(...)` | | margin 越大 → loss 越小 |

**为什么要除以参考模型？** 防止模型为了拉大差距而走极端（比如把所有概率都堆到某几个词上）。
参考模型像个"锚"，拉着模型别偏太远。

### 7.3 实现里最容易出错的地方

```python
# ① reference 必须冻结，且只跑 forward
ref.eval()
for q in ref.parameters():
    q.requires_grad = False
with torch.no_grad():                       # ← 别漏了这个
    ref_w = sequence_logprob(ref, ...)

# ② 一步里总共 4 次 forward
pi_w  = sequence_logprob(policy, chosen)    # policy × 2
pi_l  = sequence_logprob(policy, rejected)
ref_w = sequence_logprob(ref, chosen)       # ref × 2（不参与反向）
ref_l = sequence_logprob(ref, rejected)
```

### 7.4 一个漂亮的自检信号：第一步 loss 必定是 0.6931

训练刚开始时，policy 就是从 SFT 载入的，和 reference **一模一样**。所以：

```
(log π/π_ref)(chosen) = 0     （同一个模型，比值当然是 1，取对数就是 0）
(log π/π_ref)(rejected) = 0
margin = β · (0 - 0) = 0
loss = -log σ(0) = -log(0.5) = 0.6931 = ln 2
```

**如果你跑出来第一步 loss 不是 0.6931，说明代码有问题。** 这是个非常好用的正确性检查。

### 7.5 reward margin 怎么看

margin 就是上面方括号里的那个差值，**它涨了就说明模型确实学会了区分好坏**：

```
reward margin: +0.017（前 50 步） → +0.208（末 50 步），峰值 +1.33
loss:            0.690          →  0.636
```

注意 margin 抖动很大（最低到 -0.76），同样是 batch size = 1 的锅。
**看趋势，别看单步。**

---

## 8. 对比三个模型（`src/compare.py`）

```bash
$PY src/compare.py --max-new-tokens 60
```

它会把同一批指令分别喂给 base / SFT / DPO，输出并排打印，同时写进 `ckpt/compare.md`。

### 最有代表性的一题

```
把「我昨天去了图书馆借了三本书」改写成更正式的书面语

base: 改写成更正式的书面语是「我昨天去了图书馆借了三本书」改写成更正式的书面语是「我昨天…（无限循环）
SFT : 好的，以下是改写后的书面语：昨天我去图书馆借了三本书。
DPO : "昨天，我去了图书馆借了三本书。" 这句话已经比较正式了，可以改为更正式的书面语形式。…
```

base 那个不是"答得不好"，是**根本停不下来**——它没学过 `<|im_end|>`，
不知道回答完了该结束，于是一直往下续写。

### 其他对比

| 题 | base | SFT / DPO |
|---|---|---|
| 提高睡眠质量的建议 | 跑飞到英文，中英混杂乱码复读 | 稳定输出中文分条建议 |
| 翻译"今天天气很好" | 复读原句后接一串 `iente You are a helpful assistant.` 乱码 | 至少不崩坏（虽然也没翻对，毕竟只训了 500 步） |
| 介绍自己 | "我可以回答您的问题…我可以回答您的问题…" 复读 | 完整、结构化的自我介绍 |

**SFT 带来的最大收益不是"变聪明"，而是"格式稳定"**——学会对话模板、知道在哪里停。
这正好印证了任务设计里"先会说话再分好坏"的两阶段逻辑。

DPO 的效果更微妙：回答通常比 SFT 更长、更爱展开解释，但也带来了重复问题
（写诗时出现"凉爽爽，凉爽爽"）。考虑到只跑了 300 步、batch=1，这更像是训练不充分，而不是 DPO 本身的毛病。

---

## 9. 出问题了怎么办

| 症状 | 原因 | 解决 |
|---|---|---|
| `401 Unauthorized` / `cas-server.xethub.hf.co` | hf-mirror 不支持 Xet 协议 | `export HF_HUB_DISABLE_XET=1` |
| 下载卡住不动 | 同上，或没设镜像 | `export HF_ENDPOINT=https://hf-mirror.com` |
| `Could not load this library: libtorchaudio.pyd` | venv 继承了 Anaconda 的旧 torchaudio | `pip install --ignore-installed --no-deps torchaudio` |
| `Disabling PyTorch because PyTorch >= 2.5 is required` | torch 版本太低 | `pip install --upgrade "torch>=2.7"` |
| 自检显示 `[跳过]` 而不是 `[通过]` | 模型没下到 `models/Qwen2.5-0.5B` | 自检按**固定路径**找基座，位置不能改 |
| `没有筛出任何偏好对` | DPO 数据被长度过滤没了 | 调大 `--max-length`，或调小 `--max-prompt-chars` |
| 训练慢得受不了 | `--max-length` 太大 | 序列长度和耗时**成正比**，128 比 256 快一倍 |
| 内存不够（进程被杀） | DPO 要同时装两个模型（policy + ref） | 调小 `--limit`，或关掉其他程序 |
| 中文输出乱码 | Windows 控制台默认 GBK | 命令前加 `PYTHONIOENCODING=utf-8` |

---

## 10. 名词速查

| 词 | 一句话解释 |
|---|---|
| **base model** | 只做过预训练、只会续写的模型，不会对话 |
| **SFT** | 监督微调：用「人问→助手答」的样例教模型对话格式 |
| **DPO** | 直接偏好优化：用「好答案/差答案」的对比教模型审美，不需要打分模型 |
| **RLHF** | 传统对齐方法：训练打分模型 + 强化学习，比 DPO 复杂 |
| **LoRA** | 冻结原权重，只在旁边训练一对小矩阵的低成本微调法 |
| **rank（秩）** | LoRA 里小矩阵的宽度 r，越大参数越多 |
| **chat template** | 把多轮对话拍平成字符串的格式约定 |
| **loss masking** | 把不该算 loss 的位置标成 -100，让模型只学该学的部分 |
| **reference model** | DPO 里的"锚"，SFT 权重的冻结副本，防止模型跑偏 |
| **reward margin** | DPO 里「模型更喜欢好答案多少」，涨了说明学到了 |
| **token** | 模型眼里的最小单位，一个汉字可能是 1–2 个 token |
| **logprob** | 对数概率，几个概率相乘时用对数就变成相加，方便计算 |

---

## 11. 接下来还能做什么（加分项）

必做项已经跑通。如果想深入，任务 README 里列了 5 个加分项，按性价比排序：

1. **LoRA rank 消融**（S2）：把 `--rank` 改成 4 / 8 / 16 / 32 各跑一遍，看输出质量怎么变 —— 改动最小，只要一条命令跑四次
2. **SFT-only vs SFT+DPO**（S4）：画 reward margin 曲线 —— 数据已经在 `ckpt/dpo/train_args.json` 里了
3. **灾难性遗忘评估**（S3）：在 C-Eval 子集上比 base vs SFT，看微调有没有把原来的知识弄丢
4. **全量微调 vs LoRA**（S1）：显存和质量对比 —— 这台机器没显卡，会比较痛苦
5. **贯通任务五**（S5）：用 `moss-003-sft-plugin` 训一版会调工具的模型

想提升效果最直接的两招：**加大 `--limit` 和 `--steps`**（CPU 上 1.3 秒/步，跑 2000 步也就 45 分钟），
以及**加大 `--batch-size`**（能让 loss 和 margin 曲线平稳得多）。
