# 任务三实验报告：指令微调与偏好对齐

## 一、自检结果

```
[通过] lora_param_count: {"trainable": 540672, "total": 494573440, "ratio": 0.00109}
[通过] loss_masking:     {"mask_ratio": 0.643}
[通过] sft_vs_base:      {"pass": true}
```

完整结果见 [eval/result.json](eval/result.json)。

## 二、实现

| 文件 | 内容 |
|---|---|
| [src/lora.py](src/lora.py) | 手写 LoRA：`inject_lora` / `merge_lora` |
| [src/chat.py](src/chat.py) | Qwen chat template + loss masking |
| [src/data.py](src/data.py) | MOSS jsonl 解析与多轮对话转换 |
| [src/train_sft.py](src/train_sft.py) | SFT 训练循环 |
| [src/train_dpo.py](src/train_dpo.py) | DPO 训练循环 |
| [src/compare.py](src/compare.py) | base / SFT / DPO 三方对比 |

关键实现点：

1. **LoRA**：`A: in×r` 用 kaiming 初始化、`B: r×out` 零初始化（保证注入瞬间 ΔW=0，模型行为与注入前逐位一致）；缩放写成 `alpha / r`；forward 为 `y = W₀x + scaling·B(Ax)`。注入后**先冻结全部参数、再只放开 A/B** —— 不这样做的话 k_proj / o_proj / gate/up/down_proj 合计 3 亿多参数都可训，占比远超 5%。实测可训占比 **0.109%**。
2. **chat template**：手工拼接 `<|im_start|>role\ncontent<|im_end|>\n`，首条非 system 时插入 Qwen 默认 system prompt。已与 `tokenizer.apply_chat_template` 逐字比对，三种用例（有/无 system、单轮）全部一致。
3. **loss masking**：用 `offset_mapping` 拿到每个 token 的字符区间，只保留落在 assistant **回答内容**内的 token，user / system / 模板控制符全部 -100。多轮对话的**每个** assistant turn 都参与训练。mock 用例 mask 比例 0.643。
4. **DPO**：`-log σ(β·[(log π/π_ref)(chosen) - (log π/π_ref)(rejected)])`，β=0.1；reference 为 SFT 权重的冻结副本，只跑 forward。每步 4 次 forward（policy×2 + ref×2）。

## 三、训练配置与结果

| | SFT | DPO |
|---|---|---|
| 数据 | MOSS-003-sft 800 条对话（取前 2 轮） | hiyouga/DPO-En-Zh-20k 筛出 400 对 |
| 步数 / 耗时 | 500 步 / 11 分钟 | 300 步 / 18 分钟 |
| loss | 1.276 → 1.185（前/末 50 步均值） | 0.690 → 0.636 |
| 其他 | — | **reward margin +0.017 → +0.208** |

环境：i5-1135G7（4 核 8 线程）+ 16 GB 内存，**纯 CPU**，无独显。序列截断到 128 token，batch size 1。

一个值得记的正确性佐证：DPO 第一步的 loss 恰好是 **0.6931 = ln 2**。因为初始时 policy 与 reference 完全相同，log 比值之差为 0，margin=0，而 `-log σ(0) = ln 2`。数值对得上说明 DPO 损失与序列对数概率的计算都是对的。

## 四、实验观察（约 400 字）

三方对比的完整输出见 [ckpt/compare.md](ckpt/compare.md)。最直观的现象是 **base 模型存在明显的重复退化**：

- 第 7 题「把句子改写成书面语」，base 陷入死循环——"改写成更正式的书面语是「我昨天去了图书馆借了三本书」改写成更正式的书面语是……" 无限复读；SFT 给出 "好的，以下是改写后的书面语：昨天我去图书馆借了三本书。"，干净且正确。
- 第 3 题「提高睡眠质量的建议」，base 直接**跑飞到英文**（"Lauderdale, 100% sleep quality is a matter of habit…"），并伴随中英混杂的乱码式重复；SFT 和 DPO 都稳定输出中文分条建议。
- 第 4 题翻译任务三个模型都没做对（都只是复读了原句），但 base 会在复读后接上一串 "iente You are a helpful assistant." 的乱码，SFT/DPO 至少不会崩坏。

SFT 带来的最大收益不是"更聪明"，而是**格式稳定**：模型学会了对话模板，知道该在哪里停下来（`<|im_end|>`），不再无限续写。这印证了任务设计里"先会说话再分好坏"的两阶段逻辑。

DPO 的效果则更微妙。reward margin 从 +0.017 升到 +0.208，说明模型确实学会了区分 chosen / rejected；反映在输出上，DPO 的回答通常比 SFT **更长、更爱展开解释**（如第 8 题加了更多情节描述）。但 DPO 也带来了明显的**重复问题**（第 5 题写诗时出现 "凉爽爽，凉爽爽，凉爽爽" 的复读）。考虑到只跑了 300 步、batch size 为 1、margin 曲线抖动很大（-0.76 到 +1.33），这更像是训练不充分导致的退化，而不是 DPO 本身的缺陷——真正的偏好对齐需要更大的 batch 和更多数据。

## 五、踩到的坑

1. **hf-mirror 不支持 Xet 传输协议**：`snapshot_download` 报 401 Unauthorized（`cas-server.xethub.hf.co`），需设 `HF_HUB_DISABLE_XET=1` 退回传统 HTTP。
2. **venv 继承 Anaconda 的过期包**：`.venv` 的 `pyvenv.cfg` 里 `include-system-site-packages = true`，会看到 Anaconda py310 里的 torchaudio 2.1。而 transformers v5 的 `audio_utils.py` 在模块顶层无条件 `import torchaudio`，旧包配 torch 2.14 加载原生库失败，直接让自检挂掉。解决办法是往 venv 装一个 torchaudio 2.11——它已是被废弃的 0.3 MB 无依赖存根包，能盖住旧版本又不会降级 torch。
3. **大文件不必全下**：MOSS 的 `moss-003-sft-no-tools.jsonl.zip` 有 **2.68 GB**，但用 HTTP Range 请求读它的中央目录、再流式解压开头几 MB，就能拿到需要的对话（见 [data/fetch_zip_head.py](data/fetch_zip_head.py)）。省了 20 多分钟。
4. **CPU 训练比预期快得多**：原本估算 10–20 秒/步，实测约 **1.3 秒/步**（序列 128 token、batch 1）。所以训练量从"几十步意思一下"提到了 500 + 300 步。
