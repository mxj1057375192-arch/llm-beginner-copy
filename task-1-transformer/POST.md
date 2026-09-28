# 发帖正文草稿 · llm-beginner 任务一

> 提交位置：https://github.com/nndl/nndl-discussion/discussions
> 分类：**llm-beginner 实践成果**
> 用法：直接复制下面的内容发帖。图需要手动上传（在 figures/ 目录下）。

---

## 标题

```
【llm-beginner 任务一】手写 Transformer 做 ChnSentiCorp 情感分类，dev acc 0.9017
```

---

## 正文

### 1. 仓库链接

https://github.com/mxj1057375192-arch/llm-beginner-copy

实现代码在 `task-1-transformer/` 下：

| 文件 | 内容 |
|---|---|
| `src/attention.py` | M1 缩放点积注意力 + M2 多头注意力 |
| `src/block.py` | M2 Pre-LN encoder block（attention + FFN + residual + LayerNorm）|
| `src/model.py` | M3 分类器 + `load_for_eval` 工厂 |
| `src/block_lm.py` | M4 decoder-only block（causal mask）|
| `train.py` / `visualize.py` / `toy_lm.py` | 训练 / 可视化 / toy 语言模型 |
| `data/download_offline.py` | 备用下载脚本（原因见下方「踩坑」）|

### 2. `eval/result.json`

```json
[
  {
    "test": "attention_correctness",
    "pass": true,
    "max_abs_diff": 1.1920928955078125e-07
  },
  {
    "test": "causal_mask",
    "pass": true,
    "leaked_diff": 0.0
  },
  {
    "test": "classifier_accuracy",
    "pass": true,
    "accuracy": 0.9017,
    "baseline_reference": 0.85
  }
]
```

### 3. DoD Checklist

**必做 5 项**

- [x] **M1** 手写 `scaled_dot_product_attention`，自检 `attention_correctness` 通过（最大误差 1.19e-07 < 1e-5）
- [x] **M2** 手写 `MultiHeadAttention` + `TransformerBlock`，前向形状正确（`(3,7,32)`→`(3,7,32)`，注意力缓存 `(3,4,7,7)`）
- [x] **M3** ChnSentiCorp 训练分类器，dev 准确率 **0.9017**（≥ 0.80，参考基线 0.85）
- [x] **M4** 加 causal mask 跑 toy 语言模型，自检 `causal_mask` 通过（泄漏 0.0），并在唐诗上做了字符级 next-token 预测
- [x] **M5** 输出 6 张注意力热图（正面 / 负面 / 长句 / 各层对比 / 各 head 对比 / 熵分析）

**加分项**

- [x] **S1** head 数 / 层数 / 宽度消融（7 组配置 + 准确率表，见 `figures/experiments.json`）
- [x] **S2** 拆掉 residual / LayerNorm，记录收敛情况
- [x] **S3** dev 准确率 0.9017 > 0.88
- [ ] **S4** RoPE 对比（未做，留给任务二）

### 4. 注意力热图

（在帖子里上传这几张图）

- `figures/attn_positive.png` —— 正面样本，最后一层多头平均
- `figures/attn_negative.png` —— 负面样本
- `figures/attn_long.png` —— 最长样本（60 词元）
- `figures/attn_layers.png` —— 同一句子第 1~4 层对比
- `figures/attn_heads.png` —— 最后一层各 head 对比
- `figures/attn_entropy.png` —— **重点看这张**，注意力分布的熵分析

### 5. 实验观察（约 480 字）

我用 d_model=128 / n_heads=4 / n_layers=4（1.29M 参数）做 baseline，AdamW + 线性 warmup + 余弦衰减，6 轮后 dev 准确率 **0.9017**，第 5 轮最佳；第 6 轮 train acc 已到 0.951 而 dev 回落，是典型的过拟合拐点，early stopping 正好卡住。

**消融方面**做了 7 组：层数从 1 加到 4 只涨 0.6 个点（1 层 0.8958 → 4 层 0.9017），收益很快饱和；head 数在 4 附近饱和（1 头 0.8858，4 头 0.9017，8 头反而 0.8958）；把 d_model 从 128 砍到 64 掉了 5 个点，说明**宽度比深度更敏感**。S2 里去掉 residual 掉到 0.8508，且 train loss 一直是 baseline 的两倍，属于欠拟合而非过拟合——残差确实是深层可训练的前提；去掉 LayerNorm 还能收敛（0.8875），只是更抖，4 层这个深度对它没那么敏感。

**最反直觉的发现来自注意力本身。** 我算了每行注意力分布的熵：最后一层平均熵 4.092 nat，而完全均匀分布的上界 ln(60)=4.094，也就是达到了 **99.9%**；最大权重只有 0.021，而均匀值是 0.0167。也就是说这个模型**几乎没学出稀疏的词元对齐**，做的是近似平均池化。这解释了为什么层数/头数收益那么小——判别力主要来自 token embedding 和逐位置 FFN，注意力不是主要工具。所以热图看起来"一片平色"不是画错了，而且**「模型在看『不错』『失望』」这类解读在本实验里不成立**（虽然「不」「婚」「证」在 Top-5 里排前面，但领先均匀值太少，不足以当解释）。我的结论是：注意力权重只能当线索，要证明"看了哪个词"得靠消融或梯度归因。第 1 层熵略低（98.8%，最大权重 0.15），越往深层越均匀，这个层间趋势也挺有意思。

**踩坑记录**（供后来者参考）：① 本机 Anaconda 自带 MSVC 运行库 14.27 过旧，`import torch` 直接报 `WinError 1114 ... c10.dll`，换用自带 14.51 的解释器重建 venv 解决；② `datasets>=4` 已移除脚本式数据集支持，官方 `data/download.py` 必失败，我加了 `data/download_offline.py` 直取 HF 仓库的 `.arrow` 并转 parquet，同时校验样本数（9600/1200/1200）。详细过程写在 `task-1-transformer/LEARN.md`。

---

## 发帖检查清单

- [ ] 已 `git push` 到 fork，且仓库链接在浏览器能打开、能看到代码
- [ ] 6 张图已上传到帖子
- [ ] `eval/result.json` 文本已贴
- [ ] DoD checklist 已贴
- [ ] 实验观察 ≥ 200 字
- [ ] 分类选对了（llm-beginner 实践成果）
