"""任务一 M5：可视化注意力热图。

用法：
    python visualize.py

产出（figures/ 下）：
    attn_positive.png   正面样本的注意力热图（最后一层，多头平均）
    attn_negative.png   负面样本的注意力热图
    attn_long.png       长句样本的注意力热图
    attn_layers.png     同一个句子在各层的注意力对比
    attn_heads.png      同一个句子、最后一层各个 head 的对比
    attn_entropy.png    注意力分布的"均匀程度"量化（回答"热图为什么看不清"）

注意：这批热图的信息量取决于注意力是否"尖锐"。如果每行的注意力分布都接近
均匀分布，图看上去就是一片平色 —— 这不是画错了，而是模型确实没学到稀疏的
对齐关系。`attn_entropy.png` 就是用来判断这一点的量化证据。
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.model import load_for_eval

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Windows 自带的中文字体，让热图上的汉字正常显示
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False


def get_attention(model, tokenize_fn, text, max_display=60):
    """跑一次前向，取出每一层的注意力权重。

    返回 (token 列表, 每层的注意力矩阵列表 (T, T))。
    注意力矩阵已对所有 head 取平均 —— 单个 head 往往只学到某个特定模式，
    平均之后更能看出「整体上模型在关注哪些词」。
    """
    ids = tokenize_fn(text)[:max_display]
    model.eval()
    with torch.no_grad():
        model(ids.unsqueeze(0))          # forward 会把权重缓存到每个 block 上

    tokens = [ch if ch.strip() else "·" for ch in text[:len(ids)]]
    per_layer = []
    for block in model.blocks:
        w = block.attn.attn_weights        # (1, H, T, T)
        per_layer.append(w[0].mean(dim=0).cpu().numpy())   # 对 head 求平均 -> (T, T)
    return tokens, per_layer


def get_attention_per_head(model, tokenize_fn, text, max_display=60):
    """和 get_attention 一样，但**不**对 head 求平均。

    返回 (tokens, 每层注意力 (H, T, T))。
    单看某个 head 有时能看到平均后被抹掉的模式，所以两者都留着。
    """
    ids = tokenize_fn(text)[:max_display]
    model.eval()
    with torch.no_grad():
        model(ids.unsqueeze(0))
    tokens = [ch if ch.strip() else "·" for ch in text[:len(ids)]]
    return tokens, [b.attn.attn_weights[0].cpu().numpy() for b in model.blocks]


def row_entropy(weights):
    """每行注意力分布的熵（自然对数，单位 nat）。

    熵 = 「这个位置的注意力有多分散」：
        熵 ≈ ln(T)  -> 几乎均匀分布（谁都没特别关注）
        熵 ≈ 0      -> 几乎全部押在某一个位置（非常尖锐）
    """
    p = np.clip(weights, 1e-12, None)
    return -(p * np.log(p)).sum(axis=-1)


def plot_heatmap(ax, tokens, weights, title, tick_step=None):
    """把一个 (T, T) 注意力矩阵画成热图。x 轴=被关注的词，y 轴=当前词。"""
    T = len(tokens)
    tick_step = tick_step or max(1, T // 30)
    im = ax.imshow(weights, cmap="Blues", aspect="auto", vmin=0)

    ticks = list(range(0, T, tick_step))
    ax.set_xticks(ticks); ax.set_xticklabels([tokens[i] for i in ticks], fontsize=7)
    ax.set_yticks(ticks); ax.set_yticklabels([tokens[i] for i in ticks], fontsize=7)
    ax.set_xlabel("被关注的词元 (key)")
    ax.set_ylabel("当前词元 (query)")
    ax.set_title(title, fontsize=10)
    return im


def top_attended(tokens, weights, k=5):
    """找出被关注最多的词元：把每行权重按 key 求和，取 top-k。"""
    col = weights.sum(axis=0)
    order = col.argsort()[::-1][:k]
    return [(tokens[i], round(float(col[i]), 2)) for i in order]


def main():
    ckpt = ROOT / "ckpt" / "best.pt"
    if not ckpt.exists():
        raise SystemExit("找不到 ckpt/best.pt，请先运行 python train.py")

    model, tokenize_fn = load_for_eval(str(ckpt))
    dev = pd.read_parquet(ROOT / "data" / "validation.parquet")

    # 从验证集里挑三个有代表性的样本
    lengths = dev["text"].str.len()
    picks = [
        ("attn_positive", "正面样本", dev[dev["label"] == 1].iloc[0]["text"]),
        ("attn_negative", "负面样本", dev[dev["label"] == 0].iloc[0]["text"]),
        # 长句：截到 60 个词元，否则热图太密看不清
        ("attn_long", "长句样本", dev.loc[lengths.idxmax()]["text"]),
    ]

    report_lines = []
    for name, label, text in picks:
        tokens, per_layer = get_attention(model, tokenize_fn, text)
        weights = per_layer[-1]              # 取最后一层的注意力

        fig, ax = plt.subplots(figsize=(max(6, len(tokens) * 0.22), max(6, len(tokens) * 0.22)))
        im = plot_heatmap(ax, tokens, weights, f"{label}（最后一层，head 平均）")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="attention weight")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"{name}.png", dpi=150)
        plt.close(fig)

        top = top_attended(tokens, weights)
        report_lines.append(f"{label}: {text[:40]}...")
        report_lines.append(f"  被关注最多的词元: {top}")
        print(f"已保存 figures/{name}.png  长度 {len(tokens)}")
        print(f"  被关注最多的词元: {top}")

    # 额外画一张：同一个句子在不同层的注意力对比（观察层次结构）
    text = picks[0][2]
    tokens, per_layer = get_attention(model, tokenize_fn, text)
    n = len(per_layer)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.5))
    if n == 1:
        axes = [axes]
    for i, (ax, w) in enumerate(zip(axes, per_layer)):
        im = plot_heatmap(ax, tokens, w, f"第 {i+1} 层", tick_step=max(1, len(tokens) // 12))
    fig.suptitle("同一个句子在各层的注意力（正面样本）", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "attn_layers.png", dpi=150)
    plt.close(fig)
    print("已保存 figures/attn_layers.png")

    # ---- 图 5：单个 head 的对比（平均可能把某个 head 的模式抹掉）----
    tokens_h, per_layer_h = get_attention_per_head(model, tokenize_fn, text)
    last = per_layer_h[-1]                      # (H, T, T)
    n_heads = last.shape[0]
    fig, axes = plt.subplots(1, n_heads, figsize=(4.0 * n_heads, 4.5))
    if n_heads == 1:
        axes = [axes]
    for h, ax in enumerate(axes):
        print(f"  head{h+1} max weight = {last[h].max():.4f}")
        plot_heatmap(ax, tokens_h, last[h], f"head {h+1}",
                     tick_step=max(1, len(tokens_h) // 10))
    fig.suptitle(f"最后一层各个 head 的注意力（{n_heads} 个头，正面样本）", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "attn_heads.png", dpi=150)
    plt.close(fig)
    print("已保存 figures/attn_heads.png")

    # ---- 图 6：注意力到底有多"均匀"？用熵量化 ----
    # 热图若看起来一片平色，用熵可以确认"确实接近均匀分布"，
    # 而不是画图参数没调好。这是本任务最有信息量的一张分析图。
    T = len(tokens_h)
    uniform_entropy = float(np.log(T))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    for li, w in enumerate(per_layer_h):
        e = row_entropy(w).ravel()
        axes[0].hist(e, bins=25, alpha=0.55, label=f"第 {li+1} 层")
    axes[0].axvline(uniform_entropy, ls="--", c="k", lw=1.5,
                    label=f"均匀分布上界 ln(T)={uniform_entropy:.2f}")
    axes[0].set_xlabel("每行注意力的熵（nat）")
    axes[0].set_ylabel("行数")
    axes[0].set_title("注意力分布的熵：越靠右越接近均匀")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    head_ent = [row_entropy(last[h]).mean() for h in range(n_heads)]
    axes[1].bar(range(1, n_heads + 1), head_ent, color="steelblue")
    axes[1].axhline(uniform_entropy, ls="--", c="k", lw=1.5, label="均匀分布上界")
    axes[1].set_xticks(range(1, n_heads + 1))
    axes[1].set_xlabel("head 编号（最后一层）")
    axes[1].set_ylabel("平均熵（nat）")
    axes[1].set_title("各 head 的平均熵")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "attn_entropy.png", dpi=150)
    plt.close(fig)
    print("已保存 figures/attn_entropy.png")

    # ---- 把数字也写进 observations，报告里可以直接引用 ----
    report_lines.append("")
    report_lines.append("=== 注意力分布的定量刻画 ===")
    report_lines.append(f"句子长度 T = {T}，完全均匀分布的熵上界 = ln(T) = {uniform_entropy:.3f} nat")
    for li, w in enumerate(per_layer_h):
        e_avg = row_entropy(w).mean()
        e_head = [row_entropy(w[h]).mean() for h in range(w.shape[0])]
        report_lines.append(
            f"第 {li+1} 层: 平均熵 {e_avg:.3f} nat（占均匀上界 {e_avg/uniform_entropy:.1%}），"
            f"最大权重 {w.max():.4f}（均匀应为 {1/T:.4f}），各 head 熵 "
            + ", ".join(f"{x:.3f}" for x in e_head))
    report_lines.append(
        "解读：熵越接近 ln(T)、最大权重越接近 1/T，说明注意力越接近均匀分布，"
        "此时热图会呈现为一片平色，且「盯着哪个词」这种解释不成立；"
        "反之则说明模型学出了稀疏的词元对齐。")

    (FIG_DIR / "attention_observations.txt").write_text(
        "\n".join(report_lines), encoding="utf-8")
    print()
    print("\n".join(report_lines[6:]))


if __name__ == "__main__":
    main()
