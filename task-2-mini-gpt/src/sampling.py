"""采样策略（M5）：greedy / temperature / top-k / top-p（核采样）。

为什么不能直接按 softmax 概率随机抽？
    语言模型的词表有几万个 token，softmax 之后长尾上堆着大量「低概率但离谱」的候选：
    偶尔抽中一个语义完全不搭的字，整句就崩了。所以要做「截断」：
    top-k 砍掉「除前 k 名以外的所有」，top-p 砍掉「累计概率超过 p 之后的长尾」。

四种策略的组合关系：
    greedy            ：永远取最大概率 -> 完全确定
    temperature       ：先对 logits 除以 T 再 softmax。T<1 让分布更尖（保守），
                        T>1 更平（多样）。T→0 时等价于 greedy。
    top-k             ：保留概率最高的 k 个
    top-p             ：保留累计概率刚过 p 的最小集合（k 随上下文自适应，比 top-k 灵活）

工程细节（容易写错的地方）：
    1. 顺序不能乱：temperature 缩放 -> top-k 过滤 -> top-p 过滤 -> 重新归一化 -> 采样；
    2. top-p 的阈值比较要在**累计概率**上做，且必须重新归一化，否则概率和不为 1；
    3. temperature=0 要特判成 greedy，否则会出现 0/0 的 NaN；
    4. 被过滤掉的 token 填 -inf（不是 0），否则它们还会分到概率。
"""
from __future__ import annotations

from typing import Optional

import torch


def apply_top_k(logits: torch.Tensor, top_k: Optional[int]) -> torch.Tensor:
    """只保留概率最高的 k 个 logits，其余置为 -inf。logits: (B, V)"""
    if not top_k or top_k <= 0 or top_k >= logits.size(-1):
        return logits
    # 第 k 大的值作为阈值；小于它的全部屏蔽（注意用严格小于，同分的保留）
    kth = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < kth, float("-inf"))


def apply_top_p(logits: torch.Tensor, top_p: Optional[float]) -> torch.Tensor:
    """核采样：按概率降序累加，保留累计概率恰好超过 top_p 的最小集合。"""
    if not top_p or top_p >= 1.0 or top_p <= 0.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probs, dim=-1)
    # 累计概率已经超过 top_p 的位置（保留第一个越过的，shift 一下）
    remove = cumulative - probs > top_p
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    # 还原回原来的词表顺序
    return logits.scatter(-1, sorted_idx, sorted_logits)


def sample_next_token(logits: torch.Tensor, temperature: float = 1.0,
                      top_k: Optional[int] = None, top_p: Optional[float] = None,
                      greedy: bool = False) -> torch.Tensor:
    """从 (B, V) 的 logits 采样出 (B, 1) 的下一个 token id。"""
    logits = logits.float()

    # ① temperature=0（或显式 greedy）退化为 argmax
    if greedy or temperature is None or temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    # ② 温度缩放
    logits = logits / max(float(temperature), 1e-6)

    # ③ top-k 截断
    logits = apply_top_k(logits, top_k)

    # ④ top-p 截断（在温度缩放后的分布上做）
    logits = apply_top_p(logits, top_p)

    # ⑤ softmax 重新归一化后按概率采样
    probs = torch.softmax(logits, dim=-1)
    # 极端情况下（数值下溢）兜底回 greedy，避免 multinomial 报错
    if torch.isnan(probs).any() or probs.sum(dim=-1).min() <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1)


# 给 generate.py / 报告用的一组「策略组合」预设
PRESETS = {
    "greedy":      dict(greedy=True, temperature=1.0),
    "temperature": dict(temperature=1.2),
    "top_k":       dict(temperature=1.0, top_k=20),
    "top_p":       dict(temperature=1.0, top_p=0.9),
    "top_k_top_p": dict(temperature=0.8, top_k=40, top_p=0.9),
}


def decode_preset(name: str) -> dict:
    return dict(PRESETS[name])
