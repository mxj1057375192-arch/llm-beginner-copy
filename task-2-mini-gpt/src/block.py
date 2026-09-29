"""Transformer 的零件：RMSNorm、SwiGLU 前馈网络、Pre-LN decoder block。

Pre-LN vs Post-LN（README 知识点之一）
-------------------------------------
- Post-LN（原始 Transformer）：x = LN(x + Attn(x))，归一化放在残差之后。
  深层时梯度容易不稳定，通常需要 warmup 才能训起来。
- Pre-LN（现代 LLM 默认，nanoGPT/LLaMA/GPT 都是它）：x = x + Attn(LN(x))。
  残差是「干净的高速公路」，梯度可以直接回传，不需要精细的 warmup，训练更稳。
  代价是最终输出要再补一次 LN（在 model.py 的 final_norm 里）。
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.attention import CausalSelfAttention


class RMSNorm(nn.Module):
    """RMSNorm：只按均方根缩放，不减均值、无 bias。

    对比 LayerNorm：少了「减均值」这一步，参数更少、计算更省，
    在大模型上效果与 LayerNorm 相当（LLaMA 起成为主流）。
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 用 float32 算统计量，低精度训练时更稳
        dtype = x.dtype
        x32 = x.float()
        rms = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x32 * rms).to(dtype) * self.weight


class SwiGLU(nn.Module):
    """SwiGLU 前馈层：FFN(x) = W2( silu(W1 x) * W3 x )。

    比经典的两层 MLP（W2 gelu(W1 x)）多一条「门控」支路：一个 Linear 当开关，
    逐元素乘到另一个 Linear 上。实践中同等参数量下效果更好，是 LLaMA 等的标配。
    隐藏维通常取 8/3 * d_model 再对齐到 64 的倍数，与两层 MLP 的参数量相当。
    """

    def __init__(self, d_model: int, hidden_mult: float = 2.6667, dropout: float = 0.0):
        super().__init__()
        hidden = int(hidden_mult * d_model)
        hidden = 64 * ((hidden + 63) // 64)      # 对齐到 64 的倍数，kernel 更友好
        self.w1 = nn.Linear(d_model, hidden, bias=False)   # 被门控的支路
        self.w3 = nn.Linear(d_model, hidden, bias=False)   # 门控支路
        self.w2 = nn.Linear(hidden, d_model, bias=False)   # 投回 d_model
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    """一个 decoder block：Pre-LN 的「attention + FFN」两段式残差结构。"""

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 2048,
                 dropout: float = 0.0, rope_base: float = 10000.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len=max_seq_len,
                                        dropout=dropout, rope_base=rope_base)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, dropout=dropout)

    def forward(self, x: torch.Tensor,
                kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False):
        # 残差 1：注意力分支（先归一化再进子层）
        h, new_cache = self.attn(self.norm1(x), kv_cache=kv_cache, use_cache=use_cache)
        x = x + h
        # 残差 2：前馈分支
        x = x + self.ffn(self.norm2(x))
        return x, new_cache
