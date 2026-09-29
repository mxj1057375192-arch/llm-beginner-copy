"""Causal 多头自注意力 + KV cache（M2 + M3）。

KV cache 到底省了什么？
----------------------
自回归生成时，生成第 t 个词只需要看前 t 个词。**没 cache** 的做法是每一步都把
长度为 t 的整句重新 forward 一遍，第 t 步要算 t 次 QKV 投影和 t×t 的注意力矩阵：
总计算量 O(T^3)。**有 cache** 的做法是把每层已经算好的 K、V 存下来，新的一步
只算 1 个新词元的 K、V（Q 也只有 1 个），拼到历史后面：
总计算量降到 O(T^2)，每步的耗时几乎与历史长度无关（只多了拼接和一次长度 T 的注意力）。

拿一句话说清楚：cache 是「用显存换计算」——保存 K/V（每层 2 × B×H×T×d_k 个浮点数），
换掉重复的前缀计算。

两个必须小心的地方（自检 kv_cache_equivalence 就是抓这两个）：
    1. 拼接要在**序列长度维 T** 上拼，不是别的维；
    2. 新词元的 RoPE 位置必须接着历史长度算（position_offset = 历史长度），
       否则角度错、logits 和全量前向对不上。
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from src.rope import RoPE

# KV cache 的类型：每层一个 (K, V)，形状 (B, H, T, head_dim)
KVCache = List[Tuple[torch.Tensor, torch.Tensor]]


def build_causal_mask(T_q: int, T_k: int, offset: int = 0) -> torch.Tensor:
    """构造 causal mask（上三角 True = 屏蔽）。

    Args:
        T_q: 本次 query 的数量（增量解码时是 1）
        T_k: key/value 的总长度（含 cache 里的历史）
        offset: 本次第一个 query 在完整序列中的绝对位置。
            全量前向时 offset=0、T_q=T_k，得到标准下三角掩码；
            增量解码时 T_q=1，新的 query 位于第 offset 位，要能看到 0..offset 的全部 key。
    返回:
        (T_q, T_k) 的 bool 张量，True 表示该位置禁止被注意到。
    """
    q_pos = torch.arange(offset, offset + T_q).unsqueeze(1)   # (T_q, 1)
    k_pos = torch.arange(T_k).unsqueeze(0)                    # (1, T_k)
    return k_pos > q_pos                                      # 未来的位置 -> True


class CausalSelfAttention(nn.Module):
    """多头 causal self-attention：RoPE(Q,K) + softmax(QK^T/sqrt(d))V + KV cache。"""

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 2048,
                 dropout: float = 0.0, rope_base: float = 10000.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # RoPE 作用在 Q、K 上（V 不旋转！V 是「内容」，位置信息由 QK 的内积携带）
        self.rope = RoPE(self.head_dim, max_seq_len=max_seq_len, base=rope_base)
        self.dropout = nn.Dropout(dropout)
        self.attn_weights = None      # 最近一次注意力权重，调试/可视化用

    def forward(self, x: torch.Tensor,
                kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False):
        """
        Args:
            x: (B, T, d_model)。训练/全量前向时 T = 整段长度；增量解码时 T = 1
            kv_cache: 该层的历史 (K, V)，形状 (B, H, T_hist, head_dim)，没有则 None
            use_cache: 是否返回更新后的 cache
        Returns:
            out: (B, T, d_model)
            new_cache: use_cache=True 时返回 (K_all, V_all)
        """
        B, T, _ = x.shape

        # 1) 投影 + 拆多头： (B, T, d_model) -> (B, H, T, head_dim)
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # 2) 位置偏移 = 历史长度：有 cache 时新词元要从 T_hist 开始数位置
        past_len = 0 if kv_cache is None else kv_cache[0].size(-2)
        q = self.rope(q, position_offset=past_len)
        k = self.rope(k, position_offset=past_len)

        # 3) 拼接历史 K/V —— 注意在第 -2 维（序列长度维）上拼
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=-2)
            v = torch.cat([kv_cache[1], v], dim=-2)
        new_cache = (k, v) if use_cache else None

        # 4) 打分 + 缩放。QK^T 用 float32 算，避免 fp16/bf16 下 softmax 溢出
        T_k = k.size(-2)
        scores = torch.matmul(q, k.transpose(-2, -1)).float() / math.sqrt(self.head_dim)

        # 5) causal mask：挡掉「未来」的位置（填 -inf，不是乘 0）
        mask = build_causal_mask(T, T_k, offset=past_len).to(scores.device)
        scores = scores.masked_fill(mask, float("-inf"))

        # 6) softmax（沿 key 维）-> dropout -> 加权求和
        attn = torch.softmax(scores, dim=-1).to(v.dtype)
        attn = self.dropout(attn)
        self.attn_weights = attn.detach()
        out = torch.matmul(attn, v)

        # 7) 拼头 + 输出投影
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out), new_cache
