"""旋转位置编码 RoPE（Rotary Position Embedding，M2 的核心）。

一句话原理
----------
绝对位置编码是「把位置信息加到输入向量上」；RoPE 改成「把 query/key 向量按位置转一个角度」。
把 d 维向量两两配对看成 d/2 个二维平面上的点，位置 m 的第 i 对旋转角度为 m * theta_i：

    [q_{2i}  ]   [cos(m*theta_i)  -sin(m*theta_i)] [q_{2i}  ]
    [q_{2i+1}] = [sin(m*theta_i)   cos(m*theta_i)] [q_{2i+1}]

两个性质（也是它比绝对位置编码好的原因）：
    1. 内积只依赖「相对位置」：旋转后 q_m · k_n 是 (m-n) 的函数，注意力天然看到的是距离；
    2. 不需要额外参数、不改变向量长度（只是旋转），外推到更长序列时更稳。

本实现采用「前后折半」配对（GPT-NeoX / LLaMA 风格，也是 HuggingFace 的默认）：
    第 i 对 = (x[i], x[i + d/2])，而不是 (x[2i], x[2i+1])。
    两种配对数学上等价，但**训练和推理必须用同一种**，否则角度对不上、logits 全错。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RoPE(nn.Module):
    """预计算 cos/sin 表，forward 时按位置切片取用。

    表只依赖 (max_seq_len, head_dim, base)，与 batch/内容无关，所以：
        - 缓存起来，每一步生成都复用（省时间）；
        - 增量解码（KV cache）时按真实位置偏移切片，而不是从头算。
    """

    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim 必须是偶数，才能两两配对做旋转"
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        # 频率：theta_i = base^(-2i/d)，i = 0..d/2-1
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)   # (d/2,)
        self._cos: torch.Tensor | None = None
        self._sin: torch.Tensor | None = None
        self._built_len = 0

    # ------------------------------------------------------------------
    def _build_cache(self, length: int, device, dtype) -> None:
        """预计算 length 个位置的 cos/sin，形状 (length, head_dim)。"""
        t = torch.arange(length, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))        # (length, d/2)
        emb = torch.cat([freqs, freqs], dim=-1)                 # (length, d) 折半复制
        self._cos = emb.cos().to(dtype)
        self._sin = emb.sin().to(dtype)
        self._built_len = length

    def _get_cos_sin(self, length: int, device, dtype):
        need = length
        if (self._cos is None or self._built_len < need
                or self._cos.device != device or self._cos.dtype != dtype):
            self._build_cache(max(need, self.max_seq_len), device, dtype)
        return self._cos[:need], self._sin[:need]

    # ------------------------------------------------------------------
    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """前后折半配对旋转：[-x2, x1]（x1 是前半，x2 是后半）。"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        """对张量做旋转。

        Args:
            x: (B, H, T, head_dim) —— 只应作用于 Q 和 K，**不要**作用在 V 上
            position_offset: 第一个 token 的绝对位置。全量前向时为 0；
                增量解码（KV cache）时等于「已缓存的历史长度」——这正是
                「新词元的 position 要接着历史长度算」这句话的落地。
        """
        T = x.size(-2)
        cos, sin = self._get_cos_sin(position_offset + T, x.device, x.dtype)
        cos = cos[position_offset:position_offset + T].unsqueeze(0).unsqueeze(0)  # (1,1,T,d)
        sin = sin[position_offset:position_offset + T].unsqueeze(0).unsqueeze(0)
        # 二维旋转公式：x' = x*cos + rotate_half(x)*sin
        return x * cos + self.rotate_half(x) * sin
