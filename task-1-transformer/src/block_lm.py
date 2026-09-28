"""decoder-only 版 Transformer block（M4：causal mask + toy 语言模型）。

和 `src/block.py` 的编码器 block 是同一个东西，只多了一个 causal mask：
它把注意力矩阵的「上三角」屏蔽掉，让位置 i 只能看到 <= i 的词元。
这是任务二 mini-GPT 的预热。
"""
import torch.nn as nn

from src.attention import MultiHeadAttention
from src.block import FeedForward


class CausalBlock(nn.Module):
    """Pre-LN + causal self-attention + FFN 的 decoder block。

    前向就是四行：
        x = x + Attn(LN(x))     # 这里 Attn 带 causal mask
        x = x + FFN(LN(x))
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, position_offsets=None):
        x = x + self.dropout(self.attn(self.ln1(x), mask=mask,
                                       position_offsets=position_offsets))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x
