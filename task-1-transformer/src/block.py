"""Transformer encoder block（M2）：attention + FFN + residual + LayerNorm。"""
import torch.nn as nn

from src.attention import MultiHeadAttention


class FeedForward(nn.Module):
    """逐位置前馈网络：Linear -> GELU -> Dropout -> Linear。

    先升维到 d_ff（通常是 4*d_model）再降回来，给模型非线性变换的空间。
    「逐位置」意味着对每个词元单独作用，不跨位置交流信息。
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-LN 的 Transformer block。

        Pre-LN :  x = x + Attn(LN(x));  x = x + FFN(LN(x))
        Post-LN:  x = LN(x + Attn(x));  x = LN(x + FFN(x))

    这里用 Pre-LN：LayerNorm 放在子层「入口」，梯度可以沿残差这条「高速公路」
    原封不动回传，所以不用 warmup 也容易训稳。原始论文用的是 Post-LN。

    残差连接（x + ...）解决深层网络的梯度消失/退化；
    LayerNorm 把每个词元向量的激活值重新标准化，稳定训练。
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1,
                 use_residual: bool = True, use_layernorm: bool = True):
        super().__init__()
        self.use_residual = use_residual
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        # 两个开关只为加分项 S2 的消融实验准备，正常训练保持默认 True
        self.ln1 = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
        self.ln2 = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.dropout(self.attn(self.ln1(x), mask=mask))
        x = x + attn_out if self.use_residual else attn_out

        ffn_out = self.dropout(self.ffn(self.ln2(x)))
        x = x + ffn_out if self.use_residual else ffn_out
        return x
