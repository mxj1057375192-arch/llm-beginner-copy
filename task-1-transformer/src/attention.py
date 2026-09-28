"""手写缩放点积注意力（M1）与多头注意力（M2）。

接口约定（见 ../README.md「实现约定」）：
    scaled_dot_product_attention(Q, K, V, mask=None) -> Tensor
        Q/K/V 形状 (B, H, T, D)
        mask 可广播到 (B, H, T_q, T_k)，True 表示「该位置被屏蔽」
        返回 (B, H, T_q, D)
"""
import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# M1：缩放点积注意力
# ---------------------------------------------------------------------------
def scaled_dot_product_attention(Q, K, V, mask=None, return_weights=False):
    """注意力公式： softmax(Q K^T / sqrt(d_k)) V

    直觉：Q 是「我在找什么」，K 是「我能提供什么」，两者点积得到「匹配分数」，
    分数越高说明这个位置对当前查询越重要；softmax 归一成权重后，对 V（内容）加权求和。

    Args:
        Q: (B, H, T_q, D)  查询
        K: (B, H, T_k, D)  键
        V: (B, H, T_k, D)  值
        mask: 可广播到 (B, H, T_q, T_k) 的布尔张量，True = 屏蔽
        return_weights: 为 True 时额外返回注意力权重（可视化用）

    Returns:
        (B, H, T_q, D)；return_weights=True 时返回 (out, attn)，attn 形状 (B, H, T_q, T_k)
    """
    d_k = Q.size(-1)

    # 1) 打分：Q K^T / sqrt(d_k)，形状 (B, H, T_q, T_k)
    #    缩放因子是 sqrt(d_k)，不是 sqrt(d_model)！每个头只负责 d_k 维，
    #    不缩放的话点积的方差会随维度线性增长，softmax 会被推到饱和区（梯度接近 0）。
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # 2) 屏蔽：要挡住的位置填 -inf。
    #    这里必须填 -inf，不能乘 0 —— 乘 0 之后 softmax 仍会给它分到概率，
    #    信息就泄漏了（这也是自检 causal_mask 想抓的错）。
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    # 3) 归一化：在最后一维（key 方向）做 softmax，得到注意力权重，每行和为 1
    attn = torch.softmax(scores, dim=-1)

    # 4) 加权求和：把权重作用到 V 上
    out = torch.matmul(attn, V)

    if return_weights:
        return out, attn
    return out


# ---------------------------------------------------------------------------
# M2：多头注意力
# ---------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    """多头自注意力：把 d_model 切成 n_heads 个 d_k 维子空间并行做注意力。

    单个头的注意力只能表达「一种」对齐关系；多头让模型在不同子空间里
    同时关注不同模式（比如有的头盯情感词，有的头盯否定词），最后拼接融合。
    """

    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Q/K/V 各自的线性投影，以及最后的输出投影
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

        # 缓存最近一次 forward 的注意力权重，供可视化读取：(B, H, T, T)
        self.attn_weights = None

    def forward(self, x, mask=None, position_offsets=None):
        """x: (B, T, d_model) -> (B, T, d_model)

        position_offsets: 可选，用于把 mask 对齐到「带 KV cache 的增量解码」。
            全序列训练时传 None（mask 形状 (T_q, T_k) 且 T_q == T_k）。
            带 cache 生成时，query 只有 1 个位置，但它要看的 key 有 T_k 个，
            这时 mask 的 query 轴必须靠 left-pad 到 (1, T_k) 才能广播对齐，
            传 position_offsets 就是告诉 attention「我在 mask 里的 query 偏移量」。
            任务一只需要 None 这条路径（本任务不实现 KV cache）；
            这个参数是为任务二的增量解码预留的接口。
        """
        B, T, _ = x.shape
        if position_offsets is not None and mask is not None:
            # 只取当前这批 query 对应的那几行 mask
            mask = mask[..., position_offsets:position_offsets + T, :]

        # 1) 投影 + 拆头： (B, T, d_model) -> (B, T, H, d_k) -> (B, H, T, d_k)
        Q = self.q_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # 2) 每个头独立做缩放点积注意力（矩阵运算天然并行处理所有头）
        out, attn = scaled_dot_product_attention(Q, K, V, mask, return_weights=True)

        # 3) attention dropout：训练时随机丢弃一部分注意力权重，防止过拟合
        if self.training and self.attn_dropout.p > 0:
            attn = self.attn_dropout(attn)
            out = torch.matmul(attn, V)
        self.attn_weights = attn.detach()

        # 4) 拼头： (B, H, T, d_k) -> (B, T, H, d_k) -> (B, T, d_model)
        #    transpose 之后内存不连续，必须先 .contiguous() 才能用 view 展平
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)

        # 5) 输出投影：把拼接后的多头结果融合回 d_model 空间
        return self.out_proj(out)
