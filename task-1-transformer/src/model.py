"""Transformer 文本分类器（M3），以及自检脚本使用的加载工厂。"""
import torch
import torch.nn as nn

from src.block import TransformerBlock
from src.tokenizer import CharTokenizer

PAD_ID = 0


class TransformerClassifier(nn.Module):
    """Embedding + 位置编码 + N 层 Transformer block + 池化 + 分类头。

    forward(ids) 的 ids 形状是 (B, T)，返回 logits 形状 (B, num_classes)。
    padding mask 在 forward 内部根据 ids 自己算，外部不用传 —— 这样自检
    脚本直接 model(ids.unsqueeze(0)) 就能用。
    """

    def __init__(self, vocab_size, num_classes=2, d_model=128, n_heads=4,
                 n_layers=4, d_ff=512, max_len=256, dropout=0.1, pad_id=PAD_ID,
                 use_residual=True, use_layernorm=True):
        super().__init__()
        self.pad_id = pad_id
        self.max_len = max_len
        self.d_model = d_model

        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        # 可学习的绝对位置编码：第 i 个位置一个向量，让模型知道词的先后顺序
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout,
                             use_residual=use_residual, use_layernorm=use_layernorm)
            for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
        self.classifier = nn.Linear(d_model, num_classes)

        self.apply(self._init_weights)
        nn.init.zeros_(self.tok_emb.weight[pad_id])  # PAD 的向量固定为 0
        nn.init.zeros_(self.classifier.bias)

    @staticmethod
    def _init_weights(module):
        """小方差初始化：Transformer 对初始化比较敏感，std=0.02 是常见起点。"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, ids):
        """ids: (B, T) -> logits: (B, num_classes)"""
        B, T = ids.shape
        T = min(T, self.max_len)      # 位置编码表只有 max_len 行，超长要截断
        ids = ids[:, :T]

        # padding mask：(B, T)，True = 这个位置是 PAD
        pad_mask = (ids == self.pad_id)
        # 广播成注意力 mask：(B, 1, 1, T) -> 自动广播到 (B, H, T_q, T_k)
        # 这样每个 query 位置都不会去注意 PAD 位置
        attn_mask = pad_mask[:, None, None, :]

        pos = torch.arange(T, device=ids.device).unsqueeze(0)   # (1, T)
        # embedding 乘 sqrt(d_model)：把 embedding 和位置编码的量级对齐，训练更稳
        x = self.tok_emb(ids) * (self.d_model ** 0.5) + self.pos_emb(pos)
        x = self.emb_dropout(x)

        for block in self.blocks:
            x = block(x, mask=attn_mask)
        x = self.ln_final(x)

        # masked mean pooling：只对真实词元求平均，排除 PAD
        h = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        denom = (~pad_mask).sum(dim=1, keepdim=True).clamp(min=1)
        pooled = h.sum(dim=1) / denom          # (B, d_model)

        return self.classifier(pooled)


# ---------------------------------------------------------------------------
# 自检脚本约定：load_for_eval(ckpt_path) -> (model, tokenize_fn)
# ---------------------------------------------------------------------------
def load_for_eval(ckpt_path: str):
    """加载 checkpoint，返回 (模型, 分词函数)。

    checkpoint 里同时存了 state_dict、词表和超参配置，
    这样自检脚本不需要额外知道模型结构就能复原。
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model = TransformerClassifier(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    tokenizer = CharTokenizer(ckpt["vocab"])
    max_len = ckpt["config"]["max_len"]

    def tokenize_fn(text: str) -> torch.Tensor:
        """文本 -> LongTensor 形状 (T,)。"""
        return tokenizer.encode(text, max_len=max_len)

    return model, tokenize_fn
