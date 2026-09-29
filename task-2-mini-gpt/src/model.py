"""MiniGPT：decoder-only 语言模型（M2 + M3 + M5 的载体）。

结构总览（自下而上）：

    token id (B, T)
      -> token embedding            (B, T, d_model)        ← 查表，把 id 变成向量
      -> N × Block( Pre-LN:
                        x = x + Attention(RMSNorm(x))     ← 带 RoPE 和 KV cache
                        x = x + SwiGLU(RMSNorm(x)) )
      -> RMSNorm                    (B, T, d_model)        ← Pre-LN 的收尾归一化
      -> lm_head (Linear -> vocab)  (B, T, vocab)          ← 每个位置预测「下一个 token」

注意这里**没有**位置 embedding 表：位置信息全部由 RoPE 在注意力内部注入，
所以参数量与 max_seq_len 无关，序列长度也不用被 embedding 表卡住。
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.block import Block, RMSNorm
from src.sampling import sample_next_token  # noqa: F401  (M5，generate 里用)

# 训练（默认）与正式实验（唐诗档需要更大上下文）两套配置
MODEL_PRESETS = {
    # 快速跑通：CPU 上几分钟能训完，适合先验证整条 pipeline
    "tiny": dict(d_model=192, n_layers=4, n_heads=4, max_seq_len=128, dropout=0.0),
    # 唐诗默认配置：本机 CPU 约 20 分钟
    "poetry": dict(d_model=256, n_layers=4, n_heads=4, max_seq_len=256, dropout=0.05),
    # 更大规模（加分项 S1），有 GPU 再上
    "small": dict(d_model=384, n_layers=6, n_heads=6, max_seq_len=256, dropout=0.1),
    "base": dict(d_model=768, n_layers=12, n_heads=12, max_seq_len=512, dropout=0.1),
}


class MiniGPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, n_layers: int = 6,
                 n_heads: int = 4, max_seq_len: int = 256, dropout: float = 0.0,
                 rope_base: float = 10000.0, tie_weights: bool = True,
                 pos_emb: str = "rope"):
        """pos_emb: "rope"（默认，RoPE 在注意力内部注入位置）或
        "learned"（可学习的绝对位置 embedding，加分项 S2 的对照组）。"""
        super().__init__()
        assert pos_emb in ("rope", "learned"), "pos_emb 只能是 'rope' 或 'learned'"
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.pos_emb = pos_emb
        # 自检脚本会读 block_size / max_seq_len 来给困惑度分窗，两者保持一致
        self.block_size = max_seq_len
        self.tie_weights = tie_weights

        self.token_emb = nn.Embedding(vocab_size, d_model)
        # 对照组专用：绝对的、可学习的位置 embedding（长度被 max_seq_len 写死，没有外推能力）
        self.pos_embedding = (nn.Embedding(max_seq_len, d_model)
                              if pos_emb == "learned" else None)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, max_seq_len=max_seq_len,
                  dropout=dropout, rope_base=rope_base)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.apply(self._init_weights)
        if tie_weights:
            # weight tying：输入 embedding 与输出投影共用同一张表
            # （同一份「词 -> 向量」的知识，参数省 vocab*d_model，小模型上通常还有增益）
            self.lm_head.weight = self.token_emb.weight
        # 残差分支上的投影按 1/sqrt(2*n_layers) 缩小，深层时输出方差不会爆炸
        for name, p in self.named_parameters():
            if name.endswith(("out_proj.weight", "w2.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * n_layers) ** 0.5)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    # 前向
    # ------------------------------------------------------------------
    def forward(self, ids: torch.Tensor,
                kv_cache: Optional[list] = None,
                return_cache: bool = False,
                targets: Optional[torch.Tensor] = None):
        """ids: (B, T) 的 long 张量。

        Args:
            kv_cache: 长度 = n_layers 的列表，每项是 (K, V) 或 None；None 表示从头算
            return_cache: True 时返回 (logits, new_kv_cache)
            targets: 传了就顺便算交叉熵 loss（训练用）
        Returns:
            logits: (B, T, vocab_size)；return_cache=True 时返回 (logits, cache)
        """
        B, T = ids.shape
        # 允许比 max_seq_len 多 1 个 token：自检按 block_size 切窗时会取 block_size+1
        # （预测最后一个位置的 next token），RoPE 本身也能算到那里
        if T > self.max_seq_len + 1 and kv_cache is None:
            raise ValueError(f"输入长度 {T} 超过 max_seq_len={self.max_seq_len}；"
                             "请调大 max_seq_len 或缩短输入")
        x = self.drop(self.token_emb(ids))
        if self.pos_embedding is not None:            # 对照组：绝对位置编码
            if T > self.max_seq_len:
                raise ValueError(f"learned 位置编码最长只支持 {self.max_seq_len}，收到 {T}")
            pos = torch.arange(T, device=ids.device).unsqueeze(0)
            x = x + self.pos_embedding(pos)

        new_cache: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for i, block in enumerate(self.blocks):
            layer_cache = None if kv_cache is None else kv_cache[i]
            x, layer_out = block(x, kv_cache=layer_cache, use_cache=return_cache)
            if return_cache:
                new_cache.append(layer_out)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.reshape(-1), ignore_index=-100)
            return (logits, loss) if not return_cache else (logits, new_cache, loss)
        return (logits, new_cache) if return_cache else logits

    # ------------------------------------------------------------------
    # 生成（M5）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens: int = 80, top_k: Optional[int] = None,
                 top_p: Optional[float] = None, temperature: float = 1.0,
                 greedy: bool = False, repetition_penalty: float = 1.0,
                 stop_ids: Optional[List[int]] = None, use_cache: bool = True,
                 seed: Optional[int] = None, eos_id: Optional[int] = None):
        """自回归生成，返回**完整序列**（prompt + 新生成）的 id 列表。

        参数（四种采样策略，对应 README 的 M5）：
            greedy=True 或 temperature<=0 : 贪心，每次取概率最大的 token（确定性）
            top_k   : 只在概率最高的 k 个候选里采样（砍掉长尾，抑制胡言乱语）
            top_p   : 核采样，按概率从大到小累加到 p 为止，只在这个「核」里采样
            temperature: 温度，>1 更随机、<1 更保守；1 表示不缩放
        这几个可以组合：例如 temperature=0.8, top_k=40, top_p=0.9。
        """
        self.eval()
        if seed is not None:
            torch.manual_seed(seed)
        device = next(self.parameters()).device
        if isinstance(prompt_ids, (list, tuple)):
            prompt_ids = torch.tensor([list(prompt_ids)], dtype=torch.long)
        ids = prompt_ids.to(device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        if stop_ids is None and eos_id is not None:
            stop_ids = [eos_id]
        stop_ids = stop_ids or []

        # 多步生成时把 prompt 一次性喂进去建好 cache，之后每步只送 1 个新词元
        cache = None
        T_prompt = ids.size(1)
        generated = ids
        step_input = ids if use_cache else ids[:, -self.max_seq_len:]

        for step in range(max_new_tokens):
            # 训练用的是固定长度窗口，这里把输入裁到 max_seq_len 以内（没 cache 时才需要）
            if not use_cache:
                step_input = generated[:, -self.max_seq_len:]
            if use_cache:
                logits, cache = self(step_input, kv_cache=cache, return_cache=True)
            else:
                logits = self(step_input)      # 每步重算整段前缀（用来对比 cache 的价值）

            next_logits = logits[:, -1, :].float()          # 只关心最后一个位置的预测
            if repetition_penalty and repetition_penalty != 1.0:
                # 重复惩罚：对已经出现过的 token，按其正负号反向缩放 logit
                for b in range(next_logits.size(0)):
                    seen = torch.unique(generated[b])
                    vals = next_logits[b, seen]
                    next_logits[b, seen] = torch.where(vals > 0, vals / repetition_penalty,
                                                       vals * repetition_penalty)
            nxt = sample_next_token(next_logits, temperature=temperature, top_k=top_k,
                                    top_p=top_p, greedy=greedy)   # (B, 1)
            generated = torch.cat([generated, nxt], dim=1)
            if stop_ids and any(int(t) in stop_ids for t in nxt[0]):
                break

        out = generated[0].tolist()
        # 干脆把「首尾」交回调用方：是否保留 prompt 由调用者决定
        self.last_prompt_len = T_prompt
        return out

    # ------------------------------------------------------------------
    def num_parameters(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # 权重共享时 embedding 只算一次
            n -= self.token_emb.weight.numel()
            if not self.tie_weights:
                n -= self.lm_head.weight.numel()
            if self.pos_embedding is not None:
                n -= self.pos_embedding.weight.numel()
        return n

    @staticmethod
    def from_preset(preset: str, vocab_size: int, **overrides) -> "MiniGPT":
        cfg = dict(MODEL_PRESETS[preset])
        cfg.update(overrides)
        return MiniGPT(vocab_size=vocab_size, **cfg)


# ---------------------------------------------------------------------------
# 自检脚本用的加载入口
# ---------------------------------------------------------------------------
def load_for_eval(ckpt_path: str):
    """读取 ckpt/best.pt，返回 (model, tokenizer)，都已 eval() 就绪。"""
    from src.tokenizer import BPETokenizer

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state, cfg = ckpt["model_state"], ckpt["config"]
        tok_path = ckpt.get("tokenizer_path")
    else:                       # 兼容「只存了 state_dict」的 checkpoint
        state, cfg, tok_path = ckpt, {}, None

    if not tok_path:
        tok_path = str(Path(ckpt_path).resolve().parent / "tokenizer.json")
    tok = BPETokenizer.from_pretrained(tok_path)

    vocab_size = cfg.get("vocab_size", tok.vocab_size)
    cfg = {k: v for k, v in cfg.items() if k != "vocab_size"}
    model = MiniGPT(vocab_size=vocab_size, **cfg)
    model.load_state_dict(state)
    model.eval()
    return model, tok
