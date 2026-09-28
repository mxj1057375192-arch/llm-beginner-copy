"""M4：用 causal mask 跑一个 toy 语言模型（next-token prediction）。

任务一的 M4 有两个要求：
  1. 自检 `causal_mask` 通过 —— 未来词元改动后，过去位置的输出不变；
  2. "跑一个 toy 语言模型"。

`eval/run.py` 只检查第 1 条（用随机张量直接测 attention）。这个脚本做第 2 条：
用与分类器**完全同一套** attention / block 代码，只把 mask 换成上三角，
在唐诗上做字符级 next-token 预测，并把「因果性」也顺手验证一遍。

用法：
    python toy_lm.py                 # 默认 1500 步（4 核 CPU 约 8-12 分钟；600 步约 4 分钟）
    python toy_lm.py --steps 600     # 快速跑通
    python toy_lm.py --verify-only   # 只做因果性验证，不训练

产出：
    ckpt/toy_lm.pt                  toy 语言模型
    figures/toy_lm_loss.png         训练曲线
"""
import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.block_lm import CausalBlock
from src.tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT.parent / "poetryFromTang.txt"      # 仓库自带的唐诗小语料（~49KB）
PAD_ID = 0


# ---------------------------------------------------------------------------
# 模型：和分类器同构，只是 mask 换成因果 mask
# ---------------------------------------------------------------------------
class ToyLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4,
                 d_ff=512, max_len=128, dropout=0.1):
        super().__init__()
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            CausalBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        # 输出层直接复用词嵌入矩阵（weight tying），这是语言模型的常见做法：
        # 输入 embedding 和输出分类器学的是同一份「字向量」，参数量减半、效果更好。
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, ids):
        """ids: (B, T) -> logits: (B, T, vocab_size)"""
        B, T = ids.shape
        device = ids.device

        # ---- mask：两种屏蔽叠加在一起 ----
        # (1) causal：位置 i 不能看 j > i（上三角 True）—— 语言模型的核心约束
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)
        # (2) padding：任何位置都不能看 PAD
        pad = (ids == PAD_ID)[:, None, None, :]          # (B, 1, 1, T)
        mask = causal[None, None] | pad                  # 广播成 (B, 1, T, T)

        pos = torch.arange(T, device=device).unsqueeze(0)
        x = self.tok_emb(ids) * (self.tok_emb.embedding_dim ** 0.5) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x, mask=mask)
        return self.lm_head(self.ln_final(x))


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
def load_corpus(max_len):
    """把整篇语料拼成一条长文本，再切成 max_len 字一段。

    为什么不按行切？唐诗一行只有十几二十字，按行切会切不出 64 字的片段。
    语言模型本来也不在乎句子边界 —— 它只学「下一个字是什么」。
    """
    text = CORPUS.read_text(encoding="utf-8")
    text = "".join(text.split())          # 去掉所有空白/换行，拼成一条长串
    chunks = []
    for i in range(0, len(text) - max_len, max_len):
        chunks.append(text[i:i + max_len])
    return text, chunks


def make_batch(chunks, tokenizer, max_len, device):
    """把若干条等长文本拼成 (B, T+1)，前 T 个字做输入、后 T 个做标签。

    这就是 teacher forcing：输入 [x0, x1, ..., x_{T-1}]，
    标签 [x1, x2, ..., x_T] —— 每个位置都在预测「下一个字」。
    """
    seqs = [tokenizer.encode(s, max_len=max_len + 1) for s in chunks]
    ids = torch.stack(seqs).to(device)
    return ids[:, :-1], ids[:, 1:]


# ---------------------------------------------------------------------------
# 因果性验证：改未来的字，过去的输出不能变
# ---------------------------------------------------------------------------
def verify_causality(model, tokenizer, device):
    model.eval()
    text = "床前明月光，疑是地上霜。举头望明月，低头思故乡。"
    ids = tokenizer.encode(text, max_len=64)[:24].unsqueeze(0).to(device)

    with torch.no_grad():
        logits1 = model(ids)
        # 把最后 5 个位置换成完全不同的字
        ids2 = ids.clone()
        ids2[:, -5:] = torch.randint(0, len(tokenizer), (1, 5), device=device)
        logits2 = model(ids2)

    cut = ids.shape[1] - 5
    leaked = (logits1[:, :cut] - logits2[:, :cut]).abs().max().item()
    changed = (logits1[:, cut:] - logits2[:, cut:]).abs().max().item()
    print(f"  [因果性验证] 改动最后 5 个词元后：")
    print(f"      过去 {cut} 个位置的输出最大变化 = {leaked:.3e}   <- 应该约等于 0")
    print(f"      被改动位置的输出最大变化       = {changed:.3e}   <- 应该明显 > 0，说明模型确实用了未来的信息")
    return leaked


# ---------------------------------------------------------------------------
# 采样
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate(model, tokenizer, prompt, n_new=60, temperature=0.8, device="cpu"):
    model.eval()
    ids = tokenizer.encode(prompt, max_len=model.max_len - n_new).unsqueeze(0).to(device)
    for _ in range(n_new):
        ids_in = ids[:, -model.max_len:]
        logits = model(ids_in)[:, -1] / max(temperature, 1e-6)
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)
        ids = torch.cat([ids, nxt], dim=1)
    return tokenizer.decode(ids[0].tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-len", type=int, default=64)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--no-train", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    text, chunks = load_corpus(args.max_len)
    print(f"语料: {CORPUS.name} -> {len(text)} 字，切成 {len(chunks)} 条训练片段（每条 {args.max_len} 字）")

    tokenizer = CharTokenizer.build(chunks, min_freq=1)
    print(f"字符级词表大小: {len(tokenizer)}")

    model = ToyLM(len(tokenizer), d_model=args.d_model, n_heads=args.n_heads,
                  n_layers=args.n_layers, d_ff=4 * args.d_model,
                  max_len=args.max_len).to(device)
    print(f"toy LM 参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    # ---- 第一步先把「因果性」这件事验证清楚，再谈训练 ----
    print("\n[1/3] 未训练模型上的因果性检查：")
    verify_causality(model, tokenizer, device)
    if args.verify_only:
        return

    # ---- 训练：teacher forcing，每个位置都在预测「下一个字」 ----
    print(f"\n[2/3] 训练 {args.steps} 步（next-token prediction）...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                 weight_decay=0.01, betas=(0.9, 0.98))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / 100) *
        0.5 * (1 + math.cos(math.pi * s / args.steps)))

    losses, t0 = [], time.time()
    model.train()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, len(chunks), (args.batch_size,)).tolist()
        x, y = make_batch([chunks[i] for i in idx], tokenizer, args.max_len, device)

        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

        if step % 100 == 0 or step == 1:
            avg = sum(losses[-100:]) / len(losses[-100:])
            ppl = math.exp(min(avg, 20))
            print(f"  step {step:>5}/{args.steps}  loss {avg:.4f}  ppl {ppl:6.2f}"
                  f"  ({time.time()-t0:.0f}s)")

    # ---- 训练后再验证一次 + 采样 ----
    print("\n[3/3] 训练后的因果性检查与生成：")
    verify_causality(model, tokenizer, device)

    print(f"\n生成示例（temperature=0.8，只训练了 {args.steps} 步，出现不通顺很正常）：")
    for prompt in ("春眠不觉晓", "白日依山尽", "明月"):
        print(f"  {prompt} -> {generate(model, tokenizer, prompt, device=device)}")

    ROOT.joinpath("ckpt").mkdir(exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "config": dict(vocab_size=len(tokenizer), d_model=args.d_model,
                       n_heads=args.n_heads, n_layers=args.n_layers,
                       d_ff=4 * args.d_model, max_len=args.max_len),
        "vocab": tokenizer.vocab,
        "losses": losses,
    }, ROOT / "ckpt" / "toy_lm.pt")
    print(f"\n已保存 {ROOT / 'ckpt' / 'toy_lm.pt'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
        plt.rcParams["axes.unicode_minus"] = False

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(losses, lw=0.8, alpha=0.4, label="每步 loss")
        win = 50
        if len(losses) > win:
            smooth = [sum(losses[i:i + win]) / win for i in range(len(losses) - win + 1)]
            ax.plot(range(win - 1, len(losses)), smooth, lw=2, label=f"{win} 步滑动平均")
        ax.set_xlabel("step")
        ax.set_ylabel("cross entropy")
        ax.set_title("toy 语言模型（causal mask）训练曲线")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        ROOT.joinpath("figures").mkdir(exist_ok=True)
        fig.savefig(ROOT / "figures" / "toy_lm_loss.png", dpi=150)
        print(f"训练曲线已保存 {ROOT / 'figures' / 'toy_lm_loss.png'}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
