"""预训练 mini-GPT（M4）：next-token prediction + 困惑度监控。

用法（在 task-2-mini-gpt 目录下）：
    python -u src/train.py                    # 默认配置，CPU 约 20-40 分钟
    python -u src/train.py --steps 200        # 冒烟测试：1 分钟跑通
    python -u src/train.py --preset small     # 更大的模型（有 GPU 再试）

建议加 `-u`（或设 PYTHONUNBUFFERED=1）让日志实时打印，方便边跑边看。

产出（默认写到 ckpt/）：
    best.pt          验证困惑度最低时保存的模型（含 config，自检脚本据此重建模型）
    last.pt          最后一轮
    train_log.json   每步 loss / 每次 eval 的困惑度，画曲线用
    loss_curve.png   训练曲线

训练循环的四个关键点（README 常见坑）：
    1. warmup + cosine：开头 lr 从 0 线性升到峰值（避免早期大梯度炸掉），之后余弦退火。
    2. gradient clipping：裁剪梯度范数，防止偶发 loss spike。
    3. 区分 train / dev：dev 只看困惑度，不参与梯度。
    4. 早停/保存最优：唐诗语料小，dev 困惑度变差就保留之前最好的那次。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import MODEL_PRESETS, MiniGPT          # noqa: E402
from src.tokenizer import BPETokenizer                # noqa: E402


# ---------------------------------------------------------------------------
# 学习率调度：线性 warmup + cosine 退火
# ---------------------------------------------------------------------------
def lr_at(step: int, lr: float, warmup: int, total: int, min_ratio: float = 0.1) -> float:
    if step < warmup:
        return lr * (step + 1) / max(1, warmup)
    if step >= total:
        return lr * min_ratio
    progress = (step - warmup) / max(1, total - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))       # 1 -> 0
    return lr * (min_ratio + (1 - min_ratio) * coeff)


# ---------------------------------------------------------------------------
# 数据：把整段 token 流随机切窗口（get_batch）
# ---------------------------------------------------------------------------
class TokenStream:
    def __init__(self, ids, block_size: int, device="cpu"):
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.block_size = block_size
        self.device = device

    def get_batch(self, batch_size: int, generator=None):
        """随机取 batch_size 个长度为 block_size+1 的连续片段。

        x = ids[i : i+block_size]      （输入）
        y = ids[i+1 : i+block_size+1]  （标签 = 输入右移一位，就是 next-token prediction）
        """
        n = len(self.ids)
        if n <= self.block_size + 1:
            raise ValueError("语料太短：token 数不足以切出一个 block")
        ix = torch.randint(n - self.block_size - 1, (batch_size,), generator=generator)
        x = torch.stack([self.ids[i:i + self.block_size] for i in ix])
        y = torch.stack([self.ids[i + 1:i + self.block_size + 1] for i in ix])
        return x.to(self.device), y.to(self.device)


@torch.no_grad()
def evaluate(model: MiniGPT, stream: TokenStream, batch_size: int = 16,
             iters: int = 20, generator=None) -> float:
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = stream.get_batch(batch_size, generator=generator)
        _, loss = model(x, targets=y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def perplexity_like_eval(model: MiniGPT, ids, block: int, max_tokens: int = 4096) -> float:
    """完全按 eval/run.py 的口径算困惑度：dev 文本按 block 非重叠切块，逐块累加 NLL。

    为什么要单独算一遍？训练时用的是「随机窗口」（每个 token 都能看到前文），而自检是从
    文本开头按固定长度硬切，每块的头几个 token 缺少上文，NLL 天然更高。用同口径的数字盯
    进度，才能知道离「唐诗 < 50」还有多远。
    """
    model.eval()
    ids = ids[:max_tokens]
    nll, n_tok = 0.0, 0
    for i in range(0, max(1, len(ids) - 1), block):
        window = ids[i:i + block + 1]
        if len(window) < 2:
            break
        chunk = torch.tensor([window], dtype=torch.long,
                             device=next(model.parameters()).device)
        logits = model(chunk)
        nll += torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            chunk[:, 1:].reshape(-1), reduction="sum").item()
        n_tok += chunk.size(1) - 1
    model.train()
    return math.exp(nll / max(1, n_tok))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="tiny", choices=list(MODEL_PRESETS))
    ap.add_argument("--d_model", type=int, default=None)
    ap.add_argument("--n_layers", type=int, default=None)
    ap.add_argument("--n_heads", type=int, default=None)
    ap.add_argument("--max_seq_len", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--pos_emb", default="rope", choices=["rope", "learned"],
                    help="rope=旋转位置编码（默认）；learned=绝对位置编码（S2 对照组）")
    # 训练用的上下文长度（block_size）：可以小于模型上限，短窗口省时间、样本更多
    ap.add_argument("--block_size", type=int, default=None)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min_lr_ratio", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--eval_interval", type=int, default=250)
    ap.add_argument("--eval_iters", type=int, default=20)
    ap.add_argument("--max_minutes", type=float, default=0.0,
                    help=">0 时到点就停（先跑通再决定要不要加长，很实用）")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--torch_threads", type=int, default=0, help="0=自动")
    ap.add_argument("--out_dir", default=str(ROOT / "ckpt"))
    ap.add_argument("--tokenizer", default=str(ROOT / "ckpt" / "tokenizer.json"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 分词器 + 语料
    if not Path(args.tokenizer).exists():
        sys.exit(f"[错误] 找不到 {args.tokenizer}，请先跑 python src/tokenizer_train.py")
    tok = BPETokenizer.from_pretrained(args.tokenizer)
    train_text = (ROOT / "data" / "train.txt").read_text(encoding="utf-8")
    dev_text = (ROOT / "data" / "dev.txt").read_text(encoding="utf-8")
    train_ids = tok.encode(train_text)
    dev_ids = tok.encode(dev_text)
    print(f"词表 {tok.vocab_size}；train {len(train_ids)} tokens，dev {len(dev_ids)} tokens",
          flush=True)

    # 2) 模型配置
    cfg = dict(MODEL_PRESETS[args.preset])
    for k in ("d_model", "n_layers", "n_heads", "max_seq_len", "dropout"):
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
    cfg["pos_emb"] = args.pos_emb
    model = MiniGPT(vocab_size=tok.vocab_size, **cfg).to(device)
    block_size = args.block_size or min(cfg["max_seq_len"], 128)
    print(f"模型配置 {cfg}", flush=True)
    print(f"参数量 {model.num_parameters() / 1e6:.2f}M（非 embedding "
          f"{model.num_parameters(non_embedding=True) / 1e6:.2f}M）；block_size={block_size}",
          flush=True)

    train_stream = TokenStream(train_ids, block_size, device)
    dev_stream = TokenStream(dev_ids, min(block_size, max(8, len(dev_ids) // 4)), device)

    # 3) 优化器：weight decay 只加在矩阵参数上（norm/embedding 不加，常规做法）
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2 and "emb" not in name:
            decay.append(p)
        else:
            no_decay.append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8)

    gen = torch.Generator().manual_seed(args.seed)
    log = {"config": vars(args), "model": cfg, "block_size": block_size,
           "vocab_size": tok.vocab_size, "steps": [], "evals": []}
    best_ppl = float("inf")
    t0 = time.time()
    tokens_seen = 0
    stopped_by_time = False

    model.train()
    for step in range(args.steps):
        if args.max_minutes > 0 and step > 0 and (time.time() - t0) / 60 > args.max_minutes:
            print(f"[stop] 达到 --max_minutes={args.max_minutes}，提前停止", flush=True)
            stopped_by_time = True
            break
        lr = lr_at(step, args.lr, args.warmup, args.steps, args.min_lr_ratio)
        for g in optimizer.param_groups:
            g["lr"] = lr

        x, y = train_stream.get_batch(args.batch_size, generator=gen)
        _, loss = model(x, targets=y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # gradient clipping：偶发 loss spike 会让梯度范数暴涨，裁一下更稳
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        tokens_seen += x.numel()

        if step % args.log_every == 0 or step == args.steps - 1:
            sps = tokens_seen / max(1e-6, time.time() - t0)
            print(f"step {step:5d}/{args.steps} | loss {loss.item():.4f} | ppl "
                  f"{math.exp(min(loss.item(), 20)):7.2f} | lr {lr:.2e} | "
                  f"|g| {grad_norm.item():.2f} | {sps:6.0f} tok/s | "
                  f"{time.time() - t0:5.0f}s", flush=True)
            log["steps"].append({"step": step, "loss": loss.item(), "lr": lr,
                                 "grad_norm": float(grad_norm.item())})

        # 4) 验证困惑度 + 保存最优
        if (step + 1) % args.eval_interval == 0 or step == args.steps - 1:
            val_loss = evaluate(model, dev_stream, batch_size=8,
                                iters=args.eval_iters, generator=gen)
            ppl = math.exp(min(val_loss, 20))
            ppl_eval = perplexity_like_eval(model, dev_ids, block_size)
            print(f"          [eval] dev loss {val_loss:.4f}  ppl {ppl:.2f}  "
                  f"|  自检口径 ppl {ppl_eval:.2f}", flush=True)
            log["evals"].append({"step": step, "dev_loss": val_loss, "dev_ppl": ppl,
                                 "dev_ppl_eval_style": ppl_eval})
            if ppl_eval < best_ppl:
                best_ppl = ppl_eval
                torch.save({"model_state": model.state_dict(), "config": cfg,
                            "vocab_size": tok.vocab_size, "step": step,
                            "dev_ppl": ppl, "dev_ppl_eval_style": ppl_eval,
                            "tokenizer_path": str(Path(args.tokenizer).resolve())},
                           out_dir / "best.pt")
                print(f"          [save] 新的最优模型 -> {out_dir / 'best.pt'} "
                      f"(自检口径 ppl {ppl_eval:.2f})", flush=True)

    torch.save({"model_state": model.state_dict(), "config": cfg,
                "vocab_size": tok.vocab_size, "step": step,
                "tokenizer_path": str(Path(args.tokenizer).resolve())},
               out_dir / "last.pt")
    log["best_ppl"] = best_ppl
    log["elapsed_sec"] = time.time() - t0
    log["stopped_by_time"] = stopped_by_time
    (out_dir / "train_log.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

    # 5) 画曲线（loss + 困惑度）
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        steps_ = [d["step"] for d in log["steps"]]
        axes[0].plot(steps_, [d["loss"] for d in log["steps"]])
        axes[0].set_title("train loss"); axes[0].set_xlabel("step"); axes[0].grid(alpha=.3)
        if log["evals"]:
            es = [d["step"] for d in log["evals"]]
            axes[1].plot(es, [d["dev_ppl"] for d in log["evals"]], marker="o", ms=3,
                         label="random-window ppl")
            axes[1].plot(es, [d["dev_ppl_eval_style"] for d in log["evals"]],
                         marker="s", ms=3, label="eval-harness style ppl")
            axes[1].axhline(50, color="r", ls="--", lw=1, label="poetry threshold 50")
            axes[1].set_title("dev perplexity"); axes[1].set_xlabel("step")
            axes[1].set_yscale("log"); axes[1].legend(); axes[1].grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(out_dir / "loss_curve.png", dpi=130)
        print(f"[save] 曲线 -> {out_dir / 'loss_curve.png'}", flush=True)
    except Exception as e:                                   # pragma: no cover
        print(f"[warn] 画图失败（不影响训练）：{e}", flush=True)

    print(f"\n训练结束，耗时 {time.time() - t0:.0f}s，最优自检口径 ppl = {best_ppl:.2f}",
          flush=True)


if __name__ == "__main__":
    main()
