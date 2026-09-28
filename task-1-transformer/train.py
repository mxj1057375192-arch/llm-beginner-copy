"""任务一 M3：在 ChnSentiCorp 中文情感分类上训练 Transformer。

用法：
    python train.py
    python train.py --d-model 64 --n-layers 2 --n-heads 2   # 消融实验 S1
    python train.py --no-residual                            # 消融实验 S2

产出：
    ckpt/best.pt             验证集准确率最高的模型
    figures/loss_curve.png   训练曲线
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.model import TransformerClassifier
from src.tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parent


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
def encode_all(texts, tokenizer, max_len):
    return [tokenizer.encode(t, max_len) for t in texts]


def make_batches(lengths, batch_size, shuffle, rng, mega=50):
    """按长度分桶组 batch。

    一个 batch 内所有序列会被 pad 到本批最长的那条。如果随机组批，
    短句和长句混在一起会产生大量无用的 PAD 计算。先把长度接近的样本
    排到一起，能显著减少 padding 浪费（对 CPU 训练尤其明显）。
    mega 控制「大块」大小，块内打乱保证随机性。
    """
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches = []
    for start in range(0, len(order), batch_size * mega):
        chunk = order[start:start + batch_size * mega]
        if shuffle:
            rng.shuffle(chunk)
        for i in range(0, len(chunk), batch_size):
            batches.append(chunk[i:i + batch_size])
    if shuffle:
        rng.shuffle(batches)
    return batches


def collate(batch, pad_id):
    """把一批 (ids, label) 补齐成 (B, T_max) 的张量。"""
    seqs, labels = zip(*batch)
    length = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), length), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = s
    return ids, torch.tensor(labels, dtype=torch.long)


def run_epoch_batches(encoded, labels, batches, model, device, criterion,
                      optimizer=None, pad_id=0):
    """跑一个 epoch；optimizer=None 表示只做评估。"""
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss, correct, total = 0.0, 0, 0
    for idx in batches:
        ids = collate([(encoded[i], labels[i]) for i in idx], pad_id)
        x, y = ids[0].to(device), ids[1].to(device)

        with torch.set_grad_enabled(train_mode):
            logits = model(x)
            loss = criterion(logits, y)
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # 梯度裁剪：Transformer 偶尔会出现梯度爆炸，裁剪一下更稳
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        total_loss += loss.item() * len(y)
        correct += (logits.argmax(-1) == y).sum().item()
        total += len(y)
    return total_loss / total, correct / total


def main():
    p = argparse.ArgumentParser()
    # 模型超参（读一下 README 的「建议超参起点」再自己调）
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=None, help="默认 4*d_model")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max-len", type=int, default=200)
    # 训练超参
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--min-freq", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", type=str, default="", help="消融实验时给结果文件加后缀")
    # 消融开关（S2）
    p.add_argument("--no-residual", action="store_true")
    p.add_argument("--no-layernorm", action="store_true")
    p.add_argument("--out", type=str, default="ckpt/best.pt")
    args = p.parse_args()

    set_seed(args.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device} | 线程数: {torch.get_num_threads()}")

    d_ff = args.d_ff or 4 * args.d_model
    use_residual = not args.no_residual
    use_layernorm = not args.no_layernorm

    # ---- 读数据 ----
    train_df = pd.read_parquet(ROOT / "data" / "train.parquet")
    dev_df = pd.read_parquet(ROOT / "data" / "validation.parquet")
    print(f"训练集 {len(train_df)} 条 / 验证集 {len(dev_df)} 条")

    tokenizer = CharTokenizer.build(train_df["text"].tolist(), min_freq=args.min_freq)
    print(f"字符级词表大小: {len(tokenizer)}")

    train_enc = encode_all(train_df["text"], tokenizer, args.max_len)
    dev_enc = encode_all(dev_df["text"], tokenizer, args.max_len)
    train_y = train_df["label"].tolist()
    dev_y = dev_df["label"].tolist()

    # ---- 模型 ----
    config = dict(
        vocab_size=len(tokenizer),
        num_classes=int(train_df["label"].nunique()),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=d_ff,
        max_len=args.max_len,
        dropout=args.dropout,
        use_residual=use_residual,
        use_layernorm=use_layernorm,
    )
    model = TransformerClassifier(**config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {n_params/1e6:.2f} M | config: {config}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay, betas=(0.9, 0.98))
    criterion = nn.CrossEntropyLoss()

    # ---- 学习率：线性 warmup + 余弦衰减 ----
    rng = random.Random(args.seed)
    batches_per_epoch = math.ceil(len(train_enc) / args.batch_size)
    total_steps = batches_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    min_lr_ratio = 0.1

    def lr_lambda(step):
        if step < warmup_steps:                       # 预热：从 0 线性升到 lr
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)

    # ---- 训练 ----
    lengths = [len(s) for s in train_enc]
    best_acc, best_epoch, history = 0.0, -1, []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_batches = make_batches(lengths, args.batch_size, True, rng)

        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for idx in train_batches:
            ids = collate([(train_enc[i], train_y[i]) for i in idx], tokenizer.pad_id)
            x, y = ids[0].to(device), ids[1].to(device)

            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            tr_loss += loss.item() * len(y)
            tr_correct += (logits.argmax(-1) == y).sum().item()
            tr_total += len(y)

        dev_batches = make_batches([len(s) for s in dev_enc], 64, False, rng)
        dev_loss, dev_acc = run_epoch_batches(dev_enc, dev_y, dev_batches, model, device, criterion)

        tr_loss /= tr_total
        tr_acc = tr_correct / tr_total
        history.append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc,
                            dev_loss=dev_loss, dev_acc=dev_acc,
                            lr=optimizer.param_groups[0]["lr"]))

        mark = ""
        if dev_acc > best_acc:
            best_acc, best_epoch, mark = dev_acc, epoch, "  <- best"
            torch.save({
                "state_dict": model.state_dict(),
                "config": config,
                "vocab": tokenizer.vocab,
                "epoch": epoch,
                "dev_acc": dev_acc,
            }, out_path)

        print(f"[{epoch}/{args.epochs}] train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"dev loss {dev_loss:.4f} acc {dev_acc:.4f} | "
              f"lr {optimizer.param_groups[0]['lr']:.2e} | {time.time()-t0:.1f}s{mark}")

    print(f"\n最佳验证准确率 {best_acc:.4f}（第 {best_epoch} 轮），已保存到 {out_path}")

    # 消融实验时把结果追加到一份 json，方便对比
    tag = args.tag or f"d{args.d_model}_h{args.n_heads}_l{args.n_layers}"
    exp_path = ROOT / "figures" / "experiments.json"
    records = json.loads(exp_path.read_text(encoding="utf-8")) if exp_path.exists() else {}
    records[tag] = dict(best_dev_acc=round(best_acc, 4), best_epoch=best_epoch,
                        config=config, params_m=n_params / 1e6)
    exp_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 画训练曲线 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ep = [h["epoch"] for h in history]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(ep, [h["train_loss"] for h in history], marker="o", label="train")
        axes[0].plot(ep, [h["dev_loss"] for h in history], marker="s", label="dev")
        axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
        axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

        axes[1].plot(ep, [h["train_acc"] for h in history], marker="o", label="train")
        axes[1].plot(ep, [h["dev_acc"] for h in history], marker="s", label="dev")
        axes[1].axhline(0.80, ls="--", c="gray", label="pass line 0.80")
        axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy")
        axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(ROOT / "figures" / "loss_curve.png", dpi=150)
        print("训练曲线已保存到 figures/loss_curve.png")
    except ImportError:
        print("[提示] 没装 matplotlib，跳过画图")


if __name__ == "__main__":
    main()
