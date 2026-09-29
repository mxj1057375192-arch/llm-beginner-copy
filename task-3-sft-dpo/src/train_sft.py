"""SFT：在注入 LoRA 的 Qwen2.5-0.5B 上做 next-token prediction（任务三 M3）。

    python src/train_sft.py --limit 600 --max-length 128 --steps 150

CPU 上跑，所以默认参数都压得很小：只取 MOSS 前 1-2 轮、序列截到 128、
几百步就收工。产物是 ckpt/sft/ 下的 LoRA 权重。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

TASK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_ROOT))       # 支持 python src/train_sft.py 直接跑

from src.chat import encode_messages, get_tokenizer
from src.data import load_moss
from src.lora import count_trainable, inject_lora, lora_state_dict


def build_dataset(path, tokenizer, args):
    items = []
    for messages in load_moss(path, max_turns=args.max_turns,
                              max_chars=args.max_chars, limit=args.limit):
        ids, labels = encode_messages(messages, tokenizer=tokenizer,
                                      max_length=args.max_length)
        if (labels != -100).any():          # 至少要有 assistant 内容才值得训
            items.append((ids, labels))
    return items


def make_collate(pad_id: int):
    def collate(batch):
        width = max(len(ids) for ids, _ in batch)
        input_ids, labels, attn = [], [], []
        for ids, lab in batch:
            pad = width - len(ids)
            input_ids.append(torch.cat([ids, torch.full((pad,), pad_id, dtype=ids.dtype)]))
            labels.append(torch.cat([lab, torch.full((pad,), -100, dtype=lab.dtype)]))
            attn.append(torch.cat([torch.ones(len(ids), dtype=torch.long),
                                   torch.zeros(pad, dtype=torch.long)]))
        return torch.stack(input_ids), torch.stack(labels), torch.stack(attn)
    return collate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(TASK_ROOT / "models" / "Qwen2.5-0.5B"))
    p.add_argument("--data", default=str(TASK_ROOT / "data" / "moss-sft" / "sample.jsonl"))
    p.add_argument("--out", default=str(TASK_ROOT / "ckpt" / "sft"))
    p.add_argument("--limit", type=int, default=600, help="取多少条对话")
    p.add_argument("--max-turns", type=int, default=2)
    p.add_argument("--max-chars", type=int, default=300, help="单条 message 字符上限")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--target-modules", nargs="+", default=["q_proj", "v_proj"])
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    from transformers import AutoModelForCausalLM

    print(f"加载基座 {args.model} ...")
    t0 = time.time()
    tokenizer = get_tokenizer(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    print(f"  完成，{time.time() - t0:.1f}s")

    inject_lora(model, args.target_modules, r=args.rank, alpha=args.alpha)
    trainable = count_trainable(model)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA 注入完成：可训 {trainable:,} / 总 {total:,} = {trainable / total:.4%}")

    dataset = build_dataset(args.data, tokenizer, args)
    print(f"数据集：{len(dataset)} 条，token 长度中位数 "
          f"{sorted(len(i) for i, _ in dataset)[len(dataset) // 2]}")

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=make_collate(pad_id))

    model.config.use_cache = False
    model.train()
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)

    history, step, t0 = [], 0, time.time()
    done = False
    for epoch in range(1, 100):
        for input_ids, labels, attn in loader:
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            loss = out.loss
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            history.append(round(loss.item(), 4))

            if step % 10 == 0 or step == 1:
                el = time.time() - t0
                eta = el / step * (args.steps - step)
                print(f"  step {step:4d}/{args.steps}  loss {loss.item():.4f}  "
                      f"已用 {el / 60:.1f}min  预计还需 {eta / 60:.1f}min", flush=True)
            if step >= args.steps:
                done = True
                break
        if done:
            break

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(lora_state_dict(model), out_dir / "lora.pt")
    (out_dir / "train_args.json").write_text(
        json.dumps({"args": vars(args), "loss": history}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\n已保存 LoRA 权重到 {out_dir}/lora.pt")
    print(f"loss 首/末：{history[0]} -> {history[-1]}")


if __name__ == "__main__":
    main()
