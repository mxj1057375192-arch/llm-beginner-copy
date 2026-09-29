"""base / SFT / DPO 三方对比（任务三 M4）。

    python src/compare.py --max-new-tokens 60

同一批指令分别喂给三个模型，输出并排打印，同时写入 ckpt/compare.md。
为了省内存，三个模型是**顺序**加载、用完就释放的（CPU 上 0.5B fp32 约 2 GB）。
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import torch

TASK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_ROOT))       # 支持 python src/compare.py 直接跑

from src.chat import format_messages, get_tokenizer
from src.lora import inject_lora, load_lora_state_dict

DEFAULT_PROMPTS = [
    "你好，请介绍一下你自己。",
    "用一句话解释什么是深度学习。",
    "给我三个提高睡眠质量的建议。",
    "把这句话翻译成英文：今天天气很好，我想出去散步。",
    "写一首关于秋天的五言绝句。",
    "什么是过拟合？怎么缓解？",
    "帮我把「我昨天去了图书馆借了三本书」改写成更正式的书面语。",
    "推荐一部适合周末看的电影，并说明理由。",
]


def build_model(args, lora_path=None):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    if lora_path:
        inject_lora(model, args.target_modules, r=args.rank, alpha=args.alpha)
        load_lora_state_dict(model, torch.load(lora_path, map_location="cpu"))
    model.eval()
    return model


@torch.no_grad()
def run_prompts(model, tokenizer, prompts, args):
    outs = []
    for text in prompts:
        chat = format_messages([{"role": "user", "content": text}],
                               add_generation_prompt=True)
        ids = tokenizer(chat, return_tensors="pt", add_special_tokens=False).input_ids
        gen = model.generate(
            ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        outs.append(tokenizer.decode(gen[0][ids.shape[1]:], skip_special_tokens=True).strip())
    return outs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(TASK_ROOT / "models" / "Qwen2.5-0.5B"))
    p.add_argument("--sft-ckpt", default=str(TASK_ROOT / "ckpt" / "sft" / "lora.pt"))
    p.add_argument("--dpo-ckpt", default=str(TASK_ROOT / "ckpt" / "dpo" / "lora.pt"))
    p.add_argument("--out", default=str(TASK_ROOT / "ckpt" / "compare.md"))
    p.add_argument("--max-new-tokens", type=int, default=60)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--target-modules", nargs="+", default=["q_proj", "v_proj"])
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    tokenizer = get_tokenizer(args.model)

    variants = [("base", None)]
    for label, path in (("SFT", args.sft_ckpt), ("DPO", args.dpo_ckpt)):
        if Path(path).exists():
            variants.append((label, path))
        else:
            print(f"[跳过] {label}：找不到 {path}")

    results = {}
    for label, lora_path in variants:
        print(f"\n=== 加载 {label} ...")
        t0 = time.time()
        model = build_model(args, lora_path)
        results[label] = run_prompts(model, tokenizer, DEFAULT_PROMPTS, args)
        print(f"    生成完成，用时 {time.time() - t0:.0f}s")
        del model
        gc.collect()

    lines = ["# base / SFT / DPO 输出对比", ""]
    for i, prompt in enumerate(DEFAULT_PROMPTS, 1):
        lines += [f"## {i}. {prompt}", ""]
        for label, _ in variants:
            lines += [f"**{label}**", "", "```", results[label][i - 1], "```", ""]
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"\n对比结果写入 {args.out}")

    for i, prompt in enumerate(DEFAULT_PROMPTS, 1):
        print(f"\n{'=' * 70}\n[{i}] {prompt}")
        for label, _ in variants:
            print(f"\n--- {label} ---\n{results[label][i - 1][:300]}")


if __name__ == "__main__":
    main()
