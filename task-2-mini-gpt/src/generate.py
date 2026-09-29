"""用训练好的模型生成文本，对比四种采样策略（M5）。

用法（在 task-2-mini-gpt 目录下）：
    python src/generate.py                          # 默认几组 prompt，跑全部策略
    python src/generate.py --prompt "床前明月光" --max_new_tokens 60
    python src/generate.py --ckpt ckpt/best.pt --seed 0

产出：
    终端打印对比表格，同时写 docs/generation_samples.md（报告里直接引用）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import load_for_eval   # noqa: E402
from src.sampling import PRESETS      # noqa: E402

DEFAULT_PROMPTS = ["床前明月光", "春", "山中相送罢", "大江"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "ckpt" / "best.pt"))
    ap.add_argument("--prompt", default=None, help="不传则用内置的几组 prompt")
    ap.add_argument("--prompts", default=None, help="用 | 分隔多个 prompt")
    ap.add_argument("--max_new_tokens", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--repetition_penalty", type=float, default=1.15,
                    help="重复惩罚，>1 抑制复读（greedy 尤其需要）")
    ap.add_argument("--strategies", default=",".join(PRESETS.keys()))
    ap.add_argument("--out", default=str(ROOT / "docs" / "generation_samples.md"))
    args = ap.parse_args()

    if not Path(args.ckpt).exists():
        sys.exit(f"[错误] 找不到 {args.ckpt}，请先运行 python src/train.py")
    model, tok = load_for_eval(args.ckpt)
    print(f"模型：{model.num_parameters() / 1e6:.2f}M 参数，"
          f"block_size={model.block_size}，词表={tok.vocab_size}\n")

    if args.prompt:
        prompts = [args.prompt]
    elif args.prompts:
        prompts = args.prompts.split("|")
    else:
        prompts = DEFAULT_PROMPTS
    names = [s for s in args.strategies.split(",") if s in PRESETS]

    lines = ["# 任务二 · 生成样例对比（不同采样策略）", "",
             f"模型：`{Path(args.ckpt).name}`，参数量 {model.num_parameters() / 1e6:.2f}M，"
             f"上下文长度 {model.block_size}，词表 {tok.vocab_size}。",
             f"每条 prompt 生成 {args.max_new_tokens} 个新 token，固定随机种子 {args.seed}。", ""]

    for prompt in prompts:
        pid = tok.encode(prompt)
        print("=" * 78)
        print(f"prompt: {prompt}   （{len(pid)} 个 token）")
        print("=" * 78)
        lines += [f"## prompt：`{prompt}`", "",
                  "| 策略 | 参数 | 生成（含 prompt） | 新生成部分 |",
                  "|---|---|---|---|"]
        for name in names:
            kw = dict(PRESETS[name])
            desc = ", ".join(f"{k}={v}" for k, v in kw.items())
            out = model.generate(pid, max_new_tokens=args.max_new_tokens,
                                 seed=args.seed, repetition_penalty=args.repetition_penalty,
                                 **kw)
            text = tok.decode(out)
            new_text = tok.decode(out[len(pid):])
            print(f"\n[{name}]  ({desc})")
            print(f"  {text}")
            lines.append(f"| {name} | `{desc}` | {text} | {new_text} |")
        lines.append("")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n样例已写入 {out_path}")


if __name__ == "__main__":
    main()
