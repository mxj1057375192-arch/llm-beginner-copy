"""加分项 S3：KV cache 开 / 关的推理速度对比（同样生成长度计时）。

用法：
    python src/benchmark_kv_cache.py --max_new_tokens 128 --runs 3
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import load_for_eval   # noqa: E402


def time_generate(model, ids, n_new: int, use_cache: bool, runs: int) -> float:
    """返回平均耗时（秒）。先跑一次热身，不计入。"""
    # 热身（第一次会触发各种惰性初始化）
    model.generate(ids, max_new_tokens=4, greedy=True, use_cache=use_cache, seed=0)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model.generate(ids, max_new_tokens=n_new, greedy=True, use_cache=use_cache, seed=0)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "ckpt" / "best.pt"))
    ap.add_argument("--prompt", default="床前明月光，疑是地上霜。")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "docs" / "kv_cache_benchmark.md"))
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    model, tok = load_for_eval(args.ckpt)
    ids = torch.tensor([tok.encode(args.prompt)], dtype=torch.long)
    n_new = args.max_new_tokens

    t_no = time_generate(model, ids, n_new, use_cache=False, runs=args.runs)
    t_yes = time_generate(model, ids, n_new, use_cache=True, runs=args.runs)
    # 每个新词元的平均耗时（毫秒）
    per_no = t_no / n_new * 1000
    per_yes = t_yes / n_new * 1000

    print(f"prompt: {args.prompt}（{ids.size(1)} tokens），生成 {n_new} 个新 token，"
          f"取 {args.runs} 次中位数")
    print(f"  无 KV cache：{t_no:.3f}s，单步 {per_no:.1f} ms")
    print(f"  有 KV cache：{t_yes:.3f}s，单步 {per_yes:.1f} ms")
    print(f"  加速比：{t_no / t_yes:.2f}x")

    lines = ["# 加分项 S3 · KV cache 开 / 关推理速度对比", "",
             f"- 模型：`{Path(args.ckpt).name}`（{model.num_parameters() / 1e6:.2f}M 参数，"
             f"上下文 {model.block_size}）",
             f"- prompt：`{args.prompt}`（{ids.size(1)} tokens），greedy 生成 {n_new} 个新 token",
             f"- 计时：`time.perf_counter()`，重复 {args.runs} 次取中位数（CPU，无 GPU）", "",
             "| 配置 | 总耗时 (s) | 单 token 耗时 (ms) | 相对加速 |",
             "|---|---|---|---|",
             f"| 无 KV cache | {t_no:.3f} | {per_no:.1f} | 1.00x |",
             f"| 有 KV cache | {t_yes:.3f} | {per_yes:.1f} | **{t_no / t_yes:.2f}x** |", "",
             "> 原因：无 cache 时第 t 步要把长度 t 的整段前缀重新 forward 一遍（总计算量 ~T²/d 的注意力 + "
             "T 次投影），有 cache 时每步只算 1 个新词元的 QKV + 一次长度 t 的注意力（总计算量 ~T）。"
             "所以序列越长，差距越大——把 `--max_new_tokens` 调大能更明显地看到加速比上升。"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"结果写入 {out}")


if __name__ == "__main__":
    main()
