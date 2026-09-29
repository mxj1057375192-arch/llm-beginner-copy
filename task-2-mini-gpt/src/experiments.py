"""加分项 S1：参数量 / 位置编码 / 上下文长度的对照实验（扫描训练）。

一条命令跑一组配置，结果汇总到 docs/experiments.md。CPU 也能跑完，只是每组几分钟。

用法（在 task-2-mini-gpt 目录下）：
    # 1) 参数量扫描（10M 那一档 CPU 上会很慢，本地建议先跑 tiny/small 两档）
    python src/experiments.py --sweep param
    # 2) 绝对位置编码 vs RoPE 的长序列外推（S2）
    python src/experiments.py --sweep posemb
    # 3) 自定义一组
    python src/experiments.py --sweep custom --configs "d192L4:192:4:4,d384L6:384:6:6"
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

# 每组配置：名字 -> (d_model, n_layers, n_heads, max_seq_len, block_size, steps)
SWEEPS = {
    # S1：固定数据、固定步数，只改模型大小，看困惑度怎么变
    "param": {
        "d128L2": (128, 2, 4, 128, 128, 1500),
        "d192L4": (192, 4, 4, 128, 128, 1500),
        "d384L4": (384, 4, 6, 128, 128, 1500),
        "d384L8": (384, 8, 6, 128, 128, 1500),
    },
    # S2：RoPE vs 绝对位置编码（learned）
    "posemb": {
        "rope_L128": (192, 4, 4, 256, 128, 1500),
        "learned_L128": (192, 4, 4, 256, 128, 1500),
    },
    # S2 后半段：训练时用 128 上下文，测试时喂更长的序列，看谁外推得住
    "extrapolate": {
        "rope_ctx128": (192, 4, 4, 256, 128, 2000),
        "learned_ctx128": (192, 4, 4, 128, 128, 2000),
    },
}


def parse_configs(spec: str):
    """解析 --configs "name:192,4,4" 形式的自定义配置（d_model,层数,头数）。"""
    out = {}
    for item in spec.split(","):
        name, dims = item.split(":")
        d, l, h = (int(x) for x in dims.split("x"))
        out[name] = (d, l, h, 128, 128, 1500)
    return out


def run_one(name: str, cfg, pos_emb: str, extra: list[str]) -> dict:
    d_model, n_layers, n_heads, max_seq, block, steps = cfg
    out_dir = ROOT / "ckpt_sweep" / name
    cmd = [PY, "src/train.py", "--d_model", str(d_model), "--n_layers", str(n_layers),
           "--n_heads", str(n_heads), "--max_seq_len", str(max_seq), "--block_size", str(block),
           "--steps", str(steps), "--pos_emb", pos_emb, "--out_dir", str(out_dir),
           "--eval_interval", str(max(100, steps // 4)), "--warmup", "100",
           "--lr", "1e-3", "--batch_size", "32"] + extra
    print(f"\n{'=' * 70}\n>>> {name}: {' '.join(cmd[1:])}\n{'=' * 70}", flush=True)
    subprocess.call(cmd, cwd=str(ROOT))

    log_path = out_dir / "train_log.json"
    if not log_path.exists():
        return {"name": name, "error": "训练未产出 train_log.json"}
    log = json.loads(log_path.read_text(encoding="utf-8"))
    cfg_model = log["model"]
    n_params = None
    ckpt = out_dir / "best.pt"
    if ckpt.exists():
        import torch
        state = torch.load(ckpt, map_location="cpu")["model_state"]
        n_params = sum(v.numel() for v in state.values())
    return {
        "name": name,
        "d_model": cfg_model["d_model"], "n_layers": cfg_model["n_layers"],
        "n_heads": cfg_model["n_heads"], "max_seq_len": cfg_model["max_seq_len"],
        "pos_emb": cfg_model.get("pos_emb", "rope"),
        "block_size": log["block_size"], "steps": steps,
        "params_m": round(n_params / 1e6, 2) if n_params else None,
        "best_dev_ppl_random_window": round(
            min(e["dev_ppl"] for e in log["evals"]), 2) if log["evals"] else None,
        "best_dev_ppl_eval_style": round(log.get("best_ppl", float("nan")), 2),
        "ckpt": str(ckpt),
        "final_train_loss": log["steps"][-1]["loss"] if log["steps"] else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="param", choices=list(SWEEPS) + ["custom"])
    ap.add_argument("--configs", default=None, help="custom 时用，形如 name:192x4x4")
    ap.add_argument("--pos_emb", default="rope", choices=["rope", "learned"])
    ap.add_argument("--out", default=str(ROOT / "docs" / "experiments.md"))
    args = ap.parse_args()

    configs = parse_configs(args.configs) if args.sweep == "custom" else SWEEPS[args.sweep]
    results = []
    for name, cfg in configs.items():
        pos_emb = "learned" if "learned" in name else args.pos_emb
        results.append(run_one(name, cfg, pos_emb, []))

    # 汇总成 markdown 表
    lines = [f"# 加分项实验汇总 · {args.sweep}", "",
             "| 配置 | d_model | 层数 | 头数 | 位置编码 | 参数量(M) | 训练步数 | "
             "dev ppl（随机窗口） | dev ppl（自检口径） |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        if "error" in r:
            lines.append(f"| {r['name']} | - | - | - | - | - | - | {r['error']} | - |")
            continue
        lines.append(f"| {r['name']} | {r['d_model']} | {r['n_layers']} | {r['n_heads']} | "
                     f"{r['pos_emb']} | {r['params_m']} | {r['steps']} | "
                     f"{r['best_dev_ppl_random_window']} | {r['best_dev_ppl_eval_style']} |")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = out.read_text(encoding="utf-8") if out.exists() else ""
    out.write_text(existing + "\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / "ckpt_sweep" / f"results_{args.sweep}.json").parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "ckpt_sweep" / f"results_{args.sweep}.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n汇总写入 {out}")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
