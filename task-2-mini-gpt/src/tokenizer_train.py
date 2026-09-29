"""训练 BPE 词表并保存到 ckpt/tokenizer.json（M1 的产出）。

用法（在 task-2-mini-gpt 目录下运行）：
    python src/tokenizer_train.py --mode char --vocab_size 3500   # 字符级（推荐：中文小语料）
    python src/tokenizer_train.py --mode byte --vocab_size 2048   # 字节级（原始字节起步）
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tokenizer import BPETokenizer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "train.txt"),
                    help="训练 BPE 用的语料（默认 data/train.txt，只用训练集）")
    ap.add_argument("--mode", default="char", choices=["char", "byte"],
                    help="char=字符级基本单位（中文小语料推荐）/ byte=字节级")
    ap.add_argument("--vocab_size", type=int, default=3500)
    ap.add_argument("--out", default=str(ROOT / "ckpt" / "tokenizer.json"))
    ap.add_argument("--min_freq", type=int, default=2)
    args = ap.parse_args()

    text = Path(args.data).read_text(encoding="utf-8")
    print(f"语料 {args.data}：{len(text)} 字符，模式 {args.mode}")

    t0 = time.time()
    tk = BPETokenizer.train_bpe(text, vocab_size=args.vocab_size,
                                min_freq=args.min_freq, mode=args.mode)
    print(f"耗时 {time.time() - t0:.1f}s")

    # 压缩率体检：平均每个 token 覆盖多少字符（越高说明词表越划算）
    ids = tk.encode(text)
    print(f"encode 后 {len(ids)} tokens，压缩率 {len(text) / len(ids):.2f} 字符/token")

    # 落地前的自检：encode -> decode 必须逐字符还原（含 dev 里的未登录字）
    dev = ROOT / "data" / "dev.txt"
    samples = ["床前明月光", "Hello, world!", "深度学习需要数学基础"]
    if dev.exists():
        samples.append(dev.read_text(encoding="utf-8")[:400])
    for s in samples:
        got = tk.decode(tk.encode(s))
        assert got == s, f"roundtrip 失败：{s[:40]!r} -> {got[:40]!r}"
    print("roundtrip 自检通过（含 dev 未登录字回退）")

    tk.save(args.out)
    print(f"词表已保存到 {args.out}（vocab_size={tk.vocab_size}）")


if __name__ == "__main__":
    main()
