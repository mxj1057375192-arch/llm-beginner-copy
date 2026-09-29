"""chunk_size / overlap 扫描（文本层面预筛，对应加分项 S1）。

不建索引、不跑模型：只把 PDF 抽一次文本，对每个 (chunk_size, overlap) 组合算
「多少条 gold 题至少有一个 anchor 完整落在某个 chunk 里」——这正是自检
`nndl_gold_recall_at_10` 的命中口径（只是它还要真的检索得到）。

先用它挑出覆盖率最高的一档再去建索引，能省掉好几轮「重建索引 -> 重跑评测」的来回。
注意这里算的是**文本上限**：chunk 里有没有 anchor，真实 Recall 只会更低（还取决于检索排序）。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from .chunker import _spans, load_pdf_text
from .paths import GOLD_QA, KB_PDF

DEFAULT_GRID = [(128, 16), (256, 32), (256, 64), (512, 64), (512, 128), (1024, 128)]


def normalize(text) -> str:
    """去掉所有空白——自检的 anchor 命中判定就是这个口径。"""
    return re.sub(r"\s+", "", str(text))


def load_gold(path=None):
    path = Path(path or GOLD_QA)
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def coverage(full_text: str, chunk_size: int, overlap: int, gold) -> dict:
    """返回该组合下每个 chunk 的平均长度与 anchor 命中情况。"""
    chunks = [normalize(full_text[s:e]) for s, e in _spans(full_text, chunk_size, overlap)]
    chunks = [c for c in chunks if c]
    # 用 \x00 把各块拼起来再搜：anchor 不含 \x00，所以"命中拼接串"等价于
    # "完整落在某一个 chunk 里"，但只需扫一遍文本，快得多。
    blob = "\x00".join(chunks)

    hits, misses = 0, []
    for item in gold:
        anchors = [normalize(a) for a in item.get("gold_anchors", [])]
        if any(a in blob for a in anchors if a):
            hits += 1
        else:
            misses.append(item.get("id"))
    lens = [len(c) for c in chunks]
    return {
        "chunk_size": chunk_size, "overlap": overlap, "chunks": len(chunks),
        "avg_len": round(sum(lens) / len(lens), 1) if lens else 0,
        "hits": hits, "n": len(gold), "recall": round(hits / len(gold), 3),
        "misses": misses,
    }


def main():
    import argparse
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="chunk 策略扫描（文本层面 anchor 覆盖率）")
    ap.add_argument("--pdf", default=None)
    ap.add_argument("--grid", default=None,
                    help="自定义组合，如 256x32,512x64（不传则用内置网格）")
    args = ap.parse_args()

    pdf_path = Path(args.pdf or KB_PDF)
    gold = load_gold()
    full_text, _ = load_pdf_text(pdf_path)
    print(f"PDF：{pdf_path.name}  抽文本 {len(full_text)} 字符  评测题 {len(gold)} 条\n")

    # 上限：anchor 本身在不在 PDF 抽出的文本里（不在的话怎么切都没用）
    pdf_norm = normalize(full_text)
    reachable = sum(1 for it in gold
                    if any(normalize(a) in pdf_norm for a in it.get("gold_anchors", [])))
    print(f"anchor 出现在 PDF 全文里的题数（覆盖率上限）：{reachable}/{len(gold)}\n")

    grid = DEFAULT_GRID
    if args.grid:
        grid = [tuple(int(x) for x in pair.split("x")) for pair in args.grid.split(",")]

    print(f"{'chunk_size':>10} {'overlap':>8} {'chunks':>8} {'avg_len':>9}  命中  覆盖率")
    best = None
    for chunk_size, overlap in grid:
        r = coverage(full_text, chunk_size, overlap, gold)
        print(f"{r['chunk_size']:>10} {r['overlap']:>8} {r['chunks']:>8} "
              f"{r['avg_len']:>9} {r['hits']:>4}/{r['n']:<3} {r['recall']:.3f}")
        if best is None or r["recall"] > best["recall"]:
            best = r

    print(f"\n最佳组合：chunk_size={best['chunk_size']}, overlap={best['overlap']} "
          f"（anchor 覆盖率 {best['recall']}）")
    if best["misses"]:
        print(f"仍未命中的题：{', '.join(best['misses'])}")
        print("（这些题的 anchor 落在抽取文本本身就不连续的位置，或者切分点恰好切断，"
              "需要看具体文本再决定是否调 overlap）")


if __name__ == "__main__":
    main()
