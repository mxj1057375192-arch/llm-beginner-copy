"""PDF 文本抽取与切分（M1）。

自检口径（../eval/run.py::test_chunking_sanity）：`chunk_text(text, chunk_size, overlap)`
的 chunk_size / overlap 一律按**字符**算，要求 chunk 数 > 10、平均长度落在
(chunk_size * 0.5, chunk_size * 1.2)。按词元数切会直接挂。

切分策略：先把文本按句子边界打散成小片段，再贪心装进「目标长度 = chunk_size - overlap」
的块里，最后把上一块的尾巴 overlap 个字符补到下一块开头。特意留出 overlap 的余量，
是为了补完重叠之后每块仍落在 chunk_size 附近——按 chunk_size 装再补重叠，块长会稳定超标。

PDF 的两点特殊处理：
- 页与页之间用 "\\n" 拼接，并记住每页起点的字符偏移。这样跨页的 gold anchor 不会被切断
  （评测的命中口径是「整个 anchor 出现在同一个 chunk 里」），同时每块仍能报出页码。
- 页眉页脚常被抽成孤立的纯数字行（页码），这类行只会污染检索，在页首/页尾各两行内删掉。
"""
from __future__ import annotations

import re
from bisect import bisect_right
from pathlib import Path
from typing import List

from .paths import TASK_ROOT

# 句子边界：中英文句末标点 + 换行。零宽断言切分，标点留在前一片段里
_SENT_SPLIT = re.compile(r"(?<=[。！？；!?;])|(?<=\n)")
# 孤立页码行：1-4 位数字
_PAGE_NUM_LINE = re.compile(r"^\s*\d{1,4}\s*$")


def clean_page_text(text: str) -> str:
    """单页文本的轻量清洗。

    只做不会破坏 anchor 连续性的清洗——公式、表格抽出来再乱也照原样留着，
    宁可留垃圾，也不要删掉正文导致 gold anchor 命中不了。
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in text.split("\n")]
    edge = set(range(min(2, len(lines)))) | set(range(max(0, len(lines) - 2), len(lines)))
    for i in edge:
        if _PAGE_NUM_LINE.match(lines[i]):
            lines[i] = ""
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines))


def extract_pdf_text(pdf_path) -> List[tuple]:
    """从 PDF 抽文本，返回 [(页码, 该页文本)]，页码从 1 开始。"""
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    return [(i, clean_page_text(page.extract_text() or ""))
            for i, page in enumerate(reader.pages, 1)]


def _spans(text: str, chunk_size: int, overlap: int) -> List[tuple]:
    """算出每个 chunk 在 text 中的 [start, end)，chunk_text / chunk_pdf 共用。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须是正整数（按字符计）")
    overlap = max(0, min(overlap, chunk_size // 2))
    target = max(chunk_size - overlap, chunk_size // 2)   # 每块「新增」内容的长度上限

    blocks = []                      # 不含重叠的原始块：覆盖全文、首尾相接
    cur_start, cur_len, pos = None, 0, 0
    for piece in _SENT_SPLIT.split(text):
        n = len(piece)
        if not piece.strip():
            # 纯空白：不动切点，直接并进当前块（保持 offset 连续）
            if cur_start is None:
                cur_start = pos
            cur_len += n
            pos += n
            continue
        if n > target:
            # 超长片段：PDF 抽出的表格 / 公式常整段没有句末标点，按 target 硬切
            if cur_len:
                blocks.append((cur_start, pos))
                cur_start, cur_len = None, 0
            for i in range(0, n, target):
                blocks.append((pos + i, min(pos + i + target, pos + n)))
            pos += n
            continue
        if cur_len and cur_len + n > target:
            blocks.append((cur_start, pos))
            cur_start, cur_len = None, 0
        if cur_start is None:
            cur_start = pos
        cur_len += n
        pos += n
    if cur_len:
        blocks.append((cur_start, pos))

    spans = []
    for i, (start, end) in enumerate(blocks):
        # 把上一块的尾巴 overlap 个字符补到本块开头，但不越过上一块的起点
        begin = start if i == 0 else max(blocks[i - 1][0], start - overlap)
        spans.append((begin, end))
    return spans


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    """把 text 切成字符数接近 chunk_size、相邻块重叠 overlap 个字符的列表（M1 契约）。"""
    out = [text[s:e].strip() for s, e in _spans(str(text), chunk_size, overlap)]
    return [c for c in out if c]


def load_pdf_text(pdf_path) -> tuple:
    """把 PDF 各页拼成全文，返回 (full_text, page_starts)。

    page_starts 是 [(起始字符偏移, 页码)]，用来把任意字符位置映射回页码。
    拼的时候页间只加一个 "\\n"，跨页的句子在全文里仍是连续的。
    """
    pages = extract_pdf_text(pdf_path)
    page_starts, parts = [], []
    pos = 0
    for page_no, page_text in pages:
        page_starts.append((pos, page_no))
        parts.append(page_text)
        pos += len(page_text) + 1        # +1 是 join 时插入的 "\n"
    return "\n".join(parts), page_starts


def chunk_pdf(pdf_path=None, chunk_size: int = 512, overlap: int = 64) -> List[dict]:
    """从 PDF 建 chunk，返回 [{"text", "page", "source"}]。"""
    from .paths import KB_PDF

    pdf_path = Path(pdf_path or KB_PDF)
    full_text, page_starts = load_pdf_text(pdf_path)
    starts = [s for s, _ in page_starts]

    try:
        source = str(pdf_path.resolve().relative_to(TASK_ROOT))
    except ValueError:
        source = pdf_path.name

    chunks = []
    for start, end in _spans(full_text, chunk_size, overlap):
        text = full_text[start:end].strip()
        if not text:
            continue
        page = page_starts[bisect_right(starts, start) - 1][1]
        chunks.append({"text": text, "page": page, "source": source})
    return chunks


if __name__ == "__main__":
    import argparse
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="PDF 抽取 + 切分预览")
    ap.add_argument("--pdf", default=None)
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=64)
    args = ap.parse_args()

    chunks = chunk_pdf(args.pdf, args.chunk_size, args.overlap)
    lens = [len(c["text"]) for c in chunks]
    print(f"chunk 数：{len(chunks)}")
    if lens:
        print(f"平均长度：{sum(lens) / len(lens):.1f}  最短 {min(lens)}  最长 {max(lens)}")
        print(f"页码范围：p{chunks[0]['page']} - p{chunks[-1]['page']}")
        print("\n--- 示例 ---")
        for c in chunks[:2]:
            print(f"[p{c['page']}] {c['text'][:120]}...")
