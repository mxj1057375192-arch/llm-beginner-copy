"""MOSS-003-sft 数据读取与转换。

MOSS 的一条样本长这样：

    {"conversation_id": 1, "num_turns": 5, "category": "Brainstorming",
     "chat": {"turn_1": {"Human": "<|Human|>: ...<eoh>\\n",
                         "Inner Thoughts": "<|Inner Thoughts|>: None<eot>\\n",
                         "Commands": "<|Commands|>: None<eoc>\\n",
                         "Tool Responses": "<|Results|>: None<eor>\\n",
                         "MOSS": "<|MOSS|>: ...<eom>\\n"}, ...}}

对话普遍有 4-8 轮，直接整段喂进去序列会很长；这里只取前 `max_turns` 轮，
把每条样本压成短的多轮对话，CPU 上才跑得动。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_HUMAN_RE = re.compile(r"^\s*<\|Human\|>:\s*")
_MOSS_RE = re.compile(r"^\s*<\|MOSS\|>:\s*")


def _strip(text: str, head_re: re.Pattern, end_tag: str) -> str:
    text = head_re.sub("", text, count=1).rstrip()
    while text.endswith(end_tag):
        text = text[: -len(end_tag)].rstrip()
    return text.strip()


def clean_human(text: str) -> str:
    """去掉 MOSS 的 `<|Human|>: ` 前缀与 `<eoh>` 结尾。"""
    return _strip(text, _HUMAN_RE, "<eoh>")


def clean_moss(text: str) -> str:
    """去掉 MOSS 的 `<|MOSS|>: ` 前缀与 `<eom>` 结尾。"""
    return _strip(text, _MOSS_RE, "<eom>")


def row_to_messages(row: dict, max_turns: int = 2, max_chars: int = 300):
    """把一条 MOSS 样本转成 [{"role","content"}]；超长内容按字符截断。"""
    messages = []
    for turn_key in sorted(row.get("chat", {}), key=lambda k: int(k.split("_")[-1])):
        if len(messages) // 2 >= max_turns:
            break
        turn = row["chat"][turn_key]
        human = clean_human(turn.get("Human", ""))
        moss = clean_moss(turn.get("MOSS", ""))
        if not human or not moss:
            continue
        messages.append({"role": "user", "content": human[:max_chars]})
        messages.append({"role": "assistant", "content": moss[:max_chars]})
    return messages


def load_moss(path, max_turns: int = 2, max_chars: int = 300, limit: int | None = None):
    """逐行读 MOSS jsonl，产出 messages 列表。"""
    path = Path(path)
    n = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            messages = row_to_messages(json.loads(line), max_turns, max_chars)
            if len(messages) < 2:
                continue
            yield messages
            n += 1
            if limit is not None and n >= limit:
                return


if __name__ == "__main__":
    import sys

    sample = Path(__file__).resolve().parents[1] / "data" / "moss-sft" / "sample.jsonl"
    for i, msgs in enumerate(load_moss(sample, limit=2)):
        print(f"--- 样本 {i}：{len(msgs)} 条消息 ---")
        for m in msgs:
            print(f"  [{m['role']}] {m['content'][:80]}")
