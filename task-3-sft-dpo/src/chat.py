"""Qwen chat template 与 loss masking（任务三 M2）。

两个约定：
- `format_messages` 手工套 Qwen 官方模板（`<|im_start|>` / `<|im_end|>`），
  首条不是 system 时插入 Qwen 的默认 system prompt —— 与官方 chat_template 一致。
- `build_labels` 只对 assistant 的**回答内容**给标签，user / system /
  模板控制符（`<|im_start|>assistant\\n`、`<|im_end|>`）全部 -100。
"""
from __future__ import annotations

from pathlib import Path

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
# 取自模型自带 tokenizer_config.json 的 chat_template 默认值
DEFAULT_SYSTEM = "You are a helpful assistant."

_TOKENIZERS: dict[str, object] = {}


def task_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_tokenizer(model_path=None):
    """默认取自检脚本用的固定路径 models/Qwen2.5-0.5B，带缓存。"""
    from transformers import AutoTokenizer

    path = str(model_path or task_root() / "models" / "Qwen2.5-0.5B")
    if path not in _TOKENIZERS:
        _TOKENIZERS[path] = AutoTokenizer.from_pretrained(path)
    return _TOKENIZERS[path]


def _render(messages, add_generation_prompt: bool = False):
    """渲染整段文本，同时返回 assistant 回答内容在文本里的字符区间。"""
    msgs = [dict(m) for m in messages]
    if not msgs or msgs[0].get("role") != "system":
        msgs = [{"role": "system", "content": DEFAULT_SYSTEM}] + msgs

    chunks: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for m in msgs:
        head = f"{IM_START}{m['role']}\n"
        body = m["content"]
        tail = f"{IM_END}\n"
        chunks.append(head)
        cursor += len(head)
        if m["role"] == "assistant":
            spans.append((cursor, cursor + len(body)))
        chunks.append(body)
        cursor += len(body)
        chunks.append(tail)
        cursor += len(tail)
    if add_generation_prompt:
        chunks.append(f"{IM_START}assistant\n")
    return "".join(chunks), spans


def format_messages(messages, add_generation_prompt: bool = False) -> str:
    """把 [{"role": ..., "content": ...}] 套成 Qwen 官方模板字符串。"""
    return _render(messages, add_generation_prompt)[0]


def build_labels(input_ids, messages, tokenizer=None, add_generation_prompt: bool = False):
    """构造与 input_ids 同形状的 labels：assistant 内容保留 token id，其余 -100。"""
    import torch

    if not torch.is_tensor(input_ids):
        input_ids = torch.tensor(input_ids)

    tok = tokenizer or get_tokenizer()
    text, spans = _render(messages, add_generation_prompt)

    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    if len(ids) != input_ids.numel():
        raise ValueError(
            f"重新分词得到 {len(ids)} 个 token，与传入的 input_ids（{input_ids.numel()} 个）不一致；"
            "请确认两边用的是同一份 format_messages 文本与同样的分词参数。"
        )

    # 先全部填 -100（= 不参与 loss），再把 assistant 回答部分改回真实 token id
    labels = torch.full_like(input_ids, -100)
    for i, (start, end) in enumerate(offsets):
        # 判定「这个 token 的字符区间 [start, end) 和某段回答的区间 [lo, hi) 有交集」。
        # 用交集而不是严格包含，是因为 BPE 的 token 边界未必和内容边界严格对齐。
        # 模板控制符（<|im_start|>assistant\n、<|im_end|>）落在回答区间之外，自然被排除。
        if any(start < hi and end > lo for lo, hi in spans):
            labels[i] = input_ids[i]
    return labels


def encode_messages(messages, tokenizer=None, max_length: int | None = None,
                    add_generation_prompt: bool = False):
    """训练用：一次拿到 input_ids 与对齐的 labels，可选截断。"""
    import torch

    tok = tokenizer or get_tokenizer()
    text, _ = _render(messages, add_generation_prompt)
    ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    labels = build_labels(ids, messages, tokenizer=tok,
                          add_generation_prompt=add_generation_prompt)
    if max_length is not None:
        ids, labels = ids[:max_length], labels[:max_length]
    return ids, labels
