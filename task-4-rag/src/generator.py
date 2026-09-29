"""把检索结果拼成 prompt，调本地 Qwen 生成答案。

prompt 的硬要求：明确「只能用提供的上下文回答，不知道就说不知道」。否则模型会拿
预训练知识硬答——答案读起来很顺，实际上检索早就失败了，流畅的答案会掩盖召回问题。

生成用贪心解码（do_sample=False）：这个任务要的是可复现，不是文采。
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .paths import GEN_MODEL

DEFAULT_SYSTEM = "你是一个严谨的中文问答助手，只依据给出的资料回答问题。"

PROMPT_TEMPLATE = """请只依据下面【资料】回答问题。

【资料】
{context}

【问题】
{question}

【回答要求】
1. 只使用【资料】里出现的信息，不要用你自己的知识补充；
2. 如果【资料】不足以回答，直接说「根据提供的资料无法回答」，不要猜；
3. 回答尽量简洁，并在结尾用 [编号] 标出所依据的资料。"""


def build_prompt(question: str, contexts: List[dict]) -> str:
    """把召回片段拼成带编号和来源的上下文块。"""
    blocks = [f"[{i}]（来源：{c.get('source', '未知')}）\n{c['text']}"
              for i, c in enumerate(contexts, 1)]
    return PROMPT_TEMPLATE.format(context="\n\n".join(blocks), question=question)


class Generator:
    """本地 Qwen 生成器（默认 Qwen2.5-0.5B-Instruct，换成 7B 只需改 model_dir）。"""

    def __init__(self, model_dir=None, device: str = "cpu", max_new_tokens: int = 256):
        self.model_dir = str(model_dir or GEN_MODEL)
        if not Path(self.model_dir).exists():
            raise FileNotFoundError(
                f"找不到生成模型 {self.model_dir}；先下载 Qwen2.5-Instruct，"
                f"或用 RAG_GEN_MODEL 指定已有路径")
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_dir).to(device).eval()

    def _apply_chat_template(self, messages: List[dict]) -> str:
        try:
            return self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                      add_generation_prompt=True)
        except Exception:
            # 分词器没带 chat template 时退回 Qwen 的 ChatML 手工格式
            parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages]
            return "".join(parts) + "<|im_start|>assistant\n"

    @torch.no_grad()
    def generate(self, prompt: str, system: str = None, max_new_tokens: int = None) -> str:
        messages = [{"role": "system", "content": system or DEFAULT_SYSTEM},
                    {"role": "user", "content": prompt}]
        enc = self.tokenizer(self._apply_chat_template(messages),
                             return_tensors="pt").to(self.device)
        out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            do_sample=False,                      # 贪心：同一问题答案稳定，便于复现
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        new_tokens = out[0][enc["input_ids"].shape[1]:]      # 只解码新生成的部分
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


_DEFAULT: Generator = None


def get_generator(**kwargs) -> Generator:
    """进程内复用的默认 Generator。模型路径可用环境变量 RAG_GEN_MODEL 覆盖。"""
    global _DEFAULT
    if _DEFAULT is None:
        import os
        kwargs.setdefault("model_dir", os.environ.get("RAG_GEN_MODEL"))
        _DEFAULT = Generator(**kwargs)
    return _DEFAULT
