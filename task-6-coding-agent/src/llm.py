"""模型客户端：只做一件事，chat(messages, stop) -> str。

走 OpenAI 兼容端点（llama-server / Ollama / vLLM 都行），默认连本机 llama-server。
不在这里做 tool-calling 的 JSON 解析——1.5B 级模型吐 JSON 不稳，agent 那层改用
更宽容的 ReAct 文本协议，见 agent.py。
"""
from __future__ import annotations

import logging
import os
import time

# openai/httpx 每发一次请求就在 INFO 级别打一行，会把自检输出刷得没法看
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_MODEL = "qwen2.5-coder-1.5b-instruct"


class ChatModel:
    def __init__(self, base_url: str | None = None, model_name: str | None = None,
                 max_tokens: int = 512, temperature: float = 0.0, retries: int = 2):
        from openai import OpenAI

        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL)
        self.model_name = model_name or os.environ.get("AGENT_MODEL_NAME", DEFAULT_MODEL)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.retries = retries
        self.client = OpenAI(base_url=self.base_url,
                             api_key=os.environ.get("OPENAI_API_KEY", "local"),
                             timeout=600)
        self.total_tokens = 0
        self.calls = 0

    def chat(self, messages: list, stop=None) -> str:
        """一次补全。失败重试，仍失败就抛——让 agent 记进 trace 而不是静默吞掉。"""
        last = None
        for attempt in range(self.retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name, messages=messages,
                    temperature=self.temperature, max_tokens=self.max_tokens,
                    stop=list(stop) if stop else None)
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    self.total_tokens += getattr(usage, "total_tokens", 0) or 0
                self.calls += 1
                return resp.choices[0].message.content or ""
            except Exception as e:                       # 端点抖动/超时
                last = e
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"模型调用失败（{self.retries + 1} 次）：{last}")
