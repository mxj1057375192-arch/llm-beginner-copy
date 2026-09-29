"""调模型的薄封装：默认进程内跑本地 Qwen（transformers），设了 OPENAI_BASE_URL 就走 HTTP。

README 建议用 Ollama / vLLM 起一个 OpenAI 兼容服务再调，但本机没有 GPU，0.5B 走
transformers 进程内加载最省事（也省掉装服务、拉模型）；两条路都留着，换成 7B 或者
远端 endpoint 时不用改 agent 代码：设 OPENAI_BASE_URL / AGENT_MODEL_NAME 即可。
"""
from __future__ import annotations

import os

# 512 而不是 256：python_sandbox 的 code 参数是一整段代码，JSON 里还带 \n 转义，
# 256 个 token 会在 JSON 中途截断，模型得到的报错是"语法错误"，它根本不知道自己被截了。
DEFAULT_MAX_NEW_TOKENS = 512


class ChatModel:
    """只做一件事：chat(messages, stop) -> str。"""

    def __init__(self, model_dir=None, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS):
        self.max_new_tokens = max_new_tokens
        self.base_url = os.environ.get("OPENAI_BASE_URL")
        self.client = self.tokenizer = self.model = None

        if self.base_url:
            from openai import OpenAI            # 可选依赖：只有走 HTTP 时才需要
            self.client = OpenAI(base_url=self.base_url,
                                 api_key=os.environ.get("OPENAI_API_KEY", "ollama"))
            self.model_name = os.environ.get("AGENT_MODEL_NAME", "qwen2.5:7b-instruct")
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from .paths import resolve_model_dir

        self.model_dir = str(model_dir or resolve_model_dir())
        torch.set_num_threads(os.cpu_count() or 4)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_dir).eval()

    def chat(self, messages: list, stop=None) -> str:
        if self.client is not None:
            resp = self.client.chat.completions.create(
                model=self.model_name, messages=messages, temperature=0,
                max_tokens=self.max_new_tokens, stop=list(stop) if stop else None)
            return resp.choices[0].message.content or ""
        return self._local_chat(messages, list(stop or []))

    def _local_chat(self, messages: list, stop: list) -> str:
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").input_ids
        prompt_len = inputs.shape[1]
        tokenizer = self.tokenizer

        class _StopOnText(StoppingCriteria):
            """生成到停止串就提前收工。

            不是为了省时间，是为了正确性：0.5B 会顺着 few-shot 的格式自己往下编
            Observation（实测见过它直接写 "Observation: 2500"），不掐掉的话 agent
            自问自答，真工具一次都不会被调用。
            """

            def __call__(self, input_ids, scores, **kwargs):
                text = tokenizer.decode(input_ids[0][prompt_len:],
                                        skip_special_tokens=True)
                return any(marker in text for marker in stop)

        with torch.no_grad():
            output = self.model.generate(
                inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList([_StopOnText()]))
        text = self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
        for marker in stop:                       # 停止串可能只生成了前半截
            index = text.find(marker)
            if index != -1:
                text = text[:index]
        return text.strip()
