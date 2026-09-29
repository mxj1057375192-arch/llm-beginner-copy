"""任务二：从零实现 mini-GPT 的源码包。

模块划分：
    tokenizer.py  —— 手写 byte-level BPE 分词器（M1）
    rope.py       —— 旋转位置编码 RoPE（M2）
    attention.py  —— causal 多头自注意力 + KV cache（M2/M3）
    block.py      —— RMSNorm / SwiGLU / Transformer Block
    model.py      —— MiniGPT 本体 + generate + load_for_eval
    sampling.py   —— greedy / top-k / top-p / temperature 采样（M5）
"""
