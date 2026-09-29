"""任务三实现：手写 LoRA + Qwen chat template / loss masking。

阅读顺序建议（配套讲解见 ../docs/walkthrough.md）：

    1. lora.py        低秩注入与合并          —— 任务的 M1
    2. chat.py        chat template + loss mask —— 任务的 M2
    3. data.py        MOSS 数据解析
    4. train_sft.py   SFT 训练循环             —— 任务的 M3
    5. train_dpo.py   DPO 训练循环             —— 任务的 M4
    6. compare.py     base / SFT / DPO 对比
"""
