"""手写 LoRA：低秩矩阵注入与合并（任务三 M1）。

LoRA 把微调时的权重更新约束成低秩分解（论文 eq. 3）：

    h = W0 x + ΔW x = W0 x + (alpha / r) * B (A x)

- A: (in_features, r)，kaiming 初始化
- B: (r, out_features)，**零**初始化 —— 保证注入瞬间 ΔW = 0，
  模型行为与注入前逐位一致，不会一上来就把基座的输出带偏
- W0 冻结，反向只更新 A、B
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """把 nn.Linear 包成「冻结原权重 + 低秩旁路」的形式。"""

    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError(f"r 必须为正整数，收到 {r}")
        self.base = base
        self.r = r
        self.alpha = alpha
        # 缩放必须写成 alpha / r：换 rank 时等效学习率才不会跟着漂
        self.scaling = alpha / r

        in_features, out_features = base.in_features, base.out_features
        self.lora_A = nn.Parameter(torch.empty(in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 两条路：原权重那条不动，低秩旁路那条才是训练时要调的部分
        out = self.base(x)                                    # W0 · x
        h = self.lora_dropout(x)
        # 先降到 r 维、再升回去：(x @ A) 形状 (..., r)，再 @ B 回到 (..., out)
        # 中间那一层只有 r 个数，这就是"参数量小"的来源
        delta = (h @ self.lora_A.to(h.dtype)) @ self.lora_B.to(h.dtype)
        return out + (self.scaling * delta).to(out.dtype)     # + (alpha/r) · B·A·x

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        """W0 + scaling * B @ A，形状与 base.weight 一致（out, in）。"""
        delta = self.lora_B.T @ self.lora_A.T
        return self.base.weight + self.scaling * delta.to(self.base.weight.dtype)

    @torch.no_grad()
    def to_linear(self) -> nn.Linear:
        """合并成普通 nn.Linear，LoRA 分支就此摘掉。"""
        linear = nn.Linear(
            self.base.in_features,
            self.base.out_features,
            bias=self.base.bias is not None,
            device=self.base.weight.device,
            dtype=self.base.weight.dtype,
        )
        linear.weight.copy_(self.merged_weight())
        if self.base.bias is not None:
            linear.bias.copy_(self.base.bias)
        for p in linear.parameters():
            p.requires_grad = False
        return linear


def inject_lora(
    model: nn.Module,
    target_modules,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
) -> nn.Module:
    """在目标线性层旁挂低秩分支，原地修改并返回同一个 model。

    注入后**除 A、B 外的全部参数都会被冻结**——这是 `lora_param_count`
    自检（可训占比 < 5%）通过的前提：不冻结的话，k_proj / o_proj /
    gate_proj / up_proj / down_proj 加起来有 3 亿多参数，远超 5%。
    """
    targets = set(target_modules)
    replaced: list[str] = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if name.rsplit(".", 1)[-1] not in targets:
            continue
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr, LoRALinear(module, r, alpha, dropout))
        replaced.append(name)

    if not replaced:
        raise ValueError(f"没有匹配到任何目标层：{sorted(targets)}")

    # 先全部冻结，再只放开 LoRA 的 A、B
    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.lora_A.requires_grad = True
            m.lora_B.requires_grad = True

    return model


@torch.no_grad()
def merge_lora(model: nn.Module) -> nn.Module:
    """把 scaling * B @ A 合并回原权重并摘掉 LoRA 分支，原地修改并返回 model。"""
    for name, module in list(model.named_modules()):
        if not isinstance(module, LoRALinear):
            continue
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr, module.to_linear())
    return model


def lora_state_dict(model: nn.Module) -> dict:
    """只取 LoRA 的 A、B 权重，用于存 ckpt/sft、ckpt/dpo。"""
    return {
        k: v.detach().clone()
        for k, v in model.state_dict().items()
        if ".lora_A" in k or ".lora_B" in k
    }


def load_lora_state_dict(model: nn.Module, state: dict):
    """把 LoRA 权重灌回已注入的模型；返回 (missing, unexpected)。"""
    return model.load_state_dict(state, strict=False)


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
