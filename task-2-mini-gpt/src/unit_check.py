"""单元自测：在**随机初始化**的模型上验证 RoPE / KV cache / 采样这些容易写错的地方。

好处：不依赖训练好的 ckpt，几秒钟就能发现问题（训练前就该跑一遍）。
    python src/unit_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import MiniGPT                       # noqa: E402
from src.sampling import (apply_top_k, apply_top_p,  # noqa: E402
                          sample_next_token)

torch.manual_seed(0)
DEV = "cpu"
ok = True


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok
    ok = ok and bool(cond)
    print(f"{'[通过]' if cond else '[失败]'} {name} {detail}")


def build() -> MiniGPT:
    m = MiniGPT(vocab_size=128, d_model=64, n_layers=2, n_heads=4,
                max_seq_len=64, dropout=0.0).to(DEV)
    m.eval()
    return m


# ---------------------------------------------------------------------------
# 1) KV cache 等价性：逐词元增量前向 与 整段一次前向 的 logits 必须一致
# ---------------------------------------------------------------------------
def test_kv_cache():
    m = build()
    ids = torch.tensor([[1, 5, 9, 20, 33, 7, 100, 4]], device=DEV)
    with torch.no_grad():
        full = m(ids)                                    # (1, T, V)
        cache, pieces = None, []
        for i in range(ids.size(1)):
            # 注意：这里把整个前缀都重新喂？不是——只喂「最后一个新词元」，
            # 历史信息全部由 cache 提供，这才是真实的增量解码
            out, cache = m(ids[:, i:i + 1], kv_cache=cache, return_cache=True)
            pieces.append(out)
        inc = torch.cat(pieces, dim=1)
    diff = (full - inc).abs().max().item()
    check("kv_cache_equivalence", diff < 1e-4, f"max|diff|={diff:.3e}")

    # 半路开 cache 也要对（例如先喂 prompt 再增量）
    with torch.no_grad():
        prompt, rest = ids[:, :4], ids[:, 4:]
        out_p, cache = m(prompt, return_cache=True)
        logits = [out_p]
        for i in range(rest.size(1)):
            out_i, cache = m(rest[:, i:i + 1], kv_cache=cache, return_cache=True)
            logits.append(out_i)
        inc2 = torch.cat(logits, dim=1)
    diff2 = (full - inc2).abs().max().item()
    check("kv_cache_prompt_then_incremental", diff2 < 1e-4, f"max|diff|={diff2:.3e}")


# ---------------------------------------------------------------------------
# 2) causal 性：改动位置 t 之后的输入，不应该影响位置 <= t 的输出
# ---------------------------------------------------------------------------
def test_causal():
    m = build()
    a = torch.tensor([[1, 2, 3, 4, 5, 6]], device=DEV)
    b = a.clone()
    b[0, 3:] = torch.tensor([77, 88, 99], device=DEV)     # 只改后半段
    with torch.no_grad():
        la, lb = m(a), m(b)
    d_front = (la[:, :3] - lb[:, :3]).abs().max().item()
    d_back = (la[:, 3:] - lb[:, 3:]).abs().max().item()
    check("causal_mask_不泄漏未来", d_front < 1e-5 and d_back > 1e-3,
          f"前段差={d_front:.2e}（应≈0）后段差={d_back:.2e}（应>0）")


# ---------------------------------------------------------------------------
# 3) RoPE：内积只依赖相对位置
# ---------------------------------------------------------------------------
def test_rope_relative():
    from src.rope import RoPE
    rope = RoPE(head_dim=8, max_seq_len=32)
    q = torch.randn(1, 1, 1, 8)
    k = torch.randn(1, 1, 1, 8)
    def dot_at(m, n):
        qm = rope(q, position_offset=m)
        kn = rope(k, position_offset=n)
        return (qm * kn).sum().item()
    d1 = abs(dot_at(0, 3) - dot_at(10, 13))      # 相对距离都是 3
    d2 = abs(dot_at(0, 3) - dot_at(0, 5))        # 相对距离 3 vs 5
    check("RoPE_内积只依赖相对位置", d1 < 1e-5 and d2 > 1e-4,
          f"同距离差={d1:.2e}（应≈0）不同距离差={d2:.2e}（应>0）")


# ---------------------------------------------------------------------------
# 4) 采样：greedy 确定性、temperature=0 不崩、top-k/top-p 的形状与范围
# ---------------------------------------------------------------------------
def test_sampling():
    torch.manual_seed(1)
    logits = torch.randn(2, 50)
    g1 = sample_next_token(logits, greedy=True)
    g2 = sample_next_token(logits, temperature=0.0)
    check("temperature=0 等价 greedy", torch.equal(g1, g2))
    check("greedy 取 argmax", torch.equal(g1.squeeze(-1), logits.argmax(-1)))

    # top-k=1 时只剩最大值，采样结果必须等于 argmax
    s = sample_next_token(logits, temperature=1.0, top_k=1)
    check("top_k=1 等价 argmax", torch.equal(s.squeeze(-1), logits.argmax(-1)))

    # top-k / top-p 过滤后的候选集合检查
    k_f = apply_top_k(logits, 5)
    kept = (~torch.isinf(k_f)).sum(-1)
    check("top_k 保留数量正确", bool((kept == 5).all()), f"kept={kept.tolist()}")
    p_f = apply_top_p(logits, 0.9)
    cnt = (~torch.isinf(p_f)).sum(-1)
    check("top_p 保留集合非空且小于全词表", bool((cnt >= 1).all() and (cnt < 50).all()),
          f"kept={cnt.tolist()}")

    # 生成：四种策略都不应报错，且 greedy 两次结果一致
    m = build()
    p = [1, 2, 3]
    a = m.generate(p, max_new_tokens=12, greedy=True)
    b = m.generate(p, max_new_tokens=12, greedy=True)
    check("greedy 生成可复现", a == b)
    for kw in (dict(temperature=0.8, top_k=10), dict(temperature=1.0, top_p=0.9),
               dict(temperature=1.3)):
        out = m.generate(p, max_new_tokens=8, **kw)
        assert len(out) == 11, kw
    check("top-k / top-p / temperature 生成不报错", True)

    # KV cache 与不用 cache 的生成结果在 greedy 下应完全一致
    c1 = m.generate(p, max_new_tokens=12, greedy=True, use_cache=True)
    c2 = m.generate(p, max_new_tokens=12, greedy=True, use_cache=False)
    check("生成时 cache 开/关结果一致", c1 == c2)


# ---------------------------------------------------------------------------
# 5) 分词器：encode->decode 还原 + 无 <unk>
# ---------------------------------------------------------------------------
def test_tokenizer():
    from src.tokenizer import BPETokenizer
    tk = BPETokenizer.train_bpe("床前明月光，疑是地上霜。" * 30, vocab_size=300, verbose=False)
    for s in ["床前明月光", "Hello, world!", "深度学习需要数学基础", "123 45.6%"]:
        got = tk.decode(tk.encode(s))
        check(f"roundtrip: {s!r}", got == s, f"-> {got!r}")
    check("词表大小 >= 259", tk.vocab_size >= 259, f"vocab_size={tk.vocab_size}")


if __name__ == "__main__":
    test_tokenizer()
    test_rope_relative()
    test_kv_cache()
    test_causal()
    test_sampling()
    print("\n全部自测通过 ✅" if ok else "\n有自测失败 ❌")
    sys.exit(0 if ok else 1)
