"""DPO：在 SFT 的 LoRA 之上继续做偏好对齐（任务三 M4）。

    python src/train_dpo.py --limit 300 --steps 100

损失（DPO 论文 eq. 7）：

    L = -log σ( β · [ (log π/π_ref)(chosen) - (log π/π_ref)(rejected) ] )

其中 log π(y|x) 是回答部分所有 token 的对数概率之和。reference model
是 SFT 权重的冻结副本，只跑 forward。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

TASK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_ROOT))       # 支持 python src/train_dpo.py 直接跑

from src.chat import encode_messages, get_tokenizer
from src.lora import count_trainable, inject_lora, load_lora_state_dict, lora_state_dict


def sequence_logprob(model, input_ids, labels, attention_mask):
    """回答部分的对数概率之和，返回 (batch,)。labels 里 -100 的位置不计入。"""
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :].float()
    tgt = input_ids[:, 1:]
    logp = -F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="none"
    ).view(tgt.shape)
    mask = (labels[:, 1:] != -100).float()
    return (logp * mask).sum(dim=-1)


def build_pairs(path, tokenizer, args):
    """把 {conversations, chosen, rejected} 转成两两成对的 (ids, labels, mask)。"""
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    pairs, scanned = [], 0
    for row in rows:
        scanned += 1
        prompt = [{"role": "user", "content": c["value"][: args.max_prompt_chars]}
                  for c in row["conversations"] if c.get("from") == "human"]
        if not prompt:
            continue
        try:
            cand = {}
            for key in ("chosen", "rejected"):
                msgs = prompt + [{"role": "assistant",
                                  "content": row[key]["value"][: args.max_chars]}]
                ids, labels = encode_messages(msgs, tokenizer=tokenizer,
                                              max_length=args.max_length)
                if (labels != -100).sum() < 2:      # 回答被截没了就丢弃
                    raise ValueError
                cand[key] = (ids, labels)
        except ValueError:
            continue
        if len(cand["chosen"][0]) < args.max_length and len(cand["rejected"][0]) < args.max_length:
            pairs.append((cand["chosen"], cand["rejected"]))
        if len(pairs) >= args.limit:
            break
    print(f"扫描 {scanned} 条原始偏好对，筛出 {len(pairs)} 条（长度 <= {args.max_length}）")
    return pairs


def pad_batch(items, pad_id):
    width = max(len(it[0]) for it in items)
    ids, labels, attn = [], [], []
    for seq, lab in items:
        pad = width - len(seq)
        ids.append(torch.cat([seq, torch.full((pad,), pad_id, dtype=seq.dtype)]))
        labels.append(torch.cat([lab, torch.full((pad,), -100, dtype=lab.dtype)]))
        attn.append(torch.cat([torch.ones(len(seq), dtype=torch.long),
                               torch.zeros(pad, dtype=torch.long)]))
    return torch.stack(ids), torch.stack(labels), torch.stack(attn)


def load_injected(model_path, sft_ckpt, args):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    inject_lora(model, args.target_modules, r=args.rank, alpha=args.alpha)
    state = torch.load(sft_ckpt, map_location="cpu")
    missing, unexpected = load_lora_state_dict(model, state)
    if unexpected:
        raise RuntimeError(f"SFT 权重没灌进去：{unexpected[:3]}")
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(TASK_ROOT / "models" / "Qwen2.5-0.5B"))
    p.add_argument("--data", default=str(TASK_ROOT / "data" / "dpo" / "dpo_zh.json"))
    p.add_argument("--sft-ckpt", default=str(TASK_ROOT / "ckpt" / "sft" / "lora.pt"))
    p.add_argument("--out", default=str(TASK_ROOT / "ckpt" / "dpo"))
    p.add_argument("--limit", type=int, default=300)
    p.add_argument("--max-prompt-chars", type=int, default=160)
    p.add_argument("--max-chars", type=int, default=240, help="chosen/rejected 字符上限")
    p.add_argument("--max-length", type=int, default=192)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--target-modules", nargs="+", default=["q_proj", "v_proj"])
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    tokenizer = get_tokenizer(args.model)
    pairs = build_pairs(args.data, tokenizer, args)
    if not pairs:
        raise SystemExit("没有筛出任何偏好对，试试调大 --max-length 或调小 --max-prompt-chars")

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    print(f"加载 policy（基座 + SFT LoRA {args.sft_ckpt}）...")
    policy = load_injected(args.model, args.sft_ckpt, args)
    # policy 重新注入时全冻了，这里只放开 LoRA 的 A、B
    for m in policy.modules():
        if hasattr(m, "lora_A"):
            m.lora_A.requires_grad = True
            m.lora_B.requires_grad = True
    policy.config.use_cache = False
    policy.train()

    print("加载 reference（同权重，冻结、只跑 forward）...")
    ref = load_injected(args.model, args.sft_ckpt, args)
    ref.eval()
    for q in ref.parameters():
        q.requires_grad = False

    print(f"可训参数：{count_trainable(policy):,}")
    opt = torch.optim.AdamW([q for q in policy.parameters() if q.requires_grad], lr=args.lr)

    history, step, t0 = [], 0, time.time()
    done = False
    while not done:
        for i in range(0, len(pairs), 1):
            chosen, rejected = pairs[i]

            # 一步总共 4 次 forward：policy ×2 参与反向，ref ×2 只做参考
            pi_w = sequence_logprob(policy, *pad_batch([chosen], pad_id))
            pi_l = sequence_logprob(policy, *pad_batch([rejected], pad_id))
            with torch.no_grad():                      # ref 冻结，别让它进计算图
                ref_w = sequence_logprob(ref, *pad_batch([chosen], pad_id))
                ref_l = sequence_logprob(ref, *pad_batch([rejected], pad_id))

            # 方括号里就是「模型相对参考模型更喜欢好回答多少」，
            # 再乘 β 缩放、过 sigmoid 变成概率，取负对数当损失
            margin = args.beta * ((pi_w - ref_w) - (pi_l - ref_l))
            loss = -F.logsigmoid(margin).mean()
            # 自检小技巧：第一步 policy 和 ref 完全相同 -> margin=0 -> loss 应恰好是 ln2 ≈ 0.6931

            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            history.append({"loss": round(loss.item(), 4),
                            "margin": round(margin.mean().item(), 4)})

            if step % 10 == 0 or step == 1:
                el = time.time() - t0
                eta = el / step * (args.steps - step)
                print(f"  step {step:4d}/{args.steps}  loss {loss.item():.4f}  "
                      f"reward margin {margin.mean().item():+.4f}  "
                      f"已用 {el / 60:.1f}min  预计还需 {eta / 60:.1f}min", flush=True)
            if step >= args.steps:
                done = True
                break

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(lora_state_dict(policy), out_dir / "lora.pt")
    (out_dir / "train_args.json").write_text(
        json.dumps({"args": vars(args), "history": history}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\n已保存 LoRA 权重到 {out_dir}/lora.pt")
    print(f"reward margin：{history[0]['margin']:+.4f} -> {history[-1]['margin']:+.4f}")


if __name__ == "__main__":
    main()
