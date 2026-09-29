#!/usr/bin/env python
"""同时跑三个自检脚本：src/unit_check.py + eval/run.py（自检）+ 结果汇总。

用法（在 task-2-mini-gpt 目录下）：
    python run_all.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable


def run(script: str, cwd: Path) -> int:
    print(f"\n{'=' * 70}\n>>> python {script}\n{'=' * 70}", flush=True)
    return subprocess.call([PY, script], cwd=str(cwd))


def main() -> None:
    rc1 = run("src/unit_check.py", HERE)
    rc2 = run("eval/run.py", HERE)
    print(f"\n单元自测 exit={rc1}，自检 exit={rc2}")
    print("自检明细见 eval/result.json")

    # eval/run.py 把「有测试失败」也算 exit=0，这里额外读一遍 result.json 报出结论
    import json
    result = json.loads((HERE / "eval" / "result.json").read_text(encoding="utf-8"))
    failed = [r["test"] for r in result if r.get("pass") is False]
    skipped = [r["test"] for r in result if r.get("pass") is None]
    if failed:
        print(f"[注意] 有 {len(failed)} 项未通过：{failed}")
    if skipped:
        print(f"[注意] 有 {len(skipped)} 项被跳过（前置条件未就绪）：{skipped}")
    if not failed and not skipped:
        print("[完成] 全部自检通过 ✅")
    sys.exit(1 if (failed or skipped or rc1) else 0)


if __name__ == "__main__":
    main()
