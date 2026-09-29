"""跑一道题，把 ReAct 循环每一步的原文打出来，方便观察和改代码。

用法（在 task-5-tool-agent 目录下执行）：

    python src/demo.py                    # 默认跑一道算术题
    python src/demo.py --task 9           # 跑 data/tasks.json 里的第 9 题，并显示期望关键词
    python src/demo.py "地球为什么是圆的"   # 跑你自己出的问题

判分请跑 `python eval/run.py`：这个脚本只是把过程打印出来，不打分。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_ROOT))

from src.agent import ReActAgent      # noqa: E402
from src.paths import TASKS_JSON      # noqa: E402

DEFAULT_QUESTION = "计算 (123 + 456) * 789 的结果，并告诉我结果的位数。"


def pick_question(argv: list):
    """命令行参数 → (问题, 期望关键词)。"""
    if len(argv) >= 2 and argv[0] == "--task":
        tasks = json.loads(TASKS_JSON.read_text(encoding="utf-8"))
        item = next((t for t in tasks if str(t["id"]) == str(argv[1])), None)
        if item is None:
            raise SystemExit(f"data/tasks.json 里没有第 {argv[1]} 题（共 {len(tasks)} 题）")
        return item["task"], item.get("expected_answer_contains")
    if argv:
        return " ".join(argv), None
    return DEFAULT_QUESTION, None


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):     # Windows 控制台默认 GBK，会乱码
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    question, expected = pick_question(sys.argv[1:])

    agent = ReActAgent()
    model = getattr(agent.llm, "model_dir", None) or agent.llm.base_url
    print(f"模型：{model}")
    print(f"问题：{question}")
    print("=" * 64)

    trace = agent.run(question)

    for step in trace["steps"]:
        print(f"\n───── 第 {step['step']} 步 ─────")
        print(step["raw"].strip())                 # 模型这一步的原话
        if step["observation"] is not None:        # 工具返回（或错误提示）
            print(f"\nObservation: {step['observation']}")

    print("\n" + "=" * 64)
    print(f"Final Answer：{trace['final_answer']}")
    print(f"步数：{trace['n_steps']}；agent 自报 success={trace['success']}"
          f"（自检不认这个字段，只按 Final Answer 的关键词判分）")
    if expected:
        print(f"这一题的期望关键词：{expected}")


if __name__ == "__main__":
    main()
