"""本地文件检索：既按文件名找，也按内容找，并回带内容片段。

两种匹配都得做，因为评测集里两种问法都有：第 3 题要「数出所有 .md 文件」（按名），
第 7 题要「找包含 TODO 的文件」（按内容），第 10 题要「README.md 第一段写了什么」
（只回路径没用，得真把内容带回来）。pattern 先按文件名匹配，没中再当内容来搜。

路径安全：dir 会 resolve 后校验必须落在 FILE_SEARCH_ROOTS 内，挡掉 ../../ 越界；
相对路径优先按任务目录解析（而不是进程 CWD），这样从仓库根跑自检也能找到文件。
"""
from __future__ import annotations

import fnmatch
import re
from pathlib import Path

from ..paths import FILE_SEARCH_ROOTS, TASK_ROOT

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "file_search",
        "description": "在指定目录下检索文件：先按文件名匹配 pattern，没命中再按文件内容搜索，"
                       "并返回匹配文件的路径和内容片段。",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": '文件名关键字或通配符（如 ".md"、"*.md"、"README.md"），'
                                   "也可以是要在文件内容里搜索的关键字（如 TODO）",
                },
                "dir": {
                    "type": "string",
                    "description": '检索目录，相对路径按任务目录解析，例如 "data/agent-fixtures"',
                },
            },
            "required": ["pattern", "dir"],
        },
    },
}

_TEXT_SUFFIXES = {".md", ".txt", ".py", ".json", ".jsonl", ".csv", ".tsv", ".yaml",
                  ".yml", ".toml", ".ini", ".cfg", ".log", ".rst", ".html", ".sh"}
_MAX_FILE_BYTES = 1_000_000
_MAX_RESULTS = 30
_SNIPPET_CHARS = 200


def _allowed_roots():
    return [path.resolve() for path in FILE_SEARCH_ROOTS]


def _resolve_target(dir_arg: str) -> Path:
    raw = Path(dir_arg)
    candidates = [raw] if raw.is_absolute() else [TASK_ROOT / raw, Path.cwd() / raw]
    for candidate in candidates:
        if candidate.exists():
            target = candidate.resolve()
            break
    else:
        raise ValueError(f"目录或文件不存在：{dir_arg}")

    for root in _allowed_roots():
        if target == root or root in target.parents:
            return target
    raise ValueError(f"拒绝访问 {dir_arg}：超出允许的检索根目录（{_allowed_roots()}）")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _is_text(path: Path) -> bool:
    return path.suffix.lower() in _TEXT_SUFFIXES


def _name_matches(name: str, pattern: str) -> bool:
    lowered = name.lower()
    needle = pattern.lower()
    return needle in lowered or fnmatch.fnmatch(lowered, needle)


def _first_paragraph(path: Path) -> str:
    """取首个非空行当片段——评测集里「第一段写了什么」这类问题靠它回答。"""
    try:
        for line in _read_text(path).splitlines():
            if line.strip():
                return line.strip()[:_SNIPPET_CHARS]
    except OSError:
        pass
    return ""


def _search_content(path: Path, pattern: str):
    """在文件内容里找 pattern，返回 (是否命中, 命中行片段)。"""
    try:
        text = _read_text(path)
    except OSError:
        return False, ""
    try:
        match = re.search(pattern, text, re.IGNORECASE)
    except re.error:
        match = None
    if match is None:
        index = text.lower().find(pattern.lower())
        if index < 0:
            return False, ""
        start = max(0, index - _SNIPPET_CHARS // 2)
        return True, text[start:start + _SNIPPET_CHARS].replace("\n", " ").strip()

    line_start = text.rfind("\n", 0, match.start()) + 1
    line_end = text.find("\n", match.end())
    line = text[line_start:line_end if line_end != -1 else len(text)]
    return True, line.strip()[:_SNIPPET_CHARS]


def run(args: dict) -> str:
    raw_dir = str(args.get("dir") or "").strip()
    if raw_dir:
        target = _resolve_target(raw_dir)
    else:
        # 模型偶尔会写成 Action Input: {}（把文件名当成"不用参数"）。这时别直接
        # 报错把它顶回去，默认在任务目录下检索，至少让它看到真实的文件列表。
        target = TASK_ROOT.resolve()

    # dir 直接指到某个文件时（模型常这么写，尤其在「读某文件第一段」的题上），
    # 不必再要 pattern，直接把内容回给它。
    if target.is_file():
        text = _read_text(target)
        try:
            shown = target.relative_to(TASK_ROOT)
        except ValueError:
            shown = target
        return (f"{shown} 的内容（共 {len(text)} 字符）：\n"
                f"第一段：{_first_paragraph(target)}\n"
                f"开头 300 字符：{text[:300].strip()}")

    pattern = str(args.get("pattern") or "").strip() or "*"

    hits = []
    for path in sorted(target.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        if _name_matches(path.name, pattern):
            hits.append((path, _first_paragraph(path), "文件名"))
        elif _is_text(path) and path.stat().st_size <= _MAX_FILE_BYTES:
            matched, snippet = _search_content(path, pattern)
            if matched:
                hits.append((path, snippet, "内容"))
        if len(hits) >= _MAX_RESULTS:
            break

    try:
        shown_target = target.relative_to(TASK_ROOT)
    except ValueError:
        shown_target = target
    if str(shown_target) == ".":
        shown_target = "任务目录"
    if not hits:
        return f"在 {shown_target} 下没有找到匹配「{pattern}」的文件。"

    lines = [f"在 {shown_target} 下按「{pattern}」找到 {len(hits)} 个文件（按文件名或内容匹配）："]
    for path, snippet, kind in hits:
        try:
            shown = path.relative_to(TASK_ROOT)
        except ValueError:
            shown = path
        lines.append(f"- {shown}（{path.stat().st_size} 字节，{kind}命中）")
        if snippet:
            lines.append(f"  片段：{snippet}")
    lines.append(f"共 {len(hits)} 个匹配文件。")
    return "\n".join(lines)


if __name__ == "__main__":
    print(run({"pattern": "README.md", "dir": str(TASK_ROOT)}))
    print()
    print(run({"pattern": "TODO", "dir": "data/agent-fixtures"}))
    print()
    print(run({"pattern": "*.md", "dir": "data/agent-fixtures"}))
