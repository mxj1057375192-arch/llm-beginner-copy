"""任务六 M2：Skill 加载器（约 50 行），实现渐进式披露。

关键点：`list_skills()` 只读 front-matter 的元数据，**不读正文**；只有 `match()`
命中之后才用 `load()` 把完整正文取出来塞进 context。一上来就把所有 SKILL.md 正文
读进来，就不叫渐进式披露了，也白白撑爆 7B 模型的上下文。
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

_FENCE = "---"
_TOKEN_RE = re.compile(r"[a-z0-9_]+|[一-鿿]")


def _split_front_matter(text: str) -> tuple[dict, str]:
    """拆出 YAML front-matter 与正文。没有 front-matter 时元数据为空 dict。"""
    if not text.startswith(_FENCE):
        return {}, text
    parts = text.split(_FENCE, 2)
    if len(parts) < 3:
        return {}, text
    meta = yaml.safe_load(parts[1])
    return (meta if isinstance(meta, dict) else {}), parts[2].strip()


def _tokens(text: str) -> set:
    return set(_TOKEN_RE.findall(text.lower()))


class SkillLoader:
    """扫描 skills_dir/*/SKILL.md，按 description 匹配，命中后才加载正文。"""

    def __init__(self, skills_dir: str):
        self.skills_dir = Path(skills_dir)
        self._by_name: dict[str, Path] = {}

    def list_skills(self) -> list[dict]:
        """只返回元数据（name / description / path），不读正文。"""
        found = []
        if not self.skills_dir.is_dir():
            return found
        for path in sorted(self.skills_dir.glob("*/SKILL.md")):
            meta, _ = _split_front_matter(path.read_text(encoding="utf-8", errors="replace"))
            name = meta.get("name") or path.parent.name
            found.append({"name": name,
                          "description": meta.get("description") or "",
                          "path": str(path)})
            self._by_name[name] = path
        return found

    def load(self, name: str) -> str:
        """取出某个 Skill 的完整正文（front-matter 之后的部分）。"""
        path = self._by_name.get(name)
        if path is None:
            for meta in self.list_skills():          # 缓存没建时补扫一次
                if meta["name"] == name:
                    path = self._by_name[name]
                    break
        if path is None:
            raise KeyError(f"没有这个 Skill：{name}")
        _, body = _split_front_matter(path.read_text(encoding="utf-8", errors="replace"))
        return body

    def match(self, task: str, top_k: int = 1) -> list[dict]:
        """按 description 与任务文本的词重叠度挑选，命中才值得 load()。"""
        wanted = _tokens(task)
        scored = []
        for meta in self.list_skills():
            score = len(wanted & _tokens(f"{meta['name']} {meta['description']}"))
            if score > 0:
                scored.append((score, meta))
        scored.sort(key=lambda pair: -pair[0])
        return [meta for _, meta in scored[:top_k]]
