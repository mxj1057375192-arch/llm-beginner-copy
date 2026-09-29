"""维基百科查询：中文走 TextExtracts，英文走 section=0 的正文，标题对不上就搜索兜底。

两个接口不一样是有原因的：中文维基开了 TextExtracts 扩展（prop=extracts 能直接拿到
纯文本摘要），英文维基没有——同样的参数在 en 上返回空串，只能取 action=parse 的首节
正文再剥标签。

两处兜底也都是必需的：
  - **搜索兜底**：评测集里的中文查询用的是「Transformer (机器学习模型)」这种书面叫法，
    而维基上的实际标题是「Transformer架构」，精确标题查询会直接 missing；
  - **中英互备 + 重试**：本机到维基的连接是间歇性的（实测见过 TLS 被掐断的
    SSLZeroReturnError），重试两三次通常能过；某个语种连不上就换另一个语种查。
"""
from __future__ import annotations

import html
import re
import time
from urllib.parse import quote

import requests

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "wiki",
        "description": "查询维基百科里的百科知识（人物、事件、年份、概念、机构等），返回条目摘要正文。"
                       "中英文条目都支持。不要拿它查单词含义或与百科无关的问题。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": '要查询的条目名，例如 "图灵机"、"Geoffrey Hinton"',
                },
            },
            "required": ["query"],
        },
    },
}

_HEADERS = {"User-Agent": "llm-beginner-task5/1.0 (educational use)"}
_TIMEOUT = 12
_ATTEMPTS = 4
_RETRY_WAIT = 1.0
_MAX_CHARS = 1000
_ZH_API = "https://zh.wikipedia.org/w/api.php"
_EN_API = "https://en.wikipedia.org/w/api.php"
_ZH_REST = "https://zh.wikipedia.org/api/rest_v1/page/summary"
_EN_REST = "https://en.wikipedia.org/api/rest_v1/page/summary"


def _has_cjk(text: str) -> bool:
    return any("一" <= ch <= "鿿" for ch in text)


def _get(api: str, params: dict) -> dict:
    """带重试的 GET：这台机器上维基的连接偶尔被掐断，重试两三次通常就过了。"""
    last_error = None
    for attempt in range(_ATTEMPTS):
        try:
            resp = requests.get(api, headers=_HEADERS, timeout=_TIMEOUT, params=params)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            last_error = exc
            if attempt < _ATTEMPTS - 1:
                time.sleep(_RETRY_WAIT * (attempt + 1))
    raise last_error


def _strip_html(raw: str) -> str:
    raw = re.sub(r"<(script|style|sup|table)[^>]*>.*?</\1>", " ", raw,
                 flags=re.S | re.IGNORECASE)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", raw))).strip()


def _rest_summary(base: str, title: str) -> str:
    """REST 摘要接口：和 w/api.php 是两条独立的路，一条被掐断时另一条有时还能通。"""
    resp = requests.get(f"{base}/{quote(title)}", headers=_HEADERS, timeout=_TIMEOUT)
    if resp.status_code != 200:
        return ""
    return (resp.json().get("extract") or "").strip()


def _zh_extract(title: str) -> str:
    try:
        data = _get(_ZH_API, {"action": "query", "format": "json", "prop": "extracts",
                              "exintro": 1, "explaintext": 1, "redirects": 1,
                              "titles": title})
    except requests.RequestException:
        return _rest_summary(_ZH_REST, title)     # 主接口断了，换 REST 这条路
    page = next(iter(data.get("query", {}).get("pages", {}).values()), {})
    return "" if "missing" in page else page.get("extract", "")


def _en_extract(title: str) -> str:
    try:
        data = _get(_EN_API, {"action": "parse", "format": "json", "page": title,
                              "section": 0, "prop": "text", "redirects": 1})
    except requests.RequestException:
        return _rest_summary(_EN_REST, title)
    if "error" in data:
        return ""
    return _strip_html(data["parse"]["text"]["*"])


def _search(lang: str, query: str):
    """精确标题查不到时的兜底：先搜标题，再取第一条的正文。"""
    api = _ZH_API if lang == "zh" else _EN_API
    data = _get(api, {"action": "query", "format": "json", "list": "search",
                      "srsearch": query, "srlimit": 1})
    found = data.get("query", {}).get("search", [])
    if not found:
        return "", ""
    title = found[0]["title"]
    extract = _zh_extract(title) if lang == "zh" else _en_extract(title)
    return title, extract


def _lookup(lang: str, query: str):
    """返回 (标题, 摘要)；精确标题没有就搜索兜底。"""
    fetch = _zh_extract if lang == "zh" else _en_extract
    extract = fetch(query)
    if extract:
        return query, extract
    return _search(lang, query)


def run(args: dict) -> str:
    query = str(args["query"]).strip()
    if not query:
        raise ValueError("query 不能为空")

    lang = "zh" if _has_cjk(query) else "en"
    title = extract = ""
    failure = None
    for candidate in (lang, "en" if lang == "zh" else "zh"):
        try:
            title, extract = _lookup(candidate, query)
        except requests.RequestException as exc:
            failure = exc
        if extract:
            break
    if not extract:
        if failure is not None:
            raise RuntimeError(f"维基百科访问失败：{failure}") from failure
        raise ValueError(f"维基百科没有找到「{query}」这个条目")

    if len(extract) > _MAX_CHARS:
        extract = extract[:_MAX_CHARS] + "…"
    return f"维基百科「{title}」：\n{extract}"


if __name__ == "__main__":
    for text in ["图灵机", "Geoffrey Hinton", "Transformer (机器学习模型)"]:
        print(run({"query": text})[:200])
        print()
