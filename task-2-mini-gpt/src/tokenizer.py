"""手写 byte-level BPE 分词器（对应 DoD 的 M1）。

为什么选「字节级（byte-level）」而不是「字符级」？
    中文常用字有几千个，字符级词表一开始就要预留几千个 token，且训练语料里没出现过的
    字直接变成 <unk>（信息永久丢失）。字节级先把每个字符拆成 UTF-8 字节（0~255），
    词表只需要 256 个「基本单位」，任何字符（中文、emoji、生僻字）都能表示，**永远不会
    出现 <unk>**；再由 BPE 把高频字节序列合并成更大的 token（比如「明月」整体一个 token）。
    代价是一个汉字通常占 2~3 个 token，序列变长——这是工程上常见的取舍。

    字节级还有一个「看不见的坑」：UTF-8 是变长编码，一个汉字的 3 个字节必须整体还原。
    因为我们是先 encode 成字节再 decode 回字节（逐字节可逆），这个边界问题自动解决。

BPE 训练流程（本文件 train_bpe）：
    1. 预分词：用正则把文本切成「词」——连续字母/数字/汉字算一个词，标点各自成词。
       这让 merge 只在词内部发生，不会把「光」和后面诗句的「，」粘成一个 token。
    2. 每个词转成 UTF-8 字节序列，作为初始符号序列。
    3. 迭代：统计所有相邻符号对的频率 -> 取频率最高的一对合并 -> 记录进 merge 表。
       重复到词表达到目标大小。
    4. 导出：base 词表（256 字节 + 特殊 token）+ merge 表（有序！顺序就是优先级）。

encode 时按「merge 表顺序」贪心合并（BPE 的标准做法，也叫 ranks）；
decode 时把每个 token id 映射回字节串、拼起来、按 UTF-8 解码。
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 特殊 token：<pad> 补齐、<bos> 序列开始、<eos> 序列结束
# ---------------------------------------------------------------------------
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>"]


def bytes_to_unicode() -> Tuple[Dict[int, str], Dict[str, int]]:
    """GPT-2 同款「字节 -> 可打印 unicode 字符」双射表。

    为什么不直接用 chr(b)？因为 byte 32（空格）、byte 10/13（换行回车）等是不可见字符，
    写进 JSON 词表后肉眼没法检查、也容易被编辑器吃掉。这里把所有 256 个字节映射到
    256 个可见字符（例如 32 -> 'Ġ'），一一对应，写进 json 既安全又能读懂。
    """
    bs = (list(range(ord("!"), ord("~") + 1))            # 可见 ASCII
          + list(range(ord("¡"), ord("¬") + 1))           # 拉丁-1 可见区
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:                 # 剩下 68 个「不可见字节」依次分配到 256 之后
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}, {chr(c): b for b, c in zip(bs, cs)}


BYTE_TO_CHAR, CHAR_TO_BYTE = bytes_to_unicode()

# 预分词正则：先吃掉特殊 token，再把「字母/数字/汉字」聚成词，其余（标点、空白）各自成词。
# \w 在 re.UNICODE 下包含汉字，所以中文会连续成词。
_SPLIT_PATTERN = re.compile(
    r"<\|[a-z_]+\|>|"                       # 形如 <|endoftext|> 的显式特殊标记
    r"[^\W\d_]+|"                           # 字母 / 汉字 / 其它文字字符
    r"\d+|"                                 # 数字
    r"\s+|"                                 # 空白（保留，decode 才能还原）
    r"[^\s\w]+",                            # 标点符号（每个独立）
    re.UNICODE,
)


class BPETokenizer:
    """简化版 BPE。接口见任务 README「实现约定」。

    两种基本单位（`mode`）：
      - "byte"（字节级，默认）：基本单位是 256 个字节，任何字符都能表示，绝无 <unk>；
        缺点是一个汉字要 2~3 个 token，小语料上容易产生"半个汉字"的碎片 token。
      - "char"（字符级）：基本单位是语料里出现过的所有字符（中文常用字+标点+拉丁字母），
        每个 token 至少是一个完整字符；没见过的字符回退成它的 UTF-8 字节（仍然无 <unk>）。
    两种模式都用同一套「统计相邻对 -> 迭代合并」的 BPE 算法，只是起始符号表不同。
    """

    def __init__(self, merges: Optional[List[Tuple[str, str]]] = None,
                 special_tokens: Optional[List[str]] = None,
                 base_chars: Optional[List[str]] = None):
        self.special_tokens = list(special_tokens or SPECIAL_TOKENS)
        self.base_chars = list(base_chars or [])          # char 模式的起始字符表
        # token 字符串 -> id
        self.token_to_id: Dict[str, int] = {}
        self.merges: List[Tuple[str, str]] = list(merges or [])
        self._ranks: Dict[Tuple[str, str], int] = {
            pair: i for i, pair in enumerate(self.merges)
        }
        # 编码缓存：同一个「词」只算一次（中文语料里重复词很多，缓存能省大量时间）
        self._cache: Dict[str, List[int]] = {}
        self._build_vocab()

    # ------------------------------------------------------------------
    # 词表构建
    # ------------------------------------------------------------------
    def _build_vocab(self) -> None:
        """按固定顺序建 id：特殊 token -> 基本单位（256 字节或字符表）-> 每次 merge 的新 token。"""
        self.token_to_id = {}
        self.id_to_token: List[str] = []

        def add(tok: str) -> int:
            if tok in self.token_to_id:
                return self.token_to_id[tok]
            idx = len(self.id_to_token)
            self.token_to_id[tok] = idx
            self.id_to_token.append(tok)
            return idx

        self.pad_id = add(self.special_tokens[0]) if len(self.special_tokens) > 0 else 0
        self.bos_id = add(self.special_tokens[1]) if len(self.special_tokens) > 1 else 1
        self.eos_id = add(self.special_tokens[2]) if len(self.special_tokens) > 2 else 2
        if self.base_chars:                       # char 模式
            for ch in self.base_chars:
                add(ch)                           # 语料里出现过的字符（优先占好 id）
            for b in range(256):
                add(BYTE_TO_CHAR[b])              # 兜底：未登录字回退成 UTF-8 字节
            self.mode = "char"
        else:                                     # byte 模式
            for b in range(256):
                add(BYTE_TO_CHAR[b])
            self.mode = "byte"
        for a, b in self.merges:                  # 每个 merge 生成一个新 token
            add(a + b)
        self.id_to_token = list(self.id_to_token)  # 与插入顺序一致
        # 解码用：token id -> 该 token 对应的字节串
        self._bytes_of: List[bytes] = []
        for tok in self.id_to_token:
            if tok in self.special_tokens:
                self._bytes_of.append(tok.encode("utf-8"))
            elif self.mode == "char":
                # 字符模式下 token 要么是语料字符（含多字 merge），要么是兜底字节代理字符
                if all(c in CHAR_TO_BYTE for c in tok):
                    self._bytes_of.append(bytes(CHAR_TO_BYTE[c] for c in tok))
                else:
                    self._bytes_of.append(tok.encode("utf-8"))
            else:
                self._bytes_of.append(bytes(CHAR_TO_BYTE[c] for c in tok))

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------
    @classmethod
    def train_bpe(cls, text: str, vocab_size: int = 2048, verbose: bool = True,
                  min_freq: int = 2, max_merges: Optional[int] = None,
                  mode: str = "byte") -> "BPETokenizer":
        """在 text 上训练 BPE，返回词表大小约等于 vocab_size 的分词器。

        Args:
            vocab_size: 目标词表大小（含特殊 token 与基本单位）
            min_freq: 频率低于该值的字节对不再合并（避免为罕见组合浪费词表）
            max_merges: 直接限制 merge 次数（优先级高于 vocab_size），调试用
            mode: "byte" 字节级 / "char" 字符级（基本单位 = 语料里出现过的字符）
        """
        assert mode in ("byte", "char"), "mode 只能是 'byte' 或 'char'"
        base_chars = sorted(set(text)) if mode == "char" else []
        # 基本单位：字符模式 = 语料字符 + 256 个兜底字节；字节模式 = 256 个字节
        n_base = len(SPECIAL_TOKENS) + (len(base_chars) + 256 if base_chars else 256)
        if vocab_size <= n_base:
            # 字符模式下如果语料字符数已经超过目标，就只做字符级、不 merge（仍然可用）
            if base_chars:
                if verbose:
                    print(f"[BPE] 字符数 {n_base} 已达/超过 vocab_size={vocab_size}，不额外 merge")
                return cls([], base_chars=base_chars)
            raise ValueError(f"vocab_size 至少要为 {n_base + 1}（> 特殊 token 256 字节）")
        n_merges = vocab_size - n_base
        if max_merges is not None:
            n_merges = min(n_merges, max_merges)

        # 1) 预分词 -> 每个「词」变成符号元组，并统计词频（只对唯一词做 merge，快得多）
        word_freq: Counter = Counter()
        for chunk in _SPLIT_PATTERN.findall(text):
            if base_chars:                       # 字符模式：直接按字符切
                syms = tuple(chunk)
            else:                                # 字节模式：UTF-8 字节 -> 可见字符
                syms = tuple(BYTE_TO_CHAR[x] for x in chunk.encode("utf-8"))
            word_freq[syms] += 1
        if verbose:
            print(f"[BPE] 模式 {mode}，唯一词数 {len(word_freq)}，"
                  f"基本单位 {n_base}，目标 merge 次数 {n_merges}")

        # 2) 初始化「相邻对 -> 出现次数」和「pair -> 包含它的词集合」
        #    后者用于增量更新：合并后只需重算受影响的词，不必每轮全量重扫
        pair_counts: Counter = Counter()
        pair_words: Dict[Tuple[str, str], set] = defaultdict(set)
        for w, f in word_freq.items():
            for i in range(len(w) - 1):
                pair = (w[i], w[i + 1])
                pair_counts[pair] += f
                pair_words[pair].add(w)

        merges: List[Tuple[str, str]] = []
        for step in range(n_merges):
            if not pair_counts:
                break
            # 取当前频率最高的一对；分数相同时取字典序最小的，保证可复现
            best = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
            if pair_counts[best] < min_freq:
                if verbose:
                    print(f"[BPE] 最高频对频率 {pair_counts[best]} < min_freq={min_freq}，提前停止")
                break
            merges.append(best)

            # 3) 只对「包含这一对」的词做替换，并局部更新 pair_counts
            affected = list(pair_words.pop(best, ()))
            merged_tok = best[0] + best[1]
            for w in affected:
                f = word_freq.get(w, 0)
                if f == 0:
                    continue
                # 减去旧词的贡献
                for i in range(len(w) - 1):
                    p = (w[i], w[i + 1])
                    pair_counts[p] -= f
                    if pair_counts[p] <= 0:
                        del pair_counts[p]
                    else:
                        pair_words[p].discard(w)
                # 生成新词（把 best 合并成一个符号）
                new_w, i = [], 0
                while i < len(w):
                    if i < len(w) - 1 and w[i] == best[0] and w[i + 1] == best[1]:
                        new_w.append(merged_tok)
                        i += 2
                    else:
                        new_w.append(w[i])
                        i += 1
                new_w = tuple(new_w)
                word_freq.pop(w, None)
                word_freq[new_w] = word_freq.get(new_w, 0) + f
                # 加上新词的贡献
                for j in range(len(new_w) - 1):
                    p = (new_w[j], new_w[j + 1])
                    pair_counts[p] += f
                    pair_words[p].add(new_w)

            if verbose and (step + 1) % 200 == 0:
                print(f"[BPE] merge {step + 1}/{n_merges}：{best[0]!r}+{best[1]!r} "
                      f"(freq={pair_counts.get(best, 0)})")

        if verbose:
            print(f"[BPE] 训练完成，共 {len(merges)} 次 merge，词表大小 {n_base + len(merges)}")
        return cls(merges, base_chars=base_chars)

    # ------------------------------------------------------------------
    # 编码 / 解码
    # ------------------------------------------------------------------
    def _syms(self, word: str) -> List[str]:
        """把一个「词」转成初始符号序列（每个符号都已经在词表里 -> 不会出现 <unk>）。

        char 模式：优先整字；语料里没见过的字退回它的 UTF-8 字节。
        byte 模式：一律按 UTF-8 字节。
        """
        if self.mode == "char":
            out: List[str] = []
            for ch in word:
                if ch in self.token_to_id:
                    out.append(ch)
                else:
                    out.extend(BYTE_TO_CHAR[b] for b in ch.encode("utf-8"))
            return out
        return [BYTE_TO_CHAR[b] for b in word.encode("utf-8")]

    def _encode_word(self, word: str) -> List[int]:
        """把一个「词」编码成 id 列表：先拆成基本符号，再按 merge 优先级贪心合并。"""
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        symbols = self._syms(word)
        # 反复找「优先级最高（rank 最小）的相邻对」合并，直到没有可合并的对
        while len(symbols) >= 2:
            best_rank, best_i = None, -1
            for i in range(len(symbols) - 1):
                r = self._ranks.get((symbols[i], symbols[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_i < 0:
                break
            symbols = (symbols[:best_i] + [symbols[best_i] + symbols[best_i + 1]]
                       + symbols[best_i + 2:])
        ids = [self.token_to_id[s] for s in symbols]
        self._cache[word] = ids
        return ids

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """文本 -> id 列表。任何字符都不会变成 <unk>（字节级的好处）。"""
        ids: List[int] = []
        if add_bos:
            ids.append(self.bos_id)
        for chunk in _SPLIT_PATTERN.findall(text):
            ids.extend(self._encode_word(chunk))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """id 列表 -> 文本：每个 token 还原成字节串，拼接后按 UTF-8 解码。"""
        byte_buf = bytearray()
        for i in ids:
            i = int(i)
            if i < 0 or i >= self.vocab_size:
                continue
            try:
                tok = self.id_to_token[i]
            except IndexError:
                continue
            if tok in self.special_tokens:
                if skip_special:
                    continue
            byte_buf.extend(self._bytes_of[i])
        # errors="replace"：正常路径永远走得通（逐字节可逆）；万一 token 被截断，
        # 也只是显示成 U+FFFD，而不是抛异常中断生成。
        return bytes(byte_buf).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # 保存 / 读取
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 2,
            "type": f"{self.mode}-level-bpe",
            "mode": self.mode,
            "special_tokens": self.special_tokens,
            "base_chars": self.base_chars if self.mode == "char" else [],
            # 存成 [["a","b"], ...]，顺序即优先级
            "merges": [[a, b] for a, b in self.merges],
            "vocab_size": self.vocab_size,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_pretrained(cls, path: str) -> "BPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        merges = [tuple(pair) for pair in payload["merges"]]
        return cls(merges,
                   special_tokens=payload.get("special_tokens", SPECIAL_TOKENS),
                   base_chars=payload.get("base_chars") or [])


if __name__ == "__main__":
    # 小自测：两种模式各训练一个迷你分词器，检查 encode->decode 是否还原
    # （注意：语料里没出现过的字在两种模式下都会回退成字节，仍能还原但 token 变多）
    demo = "床前明月光，疑是地上霜。举头望明月，低头思故乡。世界你好123。" * 20
    s = "床前明月光，Hello 世界! 123"
    for mode, vs in (("byte", 300), ("char", 340)):
        tk = BPETokenizer.train_bpe(demo, vocab_size=vs, verbose=False, mode=mode)
        ids = tk.encode(s)
        print(f"[{mode}] vocab_size={tk.vocab_size}  tokens={len(ids)}  "
              f"roundtrip={tk.decode(ids) == s}")
        print("    ", [repr(tk.id_to_token[i]) for i in ids])
