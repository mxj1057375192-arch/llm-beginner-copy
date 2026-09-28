"""字符级分词器。

任务一考的是 Transformer，不是分词，所以这里用最朴素的中文字符级方案：
每个汉字就是一个词元，词表直接从训练集统计出来。
好处是不依赖任何预训练模型 / 外部词表，几行就能讲清楚。

（手写 BPE 是任务二的要求。）
"""
from collections import Counter

import torch

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


class CharTokenizer:
    """字符 <-> id 的双向映射。"""

    def __init__(self, vocab: dict):
        self.vocab = dict(vocab)
        self.inv_vocab = {i: ch for ch, i in self.vocab.items()}
        self.pad_id = self.vocab[PAD_TOKEN]
        self.unk_id = self.vocab[UNK_TOKEN]

    def __len__(self):
        return len(self.vocab)

    @classmethod
    def build(cls, texts, min_freq: int = 2):
        """统计训练集字符频率建词表；出现次数 < min_freq 的字符归入 <unk>。

        id 0/1 固定留给 <pad>/<unk>。频率低的字（错别字、罕见符号）不单独建 id，
        否则词表会被长尾噪声撑大，而这些字对情感判断也几乎没有信息量。
        """
        counter = Counter()
        for text in texts:
            counter.update(text)

        vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        for ch, freq in counter.most_common():
            if freq >= min_freq:
                vocab[ch] = len(vocab)
        return cls(vocab)

    def encode(self, text: str, max_len: int = 256) -> torch.Tensor:
        """文本 -> LongTensor，形状 (T,)，T <= max_len。"""
        ids = [self.vocab.get(ch, self.unk_id) for ch in text[:max_len]]
        if not ids:  # 空串兜底：避免产生长度为 0 的张量让模型报错
            ids = [self.unk_id]
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids) -> str:
        return "".join(self.inv_vocab.get(int(i), UNK_TOKEN) for i in ids)
