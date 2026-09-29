#!/usr/bin/env bash
# 本机实测：hf-mirror ~110KB/s，ModelScope ~1MB/s（快约 10 倍），所以模型改走 ModelScope 拉。
# 用法：bash data/fetch_modelscope.sh
set -u
cd "$(dirname "$0")/.." || exit 1

BASE="https://www.modelscope.cn/api/v1/models"

fetch() {  # fetch <ModelScope 模型 id> <目标目录> <文件名...>
  local model="$1" dst="$2"; shift 2
  mkdir -p "$dst"
  for f in "$@"; do
    if [ -s "$dst/$f" ]; then echo "  [跳过] $f（已存在）"; continue; fi
    printf "  [下载] %-32s " "$f"
    if curl -sL --fail --retry 3 --max-time 3600 -o "$dst/$f.part" \
         "$BASE/$model/repo?Revision=master&FilePath=$f"; then
      mv "$dst/$f.part" "$dst/$f"
      du -h "$dst/$f" | cut -f1
    else
      echo "失败"; rm -f "$dst/$f.part"
    fi
  done
}

echo "=== 1/3 bge-small-zh-v1.5（检索，关键路径）==="
fetch BAAI/bge-small-zh-v1.5 models/bge-small-zh-v1.5 \
  model.safetensors config.json tokenizer.json tokenizer_config.json vocab.txt special_tokens_map.json

echo "=== 2/3 Qwen2.5-0.5B-Instruct（生成）==="
fetch Qwen/Qwen2.5-0.5B-Instruct models/Qwen2.5-0.5B-Instruct \
  model.safetensors config.json generation_config.json \
  tokenizer.json tokenizer_config.json vocab.json merges.txt

echo "=== 3/3 bge-reranker-base（可选，放最后）==="
fetch BAAI/bge-reranker-base models/bge-reranker-base \
  model.safetensors config.json tokenizer.json tokenizer_config.json \
  special_tokens_map.json sentencepiece.bpe.model

echo "=== 全部结束 ==="
du -sh models/* 2>/dev/null
