"""ChnSentiCorp 备用下载脚本（当 data/download.py 失败时用）。

为什么需要这个脚本？
--------------------
`data/download.py` 走的是 `datasets.load_dataset("seamew/ChnSentiCorp")`。
该数据集在 Hugging Face 上是一个「带加载脚本」的数据集（仓库里有
`ChnSentiCorp.py`），而 `datasets>=4.0` 已经**移除**了对脚本式数据集的
支持，所以在新版本上会直接报：

    RuntimeError: Dataset scripts are no longer supported, but found ChnSentiCorp.py

解决办法有两条：
  1. 降级 `pip install "datasets<4"`（简单，但会牵动其它依赖）；
  2. 直接下载仓库里已经生成好的 `.arrow` 数据文件，自己转成 parquet（本脚本）。

本脚本用方案 2：从 HF 仓库取
    chn_senti_corp-train.arrow / -validation.arrow / -test.arrow
然后写成 `data/{train,validation,test}.parquet`，
列名和划分与 `download.py` 完全一致（train 9600 / validation 1200 / test 1200），
所以后续 `train.py`、`eval/run.py`、`visualize.py` 不需要任何改动。

用法
----
    python data/download_offline.py
国内网络如果直连 HF 不稳，先设镜像：
    set HF_ENDPOINT=https://hf-mirror.com
"""
import json
import os
import ssl
import sys
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).parent
RAW_DIR = DATA_DIR / "raw"

REPO = "seamew/ChnSentiCorp"
SPLITS = ("train", "validation", "test")
FILES = [f"chn_senti_corp-{s}.arrow" for s in SPLITS] + ["dataset_info.json"]


def repo_base() -> str:
    """HF 官方地址或镜像地址（可用 HF_ENDPOINT 覆盖）。"""
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    return f"{endpoint}/datasets/{REPO}/resolve/main/"


def make_opener():
    """构造一个 urllib opener：走系统代理（如果设了 HTTP_PROXY/HTTPS_PROXY）。

    这里放宽了证书校验，只是为了让公司/学校网络里常见的自签代理能通；
    如果不需要可以删掉 ctx 那两行。
    """
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    handlers = [urllib.request.HTTPSHandler(context=ctx)]
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    return urllib.request.build_opener(*handlers)


def download(url: str, out: Path):
    """带断点判断的极简下载：文件已存在且非空就跳过。"""
    if out.exists() and out.stat().st_size > 0:
        print(f"  [跳过] {out.name}（已存在，{out.stat().st_size:,} 字节）")
        return
    print(f"  [下载] {url}")
    opener = make_opener()
    with opener.open(url, timeout=120) as resp, open(out, "wb") as f:
        while True:
            chunk = resp.read(1 << 16)          # 64KB 一块，避免一次性吃满内存
            if not chunk:
                break
            f.write(chunk)
    print(f"  [完成] {out.name}（{out.stat().st_size:,} 字节）")


def main():
    print(f"数据源: {repo_base()}")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for name in FILES:
        download(repo_base() + name, RAW_DIR / name)

    # ---- 校验一下元信息，确认样本数与划分符合预期 ----
    info = json.loads((RAW_DIR / "dataset_info.json").read_text(encoding="utf-8"))
    expected = {s: info["splits"][s]["num_examples"] for s in SPLITS}
    print(f"\n官方样本数: {expected}")

    try:
        from datasets import Dataset
    except ImportError:
        sys.exit("[错误] 缺少依赖：pip install datasets pyarrow")

    for split in SPLITS:
        ds = Dataset.from_file(str(RAW_DIR / f"chn_senti_corp-{split}.arrow"))
        assert len(ds) == expected[split], f"{split} 样本数不符: {len(ds)} != {expected[split]}"
        out = DATA_DIR / f"{split}.parquet"
        ds.to_parquet(str(out))
        print(f"  {split}: {len(ds)} 条 -> {out.name}")

    print(f"\n完成。数据保存在 {DATA_DIR}")
    print("下一步：python train.py")


if __name__ == "__main__":
    main()
