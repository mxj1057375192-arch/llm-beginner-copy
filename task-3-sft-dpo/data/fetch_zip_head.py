"""只下载远端 zip 的开头一小段，流式解压出前 N 行，避免拉整个 2 GB。

做法：HTTP Range 请求读 zip 尾部的中央目录 -> 拿到第一个条目的数据偏移 ->
再 Range 请求该条目的前若干字节 -> zlib 流式解压 -> 读够 N 行就停。
"""
from __future__ import annotations

import json
import struct
import subprocess
import sys
import zlib
from pathlib import Path

URL = ("https://hf-mirror.com/datasets/OpenMOSS-Team/moss-003-sft-data/"
       "resolve/main/moss-003-sft-no-tools.jsonl.zip")


def _curl(args: list[str]) -> bytes:
    """走 curl 而不是 urllib：hf-mirror 的证书链在 Python 这边校验不过，curl 没问题。"""
    p = subprocess.run(["curl", "-sSL", "--retry", "3", *args],
                       capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f"curl 失败：{p.stderr.decode('utf-8', 'replace')[:300]}")
    return p.stdout


def get_range(url: str, start: int, end: int) -> bytes:
    return _curl(["-r", f"{start}-{end}", url])


def head_info(url: str):
    out = _curl(["-I", url]).decode("utf-8", "replace")
    size = ranges = None
    for line in out.splitlines():
        k, _, v = line.partition(":")
        k, v = k.strip().lower(), v.strip()
        if k == "content-length":
            size = int(v)
        elif k == "accept-ranges":
            ranges = v
    return size, ranges


def read_central_directory(url: str, size: int):
    """返回 [(文件名, 压缩方式, 压缩大小, 数据偏移)]。"""
    tail = get_range(url, max(0, size - 65557), size - 1)
    idx = tail.rfind(b"PK\x05\x06")
    if idx < 0:
        raise RuntimeError("没找到 EOCD 记录")
    cd_size, cd_off = struct.unpack("<II", tail[idx + 12: idx + 20])
    if cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:  # ZIP64
        z64 = tail.rfind(b"PK\x06\x06")
        cd_size, cd_off = struct.unpack("<QQ", tail[z64 + 40: z64 + 56])
    cd = get_range(url, cd_off, cd_off + cd_size - 1)

    entries, p = [], 0
    while p < len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        method, = struct.unpack("<H", cd[p + 10: p + 12])
        csize, = struct.unpack("<I", cd[p + 20: p + 24])
        nlen, elen, clen = struct.unpack("<HHH", cd[p + 28: p + 34])
        lho, = struct.unpack("<I", cd[p + 42: p + 46])
        name = cd[p + 46: p + 46 + nlen].decode("utf-8", "replace")
        entries.append((name, method, csize, lho))
        p += 46 + nlen + elen + clen
    return entries


def local_data_start(url: str, offset: int) -> tuple[int, int]:
    """读本地文件头，返回 (数据起始偏移, 压缩方式)。"""
    hdr = get_range(url, offset, offset + 29)
    method, = struct.unpack("<H", hdr[8:10])
    nlen, elen = struct.unpack("<HH", hdr[26:30])
    return offset + 30 + nlen + elen, method


def stream_lines(url: str, data_start: int, want: int, chunk_mb: int = 4):
    """流式解压，产出前 want 行。"""
    dec = zlib.decompressobj(-zlib.MAX_WBITS)
    buf = b""
    got = 0
    read = 0
    while got < want:
        raw = get_range(url, data_start + read, data_start + read + chunk_mb * 1024 * 1024 - 1)
        if not raw:
            break
        read += len(raw)
        buf += dec.decompress(raw)
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            if line.strip():
                got += 1
                yield line
                if got >= want:
                    return
        print(f"  已解压 {read/1e6:.1f} MB 压缩数据 -> {got} 行", file=sys.stderr)


def main():
    want = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    out = Path(__file__).parent / "moss-sft" / "sample.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    size, ranges = head_info(URL)
    print(f"远端 zip 大小 {size/1e9:.2f} GB，Accept-Ranges={ranges}")
    entries = read_central_directory(URL, size)
    print(f"包内 {len(entries)} 个条目：")
    for name, method, csize, lho in entries:
        print(f"  {name}  压缩后 {csize/1e6:.1f} MB  method={method}")

    name, method, csize, lho = entries[0]
    data_start, method = local_data_start(URL, lho)
    print(f"\n从 {name} 流式解压前 {want} 行 ...")
    if method != 8:
        print(f"[警告] 压缩方式 {method} 不是 deflate，可能无法流式解压")

    n = 0
    with out.open("w", encoding="utf-8") as f:
        for line in stream_lines(URL, data_start, want):
            f.write(line.decode("utf-8") + "\n")
            n += 1
    print(f"\n写入 {out}：{n} 行，{out.stat().st_size/1e6:.1f} MB")

    first = json.loads(out.open(encoding="utf-8").readline())
    print("第一条样例的键：", list(first.keys()))
    print(json.dumps(first, ensure_ascii=False)[:600])


if __name__ == "__main__":
    main()
