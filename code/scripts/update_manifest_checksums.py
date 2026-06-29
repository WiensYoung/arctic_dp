"""更新 data_manifest.json: 填充所有已下载文件的 SHA256 校验和及下载日期。

用法:
    python scripts/update_manifest_checksums.py
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "data" / "sci1_sources" / "data_manifest.json"
DATA_ROOT = PROJECT_ROOT / "data" / "sci1_sources"


def sha256_file(path: Path) -> str:
    """计算单个文件的 SHA256 十六进制字符串。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def find_files_for_source(local_path: str) -> dict[str, str]:
    """找到 local_path 下所有文件及其 SHA256。

    返回: {filename: sha256_hex, ...}
    """
    # local_path 是相对于项目根目录的 (如 "data/sci1_sources/nsidc_cdr_sic/")
    full = PROJECT_ROOT / local_path
    if not full.exists():
        return {}

    if full.is_file():
        return {full.name: sha256_file(full)}

    if full.is_dir():
        result = {}
        for f in sorted(full.glob("*.nc")):
            result[f.name] = sha256_file(f)
        for f in sorted(full.glob("*.csv")):
            result[f.name] = sha256_file(f)
        for f in sorted(full.glob("*.dat")):
            result[f.name] = sha256_file(f)
        for f in sorted(full.glob("*.zip")):
            result[f.name] = sha256_file(f)
        for f in sorted(full.glob("*.xlsx")):
            result[f.name] = sha256_file(f)
        for f in sorted(full.glob("*.txt")):
            result[f.name] = sha256_file(f)
        return result

    return {}


def get_file_mtime(path_str: str) -> str | None:
    """获取文件/目录中最新文件的修改时间。"""
    full = PROJECT_ROOT / path_str
    if not full.exists():
        return None

    if full.is_file():
        mtime = full.stat().st_mtime
    elif full.is_dir():
        files = list(full.glob("*"))
        if not files:
            return None
        mtime = max(f.stat().st_mtime for f in files)
    else:
        return None

    return datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%d")


def main():
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)

    updated_count = 0
    for src in manifest["sources"]:
        status = src.get("status", "")
        if status != "downloaded":
            continue

        local_path = src.get("local_path", "")
        if not local_path:
            continue

        # 计算校验和
        file_checksums = find_files_for_source(local_path)
        if not file_checksums:
            print(f"  [WARN] No files found for {src['name']}: {local_path}")
            continue

        # 单文件 → 直接字符串; 多文件 → 字典
        if len(file_checksums) == 1:
            src["checksum_sha256"] = list(file_checksums.values())[0]
        else:
            src["checksum_sha256"] = file_checksums

        # 填充下载日期（如果缺失）
        if not src.get("download_date"):
            mtime = get_file_mtime(local_path)
            if mtime:
                src["download_date"] = mtime

        updated_count += 1
        file_count = len(file_checksums) if isinstance(file_checksums, dict) else 1
        print(f"  [OK] {src['name']}: {file_count} file(s), checksum updated")

    # 更新元数据
    manifest["last_checksum_update_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # 写入
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    total = len(manifest["sources"])
    print(f"\nDone: {updated_count}/{total} sources updated with SHA256 checksums.")
    print(f"Manifest saved: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
