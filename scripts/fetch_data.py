import gzip
import hashlib
import json
import os
import tempfile
import urllib.request
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_lock() -> dict:
    return json.loads((ROOT / "data/sources.lock.json").read_text(encoding="utf-8"))


def verify_payload(item: dict, payload: bytes) -> None:
    if (
        len(payload) != item["upstream_bytes"]
        or hashlib.sha256(payload).hexdigest() != item["upstream_sha256"]
    ):
        raise ValueError(f"Размер или SHA-256 не совпал: {item['file']}")


def read_cached_payload(item: dict) -> bytes:
    path = ROOT / item["file"]
    try:
        stored = path.read_bytes()
        payload = stored if item["url"].endswith(".gz") else gzip.decompress(stored)
        verify_payload(item, payload)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Запустите python scripts/fetch_data.py: отсутствует {path}"
        ) from exc
    except (gzip.BadGzipFile, EOFError, zlib.error, ValueError) as exc:
        raise ValueError(f"Повреждён кеш {path}. Удалите этот файл и повторите загрузку.") from exc
    return payload


def read_snapshot_text(item: dict) -> str:
    payload = read_cached_payload(item)
    if item["url"].endswith(".gz"):
        payload = gzip.decompress(payload)
    return payload.decode("utf-8")


def main() -> None:
    lock = read_lock()
    for item in [*lock["files"], *lock.get("audit_sources", [])]:
        path = ROOT / item["file"]
        if path.exists():
            read_cached_payload(item)
            print(f"Кеш проверен: {path.name}")
            continue
        with urllib.request.urlopen(item["url"], timeout=60) as response:
            payload = response.read(item["upstream_bytes"] + 1)
        verify_payload(item, payload)
        stored = payload if item["url"].endswith(".gz") else gzip.compress(payload, mtime=0)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, filename = tempfile.mkstemp(dir=path.parent)
        temp_path = Path(filename)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(stored)
            temp_path.replace(path)
        finally:
            temp_path.unlink(missing_ok=True)
        print(f"Загружен: {path.name}")


if __name__ == "__main__":
    main()
