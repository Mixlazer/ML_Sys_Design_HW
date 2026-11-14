"""Restore exact public snapshots using lockfile URLs and SHA-256 guards."""

import gzip
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    lock = json.loads((ROOT / "data/sources.lock.json").read_text())
    for item in lock["files"]:
        path = ROOT / item["file"]
        if path.exists():
            local = path.read_bytes()
            upstream = local if item["url"].endswith(".gz") else gzip.decompress(local)
            if hashlib.sha256(upstream).hexdigest() != item["upstream_sha256"]:
                raise ValueError(f"Corrupt local snapshot: {path}; remove it explicitly to restore")
            print(f"Verified: {path.name}")
            continue
        with urllib.request.urlopen(item["url"], timeout=60) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != item["upstream_sha256"]:
            raise ValueError(f"Upstream checksum mismatch: {path}")
        stored = content if item["url"].endswith(".gz") else gzip.compress(content, mtime=0)
        # Semantic source bytes are stable across Python/zlib gzip container versions.
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(stored)
        print(f"Restored: {path.name}")


if __name__ == "__main__":
    main()
