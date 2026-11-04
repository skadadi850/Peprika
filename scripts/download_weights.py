"""
Downloads the weights of Peprika, checks the checksum and unpacks them into weights/.

Usage:
    python scripts/download_weights.py
    PEPRIKA_WEIGHTS_URL=file:///path/to/peprika_weights.zip python scripts/download_weights.py
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
URL = os.environ.get("PEPRIKA_WEIGHTS_URL", "https://github.com/shouryaverma/peprika/releases/download/v1.0/peprika_weights.zip")
ARCHIVE = ROOT / "dist" / "peprika_weights.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main() -> None:
    expected = (ROOT / "weights.sha256").read_text().split()[0]
    ARCHIVE.parent.mkdir(exist_ok=True)

    if not ARCHIVE.exists():
        partial = ARCHIVE.with_suffix(".part")
        print(f"downloading {URL}", flush=True)
        urllib.request.urlretrieve(URL, partial)
        partial.replace(ARCHIVE)

    actual = sha256(ARCHIVE)

    if actual != expected:
        ARCHIVE.unlink()
        raise SystemExit(f"checksum mismatch\nexpected {expected}\nactual   {actual}")

    with zipfile.ZipFile(ARCHIVE) as handle:
        handle.extractall(ROOT)

    print(f"weights in {ROOT / 'weights'}", flush=True)


if __name__ == "__main__":
    main()
