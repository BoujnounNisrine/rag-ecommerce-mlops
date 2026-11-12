"""Étape DVC `download` : télécharge le CSV Bitext depuis Hugging Face.

Usage : python -m src.data.download
"""

import hashlib
import urllib.request

from src.config import load_params, resolve


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    params = load_params()
    url = params["data"]["bitext_url"]
    dest = resolve(params["data"]["raw_bitext"])
    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"Téléchargement de {url}")
    urllib.request.urlretrieve(url, dest)
    size_mb = dest.stat().st_size / 1e6
    print(f"OK : {dest} ({size_mb:.1f} Mo, sha256={sha256(dest)[:12]})")


if __name__ == "__main__":
    main()
