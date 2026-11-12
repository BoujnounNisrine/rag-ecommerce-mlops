"""Étape DVC `chunk` : documents.jsonl -> chunks.jsonl.

Usage : python -m src.data.chunk
"""

from src.config import load_params, resolve
from src.data.chunking import chunk_documents
from src.data.io import read_jsonl, write_jsonl


def main() -> None:
    params = load_params()
    cfg, d = params["chunking"], params["data"]

    docs = read_jsonl(resolve(d["documents"]))
    chunks = chunk_documents(docs, cfg["chunk_size"], cfg["chunk_overlap"])
    n = write_jsonl(chunks, resolve(d["chunks"]))
    print(
        f"{n} chunks écrits à partir de {len(docs)} documents "
        f"(size={cfg['chunk_size']}, overlap={cfg['chunk_overlap']})"
    )


if __name__ == "__main__":
    main()
