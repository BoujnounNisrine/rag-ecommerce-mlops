"""Étape DVC `build_index` : chunks.jsonl -> artifacts/index/.

Usage : python -m src.rag.build_index
"""

import time

from src.config import load_params, resolve
from src.data.io import read_jsonl
from src.rag.embeddings import load_embedder
from src.rag.index import VectorIndex


def main() -> None:
    params = load_params()
    d = params["data"]

    chunks = read_jsonl(resolve(d["chunks"]))
    embedder = load_embedder(params)

    start = time.perf_counter()
    index = VectorIndex.build(chunks, embedder)
    elapsed = time.perf_counter() - start

    index.save(resolve(d["index_dir"]))
    print(
        f"Index construit : {len(index)} chunks, dim={index.meta['dim']}, "
        f"modèle={index.meta['embedding_model']} ({elapsed:.1f} s)"
    )


if __name__ == "__main__":
    main()
