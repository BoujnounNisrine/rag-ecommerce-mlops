"""Index vectoriel unique (RAG monolithique).

Recherche exacte par similarité cosinus avec numpy. Avec ~150 à 300 chunks, une
recherche exacte prend moins d'une milliseconde : un index approximatif (FAISS,
HNSW…) n'apporterait rien et ajouterait une dépendance. Ce choix est à justifier
dans le rapport ; FAISS reste une extension possible si le corpus grandit.

Format sur disque (dossier `index_dir`) :
  vectors.npy   matrice (n_chunks, dim) float32, normalisée
  chunks.jsonl  texte + métadonnées de provenance de chaque chunk, même ordre
  meta.json     modèle d'embeddings, dimension, nombre de chunks, empreinte
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.data.io import read_jsonl, write_jsonl


@dataclass
class SearchResult:
    chunk: dict
    score: float
    rank: int


def chunks_fingerprint(chunks: list[dict]) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c["chunk_id"].encode("utf-8"))
        h.update(c["text"].encode("utf-8"))
    return h.hexdigest()[:16]


class VectorIndex:
    def __init__(self, vectors: np.ndarray, chunks: list[dict], meta: dict):
        if len(vectors) != len(chunks):
            raise ValueError(f"{len(vectors)} vecteurs pour {len(chunks)} chunks")
        self.vectors = vectors
        self.chunks = chunks
        self.meta = meta

    def __len__(self) -> int:
        return len(self.chunks)

    @classmethod
    def build(cls, chunks: list[dict], embedder) -> "VectorIndex":
        if not chunks:
            raise ValueError("Impossible de construire un index sans chunk")
        vectors = embedder.encode([c["text"] for c in chunks])
        meta = {
            "embedding_model": embedder.name,
            "dim": int(vectors.shape[1]),
            "n_chunks": len(chunks),
            "chunks_fingerprint": chunks_fingerprint(chunks),
        }
        return cls(vectors, chunks, meta)

    def search(self, query_vector: np.ndarray, k: int) -> list[SearchResult]:
        query_vector = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        scores = self.vectors @ query_vector
        k = min(k, len(self.chunks))
        top = np.argsort(-scores, kind="stable")[:k]
        return [
            SearchResult(chunk=self.chunks[i], score=float(scores[i]), rank=r)
            for r, i in enumerate(top, start=1)
        ]

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        write_jsonl(self.chunks, directory / "chunks.jsonl")
        (directory / "meta.json").write_text(json.dumps(self.meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "VectorIndex":
        directory = Path(directory)
        if not (directory / "vectors.npy").exists():
            raise FileNotFoundError(
                f"Index introuvable dans {directory}. Lancez d'abord : dvc repro build_index"
            )
        vectors = np.load(directory / "vectors.npy")
        chunks = read_jsonl(directory / "chunks.jsonl")
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        return cls(vectors, chunks, meta)
