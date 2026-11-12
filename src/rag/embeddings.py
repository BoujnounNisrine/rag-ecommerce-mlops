"""Modèles d'embeddings.

- SentenceTransformerEmbedder : le vrai modèle, chargé à la demande (import paresseux,
  pour que les tests et la CI n'aient pas besoin de PyTorch).
- HashingEmbedder : embedder factice, déterministe et sans dépendance, utilisé par
  les tests. Il ne capture que le vocabulaire commun, pas le sens.

Tous les embedders renvoient des vecteurs normalisés (norme L2 = 1), donc le
produit scalaire est égal à la similarité cosinus.
"""

import hashlib
import re

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9]+")


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


class HashingEmbedder:
    """Sac de mots haché dans un vecteur de taille fixe (tests uniquement)."""

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _index(self, token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        return int(digest, 16) % self.dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in TOKEN_RE.findall(text.lower()):
                out[i, self._index(token)] += 1.0
        return l2_normalize(out)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, batch_size: int = 32):
        from sentence_transformers import SentenceTransformer  # import paresseux

        self.name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device="cpu")

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)


def load_embedder(params: dict):
    cfg = params["embeddings"]
    return SentenceTransformerEmbedder(cfg["model"], cfg.get("batch_size", 32))
