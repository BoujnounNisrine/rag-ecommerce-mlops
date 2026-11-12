"""Découpage des documents en chunks (fenêtre glissante en mots)."""


def split_words(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size doit être > 0")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap doit être dans [0, chunk_size[")

    words = text.split()
    if not words:
        return []
    step = chunk_size - chunk_overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + chunk_size]))
        if start + chunk_size >= len(words):
            break
    return chunks


def chunk_documents(docs: list[dict], chunk_size: int, chunk_overlap: int) -> list[dict]:
    """Chaque chunk hérite des métadonnées de son document (provenance)."""
    out = []
    for doc in docs:
        pieces = split_words(doc["text"], chunk_size, chunk_overlap)
        for i, piece in enumerate(pieces):
            meta = {k: v for k, v in doc.items() if k != "text"}
            out.append(
                {
                    **meta,
                    "chunk_id": f"{doc['doc_id']}#{i}",
                    "chunk_index": i,
                    "n_chunks": len(pieces),
                    "text": piece,
                }
            )
    return out
