"""
Generate and cache Sentence-BERT embeddings for passages and queries.

Model: all-MiniLM-L6-v2  (384-dim, fast, strong semantic quality)
Embeddings are L2-normalised so inner product == cosine similarity.
"""

import numpy as np
from pathlib import Path

from sentence_transformers import SentenceTransformer

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 256


def _embedder() -> SentenceTransformer:
    print(f"Loading model: {MODEL_NAME}")
    return SentenceTransformer(MODEL_NAME)


def encode_passages(
    passages: dict,
    force_recompute: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """
    Returns:
        embeddings  : float32 array of shape (N, 384), L2-normalised
        pid_list    : list of passage IDs in the same row order
    """
    n = len(passages)
    cache_emb = PROCESSED_DIR / f"passage_embeddings_{n}.npy"
    cache_ids = PROCESSED_DIR / f"passage_ids_{n}.npy"

    if not force_recompute and cache_emb.exists() and cache_ids.exists():
        print("Loading cached passage embeddings...")
        embeddings = np.load(cache_emb)
        pid_list   = list(np.load(cache_ids, allow_pickle=True))
        print(f"  Shape: {embeddings.shape}")
        return embeddings, pid_list

    model = _embedder()
    pid_list = list(passages.keys())
    texts    = [passages[pid]["cleaned"] for pid in pid_list]

    print(f"Encoding {n:,} passages in batches of {BATCH_SIZE}...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,   # L2 normalise
        convert_to_numpy=True,
    ).astype(np.float32)

    np.save(cache_emb, embeddings)
    np.save(cache_ids, np.array(pid_list, dtype=object))
    print(f"Saved embeddings to {cache_emb}")
    return embeddings, pid_list


def encode_queries(
    queries: dict,
    force_recompute: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """
    Returns:
        embeddings  : float32 array of shape (Q, 384), L2-normalised
        qid_list    : list of query IDs in the same row order
    """
    n = len(queries)
    cache_emb = PROCESSED_DIR / f"query_embeddings_{n}.npy"
    cache_ids = PROCESSED_DIR / f"query_ids_{n}.npy"

    if not force_recompute and cache_emb.exists() and cache_ids.exists():
        print("Loading cached query embeddings...")
        embeddings = np.load(cache_emb)
        qid_list   = list(np.load(cache_ids, allow_pickle=True))
        print(f"  Shape: {embeddings.shape}")
        return embeddings, qid_list

    model = _embedder()
    qid_list = list(queries.keys())
    texts    = [queries[qid]["cleaned"] for qid in qid_list]

    print(f"Encoding {n:,} queries...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    np.save(cache_emb, embeddings)
    np.save(cache_ids, np.array(qid_list, dtype=object))
    print(f"Saved embeddings to {cache_emb}")
    return embeddings, qid_list
