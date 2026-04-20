"""
Four retrieval pipelines:
  1. BM25          – sparse keyword baseline (rank_bm25)
  2. ExactSearch   – brute-force FAISS IndexFlatIP (cosine via L2-norm)
  3. IVFFlatSearch – FAISS IVFFlat ANN
  4. HNSWSearch    – FAISS HNSWFlat ANN

All dense searchers expose the same interface:
    searcher.search(query_vectors, k) -> (distances, pid_indices)
"""

import time
from pathlib import Path

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from tqdm import tqdm

# Force single-threaded FAISS — prevents segfaults on Apple Silicon (M1/M2/M3)
# where the OpenMP k-means in IVF training crashes with multiple threads.
faiss.omp_set_num_threads(1)


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

class BM25Retriever:
    name = "BM25"

    def __init__(self, passages: dict, pid_list: list[str]):
        self.pid_list = pid_list
        tokenized = [passages[pid]["cleaned"].split() for pid in pid_list]
        print("Building BM25 index...")
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query_text: str, k: int) -> list[str]:
        """Returns top-k passage IDs for a single text query."""
        tokens = query_text.split()
        scores = self.bm25.get_scores(tokens)
        top_k  = np.argsort(scores)[::-1][:k]
        return [self.pid_list[i] for i in top_k]

    def batch_search(
        self, queries: dict, qid_list: list[str], k: int
    ) -> tuple[dict[str, list[str]], float]:
        """
        Returns:
            results     : qid -> list of top-k pids
            avg_latency : ms per query
        """
        results = {}
        latencies = []
        for qid in tqdm(qid_list, desc=f"BM25 search k={k}"):
            q_text = queries[qid]["cleaned"]
            t0 = time.perf_counter()
            results[qid] = self.search(q_text, k)
            latencies.append((time.perf_counter() - t0) * 1000)
        return results, float(np.mean(latencies))


# ---------------------------------------------------------------------------
# Dense base class
# ---------------------------------------------------------------------------

class _DenseRetriever:
    name: str
    index: faiss.Index

    def __init__(self, pid_list: list[str]):
        self.pid_list = pid_list

    def batch_search(
        self,
        query_vectors: np.ndarray,
        qid_list: list[str],
        k: int,
    ) -> tuple[dict[str, list[str]], float]:
        """
        Returns:
            results     : qid -> list of top-k pids
            avg_latency : ms per query
        """
        results = {}
        latencies = []
        for i, qid in enumerate(tqdm(qid_list, desc=f"{self.name} search k={k}")):
            qvec = np.ascontiguousarray(query_vectors[i : i + 1], dtype=np.float32)
            t0 = time.perf_counter()
            _, indices = self.index.search(qvec, k)
            latencies.append((time.perf_counter() - t0) * 1000)
            results[qid] = [
                self.pid_list[idx] for idx in indices[0] if idx != -1
            ]
        return results, float(np.mean(latencies))


# ---------------------------------------------------------------------------
# Exact search
# ---------------------------------------------------------------------------

CACHE_DIR = Path(__file__).parent.parent / "data" / "processed"


def _load_or_build(cache_path: Path, build_fn):
    """Load a FAISS index from disk if it exists, otherwise build and save it."""
    if cache_path.exists():
        print(f"Loading cached FAISS index from {cache_path.name}...")
        return faiss.read_index(str(cache_path))
    index = build_fn()
    faiss.write_index(index, str(cache_path))
    print(f"Saved FAISS index to {cache_path.name}")
    return index


class ExactSearch(_DenseRetriever):
    name = "ExactSearch"

    def __init__(self, passage_embeddings: np.ndarray, pid_list: list[str]):
        super().__init__(pid_list)
        vecs = np.ascontiguousarray(passage_embeddings, dtype=np.float32)
        dim  = vecs.shape[1]
        n    = vecs.shape[0]
        cache = CACHE_DIR / f"faiss_exact_{n}.index"

        def build():
            print(f"Building ExactSearch index (dim={dim}, n={n:,})...")
            idx = faiss.IndexFlatIP(dim)
            idx.add(vecs)
            return idx

        self.index = _load_or_build(cache, build)


# ---------------------------------------------------------------------------
# IVFFlat
# ---------------------------------------------------------------------------

class IVFFlatSearch(_DenseRetriever):
    name = "IVFFlat"

    def __init__(
        self,
        passage_embeddings: np.ndarray,
        pid_list: list[str],
        nlist: int = 128,
        nprobe: int = 16,
    ):
        super().__init__(pid_list)
        self.nprobe = nprobe
        vecs  = np.ascontiguousarray(passage_embeddings, dtype=np.float32)
        dim   = vecs.shape[1]
        n     = vecs.shape[0]
        nlist = min(nlist, n)
        self.nlist = nlist
        cache = CACHE_DIR / f"faiss_ivf_nlist{nlist}_{n}.index"

        def build():
            print(f"Building IVFFlat index (nlist={nlist}, nprobe={nprobe}, n={n:,})...")
            quantizer = faiss.IndexFlatIP(dim)
            idx = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            idx.train(vecs)
            idx.add(vecs)
            return idx

        self.index = _load_or_build(cache, build)
        self.index.nprobe = nprobe

    @property
    def name(self):
        return f"IVFFlat(nlist={self.nlist},nprobe={self.nprobe})"


# ---------------------------------------------------------------------------
# HNSW
# ---------------------------------------------------------------------------

class HNSWSearch(_DenseRetriever):
    name = "HNSW"

    def __init__(
        self,
        passage_embeddings: np.ndarray,
        pid_list: list[str],
        M: int = 32,
        ef_construction: int = 200,
        ef_search: int = 64,
    ):
        super().__init__(pid_list)
        self.M         = M
        self.ef_search = ef_search
        vecs  = np.ascontiguousarray(passage_embeddings, dtype=np.float32)
        dim   = vecs.shape[1]
        n     = vecs.shape[0]
        cache = CACHE_DIR / f"faiss_hnsw_M{M}_{n}.index"

        def build():
            print(f"Building HNSW index (M={M}, ef_search={ef_search}, n={n:,})...")
            idx = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
            idx.hnsw.efConstruction = ef_construction
            idx.hnsw.efSearch       = ef_search
            idx.add(vecs)
            return idx

        self.index = _load_or_build(cache, build)
        self.index.hnsw.efSearch = ef_search  # restore after load

    @property
    def name(self):
        return f"HNSW(M={self.M},ef={self.ef_search})"
