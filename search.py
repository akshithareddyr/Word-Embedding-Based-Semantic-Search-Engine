"""
Interactive search demo.

Usage:
    python search.py                      # uses HNSW by default, k=5
    python search.py --pipeline bm25      # use BM25 instead
    python search.py --pipeline exact     # use ExactSearch
    python search.py --k 10               # return top 10 results
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

# suppress noisy HuggingFace / transformers load warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
warnings.filterwarnings("ignore")

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import load_msmarco
from embeddings  import encode_passages
from retrieval   import BM25Retriever, ExactSearch, HNSWSearch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pipeline", choices=["hnsw", "exact", "bm25"], default="hnsw")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--max-passages", type=int, default=100_000)
    return p.parse_args()


def build_pipeline(name, passages, pid_list, passage_emb):
    if name == "bm25":
        print("Loading BM25 index...")
        return BM25Retriever(passages, pid_list)
    elif name == "exact":
        print("Loading ExactSearch index...")
        return ExactSearch(passage_emb, pid_list)
    else:
        print("Loading HNSW index...")
        return HNSWSearch(passage_emb, pid_list, M=32, ef_search=64)


def search_bm25(retriever, query_text, k):
    from data_loader import _clean
    cleaned = _clean(query_text)
    return retriever.search(cleaned, k)


def search_dense(retriever, query_text, k):
    from sentence_transformers import SentenceTransformer
    from data_loader import _clean
    model  = SentenceTransformer("all-MiniLM-L6-v2")
    cleaned = _clean(query_text)
    qvec   = model.encode([cleaned], normalize_embeddings=True).astype("float32")
    _, indices = retriever.index.search(qvec, k)
    return [retriever.pid_list[i] for i in indices[0] if i != -1]


def main():
    args = parse_args()

    # load cached data and embeddings
    print("Loading data...")
    passages, _, _ = load_msmarco(max_passages=args.max_passages)
    pid_list = list(passages.keys())

    passage_emb = None
    if args.pipeline in ("hnsw", "exact"):
        passage_emb, pid_list = encode_passages(passages)

    retriever = build_pipeline(args.pipeline, passages, pid_list, passage_emb)

    # keep a single SentenceTransformer loaded for dense pipelines
    encoder = None
    if args.pipeline in ("hnsw", "exact"):
        from sentence_transformers import SentenceTransformer
        from data_loader import _clean as clean_fn
        encoder = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"\nPipeline : {args.pipeline.upper()}  |  k = {args.k}")
    print("Type your query and press Enter. Type 'quit' to exit.\n")

    while True:
        try:
            query = input("Query > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        # retrieve
        if args.pipeline == "bm25":
            from data_loader import _clean
            cleaned = _clean(query)
            top_pids = retriever.search(cleaned, args.k)
        else:
            from data_loader import _clean
            cleaned = _clean(query)
            qvec    = encoder.encode([cleaned], normalize_embeddings=True).astype("float32")
            _, indices = retriever.index.search(qvec, args.k)
            top_pids = [retriever.pid_list[i] for i in indices[0] if i != -1]

        # display
        print(f"\nTop {args.k} results for: \"{query}\"\n" + "-" * 60)
        for rank, pid in enumerate(top_pids, start=1):
            text = passages[pid]["text"]
            # wrap at 100 chars for readability
            display = text if len(text) <= 300 else text[:300] + "..."
            print(f"[{rank}] {display}\n")
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
