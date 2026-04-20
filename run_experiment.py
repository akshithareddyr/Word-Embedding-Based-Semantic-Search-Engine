"""
End-to-end experiment runner.

Usage:
    python run_experiment.py [--max-passages N] [--max-queries Q] [--k 1 3 5 10]

Steps:
  1. Load MS MARCO data
  2. Encode passages and queries with SBERT
  3. Build all four retrieval indexes
  4. Evaluate each pipeline with Precision@k, Recall@k, NDCG@k, MRR, Latency
  5. Save results CSV and plots
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader  import load_msmarco
from embeddings   import encode_passages, encode_queries
from retrieval    import BM25Retriever, ExactSearch, IVFFlatSearch, HNSWSearch
from metrics      import evaluate_all
from visualize    import (
    plot_tradeoff, plot_metric_bars, plot_embedding_space, qualitative_analysis,
    plot_precision_at_k, plot_recall_at_k, plot_ndcg_at_k, plot_mrr, plot_latency,
)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max-passages", type=int, default=100_000)
    p.add_argument("--max-queries",  type=int, default=1_000,
                   help="Number of dev queries to evaluate (subset for speed)")
    p.add_argument("--k", nargs="+", type=int, default=[1, 3, 5, 10])
    p.add_argument("--skip-bm25",  action="store_true")
    p.add_argument("--embedding-viz", action="store_true",
                   help="Generate t-SNE embedding visualisation (slow)")
    return p.parse_args()


def subsample_queries(queries, qrels, max_q):
    """Keep only queries that have at least one relevant passage, up to max_q."""
    valid = {qid: v for qid, v in queries.items() if qid in qrels}
    qids  = list(valid.keys())[:max_q]
    return {qid: valid[qid] for qid in qids}, {qid: qrels[qid] for qid in qids}


def main():
    args = parse_args()
    k_values = args.k

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 1: Loading MS MARCO data")
    print("="*60)
    passages, queries_all, qrels_all = load_msmarco(max_passages=args.max_passages)

    queries, qrels = subsample_queries(queries_all, qrels_all, args.max_queries)
    print(f"\nEvaluating on {len(queries):,} queries | {len(passages):,} passages")

    pid_list = list(passages.keys())
    qid_list = list(queries.keys())

    # ------------------------------------------------------------------
    # 2. Embeddings
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 2: Generating embeddings")
    print("="*60)
    passage_emb, passage_ids = encode_passages(passages)
    query_emb,   query_ids   = encode_queries(queries)

    # align qid_list to actual embedded order
    qid_list = query_ids

    # ------------------------------------------------------------------
    # 3. Build indexes & run all pipelines
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 3: Retrieval")
    print("="*60)

    all_results = []
    max_k = max(k_values)

    # BM25
    if not args.skip_bm25:
        bm25 = BM25Retriever(passages, pid_list)
        bm25_results, bm25_lat = bm25.batch_search(queries, qid_list, k=max_k)
        all_results.append(("BM25", bm25_results, bm25_lat))

    # Exact search
    exact = ExactSearch(passage_emb, passage_ids)
    exact_results, exact_lat = exact.batch_search(query_emb, qid_list, k=max_k)
    all_results.append(("ExactSearch", exact_results, exact_lat))

    # IVFFlat – default config + a fast variant
    for nlist, nprobe in [(128, 16), (256, 8)]:
        ivf = IVFFlatSearch(passage_emb, passage_ids, nlist=nlist, nprobe=nprobe)
        ivf_results, ivf_lat = ivf.batch_search(query_emb, qid_list, k=max_k)
        all_results.append((ivf.name, ivf_results, ivf_lat))

    # HNSW – default config + a fast variant
    for M, ef in [(32, 64), (16, 32)]:
        hnsw = HNSWSearch(passage_emb, passage_ids, M=M, ef_search=ef)
        hnsw_results, hnsw_lat = hnsw.batch_search(query_emb, qid_list, k=max_k)
        all_results.append((hnsw.name, hnsw_results, hnsw_lat))

    # ------------------------------------------------------------------
    # 4. Evaluate
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 4: Evaluation")
    print("="*60)

    rows = []
    for name, results, lat in all_results:
        row = evaluate_all(results, qrels, k_values=k_values,
                           avg_latency_ms=lat, pipeline_name=name)
        rows.append(row)
        print(f"\n  {name}")
        for key, val in row.items():
            if key != "pipeline":
                print(f"    {key:20s}: {val}")

    df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # ------------------------------------------------------------------
    # 5. Visualisations
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 5: Visualisations")
    print("="*60)

    pipeline_metrics = df.to_dict(orient="records")

    # per-metric line plots across k
    plot_precision_at_k(df, k_values)
    plot_recall_at_k(df, k_values)
    plot_ndcg_at_k(df, k_values)

    # single-value metric bar charts
    plot_mrr(df)
    plot_latency(df)

    # speed vs accuracy tradeoff
    plot_tradeoff(pipeline_metrics, quality_key=f"R@{max_k}")

    # summary grouped bar chart (all metrics side by side)
    metric_cols = [f"P@{k}" for k in k_values] + [f"R@{k}" for k in k_values] + ["MRR"]
    plot_metric_bars(df, metrics=metric_cols)

    if args.embedding_viz:
        plot_embedding_space(passage_emb, labels=[], method="tsne")

    # Qualitative analysis: requires both BM25 and at least one dense pipeline
    bm25_res   = next((r for name, r, _ in all_results if name == "BM25"),   None)
    exact_res  = next((r for name, r, _ in all_results if name == "ExactSearch"), None)
    if bm25_res and exact_res:
        qualitative_analysis(
            queries, passages, qrels,
            bm25_results=bm25_res,
            dense_results=exact_res,
            dense_name="ExactSearch",
        )

    print("\nDone. All outputs in ./results/")
    return df


if __name__ == "__main__":
    main()
