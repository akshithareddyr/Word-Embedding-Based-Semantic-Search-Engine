"""
Standard IR evaluation metrics.

All functions accept:
    results : dict  qid -> list[pid]   (ranked, top-k)
    qrels   : dict  qid -> set[pid]    (ground truth relevant pids)
    k       : int
"""

import math
import numpy as np
from typing import Optional


def precision_at_k(results: dict, qrels: dict, k: int) -> float:
    scores = []
    for qid, retrieved in results.items():
        if qid not in qrels:
            continue
        relevant = qrels[qid]
        top_k    = retrieved[:k]
        hits     = sum(1 for pid in top_k if pid in relevant)
        scores.append(hits / k)
    return float(np.mean(scores)) if scores else 0.0


def recall_at_k(results: dict, qrels: dict, k: int) -> float:
    scores = []
    for qid, retrieved in results.items():
        if qid not in qrels:
            continue
        relevant = qrels[qid]
        top_k    = retrieved[:k]
        hits     = sum(1 for pid in top_k if pid in relevant)
        denom    = len(relevant)
        scores.append(hits / denom if denom > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def mrr(results: dict, qrels: dict, k: Optional[int] = None) -> float:
    """Mean Reciprocal Rank. Pass k to cap at top-k."""
    scores = []
    for qid, retrieved in results.items():
        if qid not in qrels:
            continue
        relevant = qrels[qid]
        ranked   = retrieved[:k] if k else retrieved
        rr = 0.0
        for rank, pid in enumerate(ranked, start=1):
            if pid in relevant:
                rr = 1.0 / rank
                break
        scores.append(rr)
    return float(np.mean(scores)) if scores else 0.0


def ndcg_at_k(results: dict, qrels: dict, k: int) -> float:
    """
    Binary NDCG@k: relevance is 1 if pid in qrels, else 0.
    Ideal DCG assumes all relevant docs are at the top.
    """
    scores = []
    for qid, retrieved in results.items():
        if qid not in qrels:
            continue
        relevant = qrels[qid]
        top_k    = retrieved[:k]

        dcg  = sum(
            1.0 / math.log2(rank + 1)
            for rank, pid in enumerate(top_k, start=1)
            if pid in relevant
        )
        # ideal: place all min(|relevant|, k) hits at rank 1..
        n_ideal = min(len(relevant), k)
        idcg    = sum(1.0 / math.log2(rank + 1) for rank in range(1, n_ideal + 1))
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def evaluate_all(
    results: dict,
    qrels: dict,
    k_values: list[int] = (1, 3, 5, 10),
    avg_latency_ms: float = 0.0,
    pipeline_name: str = "",
) -> dict:
    """
    Returns a flat dict of all metrics for a single pipeline.
    """
    row = {"pipeline": pipeline_name, "avg_latency_ms": round(avg_latency_ms, 3)}
    for k in k_values:
        row[f"P@{k}"]     = round(precision_at_k(results, qrels, k), 4)
        row[f"R@{k}"]     = round(recall_at_k(results, qrels, k), 4)
        row[f"NDCG@{k}"]  = round(ndcg_at_k(results, qrels, k), 4)
    row["MRR"] = round(mrr(results, qrels), 4)
    return row
