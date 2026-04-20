"""
Visualizations:
  1. t-SNE / UMAP of passage embeddings
  2. Speed vs accuracy tradeoff curve
  3. Results summary grouped bar chart
  4. Qualitative analysis: BM25 vs dense search examples
  5. Precision@k line plot (per pipeline, across k)
  6. Recall@k line plot
  7. NDCG@k line plot
  8. MRR bar chart
  9. Query latency bar chart
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def plot_embedding_space(
    embeddings: np.ndarray,
    labels: list[str],
    title: str = "Passage Embedding Space",
    method: str = "tsne",
    n_sample: int = 5000,
    save_path: str | None = None,
):
    """2-D projection of passage embeddings coloured by label (if provided)."""
    idx = np.random.choice(len(embeddings), size=min(n_sample, len(embeddings)), replace=False)
    emb_sample   = embeddings[idx]
    label_sample = [labels[i] for i in idx] if labels else None

    if method == "umap":
        from umap import UMAP
        reducer = UMAP(n_components=2, random_state=42)
    else:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)

    print(f"Running {method.upper()} on {len(emb_sample):,} points...")
    proj = reducer.fit_transform(emb_sample)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(proj[:, 0], proj[:, 1], s=2, alpha=0.5, c="steelblue")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    plt.tight_layout()

    out = save_path or str(RESULTS_DIR / f"embedding_{method}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_tradeoff(
    pipeline_metrics: list[dict],
    latency_key: str = "avg_latency_ms",
    quality_key: str = "R@10",
    save_path: str | None = None,
):
    """Scatter: latency (x) vs quality metric (y), one point per pipeline."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for row in pipeline_metrics:
        name    = row["pipeline"]
        latency = row[latency_key]
        quality = row[quality_key]
        ax.scatter(latency, quality, s=100, zorder=3)
        ax.annotate(name, (latency, quality), textcoords="offset points", xytext=(6, 4), fontsize=9)

    ax.set_xlabel("Avg Query Latency (ms)", fontsize=11)
    ax.set_ylabel(quality_key, fontsize=11)
    ax.set_title("Speed vs. Retrieval Quality Trade-off", fontsize=13)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()

    out = save_path or str(RESULTS_DIR / "tradeoff.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def qualitative_analysis(
    queries: dict,
    passages: dict,
    qrels: dict,
    bm25_results: dict,
    dense_results: dict,
    dense_name: str = "ExactSearch",
    n_examples: int = 5,
    save_path: str | None = None,
):
    """
    Print and save examples where dense search beats BM25 and vice versa.
    Useful for showing semantic search advantages (synonym/paraphrase cases).
    """
    wins_dense = []   # dense found relevant, BM25 did not
    wins_bm25  = []   # BM25 found relevant, dense did not

    for qid in queries:
        if qid not in qrels:
            continue
        relevant = qrels[qid]
        dense_hit = any(pid in relevant for pid in dense_results.get(qid, [])[:5])
        bm25_hit  = any(pid in relevant for pid in bm25_results.get(qid, [])[:5])

        if dense_hit and not bm25_hit:
            wins_dense.append(qid)
        elif bm25_hit and not dense_hit:
            wins_bm25.append(qid)

    lines = []
    lines.append("=" * 70)
    lines.append(f"QUALITATIVE ANALYSIS: BM25 vs {dense_name} (top-5)")
    lines.append("=" * 70)

    lines.append(f"\n>>> Cases where {dense_name} finds relevant doc but BM25 misses ({len(wins_dense)} total)\n")
    for qid in wins_dense[:n_examples]:
        lines.append(f"  Query : {queries[qid]['text']}")
        rel_pid = next(pid for pid in dense_results[qid][:5] if pid in qrels[qid])
        lines.append(f"  Found : {passages[rel_pid]['text'][:200]}...")
        lines.append("")

    lines.append(f"\n>>> Cases where BM25 finds relevant doc but {dense_name} misses ({len(wins_bm25)} total)\n")
    for qid in wins_bm25[:n_examples]:
        lines.append(f"  Query : {queries[qid]['text']}")
        rel_pid = next(pid for pid in bm25_results[qid][:5] if pid in qrels[qid])
        lines.append(f"  Found : {passages[rel_pid]['text'][:200]}...")
        lines.append("")

    report = "\n".join(lines)
    print(report)

    out = save_path or str(RESULTS_DIR / "qualitative_analysis.txt")
    with open(out, "w") as f:
        f.write(report)
    print(f"Saved: {out}")
    return wins_dense, wins_bm25


def plot_metric_bars(
    df: pd.DataFrame,
    metrics: list[str],
    save_path: str | None = None,
):
    """Grouped bar chart comparing all pipelines on selected metrics."""
    n_metrics   = len(metrics)
    n_pipelines = len(df)
    x = np.arange(n_metrics)
    width = 0.8 / n_pipelines

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (_, row) in enumerate(df.iterrows()):
        vals = [row[m] for m in metrics]
        offset = (i - n_pipelines / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=row["pipeline"])

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Pipeline Comparison Across IR Metrics", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="y", which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()

    out = save_path or str(RESULTS_DIR / "metric_bars.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Per-metric line plots across k values
# ---------------------------------------------------------------------------

def _line_plot_at_k(
    df: pd.DataFrame,
    prefix: str,
    k_values: list[int],
    ylabel: str,
    title: str,
    filename: str,
):
    """Generic helper: one line per pipeline showing metric@k vs k."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for _, row in df.iterrows():
        scores = [row[f"{prefix}@{k}"] for k in k_values]
        ax.plot(k_values, scores, marker="o", linewidth=2, label=row["pipeline"])

    ax.set_xlabel("k (number of retrieved documents)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(k_values)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()

    out = str(RESULTS_DIR / filename)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_precision_at_k(df: pd.DataFrame, k_values: list[int], save_path: str | None = None):
    _line_plot_at_k(
        df, prefix="P", k_values=k_values,
        ylabel="Precision@k", title="Precision@k — All Pipelines",
        filename="precision_at_k.png",
    )


def plot_recall_at_k(df: pd.DataFrame, k_values: list[int], save_path: str | None = None):
    _line_plot_at_k(
        df, prefix="R", k_values=k_values,
        ylabel="Recall@k", title="Recall@k — All Pipelines",
        filename="recall_at_k.png",
    )


def plot_ndcg_at_k(df: pd.DataFrame, k_values: list[int], save_path: str | None = None):
    _line_plot_at_k(
        df, prefix="NDCG", k_values=k_values,
        ylabel="NDCG@k", title="NDCG@k — All Pipelines",
        filename="ndcg_at_k.png",
    )


# ---------------------------------------------------------------------------
# MRR bar chart
# ---------------------------------------------------------------------------

def plot_mrr(df: pd.DataFrame, save_path: str | None = None):
    """Horizontal bar chart of MRR per pipeline."""
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    pipelines = df["pipeline"].tolist()
    scores    = df["MRR"].tolist()

    bars = ax.barh(pipelines, scores, color=colors[: len(pipelines)], edgecolor="white")
    ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=9)
    ax.set_xlabel("MRR Score", fontsize=11)
    ax.set_title("Mean Reciprocal Rank (MRR) — All Pipelines", fontsize=13)
    ax.set_xlim(0, min(1.0, max(scores) * 1.25))
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="x", which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()

    out = save_path or str(RESULTS_DIR / "mrr.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Latency bar chart
# ---------------------------------------------------------------------------

def plot_latency(df: pd.DataFrame, save_path: str | None = None):
    """Horizontal bar chart of average query latency per pipeline."""
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    pipelines = df["pipeline"].tolist()
    latencies = df["avg_latency_ms"].tolist()

    bars = ax.barh(pipelines, latencies, color=colors[: len(pipelines)], edgecolor="white")
    ax.bar_label(bars, fmt="%.2f ms", padding=4, fontsize=9)
    ax.set_xlabel("Avg Query Latency (ms)", fontsize=11)
    ax.set_title("Query Latency — All Pipelines", fontsize=13)
    ax.set_xlim(0, max(latencies) * 1.25)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="x", which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()

    out = save_path or str(RESULTS_DIR / "latency.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")
