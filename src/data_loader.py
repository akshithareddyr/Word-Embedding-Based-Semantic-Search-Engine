"""
Load and preprocess MS MARCO passage retrieval dataset.

Downloads via HuggingFace datasets. On first run this will fetch ~1GB;
subsequent runs reuse the local cache.
"""

import re
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


def _clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_msmarco(max_passages: int = 100_000, force_reload: bool = False):
    """
    Returns:
        passages  : dict  pid -> {"text": str, "cleaned": str}
        queries   : dict  qid -> {"text": str, "cleaned": str}
        qrels     : dict  qid -> set of relevant pids
    """
    passages_path = PROCESSED_DIR / f"passages_{max_passages}.json"
    queries_path  = PROCESSED_DIR / "queries.json"
    qrels_path    = PROCESSED_DIR / "qrels.json"

    if not force_reload and passages_path.exists() and queries_path.exists() and qrels_path.exists():
        print("Loading preprocessed data from cache...")
        with open(passages_path) as f:
            passages = json.load(f)
        with open(queries_path) as f:
            queries = json.load(f)
        with open(qrels_path) as f:
            qrels_raw = json.load(f)
        qrels = {qid: set(pids) for qid, pids in qrels_raw.items()}
        print(f"  Passages: {len(passages):,}  |  Queries: {len(queries):,}  |  Qrels: {len(qrels):,}")
        return passages, queries, qrels

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # --- passages ---
    # ms_marco v2.1 on HuggingFace has no global passage_id field.
    # Passages are local lists per query row, so we build a global ID as
    # "{query_id}_{passage_index}" to guarantee uniqueness.
    print("Downloading MS MARCO corpus (this may take a few minutes)...")
    corpus_ds = load_dataset("ms_marco", "v2.1", split="train")

    passages = {}
    seen = 0
    for row in tqdm(corpus_ds, desc="Building passage index"):
        qid = str(row["query_id"])
        for idx, ptext in enumerate(row["passages"]["passage_text"]):
            pid_str = f"{qid}_{idx}"
            if pid_str not in passages:
                passages[pid_str] = {"text": ptext, "cleaned": _clean(ptext)}
                seen += 1
            if seen >= max_passages:
                break
        if seen >= max_passages:
            break

    # --- queries + qrels ---
    print("Building query index and relevance judgments...")
    queries = {}
    qrels: dict[str, set] = {}

    for split in ("train", "validation"):
        ds = load_dataset("ms_marco", "v2.1", split=split)
        for row in tqdm(ds, desc=f"  {split}"):
            qid = str(row["query_id"])
            qtext = row["query"]
            queries[qid] = {"text": qtext, "cleaned": _clean(qtext)}

            relevant_pids = set()
            for idx, is_sel in enumerate(row["passages"]["is_selected"]):
                pid_str = f"{qid}_{idx}"
                if is_sel == 1 and pid_str in passages:
                    relevant_pids.add(pid_str)
            if relevant_pids:
                qrels[qid] = relevant_pids

    # keep only queries that have at least one relevant passage in our corpus
    queries = {qid: v for qid, v in queries.items() if qid in qrels}
    print(f"  Queries with relevant passages in corpus: {len(queries):,}")

    # --- persist ---
    with open(passages_path, "w") as f:
        json.dump(passages, f)
    with open(queries_path, "w") as f:
        json.dump(queries, f)
    with open(qrels_path, "w") as f:
        json.dump({qid: list(pids) for qid, pids in qrels.items()}, f)

    print(f"Saved to {PROCESSED_DIR}")
    return passages, queries, qrels


if __name__ == "__main__":
    passages, queries, qrels = load_msmarco(max_passages=100_000)
    print(f"\nPassages : {len(passages):,}")
    print(f"Queries  : {len(queries):,}")
    print(f"Qrels    : {len(qrels):,}")
