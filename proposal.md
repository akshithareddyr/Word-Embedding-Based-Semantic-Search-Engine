# Word Embedding-Based Semantic Search Engine


## Abstract

Traditional keyword-based search engines fail to capture the semantic intent behind a user's query, struggling with synonyms and contextual nuances. This project builds a semantic search system using dense word embeddings to improve document retrieval over a corpus with real relevance judgments. We implement and benchmark four retrieval pipelines — a sparse BM25 baseline, exact vector search, and two Approximate Nearest Neighbor (ANN) algorithms (IVFFlat and HNSW) — evaluating the trade-offs between retrieval quality and query latency using standard Information Retrieval metrics including Precision@k, Recall@k, MRR, and NDCG@k.

---

## 1. Introduction and Project Goal

Efficiently retrieving relevant documents from large corpora is a fundamental task in Information Retrieval (IR). Traditional keyword-based methods such as TF-IDF and BM25 rely on exact term matching, which frequently misses the semantic meaning or broader intent behind a user's query.

Our primary goal is to build a modern semantic search system using deep learning-based text embeddings, rigorously evaluated against a dataset with genuine relevance judgments. Specifically, our objectives are to:

- Generate dense semantic embeddings for a large passage corpus using a pre-trained Sentence Transformer model.
- Implement four distinct retrieval pipelines — from a classical sparse baseline to highly optimized ANN indexing — to find the optimal balance between query latency and retrieval accuracy.
- Benchmark these methods using standard IR metrics on a dataset with real human-annotated relevance labels, eliminating the need for a weak category-match proxy.

---

## 2. Dataset: MS MARCO Passage Retrieval

We use the **MS MARCO Passage Retrieval** dataset, the de-facto standard benchmark for dense passage retrieval research.

| Property | Value |
|---|---|
| Corpus size | ~8.8 million passages (web documents) |
| Query set | ~6,980 development queries with relevance labels |
| Relevance judgments | Human-annotated (real Microsoft Bing search logs) |
| Avg. passage length | ~60 words |

**Why MS MARCO over AG News:**
- AG News has only 4 broad topic categories, making the category-match relevance proxy too coarse (e.g., a query about "football injury" and a document about "cricket trade" both score as "relevant" under the Sports label).
- MS MARCO provides actual query-passage relevance judgments from human annotators, enabling meaningful Precision, Recall, MRR, and NDCG computation.
- MS MARCO is the standard benchmark for Sentence-BERT and FAISS-based retrieval; our results will be directly comparable to published literature.
- For computational tractability in a class project setting, we work with a **100K-passage subset** plus all 6,980 dev queries.

---

## 3. Proposed Methodology

### 3.1 Preprocessing

Before generating embeddings, we clean the passage and query text by:
- Converting to lowercase
- Removing punctuation and excess whitespace
- Truncating passages to a maximum of 256 word-piece tokens (SBERT's effective range)

### 3.2 Embedding Generation

We use **Sentence-BERT** with the `all-MiniLM-L6-v2` architecture to encode all passages and queries into a shared 384-dimensional dense vector space. Geometric proximity in this space correlates with semantic similarity. Embeddings are L2-normalized before indexing so that inner product search is equivalent to cosine similarity.

### 3.3 Retrieval Pipelines

We implement four retrieval methods to cover the full spectrum from classical sparse retrieval to optimized ANN search:

| Pipeline | Method | Library | Description |
|---|---|---|---|
| **BM25** | Sparse keyword baseline | `rank_bm25` | Classic probabilistic term-matching; strong traditional baseline |
| **Exact Search** | Brute-force cosine | `FAISS IndexFlatIP` | Exhaustive inner product; upper-bound accuracy, slow at scale |
| **IVFFlat** | Inverted file ANN | `FAISS IndexIVFFlat` | Partitions space into clusters; searches only nearby clusters (`nprobe` tuned) |
| **HNSW** | Graph-based ANN | `FAISS IndexHNSWFlat` | Layered proximity graph; sub-millisecond search (`M` and `ef_search` tuned) |

BM25 replaces the previously proposed MinHash/SimHash hashing baselines, which estimated Jaccard and cosine similarity over sparse token sets and are inappropriate for comparing against dense semantic embeddings.

---

## 4. Evaluation Strategy

### 4.1 Ground Truth

We use the **official MS MARCO dev set relevance judgments** (qrels). Each query has one or more annotated relevant passages. This eliminates the weak category-proxy approach and enables precise IR metric computation.

### 4.2 Evaluation Metrics

We evaluate all four pipelines at k ∈ {1, 3, 5, 10}:

| Metric | Description |
|---|---|
| **Precision@k** | Fraction of top-k retrieved documents that are relevant |
| **Recall@k** | Fraction of all relevant documents that appear in the top-k results |
| **MRR (Mean Reciprocal Rank)** | Average of reciprocal rank of the first relevant document |
| **NDCG@k** | Normalized Discounted Cumulative Gain; measures ranking quality with binary relevance |
| **Query Latency** | Average wall-clock search time per query (ms), benchmarked over all dev queries |

### 4.3 Experimental Design

- All dense pipelines (Exact, IVFFlat, HNSW) use identical `all-MiniLM-L6-v2` embeddings, isolating the effect of the indexing algorithm on quality and speed.
- IVFFlat: sweep `nlist` ∈ {64, 128, 256} and `nprobe` ∈ {4, 8, 16, 32}.
- HNSW: sweep `M` ∈ {16, 32, 64} and `ef_search` ∈ {32, 64, 128}.
- Report a **speed vs. accuracy tradeoff curve** (Recall@10 vs. latency) for ANN methods.

---

## 5. Deliverables

1. **Codebase** — modular Python implementation with separate modules for data loading, embedding, indexing, and evaluation.
2. **Results table** — Precision@k, Recall@k, MRR, NDCG@k, and latency for all four pipelines.
3. **Tradeoff analysis** — latency vs. Recall@10 curves for IVFFlat and HNSW parameter sweeps.
4. **Qualitative analysis** — example queries where dense retrieval outperforms BM25 (synonym/paraphrase cases) and vice versa (exact keyword match cases).
5. **Embedding visualization** — t-SNE/UMAP plot of a passage embedding sample to illustrate semantic clustering.

---

## 6. References

1. N. Reimers and I. Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," in *Proceedings of EMNLP*, 2019.
2. T. Nguyen et al., "MS MARCO: A Human Generated MAchine Reading COmprehension Dataset," in *NeurIPS Workshop on Reasoning, Attention, Memory*, 2016.
3. J. Johnson, M. Douze, and H. Jégou, "Billion-scale similarity search with GPUs," *IEEE Transactions on Big Data*, 2021.
4. S. E. Robertson and S. Walker, "Some simple effective approximations to the 2-Poisson model for probabilistic weighted retrieval," in *Proceedings of SIGIR*, 1994. *(BM25 foundation)*
5. Y. Malkov and D. Yashunin, "Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs," *IEEE TPAMI*, 2020.
