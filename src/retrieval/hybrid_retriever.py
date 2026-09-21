"""
Hybrid policy retriever.

Local mode:
    BM25 + SentenceTransformer dense retrieval
    + Reciprocal Rank Fusion (RRF)
    + CrossEncoder reranking

Render/lightweight mode:
    BM25 + lightweight dense latent-vector retrieval
    + Reciprocal Rank Fusion (RRF)
    + deterministic score-based reranking

The lightweight mode avoids loading PyTorch/SentenceTransformer/CrossEncoder
at API startup, which keeps the service within Render's 512 MB free-instance
memory limit while preserving the hybrid-retrieval workflow.

All returned evidence keeps:
    source
    page
    section
    chunk_id
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = PROJECT_ROOT / "data" / "policy_chunks.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Render sets this through render.yaml.
DEPLOYMENT_MODE = os.getenv("DEPLOYMENT_MODE", "local").strip().lower()
LIGHTWEIGHT_MODE = DEPLOYMENT_MODE in {"lightweight", "render", "production"}


class HybridRetriever:
    """BM25 + dense retrieval + RRF + reranking."""

    _instance = None

    @classmethod
    def get_instance(
        cls,
        chunks_path: Path = CHUNKS_PATH,
        embedding_model_name: str = EMBEDDING_MODEL,
        reranker_model_name: str = RERANKER_MODEL,
    ):
        """Return a shared retriever instance."""
        if cls._instance is None:
            cls._instance = cls(
                chunks_path=chunks_path,
                embedding_model_name=embedding_model_name,
                reranker_model_name=reranker_model_name,
            )

        return cls._instance
    def __init__(
        self,
        chunks_path: Path = CHUNKS_PATH,
        embedding_model_name: str = EMBEDDING_MODEL,
        reranker_model_name: str = RERANKER_MODEL,
    ) -> None:
        if not chunks_path.exists():
            raise FileNotFoundError(
                f"Policy chunks not found: {chunks_path}\n"
                "Run policy ingestion first."
            )

        data = json.loads(chunks_path.read_text(encoding="utf-8"))
        self.chunks: list[dict[str, Any]] = data["chunks"]

        if not self.chunks:
            raise ValueError("No policy chunks found.")

        self.documents = [chunk["text"] for chunk in self.chunks]

        # ---------------------------------------------------------
        # 1. BM25 sparse retrieval
        # ---------------------------------------------------------
        tokenized_documents = [
            self._tokenize(text) for text in self.documents
        ]
        self.bm25 = BM25Okapi(tokenized_documents)

        # ---------------------------------------------------------
        # 2. Retrieval backend
        # ---------------------------------------------------------
        if LIGHTWEIGHT_MODE:
            self._init_lightweight_dense()
        else:
            self._init_neural_dense(
                embedding_model_name=embedding_model_name,
                reranker_model_name=reranker_model_name,
            )

        print(
            f"Retriever ready with {len(self.chunks)} policy chunks "
            f"(mode={DEPLOYMENT_MODE})."
        )

    # =============================================================
    # Lightweight Render backend
    # =============================================================

    def _init_lightweight_dense(self) -> None:
        """
        Build a compact dense latent-vector index without PyTorch.

        TF-IDF captures lexical/phrase information. TruncatedSVD projects
        it into a small dense latent space, giving a lightweight vector
        retrieval stage suitable for a 512 MB deployment.
        """
        self.lightweight_vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=8000,
        )

        tfidf_matrix = self.lightweight_vectorizer.fit_transform(
            self.documents
        )

        # Keep the latent dimension small because the policy corpus is small.
        max_components = min(
            64,
            max(2, tfidf_matrix.shape[0] - 1),
            max(2, tfidf_matrix.shape[1] - 1),
        )

        self.lightweight_svd = TruncatedSVD(
            n_components=max_components,
            random_state=42,
        )

        dense_matrix = self.lightweight_svd.fit_transform(tfidf_matrix)

        # Unit-normalized dense vectors -> dot product behaves like cosine
        # similarity.
        self.lightweight_document_vectors = normalize(
            dense_matrix
        ).astype("float32")

        # Lightweight mode does not load neural models.
        self.embedding_model = None
        self.reranker = None

    # =============================================================
    # Full local neural backend
    # =============================================================

    def _init_neural_dense(
        self,
        embedding_model_name: str,
        reranker_model_name: str,
    ) -> None:
        # These imports are deliberately local so Render does not load
        # PyTorch/SentenceTransformers in lightweight deployment mode.
        import faiss
        from sentence_transformers import CrossEncoder, SentenceTransformer

        print("Loading embedding model...")

        self.embedding_model = SentenceTransformer(
            embedding_model_name
        )

        embeddings = self.embedding_model.encode(
            self.documents,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype("float32")

        dimension = embeddings.shape[1]

        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(embeddings)

        print("Loading reranker...")

        self.reranker = CrossEncoder(reranker_model_name)

    # =============================================================
    # Tokenization
    # =============================================================

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple lexical tokenizer for BM25."""
        return (
            text.lower()
            .replace("/", " ")
            .replace("-", " ")
            .split()
        )

    # =============================================================
    # BM25 sparse retrieval
    # =============================================================

    def _bm25_search(
        self,
        query: str,
        top_k: int,
    ) -> tuple[list[int], np.ndarray]:
        query_tokens = self._tokenize(query)
        scores = np.asarray(
            self.bm25.get_scores(query_tokens),
            dtype="float32",
        )

        ranked_indices = np.argsort(scores)[::-1]

        indices = [
            int(index)
            for index in ranked_indices[:top_k]
        ]

        return indices, scores

    # =============================================================
    # Dense retrieval
    # =============================================================

    def _dense_search(
        self,
        query: str,
        top_k: int,
    ) -> tuple[list[int], np.ndarray]:
        if LIGHTWEIGHT_MODE:
            query_tfidf = self.lightweight_vectorizer.transform([query])
            query_dense = self.lightweight_svd.transform(query_tfidf)
            query_dense = normalize(query_dense).astype("float32")

            scores = (
                self.lightweight_document_vectors @ query_dense.T
            ).ravel()

            ranked_indices = np.argsort(scores)[::-1]

            indices = [
                int(index)
                for index in ranked_indices[:top_k]
            ]

            selected_scores = np.asarray(
                [scores[index] for index in indices],
                dtype="float32",
            )

            return indices, selected_scores

        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = self.faiss_index.search(
            query_embedding,
            top_k,
        )

        dense_scores = scores[0]
        dense_indices = [
            int(index)
            for index in indices[0]
            if index >= 0
        ]

        return dense_indices, dense_scores

    # =============================================================
    # Reciprocal Rank Fusion
    # =============================================================

    @staticmethod
    def _rrf_fusion(
        bm25_results: list[int],
        dense_results: list[int],
        k: int = 60,
    ) -> list[tuple[int, float]]:
        scores: dict[int, float] = {}

        for rank, index in enumerate(bm25_results, start=1):
            scores[index] = scores.get(index, 0.0) + (
                1.0 / (k + rank)
            )

        for rank, index in enumerate(dense_results, start=1):
            scores[index] = scores.get(index, 0.0) + (
                1.0 / (k + rank)
            )

        return sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

    # =============================================================
    # Score helpers
    # =============================================================

    @staticmethod
    def _minmax(values: list[float]) -> dict[int, float]:
        if not values:
            return {}

        arr = np.asarray(values, dtype="float32")
        min_value = float(arr.min())
        max_value = float(arr.max())

        if max_value - min_value < 1e-8:
            return {i: 1.0 for i in range(len(values))}

        normalized = (arr - min_value) / (max_value - min_value)

        return {
            i: float(value)
            for i, value in enumerate(normalized)
        }

    # =============================================================
    # Complete retrieval
    # =============================================================

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        retrieval_k: int = 15,
    ) -> list[dict[str, Any]]:
        if not query.strip():
            return []

        # Sparse retrieval.
        bm25_results, bm25_scores = self._bm25_search(
            query,
            retrieval_k,
        )

        # Dense retrieval.
        dense_results, dense_scores = self._dense_search(
            query,
            retrieval_k,
        )

        # RRF fusion.
        fused_results = self._rrf_fusion(
            bm25_results,
            dense_results,
        )

        candidate_indices = [
            index
            for index, _ in fused_results[:retrieval_k]
        ]

        if not candidate_indices:
            return []

        if LIGHTWEIGHT_MODE:
            # -----------------------------------------------------
            # Lightweight deterministic reranking.
            # -----------------------------------------------------
            #
            # RRF decides the candidate pool. We then combine normalized
            # BM25 and dense-vector similarity to rerank those candidates.
            # This avoids loading a CrossEncoder model in the 512 MB
            # Render instance.
            bm25_lookup = {
                index: float(bm25_scores[index])
                for index in candidate_indices
            }

            dense_lookup = {
                index: float(
                    self._dense_score_for_index(
                        index,
                        dense_results,
                        dense_scores,
                    )
                )
                for index in candidate_indices
            }

            bm25_values = list(bm25_lookup.values())
            dense_values = list(dense_lookup.values())

            def normalize_score(
                value: float,
                values: list[float],
            ) -> float:
                if not values:
                    return 0.0

                low = min(values)
                high = max(values)

                if high - low < 1e-8:
                    return 1.0

                return (value - low) / (high - low)

            reranked = sorted(
                candidate_indices,
                key=lambda index: (
                    0.55 * normalize_score(
                        dense_lookup[index],
                        dense_values,
                    )
                    + 0.30 * normalize_score(
                        bm25_lookup[index],
                        bm25_values,
                    )
                    + 0.15 * self._rrf_score(
                        index,
                        fused_results,
                    )
                ),
                reverse=True,
            )

            results: list[dict[str, Any]] = []

            for rank, index in enumerate(
                reranked[:top_k],
                start=1,
            ):
                chunk = self.chunks[index]

                final_score = (
                    0.55 * normalize_score(
                        dense_lookup[index],
                        dense_values,
                    )
                    + 0.30 * normalize_score(
                        bm25_lookup[index],
                        bm25_values,
                    )
                    + 0.15 * self._rrf_score(
                        index,
                        fused_results,
                    )
                )

                results.append(
                    {
                        "rank": rank,
                        "chunk_id": chunk["chunk_id"],
                        "source": chunk["source"],
                        "page": chunk["page"],
                        "section": chunk["section"],
                        "text": chunk["text"],
                        "rerank_score": float(final_score),
                    }
                )

            return results

        # ---------------------------------------------------------
        # Full neural CrossEncoder reranking.
        # ---------------------------------------------------------
        pairs = [
            (
                query,
                self.documents[index],
            )
            for index in candidate_indices
        ]

        reranker_scores = self.reranker.predict(pairs)

        reranked = sorted(
            zip(candidate_indices, reranker_scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )

        results = []

        for rank, (index, rerank_score) in enumerate(
            reranked[:top_k],
            start=1,
        ):
            chunk = self.chunks[index]

            results.append(
                {
                    "rank": rank,
                    "chunk_id": chunk["chunk_id"],
                    "source": chunk["source"],
                    "page": chunk["page"],
                    "section": chunk["section"],
                    "text": chunk["text"],
                    "rerank_score": float(rerank_score),
                }
            )

        return results

    @staticmethod
    def _rrf_score(
        index: int,
        fused_results: list[tuple[int, float]],
    ) -> float:
        for _, (candidate_index, score) in enumerate(fused_results):
            if candidate_index == index:
                # RRF values are small; scale them into a stable 0-1
                # contribution for the lightweight reranker.
                return min(1.0, float(score) * 60.0)
        return 0.0

    @staticmethod
    def _dense_score_for_index(
        index: int,
        dense_results: list[int],
        dense_scores: np.ndarray,
    ) -> float:
        for position, result_index in enumerate(dense_results):
            if result_index == index:
                if position < len(dense_scores):
                    return float(dense_scores[position])
                break
        return 0.0


def print_results(
    retriever: HybridRetriever,
    query: str,
    top_k: int = 5,
) -> None:
    print("\n" + "=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)

    results = retriever.retrieve(
        query,
        top_k=top_k,
    )

    for result in results:
        print(
            f"\n[{result['rank']}] "
            f"Page {result['page']} | "
            f"{result['section']}"
        )
        print(f"Chunk: {result['chunk_id']}")
        print(
            f"Rerank score: "
            f"{result['rerank_score']:.4f}"
        )
        print(f"Text: {result['text'][:500]}...")


if __name__ == "__main__":
    retriever = HybridRetriever()

    test_queries = [
        "hospitalization minimum 24 hours",
        "room expenses limit",
        "cosmetic treatment exclusion",
        "pre hospitalization and post hospitalization expenses",
        "experimental unproven treatment",
    ]

    for query in test_queries:
        print_results(
            retriever,
            query,
            top_k=5,
        )
