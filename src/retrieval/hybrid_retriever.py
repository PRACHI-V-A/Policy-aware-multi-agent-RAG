"""
Hybrid policy retriever.

Pipeline:
    Query
      ↓
    BM25 (sparse lexical retrieval)
      +
    FAISS (dense semantic retrieval)
      ↓
    Reciprocal Rank Fusion (RRF)
      ↓
    Cross-Encoder reranking
      ↓
    Final policy evidence

All returned evidence keeps:
    source
    page
    section
    chunk_id
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHUNKS_PATH = PROJECT_ROOT / "data" / "policy_chunks.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class HybridRetriever:
    """BM25 + dense FAISS + RRF + CrossEncoder policy retriever."""

    # -------------------------------------------------------------
    # Shared retriever cache
    # -------------------------------------------------------------

    _instance: "HybridRetriever | None" = None

    @classmethod
    def get_instance(
        cls,
        chunks_path: Path = CHUNKS_PATH,
        embedding_model_name: str = EMBEDDING_MODEL,
        reranker_model_name: str = RERANKER_MODEL,
    ) -> "HybridRetriever":
        """
        Return one shared HybridRetriever instance.

        The expensive embedding model, FAISS index, BM25 index,
        and CrossEncoder are initialized only once per process.
        """

        if cls._instance is None:
            cls._instance = cls(
                chunks_path=chunks_path,
                embedding_model_name=embedding_model_name,
                reranker_model_name=reranker_model_name,
            )

        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """
        Reset the shared retriever.

        Mainly useful for testing or when the policy index changes.
        """

        cls._instance = None

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

        # ---------------------------------------------------------
        # 1. Load policy chunks
        # ---------------------------------------------------------

        data = json.loads(
            chunks_path.read_text(encoding="utf-8")
        )

        self.chunks: list[dict[str, Any]] = data["chunks"]

        if not self.chunks:
            raise ValueError("No policy chunks found.")

        self.documents = [
            chunk["text"]
            for chunk in self.chunks
        ]

        # ---------------------------------------------------------
        # 2. BM25 sparse retrieval
        # ---------------------------------------------------------

        tokenized_documents = [
            self._tokenize(text)
            for text in self.documents
        ]

        self.bm25 = BM25Okapi(tokenized_documents)

        # ---------------------------------------------------------
        # 3. Dense embedding model
        # ---------------------------------------------------------

        print("Loading embedding model...")

        self.embedding_model = SentenceTransformer(
            embedding_model_name
        )

        embeddings = self.embedding_model.encode(
            self.documents,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        embeddings = embeddings.astype("float32")

        # FAISS inner product + normalized embeddings
        # = cosine similarity.
        dimension = embeddings.shape[1]

        self.faiss_index = faiss.IndexFlatIP(dimension)

        self.faiss_index.add(embeddings)

        # ---------------------------------------------------------
        # 4. CrossEncoder reranker
        # ---------------------------------------------------------

        print("Loading reranker...")

        self.reranker = CrossEncoder(
            reranker_model_name
        )

        print(
            f"Retriever ready with {len(self.chunks)} policy chunks."
        )

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
    # BM25 retrieval
    # =============================================================

    def _bm25_search(
        self,
        query: str,
        top_k: int,
    ) -> list[int]:

        query_tokens = self._tokenize(query)

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = np.argsort(scores)[::-1]

        return [
            int(index)
            for index in ranked_indices[:top_k]
        ]

    # =============================================================
    # Dense retrieval
    # =============================================================

    def _dense_search(
        self,
        query: str,
        top_k: int,
    ) -> list[int]:

        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = self.faiss_index.search(
            query_embedding,
            top_k,
        )

        return [
            int(index)
            for index in indices[0]
            if index >= 0
        ]

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

        # BM25 contribution
        for rank, index in enumerate(
            bm25_results,
            start=1,
        ):
            scores[index] = scores.get(index, 0.0) + (
                1.0 / (k + rank)
            )

        # Dense contribution
        for rank, index in enumerate(
            dense_results,
            start=1,
        ):
            scores[index] = scores.get(index, 0.0) + (
                1.0 / (k + rank)
            )

        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return ranked

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

        # ---------------------------------------------
        # Sparse retrieval
        # ---------------------------------------------

        bm25_results = self._bm25_search(
            query,
            retrieval_k,
        )

        # ---------------------------------------------
        # Dense retrieval
        # ---------------------------------------------

        dense_results = self._dense_search(
            query,
            retrieval_k,
        )

        # ---------------------------------------------
        # RRF
        # ---------------------------------------------

        fused_results = self._rrf_fusion(
            bm25_results,
            dense_results,
        )

        # Take candidates for reranking.
        candidate_indices = [
            index
            for index, _ in fused_results[:retrieval_k]
        ]

        if not candidate_indices:
            return []

        # ---------------------------------------------
        # CrossEncoder reranking
        # ---------------------------------------------

        pairs = [
            (
                query,
                self.documents[index],
            )
            for index in candidate_indices
        ]

        reranker_scores = self.reranker.predict(
            pairs
        )

        reranked = sorted(
            zip(candidate_indices, reranker_scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )

        # ---------------------------------------------
        # Build evidence objects
        # ---------------------------------------------

        results: list[dict[str, Any]] = []

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


# ================================================================
# Simple manual testing
# ================================================================

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

        print(
            f"Chunk: {result['chunk_id']}"
        )

        print(
            f"Rerank score: "
            f"{result['rerank_score']:.4f}"
        )

        print(
            f"Text: {result['text'][:500]}..."
        )


if __name__ == "__main__":

    retriever = HybridRetriever.get_instance()

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