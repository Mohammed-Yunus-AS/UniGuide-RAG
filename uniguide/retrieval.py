from __future__ import annotations

import os
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import settings


class Retriever:
    """Hybrid retrieval: persisted Gemini embeddings when available; TF-IDF fallback.

    Embeddings are stored with each chunk in the local JSON knowledge base. If a
    key/model is unavailable, the same interface falls back to TF-IDF so the
    application remains testable and usable without external embedding calls.
    """

    def __init__(self, chunks: list[dict[str, Any]], client=None):
        self.chunks = chunks
        self.client = client
        self.vectorizer: TfidfVectorizer | None = None
        self.tfidf_matrix = None
        self.mode = "TF-IDF"
        self._prepare_tfidf()

    def _prepare_tfidf(self) -> None:
        texts = [c.get("text", "") for c in self.chunks]
        if texts:
            self.vectorizer = TfidfVectorizer(
                lowercase=True, strip_accents="unicode", ngram_range=(1, 2),
                max_features=60000, sublinear_tf=True
            )
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)

    @staticmethod
    def _normalise_rows(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)

    def embed_texts(self, texts: list[str], task_type: str) -> list[list[float]] | None:
        if not self.client or not texts:
            return None
        try:
            response = self.client.models.embed_content(
                model=settings.embedding_model,
                contents=texts,
                config={"task_type": task_type, "output_dimensionality": 768},
            )
            return [list(item.values) for item in response.embeddings]
        except Exception:
            return None

    def add_embeddings(self, chunks: list[dict[str, Any]], batch_size: int = 64) -> tuple[int, str]:
        """Attach Gemini retrieval-document embeddings to chunks, when API is available."""
        if not self.client:
            return 0, "No Gemini client: retrieval will use TF-IDF."
        completed = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            vectors = self.embed_texts([c["text"] for c in batch], "RETRIEVAL_DOCUMENT")
            if vectors is None or len(vectors) != len(batch):
                for item in chunks:
                    item.pop("embedding", None)
                return 0, "Gemini embeddings unavailable: using TF-IDF retrieval."
            for item, vector in zip(batch, vectors):
                item["embedding"] = vector
                completed += 1
        return completed, f"Generated {completed} Gemini document embeddings."

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        if not query.strip() or not self.chunks:
            return []
        k = max(1, top_k or settings.top_k)

        # Use persisted Gemini vectors only if all indexed chunks have one and a
        # query embedding can be generated. Otherwise use lexical TF-IDF.
        if self.client and all(isinstance(c.get("embedding"), list) for c in self.chunks):
            query_vector = self.embed_texts([query], "RETRIEVAL_QUERY")
            if query_vector:
                try:
                    matrix = np.asarray([c["embedding"] for c in self.chunks], dtype=np.float32)
                    q = np.asarray(query_vector[0], dtype=np.float32).reshape(1, -1)
                    if matrix.shape[1] == q.shape[1]:
                        matrix = self._normalise_rows(matrix)
                        q = self._normalise_rows(q)
                        scores = (matrix @ q.T).reshape(-1)
                        self.mode = "Gemini Embeddings"
                        return self._rank(scores, k)
                except (ValueError, TypeError):
                    pass

        self.mode = "TF-IDF"
        if self.vectorizer is None or self.tfidf_matrix is None:
            return []
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.tfidf_matrix).ravel()
        return self._rank(scores, k)

    def _rank(self, scores: np.ndarray, k: int) -> list[dict[str, Any]]:
        order = np.argsort(scores)[::-1][:k]
        results = []
        for index in order:
            item = dict(self.chunks[int(index)])
            item["score"] = float(scores[int(index)])
            results.append(item)
        return results


def source_label(item: dict[str, Any]) -> str:
    return f"{item.get('source', 'Unknown document')}, page {item.get('page', '?')}"
