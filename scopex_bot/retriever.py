"""Hybrid retrieval: keyword (TF-IDF) + semantic (fastembed), merged with
reciprocal rank fusion.

Keyword search is good at exact terms like "IFSC", "NRE" or "SEPA".
Semantic search is good at paraphrases like "my money didn't reach my mom".
Fusing both ranks gives better results than either one alone.

If the embedding model can't be loaded (no internet on first run, etc.)
the retriever falls back to keyword search only.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .knowledge import Article

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RRF_K = 60  # standard constant from the RRF paper


@dataclass
class Hit:
    article: Article
    score: float          # fused RRF score, only meaningful for ranking
    keyword_sim: float    # cosine similarity from TF-IDF
    semantic_sim: float | None  # cosine similarity from embeddings (None if disabled)


class HybridRetriever:
    def __init__(self, articles: list[Article], use_embeddings: bool = True):
        self.articles = articles
        texts = [a.text for a in articles]

        self.tfidf = TfidfVectorizer(
            ngram_range=(1, 2), sublinear_tf=True, stop_words="english"
        )
        self.tfidf_matrix = self.tfidf.fit_transform(texts)

        self.embedder = None
        self.doc_vectors = None
        if use_embeddings:
            try:
                from fastembed import TextEmbedding

                self.embedder = TextEmbedding(model_name=EMBED_MODEL)
                self.doc_vectors = self._normalize(
                    np.array(list(self.embedder.passage_embed(texts)))
                )
            except Exception as err:  # model download failed, offline, etc.
                print(f"[retriever] embeddings disabled, using keyword search only: {err}")
                self.embedder = None

    @property
    def mode(self) -> str:
        return "hybrid" if self.embedder else "keyword"

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(v, axis=-1, keepdims=True)
        return v / np.clip(norms, 1e-12, None)

    def _keyword_scores(self, query: str) -> np.ndarray:
        q = self.tfidf.transform([query])
        # TF-IDF rows are already L2-normalised, so dot product = cosine
        return (self.tfidf_matrix @ q.T).toarray().ravel()

    def _semantic_scores(self, query: str) -> np.ndarray | None:
        if not self.embedder:
            return None
        q = self._normalize(np.array(list(self.embedder.query_embed([query])))[0])
        return self.doc_vectors @ q

    def search(self, query: str, k: int = 4) -> list[Hit]:
        kw = self._keyword_scores(query)
        sem = self._semantic_scores(query)

        rankings = [np.argsort(-kw)]
        if sem is not None:
            rankings.append(np.argsort(-sem))

        fused = np.zeros(len(self.articles))
        for ranking in rankings:
            for rank, idx in enumerate(ranking):
                fused[idx] += 1.0 / (RRF_K + rank + 1)

        top = np.argsort(-fused)[:k]
        return [
            Hit(
                article=self.articles[i],
                score=float(fused[i]),
                keyword_sim=float(kw[i]),
                semantic_sim=float(sem[i]) if sem is not None else None,
            )
            for i in top
        ]
