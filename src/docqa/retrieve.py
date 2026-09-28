"""Retrieval: BM25 (lexical) + dense (bge-small embeddings in FAISS) + reciprocal-rank fusion,
optionally followed by a cross-encoder rerank. Every hit carries its document and page.

Build the index once:  python -m docqa.retrieve   (writes artifacts/embeddings.npy)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from . import config

_TOKEN = re.compile(r"[a-z0-9][a-z0-9\-']*")
_STOP = frozenset("a an the of to in and or for on with by is are be as at it its this that from "
                  "which what how when who does do can should".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    page: int
    page_label: str
    text: str
    score: float
    rank: int = 0


@lru_cache(maxsize=1)
def _embedder():
    from fastembed import TextEmbedding

    return TextEmbedding(config.EMBED_MODEL)


@lru_cache(maxsize=1)
def _reranker():
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(config.RERANK_MODEL)


def embed_passages(texts: list[str]) -> np.ndarray:
    vecs = np.asarray(list(_embedder().embed(texts, batch_size=64)), dtype="float32")
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def embed_query(query: str) -> np.ndarray:
    v = np.asarray(list(_embedder().embed([config.BGE_QUERY_PREFIX + query])), dtype="float32")
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal-rank fusion: robust way to merge lexical and dense rankings without tuning scales."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for pos, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + pos + 1)
    return sorted(scores.items(), key=lambda kv: -kv[1])


class Retriever:
    def __init__(self, chunks: pd.DataFrame, embeddings: np.ndarray | None):
        self.chunks = chunks.reset_index(drop=True)
        self.bm25 = BM25Okapi([tokenize(t) for t in self.chunks.text])
        self.index = None
        if embeddings is not None:
            self.index = faiss.IndexFlatIP(embeddings.shape[1])  # exact inner product = cosine (unit vectors)
            self.index.add(embeddings)

    @classmethod
    def load(cls, artifacts=config.ARTIFACTS) -> Retriever:
        chunks = pd.read_parquet(artifacts / "chunks.parquet")
        emb_path = artifacts / "embeddings.npy"
        return cls(chunks, np.load(emb_path) if emb_path.exists() else None)

    # -- single retrievers -------------------------------------------------------------
    def _bm25_rank(self, query: str, k: int) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        return [int(i) for i in np.argsort(-scores)[:k]]

    def _dense_rank(self, query: str, k: int) -> list[int]:
        _, ids = self.index.search(embed_query(query), k)
        return [int(i) for i in ids[0] if i >= 0]  # faiss pads with -1 when k > corpus size

    def _hit(self, idx: int, score: float, rank: int) -> Hit:
        r = self.chunks.iloc[idx]
        return Hit(r.chunk_id, r.doc_id, int(r.page), str(r.page_label), r.text, float(score), rank)

    # -- public ------------------------------------------------------------------------
    def search(self, query: str, mode: str = "hybrid+rerank", k: int = 10) -> list[Hit]:
        """mode: bm25 | dense | hybrid | hybrid+rerank"""
        n = config.RETRIEVE_K
        if mode == "bm25":
            ids = self._bm25_rank(query, k)
            return [self._hit(i, 0.0, r + 1) for r, i in enumerate(ids)]
        if mode == "dense":
            ids = self._dense_rank(query, k)
            return [self._hit(i, 0.0, r + 1) for r, i in enumerate(ids)]
        fused = rrf([self._bm25_rank(query, n), self._dense_rank(query, n)])
        if mode == "hybrid":
            return [self._hit(i, s, r + 1) for r, (i, s) in enumerate(fused[:k])]
        if mode == "hybrid+rerank":
            cand = [i for i, _ in fused[: config.RERANK_TOP]]
            scores = list(_reranker().rerank(query, [self.chunks.text.iloc[i] for i in cand]))
            order = sorted(zip(cand, scores, strict=True), key=lambda t: -t[1])[:k]
            return [self._hit(i, s, r + 1) for r, (i, s) in enumerate(order)]
        raise ValueError(f"unknown mode {mode!r}")


def build_embeddings() -> None:
    chunks = pd.read_parquet(config.ARTIFACTS / "chunks.parquet")
    emb = embed_passages(chunks.text.tolist())
    np.save(config.ARTIFACTS / "embeddings.npy", emb)
    print(f"embedded {len(chunks)} chunks -> {emb.shape}")


if __name__ == "__main__":
    build_embeddings()
