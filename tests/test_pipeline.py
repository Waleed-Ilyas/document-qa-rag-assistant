"""Offline tests: no PDFs, models or network needed (a hash-based fake embedder stands in for bge)."""

import hashlib

import numpy as np
import pandas as pd
import pytest

from docqa import evaluate, rag, retrieve
from docqa.generate import ABSTAIN_TEXT, ExtractiveGenerator, build_messages, parse_citations
from docqa.ingest import chunk_page, clean_text, split_sentences, strip_running_lines
from docqa.retrieve import Hit, Retriever, rrf, tokenize


# ---------------------------------------------------------------- ingestion
def test_clean_text_repairs_ligatures_and_hyphenation():
    assert clean_text("eﬀective risk man-\nagement") == "effective risk management"


def test_strip_running_lines_removes_headers_but_keeps_body():
    pages = [f"NIST AI 100-1 AI RMF 1.0\nBody text number {i} is unique.\n{i}" for i in range(10)]
    out = strip_running_lines(pages)
    assert all("NIST AI 100-1" not in p for p in out)
    assert all(f"Body text number {i}" in p for i, p in enumerate(out))


def test_split_sentences_handles_bullets_and_abbreviation_free_text():
    text = "First sentence here. Second one follows.\n\n- A bullet item.\n- Another bullet."
    assert len(split_sentences(text)) >= 3


def test_chunk_page_respects_max_size_and_overlaps():
    sents = [f"This is sentence number {i} with some filler words to add length." for i in range(60)]
    chunks = chunk_page(" ".join(sents))
    assert len(chunks) > 3
    assert max(len(c) for c in chunks) < 1500
    # one-sentence overlap: the last sentence of a chunk starts the next
    assert chunks[0].split(". ")[-1][:20] in chunks[1]


# ---------------------------------------------------------------- retrieval
def _fake_embed(texts):
    out = []
    for t in texts:
        v = np.zeros(1024, dtype="float32")
        for w in tokenize(t):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 1024] += 1.0
        out.append(v / (np.linalg.norm(v) or 1.0))
    return np.stack(out)


@pytest.fixture()
def retriever(monkeypatch):
    monkeypatch.setattr(retrieve, "embed_passages", _fake_embed)
    monkeypatch.setattr(retrieve, "embed_query", lambda q: _fake_embed([q]))
    filler = [f"unrelated filler passage {i} about procurement budgets and office logistics" for i in range(8)]
    chunks = pd.DataFrame({
        "chunk_id": ["a:1:0", "a:2:0", "b:1:0", "b:2:0"] + [f"f:{i}:0" for i in range(8)],
        "doc_id": ["ai-rmf", "ai-rmf", "csf", "csf"] + ["csf"] * 8,
        "page": [1, 2, 1, 2] + list(range(10, 18)), "page_label": ["1", "2", "1", "2"] + ["x"] * 8,
        "text": ["the framework has four functions govern map measure manage",
                 "trustworthy systems are valid reliable safe secure",
                 "cybersecurity functions govern identify protect detect respond recover",
                 "tiers describe partial risk informed repeatable adaptive"] + filler,
    })
    return Retriever(chunks, _fake_embed(chunks.text.tolist()))


def test_rrf_prefers_items_ranked_well_by_both():
    fused = rrf([[1, 2, 3], [3, 1, 2]])
    assert fused[0][0] == 1  # ranked 1st and 2nd beats 3rd/1st on the sum of reciprocal ranks


def test_hybrid_search_returns_the_relevant_page(retriever):
    for mode in ["bm25", "dense", "hybrid"]:
        hits = retriever.search("which functions do govern map measure manage", mode, 2)
        assert (hits[0].doc_id, hits[0].page) == ("ai-rmf", 1), mode


# ---------------------------------------------------------------- generation helpers
def test_parse_citations_keeps_valid_and_drops_invalid():
    p = parse_citations("Govern is one function [1]. Also see [2, 7] and [9].", 3)
    assert p.cited == [1, 2]
    assert p.invalid == [7, 9]
    assert "[7]" not in p.text and "[9]" not in p.text


def test_parse_citations_detects_abstention():
    assert parse_citations(ABSTAIN_TEXT, 3).abstained
    assert not parse_citations("The answer is four [1].", 3).abstained


def test_build_messages_numbers_passages_and_names_the_document():
    hit = Hit("c", "csf", 8, "8", "GOVERN is a function.", 1.0)
    msgs = build_messages("What is GOVERN?", [hit])
    assert "[1] (CSF 2.0, page 8)" in msgs[1]["content"]
    assert "ONLY" in msgs[0]["content"]


class _FixedRetriever:
    def __init__(self, hits):
        self.hits = hits

    def search(self, q, mode, k):
        return self.hits[:k]


class _EchoGen:
    name = "echo"

    def stream(self, q, hits):
        yield "GOVERN is the first function [1]."


def test_pipeline_abstains_when_retrieval_is_weak():
    weak = [Hit("c", "csf", 8, "8", "unrelated", -9.0)]
    pipe = rag.RAGPipeline(_FixedRetriever(weak), _EchoGen(), abstain_threshold=-3.0)
    ans = pipe.answer("something not in the library")
    assert ans.abstained and ans.reason == "low_retrieval_score" and ans.text == ABSTAIN_TEXT


def test_pipeline_answers_and_maps_citation_to_source_page():
    strong = [Hit("c", "csf", 8, "8", "GOVERN is a function.", 5.0)]
    ans = rag.RAGPipeline(_FixedRetriever(strong), _EchoGen()).answer("What is GOVERN?")
    assert not ans.abstained and ans.cited == [1]
    assert (ans.sources[0].doc_id, ans.sources[0].page) == ("csf", 8)


def test_extractive_generator_cites_a_passage(monkeypatch):
    from docqa import generate

    class R:
        def rerank(self, q, docs):
            return [float(len(d)) for d in docs]

    monkeypatch.setattr(generate, "_reranker", lambda: R())
    hits = [Hit("c", "csf", 8, "8", "The GOVERN Function provides outcomes to inform strategy. "
                "Detect helps find attacks and compromises early.", 1.0)]
    out = "".join(ExtractiveGenerator().stream("What does GOVERN provide?", hits))
    assert "[1]" in out


# ---------------------------------------------------------------- evaluation metrics
def test_coverage_counts_groups_not_alternatives():
    groups = [["govern"], ["map", "mapping"], ["manage"]]
    assert evaluate.coverage("Govern and Mapping", groups) == pytest.approx(2 / 3)


def test_wilson_interval_is_sane():
    lo, hi = evaluate.wilson(8, 10)
    assert 0.45 < lo < 0.8 < hi < 0.97
    assert evaluate.wilson(0, 0) == (0.0, 0.0)


def test_tune_threshold_separates_the_classes_and_sits_inside_the_plateau():
    t = evaluate.tune_threshold([5.0, 4.0, 3.0], [-6.0, -5.0])
    assert -5.0 < t <= 3.0
    assert all(s >= t for s in [5.0, 4.0, 3.0]) and all(s < t for s in [-6.0, -5.0])


def test_retrieval_metrics_hit_and_sufficiency():
    q = {"doc": "csf", "gold_pages": [8], "must_include": [["govern"], ["identify"]]}
    hits = [Hit("x", "csf", 3, "3", "unrelated", 1.0), Hit("y", "csf", 8, "8", "GOVERN and IDENTIFY", 0.9)]
    m = evaluate.retrieval_metrics([(q, hits)], ks=(1, 2))
    assert m["hit@1"] == 0 and m["hit@2"] == 1 and m["suff@2"] == 1 and m["mrr@10"] == 0.5


# ---------------------------------------------------------------- post-hoc attribution
def test_attribute_credits_the_best_passage_per_sentence(monkeypatch):
    from docqa import attribute as attr

    class R:  # scores a passage by how many words it shares with the sentence
        def rerank(self, q, docs):
            qs = set(q.lower().split())
            return [float(len(qs & set(d.lower().split()))) for d in docs]

    monkeypatch.setattr(attr, "_reranker", lambda: R())
    monkeypatch.setattr(attr, "SECOND_MARGIN", 0.5)
    hits = [Hit("a", "csf", 1, "1", "govern identify protect detect respond recover", 0.0),
            Hit("b", "ssdf", 2, "2", "prepare protect produce respond software groups", 0.0)]
    text, cited = attr.attribute("The SSDF has four groups of software practices. It helps prepare teams.", hits)
    assert cited == [2]
    assert text.count("[2]") == 2 and "[1]" not in text


def test_attribute_leaves_abstentions_alone():
    from docqa.attribute import attribute

    text, cited = attribute(ABSTAIN_TEXT, [Hit("a", "csf", 1, "1", "x", 0.0)])
    assert text == ABSTAIN_TEXT and cited == []


def test_finalize_keeps_model_citations_and_falls_back_to_posthoc(monkeypatch):
    from docqa import rag

    hits = [Hit("a", "csf", 1, "1", "GOVERN is a function", 5.0)]
    parsed, how = rag.finalize("GOVERN is a function [1].", hits)
    assert how == "model" and parsed.cited == [1]
    monkeypatch.setattr(rag, "attribute", lambda text, h: (text + " [1]", [1]))
    parsed, how = rag.finalize("GOVERN is a function.", hits)
    assert how == "post-hoc" and parsed.cited == [1]
