"""The end-to-end pipeline: retrieve -> (abstain if the evidence is weak) -> generate -> validate citations."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from . import config
from .attribute import attribute
from .generate import ABSTAIN_TEXT, ExtractiveGenerator, Parsed, parse_citations
from .retrieve import Hit, Retriever

# Cross-encoder score below which the top passage is judged not to answer the question.
# Chosen on the dev split of the evaluation set (see run_eval.py), not on the test split.
DEFAULT_ABSTAIN_THRESHOLD = -3.0


@dataclass
class Answer:
    question: str
    text: str
    cited: list[int]
    contexts: list[Hit]
    abstained: bool
    reason: str = ""  # "low_retrieval_score" | "model_abstained" | ""
    invalid_citations: list[int] = field(default_factory=list)
    top_score: float = 0.0
    seconds: float = 0.0
    attribution: str = "model"  # "model": the LLM wrote the [n] markers | "post-hoc": assigned afterwards

    @property
    def sources(self) -> list[Hit]:
        return [self.contexts[n - 1] for n in self.cited]


def finalize(raw: str, hits: list[Hit]) -> tuple[Parsed, str]:
    """Validate the model's citations; if it gave none, attribute each sentence to its best passage."""
    parsed = parse_citations(raw, len(hits))
    if parsed.abstained or parsed.cited:
        return parsed, "model"
    text, cited = attribute(parsed.text, hits)
    return Parsed(text, cited, parsed.invalid, False), "post-hoc"


class RAGPipeline:
    def __init__(self, retriever: Retriever, generator=None, mode: str = "hybrid+rerank",
                 abstain_threshold: float = DEFAULT_ABSTAIN_THRESHOLD, context_k: int = config.CONTEXT_K):
        self.retriever = retriever
        self.generator = generator or ExtractiveGenerator()
        self.mode, self.abstain_threshold, self.context_k = mode, abstain_threshold, context_k

    def retrieve(self, question: str) -> list[Hit]:
        return self.retriever.search(question, self.mode, self.context_k)

    def stream(self, question: str, hits: list[Hit] | None = None) -> Iterator[str]:
        hits = hits if hits is not None else self.retrieve(question)
        if not hits or hits[0].score < self.abstain_threshold:
            yield ABSTAIN_TEXT
            return
        yield from self.generator.stream(question, hits)

    def answer(self, question: str) -> Answer:
        t0 = time.time()
        hits = self.retrieve(question)
        top = hits[0].score if hits else float("-inf")
        if not hits or top < self.abstain_threshold:
            return Answer(question, ABSTAIN_TEXT, [], hits, True, "low_retrieval_score", [], top,
                          time.time() - t0)
        raw = "".join(self.generator.stream(question, hits))
        parsed, how = finalize(raw, hits)
        return Answer(question, parsed.text, parsed.cited, hits, parsed.abstained,
                      "model_abstained" if parsed.abstained else "", parsed.invalid, top,
                      time.time() - t0, how)
