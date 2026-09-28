"""Answer generation: a small local LLM (llama.cpp GGUF) and an LLM-free extractive baseline.

Both return text with [n] markers that point at the numbered context passages, so every claim
can be traced to a document and page. parse_citations() validates the markers afterwards.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from dataclasses import dataclass, field

from . import config
from .retrieve import Hit, _reranker

ABSTAIN_TEXT = "I could not find this in the document library."

SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered context passages provided. "
    "After every claim, cite the passage it came from as [1], [2], etc. "
    "Never use outside knowledge. If the passages do not contain the answer, reply exactly: "
    f"{ABSTAIN_TEXT} "
    "Be concise: at most 4 sentences."
)

_CITE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")


def format_context(hits: list[Hit]) -> str:
    from .config import DOC_BY_ID

    blocks = []
    for i, h in enumerate(hits, start=1):
        d = DOC_BY_ID[h.doc_id]
        blocks.append(f"[{i}] ({d.short}, page {h.page})\n{h.text}")
    return "\n\n".join(blocks)


def build_messages(question: str, hits: list[Hit]) -> list[dict]:
    user = f"Context passages:\n\n{format_context(hits)}\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


@dataclass
class Parsed:
    text: str  # answer with invalid markers removed
    cited: list[int]  # valid context numbers (1-based), in order of first use
    invalid: list[int] = field(default_factory=list)
    abstained: bool = False


def parse_citations(text: str, n_contexts: int) -> Parsed:
    """Extract [n] markers, keep those that point at a real passage, drop the rest."""
    cited: list[int] = []
    invalid: list[int] = []

    def repl(m: re.Match) -> str:
        nums = [int(x) for x in re.split(r"[,;]", m.group(1))]
        good = [n for n in nums if 1 <= n <= n_contexts]
        invalid.extend(n for n in nums if not 1 <= n <= n_contexts)
        for n in good:
            if n not in cited:
                cited.append(n)
        return "".join(f"[{n}]" for n in good)

    cleaned = _CITE.sub(repl, text).strip()
    abstained = ABSTAIN_TEXT.lower().rstrip(".") in cleaned.lower()
    return Parsed(cleaned, cited, invalid, abstained)


class ExtractiveGenerator:
    """No LLM: return the 1-3 most question-relevant sentences from the retrieved passages."""

    name = "extractive"

    def stream(self, question: str, hits: list[Hit]) -> Iterator[str]:
        from .ingest import split_sentences

        cand: list[tuple[str, int]] = []
        for i, h in enumerate(hits, start=1):
            cand.extend((s, i) for s in split_sentences(h.text) if 40 <= len(s) <= 400)
        if not cand:
            yield ABSTAIN_TEXT
            return
        scores = list(_reranker().rerank(question, [s for s, _ in cand]))
        top = sorted(zip(cand, scores, strict=True), key=lambda t: -t[1])[:3]
        yield " ".join(f"{s} [{i}]" for (s, i), _ in top)


class LlamaCppGenerator:
    """Local GGUF model via llama-cpp-python (CPU). Loaded lazily, downloaded from the HF Hub once."""

    def __init__(self, repo: str = config.LLM_REPO, filename: str = config.LLM_FILE,
                 n_ctx: int = 3072, n_threads: int | None = None):
        self.repo, self.filename, self.n_ctx = repo, filename, n_ctx
        self.n_threads = n_threads or os.cpu_count() or 4
        self.name = filename.split("-instruct")[0]
        self._llm = None

    def _load(self):
        if self._llm is None:
            from huggingface_hub import hf_hub_download
            from llama_cpp import Llama

            path = hf_hub_download(self.repo, self.filename)
            self._llm = Llama(model_path=path, n_ctx=self.n_ctx, n_threads=self.n_threads, verbose=False)
        return self._llm

    def stream(self, question: str, hits: list[Hit]) -> Iterator[str]:
        llm = self._load()
        for chunk in llm.create_chat_completion(
            messages=build_messages(question, hits), max_tokens=260, temperature=0.0, stream=True
        ):
            delta = chunk["choices"][0]["delta"].get("content")
            if delta:
                yield delta
