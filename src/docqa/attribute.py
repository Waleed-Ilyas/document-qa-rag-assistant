"""Post-hoc attribution: cite the passage that best supports each answer sentence.

Small local models often answer correctly but ignore the "[n]" citation instruction (measured in
the README: Qwen2.5 0.5B/1.5B cited on ~3% of answers). So when the model gives no valid marker we
assign citations ourselves: score every (answer sentence, retrieved passage) pair with the
cross-encoder and attach the best passage(s). This tells the reader *where to verify* a sentence;
it is a support heuristic, not proof that the model derived the sentence from that passage.
"""

from __future__ import annotations

from .generate import ABSTAIN_TEXT
from .ingest import split_sentences
from .retrieve import Hit, _reranker

MIN_SUPPORT = -4.0  # below this cross-encoder score a passage is not credited for the sentence
SECOND_MARGIN = 1.5  # also credit a runner-up passage scoring within this margin of the best


def attribute(text: str, hits: list[Hit]) -> tuple[str, list[int]]:
    """Return (text with [n] markers appended per sentence, ordered list of cited passage numbers)."""
    if not text.strip() or text.strip().startswith(ABSTAIN_TEXT.rstrip(".")) or not hits:
        return text, []
    sentences = split_sentences(text) or [text]
    rr = _reranker()
    cited: list[int] = []
    out: list[str] = []
    for s in sentences:
        scores = list(rr.rerank(s, [h.text for h in hits]))
        order = sorted(range(len(hits)), key=lambda i: -scores[i])
        picks = [order[0]] if scores[order[0]] >= MIN_SUPPORT else []
        if picks and len(order) > 1 and scores[order[1]] >= scores[order[0]] - SECOND_MARGIN \
                and scores[order[1]] >= MIN_SUPPORT:
            picks.append(order[1])
        marks = "".join(f"[{i + 1}]" for i in sorted(picks))
        for i in picks:
            if i + 1 not in cited:
                cited.append(i + 1)
        out.append(f"{s} {marks}".strip())
    return " ".join(out), cited
