"""Metrics for the hand-verified evaluation set (eval/qa_set.json). Pure functions, unit-tested."""

from __future__ import annotations

import json
import math
from pathlib import Path

from . import config


def load_qa(path: Path | None = None) -> list[dict]:
    return json.loads((path or config.EVAL_DIR / "qa_set.json").read_text(encoding="utf-8"))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a proportion: honest error bars for a small evaluation set."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def is_gold(hit_doc: str, hit_page: int, q: dict) -> bool:
    return hit_doc == q["doc"] and hit_page in q["gold_pages"]


def retrieval_metrics(results: list[tuple[dict, list]], ks=(1, 3, 4, 5, 10)) -> dict:
    """results: (question, ranked hits) for answerable questions.

    hit@k   strict: a top-k chunk sits on one of the annotated gold pages (penalises a correct chunk
            that is on a different page that repeats the same fact)
    suff@k  context sufficiency: the union of the top-k chunks contains every required fact group,
            i.e. the LLM is handed enough to answer completely, whichever pages they come from
    """
    n = len(results)
    out: dict = {"n": n}
    for k in ks:
        got = sum(any(is_gold(h.doc_id, h.page, q) for h in hits[:k]) for q, hits in results)
        out[f"hit@{k}"] = got / n
        out[f"hit@{k}_ci"] = wilson(got, n)
        suf = sum(coverage(" ".join(h.text for h in hits[:k]), q["must_include"]) == 1.0 for q, hits in results)
        out[f"suff@{k}"] = suf / n
        out[f"suff@{k}_ci"] = wilson(suf, n)
    rr = []
    for q, hits in results:
        rank = next((i for i, h in enumerate(hits[:10], 1) if is_gold(h.doc_id, h.page, q)), None)
        rr.append(1 / rank if rank else 0.0)
    out["mrr@10"] = sum(rr) / n
    out["doc_hit@1"] = sum(bool(hits) and hits[0].doc_id == q["doc"] for q, hits in results) / n
    return out


def coverage(answer: str, must_include: list[list[str]]) -> float:
    """Fraction of required-fact groups present in the answer (case-insensitive substring match)."""
    a = answer.lower()
    return sum(any(alt.lower() in a for alt in grp) for grp in must_include) / len(must_include)


def answer_record(q: dict, ans) -> dict:
    """Per-question result row for one answering system."""
    rec = {"id": q["id"], "answerable": q["answerable"], "split": q["split"], "abstained": ans.abstained,
           "reason": ans.reason, "text": ans.text, "cited": ans.cited, "seconds": round(ans.seconds, 2),
           "top_score": round(ans.top_score, 3), "invalid_citations": ans.invalid_citations,
           "attribution": ans.attribution}
    if q["answerable"]:
        rec["coverage"] = coverage(ans.text, q["must_include"]) if not ans.abstained else 0.0
        rec["complete"] = rec["coverage"] == 1.0
        srcs = ans.sources
        rec["cited_gold"] = [is_gold(h.doc_id, h.page, q) for h in srcs]
        rec["gold_in_context"] = any(is_gold(h.doc_id, h.page, q) for h in ans.contexts)
    return rec


def summarise(records: list[dict]) -> dict:
    a = [r for r in records if r["answerable"]]
    u = [r for r in records if not r["answerable"]]
    answered = [r for r in a if not r["abstained"]]
    cited_flags = [f for r in answered for f in r["cited_gold"]]
    s = {
        "n_answerable": len(a), "n_unanswerable": len(u),
        "complete_rate": sum(r["complete"] for r in a) / max(1, len(a)),
        "complete_ci": wilson(sum(r["complete"] for r in a), len(a)),
        "mean_coverage": sum(r["coverage"] for r in a) / max(1, len(a)),
        "false_abstain_rate": sum(r["abstained"] for r in a) / max(1, len(a)),
        "cited_rate": sum(bool(r["cited"]) for r in answered) / max(1, len(answered)),
        "citation_precision": sum(cited_flags) / max(1, len(cited_flags)),
        "any_gold_cited_rate": sum(any(r["cited_gold"]) for r in answered) / max(1, len(answered)),
        "invalid_citation_rate": sum(bool(r["invalid_citations"]) for r in answered) / max(1, len(answered)),
        "model_cited_rate": sum(r.get("attribution") == "model" and bool(r["cited"]) for r in answered)
        / max(1, len(answered)),
        "correct_abstain_rate": sum(r["abstained"] for r in u) / max(1, len(u)),
        "correct_abstain_ci": wilson(sum(r["abstained"] for r in u), len(u)),
        "mean_seconds": sum(r["seconds"] for r in records) / max(1, len(records)),
    }
    return s


def tune_threshold(scores_ans: list[float], scores_unans: list[float]) -> float:
    """Pick the abstain threshold that maximises balanced accuracy on the dev split.

    Many thresholds usually tie (any value between the highest-scoring unanswerable and the lowest-
    scoring answerable question separates them perfectly), so take the centre of that plateau rather
    than an edge: an edge is fitted to one or two dev questions and would not generalise."""
    pts = sorted(set(scores_ans + scores_unans))
    cands = [pts[0] - 1.0] + [(a + b) / 2 for a, b in zip(pts, pts[1:], strict=False)] + [pts[-1] + 1.0]

    def bal(t: float) -> float:
        tpr = sum(s >= t for s in scores_ans) / max(1, len(scores_ans))  # answered when answerable
        tnr = sum(s < t for s in scores_unans) / max(1, len(scores_unans))  # refused when unanswerable
        return (tpr + tnr) / 2

    best = max(bal(t) for t in cands)
    plateau = sorted(t for t in cands if bal(t) >= best - 1e-9)
    return plateau[len(plateau) // 2]
