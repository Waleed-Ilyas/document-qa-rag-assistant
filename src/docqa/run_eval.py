"""Run the full evaluation and write artifacts/eval_results.json (+ MLflow runs in mlflow.db).

  python -m docqa.run_eval retrieval     # BM25 vs dense vs hybrid vs hybrid+rerank
  python -m docqa.run_eval generation    # extractive vs Qwen2.5-0.5B vs Qwen2.5-1.5B (slow: runs the LLMs)
"""

from __future__ import annotations

import json
import sys

import mlflow

from . import config
from .evaluate import answer_record, load_qa, retrieval_metrics, summarise, tune_threshold
from .generate import ExtractiveGenerator, LlamaCppGenerator
from .rag import RAGPipeline
from .retrieve import Retriever

MODES = ["bm25", "dense", "hybrid", "hybrid+rerank"]
RESULTS = config.ARTIFACTS / "eval_results.json"


def _load_results() -> dict:
    return json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}


def _save(res: dict) -> None:
    RESULTS.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


def _mlflow() -> None:
    mlflow.set_tracking_uri(f"sqlite:///{config.ROOT / 'mlflow.db'}")
    mlflow.set_experiment("docqa")


def run_retrieval() -> None:
    _mlflow()
    qa = [q for q in load_qa() if q["answerable"]]
    retr = Retriever.load()
    res = _load_results()
    res["retrieval"] = {}
    for mode in MODES:
        results = [(q, retr.search(q["question"], mode, 10)) for q in qa]
        m = retrieval_metrics(results)
        res["retrieval"][mode] = m
        with mlflow.start_run(run_name=f"retrieval-{mode}"):
            mlflow.log_params({"stage": "retrieval", "mode": mode, "n_questions": len(qa),
                               "embed_model": config.EMBED_MODEL, "rerank_model": config.RERANK_MODEL,
                               "n_chunks": len(retr.chunks)})
            mlflow.log_metrics({k.replace("@", "_at_"): v for k, v in m.items() if isinstance(v, float)})
        print(f"{mode:14s} hit@1={m['hit@1']:.3f} hit@4={m['hit@4']:.3f} hit@10={m['hit@10']:.3f} "
              f"mrr={m['mrr@10']:.3f} | suff@1={m['suff@1']:.3f} suff@4={m['suff@4']:.3f} suff@10={m['suff@10']:.3f}")
    # where the best retriever still misses
    best = [(q, retr.search(q["question"], "hybrid+rerank", 5)) for q in qa]
    res["retrieval_misses_top5"] = [
        {"id": q["id"], "question": q["question"], "gold": [q["doc"], q["gold_pages"]],
         "got": [(h.doc_id, h.page) for h in hits]}
        for q, hits in best if not any(h.doc_id == q["doc"] and h.page in q["gold_pages"] for h in hits)]
    _save(res)


def run_generation(systems: list[str] | None = None) -> None:
    _mlflow()
    qa = load_qa()
    retr = Retriever.load()
    base = RAGPipeline(retr)
    # 1) cache retrieval once; tune the abstain threshold on the DEV split only
    hits = {q["id"]: base.retrieve(q["question"]) for q in qa}
    dev = [q for q in qa if q["split"] == "dev"]
    thr = tune_threshold([hits[q["id"]][0].score for q in dev if q["answerable"]],
                         [hits[q["id"]][0].score for q in dev if not q["answerable"]])
    print(f"abstain threshold (tuned on dev) = {thr:.3f}")
    res = _load_results()
    res["abstain_threshold"] = thr
    res.setdefault("generation", {})
    res.setdefault("answers", {})

    gens = {
        "extractive": ExtractiveGenerator(),
        "qwen2.5-0.5b": LlamaCppGenerator(config.LLM_SMALL_REPO, config.LLM_SMALL_FILE),
        "qwen2.5-1.5b": LlamaCppGenerator(config.LLM_REPO, config.LLM_FILE),
    }
    for name, gen in gens.items():
        if systems and name not in systems:
            continue
        pipe = RAGPipeline(retr, gen, abstain_threshold=thr)
        records = []
        for i, q in enumerate(qa, 1):
            ans = pipe.answer(q["question"])
            records.append(answer_record(q, ans))
            print(f"  [{name}] {i}/{len(qa)} {q['id']} {ans.seconds:.1f}s", flush=True)
        test = [r for r in records if r["split"] == "test"]
        summary = {"all": summarise(records), "test": summarise(test)}
        res["generation"][name] = summary
        res["answers"][name] = records
        _save(res)
        with mlflow.start_run(run_name=f"generation-{name}"):
            mlflow.log_params({"stage": "generation", "system": name, "abstain_threshold": thr,
                               "context_k": config.CONTEXT_K, "retrieval": "hybrid+rerank"})
            mlflow.log_metrics({f"test_{k}": v for k, v in summary["test"].items() if isinstance(v, float)})
        t = summary["test"]
        print(f"{name}: complete={t['complete_rate']:.2f} coverage={t['mean_coverage']:.2f} "
              f"false_abstain={t['false_abstain_rate']:.2f} cite_prec={t['citation_precision']:.2f} "
              f"correct_abstain={t['correct_abstain_rate']:.2f} sec={t['mean_seconds']:.1f}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "retrieval"
    if stage == "retrieval":
        run_retrieval()
    else:
        run_generation(sys.argv[2:] or None)
