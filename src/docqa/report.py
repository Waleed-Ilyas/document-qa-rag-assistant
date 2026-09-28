"""Figures for the README/notebook from artifacts/eval_results.json. Run: python -m docqa.report"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import config  # noqa: E402
from .evaluate import load_qa  # noqa: E402

GOLD, CYAN, IVORY, BG, GREY = "#D9B26A", "#4FD1C5", "#F2EDE4", "#0A0A0C", "#8a8578"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": IVORY,
                     "axes.labelcolor": IVORY, "xtick.color": IVORY, "ytick.color": IVORY,
                     "axes.edgecolor": GREY, "font.size": 11})


def _res() -> dict:
    return json.loads((config.ARTIFACTS / "eval_results.json").read_text(encoding="utf-8"))


def fig_retrieval(res: dict) -> None:
    modes = list(res["retrieval"])
    metrics = [("hit@1", "hit@1 (gold page)"), ("hit@4", "hit@4 (gold page)"), ("suff@4", "context sufficiency @4")]
    x = np.arange(len(modes))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, (key, label) in enumerate(metrics):
        vals = [res["retrieval"][m][key] for m in modes]
        ci = np.array([res["retrieval"][m][f"{key}_ci"] for m in modes])
        err = np.array([np.array(vals) - ci[:, 0], ci[:, 1] - np.array(vals)])
        ax.bar(x + (i - 1) * w, vals, w, label=label, color=[GOLD, CYAN, IVORY][i], yerr=err,
               error_kw={"ecolor": GREY, "capsize": 3, "lw": 1})
    ax.set_xticks(x, modes)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("share of 40 answerable questions")
    ax.set_title("Retrieval: BM25 vs dense vs hybrid vs hybrid + rerank (95% Wilson intervals)", fontsize=11)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(config.FIGURES / "01_retrieval.png", dpi=150)


def fig_generation(res: dict) -> None:
    systems = list(res["generation"])
    panels = [("complete_rate", "answer complete\n(all key facts)"), ("citation_precision", "cited page is\na gold page"),
              ("false_abstain_rate", "wrongly refused\n(lower is better)"),
              ("correct_abstain_rate", "correctly refused\n(unanswerable)")]
    x = np.arange(len(panels))
    w = 0.8 / len(systems)
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for i, s in enumerate(systems):
        t = res["generation"][s]["test"]
        ax.bar(x + (i - (len(systems) - 1) / 2) * w, [t[k] for k, _ in panels], w, label=s,
               color=[GREY, CYAN, GOLD][i % 3])
    ax.set_xticks(x, [p[1] for p in panels])
    ax.set_ylim(0, 1.05)
    ax.set_title("Answer quality on the held-out test split (30 answerable + 6 unanswerable)", fontsize=11)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "02_generation.png", dpi=150)


def fig_scores(res: dict) -> None:
    qa = {q["id"]: q for q in load_qa()}
    recs = res["answers"][next(iter(res["answers"]))]
    a = [r["top_score"] for r in recs if qa[r["id"]]["answerable"]]
    u_off = [r["top_score"] for r in recs if qa[r["id"]].get("kind") == "off"]
    u_near = [r["top_score"] for r in recs if qa[r["id"]].get("kind") == "near"]
    bins = np.linspace(-12, 11, 24)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.hist(a, bins, color=GOLD, alpha=0.9, label="answerable (40)")
    ax.hist(u_near, bins, color=CYAN, alpha=0.9, label="unanswerable, same topic (6)")
    ax.hist(u_off, bins, color=GREY, alpha=0.9, label="unanswerable, off-topic (2)")
    ax.axvline(res["abstain_threshold"], color=IVORY, ls="--", lw=1.2)
    ax.text(res["abstain_threshold"] + 0.2, ax.get_ylim()[1] * 0.9, "abstain threshold\n(tuned on dev split)", fontsize=9)
    ax.set_xlabel("top reranker score of the best passage")
    ax.set_ylabel("questions")
    ax.set_title("Retrieval score alone catches 6 of 8 unanswerable questions; the other 2 need the LLM to refuse", fontsize=11)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(config.FIGURES / "03_abstention_scores.png", dpi=150)


if __name__ == "__main__":
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    r = _res()
    fig_retrieval(r)
    if r.get("generation"):
        fig_generation(r)
        fig_scores(r)
    print("figures written to", config.FIGURES)
