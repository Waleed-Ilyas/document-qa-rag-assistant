"""Streamlit chat UI: ask a question, get a cited answer from a library of 8 NIST publications."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docqa import config  # noqa: E402
from docqa.generate import (  # noqa: E402
    ABSTAIN_TEXT,
    ExtractiveGenerator,
    LlamaCppGenerator,
)
from docqa.rag import DEFAULT_ABSTAIN_THRESHOLD, RAGPipeline, finalize  # noqa: E402
from docqa.retrieve import Retriever  # noqa: E402

st.set_page_config(page_title="Document Q&A (RAG)", page_icon="📄", layout="wide")

EXAMPLES = [
    "What are the four functions of the AI RMF Core?",
    "What are the six Functions of the CSF 2.0 Core?",
    "What does the policy enforcement point do in zero trust?",
    "Which generative AI risk covers made-up content, and what is it called?",
    "What are the four groups of practices in the SSDF?",
    "What is the maximum fine for a GDPR violation?",
]


@st.cache_resource(show_spinner="Loading the search index...")
def get_retriever() -> Retriever:
    return Retriever.load()


@st.cache_resource(show_spinner=False)
def get_llm() -> LlamaCppGenerator | None:
    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        return None
    return LlamaCppGenerator()


@st.cache_data
def eval_results() -> dict:
    p = config.ARTIFACTS / "eval_results.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def pdf_link(doc_id: str, page: int) -> str:
    return f"{config.DOC_BY_ID[doc_id].url}#page={page}"


res = eval_results()
threshold = res.get("abstain_threshold", DEFAULT_ABSTAIN_THRESHOLD)
retriever = get_retriever()
llm = get_llm()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("📄 Document Q&A")
    st.caption("Cited answers over 8 public NIST publications. Runs a small open model locally, no API keys.")
    options = ["Local LLM (Qwen2.5-1.5B) - fuller answers, slower", "Extractive - best sentences, fastest"]
    if llm is None:
        options = options[1:]
        st.warning("The local LLM runtime is not installed here, so answers are extractive.")
    mode = st.radio("Answer style", options, index=0)
    use_llm = mode.startswith("Local")
    st.divider()
    st.subheader("Library")
    for d in config.DOCS:
        st.markdown(f"**{d.short}** · {d.published}  \n[{d.title}]({d.url})")
    st.divider()
    with st.expander("Limitations (read this)"):
        st.markdown(
            "- A small 1.5B model can **misread or over-compress** a passage. Always check the cited page.\n"
            "- It only knows these 8 documents; it is told to refuse otherwise, but a refusal is not guaranteed "
            "(see the evaluation in the README).\n"
            "- Retrieval sends only the top 4 passages, so questions needing many pages or a whole-document "
            "summary will be incomplete.\n"
            "- Tables and figures in the PDFs are extracted as plain text and can be garbled."
        )

# ------------------------------------------------------------------ chat
st.title("Ask the document library")
st.caption("Every answer cites the document and page it came from. Click a source to read the passage.")

if "history" not in st.session_state:
    st.session_state.history = []


def render_sources(hits, cited):
    st.markdown("**Sources**")
    for i, h in enumerate(hits, start=1):
        d = config.DOC_BY_ID[h.doc_id]
        used = "✅ cited" if i in cited else "retrieved"
        label = f"[{i}] {d.short} · page {h.page}" + (f" (printed {h.page_label})" if h.page_label != str(h.page) else "")
        with st.expander(f"{label}  ·  {used}", expanded=i in cited and len(cited) <= 2):
            st.write(h.text)
            st.markdown(f"[Open the PDF at this page]({pdf_link(h.doc_id, h.page)})")


for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["q"])
    with st.chat_message("assistant"):
        st.write(turn["text"])
        if turn["hits"]:
            render_sources(turn["hits"], turn["cited"])
        st.caption(turn["meta"])

cols = st.columns(3)
clicked = None
for i, ex in enumerate(EXAMPLES):
    if cols[i % 3].button(ex, key=f"ex{i}", use_container_width=True):
        clicked = ex

question = st.chat_input("Ask about AI risk, cybersecurity, privacy, zero trust, secure software...") or clicked
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        generator = llm if (use_llm and llm is not None) else ExtractiveGenerator()
        pipe = RAGPipeline(retriever, generator, abstain_threshold=threshold)
        t0 = time.time()
        with st.spinner("Searching the library..."):
            hits = pipe.retrieve(question)
        top = hits[0].score if hits else float("-inf")
        if not hits or top < threshold:
            text, cited, hit_list = ABSTAIN_TEXT, [], []
            st.write(text)
            st.caption(f"Best passage scored {top:.1f} (threshold {threshold:.1f}): not confident enough to answer.")
            meta = f"abstained · retrieval score {top:.1f} < {threshold:.1f}"
        else:
            if use_llm:
                st.caption("First answer after a cold start downloads the ~1 GB model; later answers stream quickly.")
            raw = st.write_stream(pipe.stream(question, hits))
            parsed, how = finalize(raw, hits)
            text, cited, hit_list = parsed.text, parsed.cited, hits
            if parsed.abstained:
                hit_list = []
                st.write(parsed.text)
            else:
                if how == "post-hoc":
                    st.info("The model did not cite on its own, so each sentence was matched to the passage "
                            "that best supports it. Check the cited page to confirm.")
                    st.write(parsed.text)
                if not cited:
                    st.warning("No retrieved passage clearly supports this answer. Treat it with caution.")
                render_sources(hit_list, cited)
            meta = f"{generator.name} · {time.time() - t0:.1f}s · top retrieval score {top:.1f}"
            st.caption(meta)
    st.session_state.history.append({"q": question, "text": text, "hits": hit_list, "cited": cited, "meta": meta})
