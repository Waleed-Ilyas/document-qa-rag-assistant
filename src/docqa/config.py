"""Paths, the document registry and model names. Everything tunable lives here."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
ARTIFACTS = ROOT / "artifacts"
EVAL_DIR = ROOT / "eval"
FIGURES = ROOT / "reports" / "figures"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # 384-d, ONNX via fastembed (no torch needed at runtime)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"  # cross-encoder, ONNX via fastembed
LLM_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"  # Apache-2.0
LLM_FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
LLM_SMALL_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
LLM_SMALL_FILE = "qwen2.5-0.5b-instruct-q4_k_m.gguf"

# chunking: sentence-aware, never crosses a page (so every chunk has an exact page citation)
CHUNK_TARGET_CHARS = 900
CHUNK_MAX_CHARS = 1300
CHUNK_OVERLAP_SENTENCES = 1
MIN_CHUNK_CHARS = 200

RETRIEVE_K = 30  # candidates pulled from each retriever before fusion
RERANK_TOP = 20  # candidates sent to the cross-encoder
CONTEXT_K = 4  # chunks handed to the LLM


@dataclass(frozen=True)
class Doc:
    doc_id: str
    title: str
    short: str
    filename: str
    url: str
    published: str
    topic: str


# 8 public US government (NIST) publications: freely reusable, no login, real page citations.
# Current versions only: SP 800-61r2 and SP 800-63-3 were tried first and dropped because NIST has
# withdrawn them (a document library that answers from superseded policy is a real-world RAG failure).
DOCS: list[Doc] = [
    Doc("ai-rmf", "AI Risk Management Framework 1.0 (NIST AI 100-1)", "AI RMF 1.0",
        "NIST.AI.100-1.pdf", "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf",
        "January 2023", "AI governance"),
    Doc("ai-genai", "Generative AI Profile of the AI RMF (NIST AI 600-1)", "GenAI Profile",
        "NIST.AI.600-1.pdf", "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf",
        "July 2024", "AI governance"),
    Doc("csf", "Cybersecurity Framework 2.0 (NIST CSWP 29)", "CSF 2.0",
        "NIST.CSWP.29.pdf", "https://nvlpubs.nist.gov/nistpubs/CSWP/NIST.CSWP.29.pdf",
        "February 2024", "Cybersecurity"),
    Doc("privacy", "Privacy Framework 1.0", "Privacy Framework",
        "NIST.CSWP.01162020.pdf", "https://nvlpubs.nist.gov/nistpubs/CSWP/NIST.CSWP.01162020.pdf",
        "January 2020", "Privacy"),
    Doc("ssdf", "Secure Software Development Framework 1.1 (NIST SP 800-218)", "SSDF 1.1",
        "NIST.SP.800-218.pdf", "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-218.pdf",
        "February 2022", "Secure software"),
    Doc("zero-trust", "Zero Trust Architecture (NIST SP 800-207)", "Zero Trust",
        "NIST.SP.800-207.pdf", "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-207.pdf",
        "August 2020", "Cybersecurity"),
    Doc("incident", "Incident Response Recommendations and Considerations (NIST SP 800-61 Rev. 3)",
        "Incident Response", "NIST.SP.800-61r3.pdf",
        "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-61r3.pdf",
        "April 2025", "Incident response"),
    Doc("identity", "Digital Identity Guidelines (NIST SP 800-63-4)", "Digital Identity",
        "NIST.SP.800-63-4.pdf", "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-63-4.pdf",
        "July 2025", "Identity"),
]
DOC_BY_ID = {d.doc_id: d for d in DOCS}
