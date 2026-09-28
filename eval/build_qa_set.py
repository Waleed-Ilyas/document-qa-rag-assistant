"""Builds eval/qa_set.json from the hand-written questions below.

Every answerable question was written by reading the source PDF (single annotator). For each one:
  evidence    regexes that must appear in the text of a gold page (proves the fact is in the source)
  must_include  groups of alternatives; an answer is "complete" only if EVERY group has a match
Gold pages are computed as the pages of the gold document whose (non-table-of-contents) chunks match
an evidence regex, and this script refuses to write the file if any question has no gold page, any
must_include group is not present in the gold pages' text, or the gold page set is suspiciously wide.

Run from the repo root:  python eval/build_qa_set.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
chunks = pd.read_parquet(ROOT / "artifacts" / "chunks.parquet")
chunks = chunks[~chunks.text.str.contains(r"\.{6,}|(?:\. ){6,}", regex=True)]  # drop table-of-contents chunks

# (doc, question, [evidence regexes], [[must_include alternatives], ...])
A = [
    # ---- AI RMF 1.0
    ("ai-rmf", "What are the four functions of the AI RMF Core?", [r"composed of four functions: GOVERN"],
     [["govern"], ["map"], ["measure"], ["manage"]]),
    ("ai-rmf", "Which characteristics of trustworthy AI systems does the framework describe?",
     [r"valid and reliable, safe, secure and resilient, accountable and transparent"],
     [["valid and reliable"], ["explainable and interpretable"], ["privacy"], ["fair"]]),
    ("ai-rmf", "What does TEVV stand for in the AI RMF?", [r"test, evaluation, verification, and validation \(TEVV\)"],
     [["evaluation, verification"]]),
    ("ai-rmf", "Is use of the AI RMF mandatory?", [r"voluntary, rights-preserving"], [["voluntary"]]),
    ("ai-rmf", "Which characteristic is described as a necessary condition of trustworthiness?",
     [r"Valid & Reliable is a necessary condition of trustworthiness"], [["valid"], ["reliable"]]),
    ("ai-rmf", "What are AI actors?", [r"referred to here as AI actors"], [["organizations and individuals"]]),
    # ---- Generative AI profile
    ("ai-genai", "Which risk is the confident production of false content, and what is it colloquially called?",
     [r"Confabulation: The production of confidently stated but erroneous"], [["confabulation"], ["hallucination"]]),
    ("ai-genai", "Which generative AI risk relates to chemical, biological, radiological or nuclear weapons?",
     [r"chemical, biological, radiological, or nuclear \(CBRN\)"], [["cbrn", "chemical, biological"]]),
    ("ai-genai", "How does the profile define the Data Privacy risk?",
     [r"Data Privacy: Impacts due to leakage and unauthorized use"], [["leakage", "unauthorized use"], ["personally identifiable", "biometric"]]),
    ("ai-genai", "What is the Human-AI Configuration risk?",
     [r"Human-AI Configuration: Arrangements of or interactions between a human and an AI system"],
     [["automation bias", "over-reliance", "algorithmic aversion"]]),
    ("ai-genai", "What does the Value Chain and Component Integration risk refer to?",
     [r"Value Chain and Component Integration: Non-transparent or untraceable integration"],
     [["third-party"], ["non-transparent", "untraceable"]]),
    ("ai-genai", "What kind of harm does the Environmental Impacts risk describe?",
     [r"Environmental Impacts: Impacts due to high compute resource utilization"], [["compute"]]),
    # ---- CSF 2.0
    ("csf", "What are the six Functions of the CSF 2.0 Core?",
     [r"GOVERN, IDENTIFY, PROTECT, DETECT, RESPOND, and RECOVER"],
     [["govern"], ["identify"], ["protect"], ["detect"], ["respond"], ["recover"]]),
    ("csf", "What are the four CSF Tiers?", [r"Partial \(Tier 1\), Risk Informed \(Tier 2\), Repeatable \(Tier 3\)"],
     [["partial"], ["risk informed"], ["repeatable"], ["adaptive"]]),
    ("csf", "What is a CSF Organizational Profile?",
     [r"mechanism for describing an organization's current and/or target cybersecurity posture"],
     [["current"], ["target"]]),
    ("csf", "What are Informative References in the CSF?",
     [r"Informative References that point to sources of guidance"], [["guidance", "standards"]]),
    ("csf", "Should the CSF Functions be addressed one after another?",
     [r"Functions should be addressed concurrently"], [["concurrent"]]),
    ("csf", "What does the GOVERN Function provide?",
     [r"The GOVERN Function provides outcomes to inform what an organization may do"],
     [["prioritize"], ["other five"]]),
    # ---- Privacy Framework
    ("privacy", "What are the five Functions of the Privacy Framework Core?",
     [r"five Functions, Identify-P, Govern-P, Control-P, Communicate-P, and Protect-P"],
     [["identify-p"], ["govern-p"], ["control-p"], ["communicate-p"], ["protect-p"]]),
    ("privacy", "What are the three parts that make up the Privacy Framework?",
     [r"three parts: Core, Profiles, and Implementation Tiers"],
     [["core"], ["profiles"], ["implementation tiers"]]),
    ("privacy", "How can the Privacy Framework be used together with the Cybersecurity Framework Functions?",
     [r"use all five of the Cybersecurity Framework Functions in conjunction with Identify-P"],
     [["cybersecurity framework"], ["identify-p"]]),
    ("privacy", "Why are the Privacy Framework Functions labelled with a '-P' suffix?",
     [r"in order to avoid confusion with Cybersecurity Framework Functions"], [["confusion"]]),
    # ---- SSDF
    ("ssdf", "What are the four groups of practices in the SSDF?", [r"organized into four groups"],
     [["prepare the organization"], ["protect the software"], ["produce well-secured software"],
      ["respond to vulnerabilities"]]),
    ("ssdf", "What does SSDF practice PS.1 aim to prevent?",
     [r"Protect All Forms of Code from Unauthorized Access and Tampering \(PS\.1\)"],
     [["unauthorized"], ["tampering", "changes"]]),
    ("ssdf", "When was Executive Order 14028 issued and what is it titled?",
     [r"Improving the Nation's Cybersecurity \(14028\)\" issued on May 12, 2021"],
     [["may 12, 2021"], ["improving the nation's cybersecurity"]]),
    ("ssdf", "What does practice PO.1 ask organizations to define?",
     [r"Define Security Requirements for Software Development \(PO\.1\)"], [["security requirements"]]),
    ("ssdf", "What does practice RV.1 focus on?",
     [r"Identify and Confirm Vulnerabilities on an Ongoing Basis \(RV\.1\)"], [["vulnerabilit"]]),
    # ---- Zero trust
    ("zero-trust", "What two logical components make up the policy decision point?",
     [r"policy decision point \(PDP\) is broken down into two logical components"],
     [["policy engine"], ["policy administrator"]]),
    ("zero-trust", "What does zero trust assume about network location?",
     [r"no implicit trust granted to assets or user accounts based solely on their physical or network location"],
     [["implicit trust", "no trust"]]),
    ("zero-trust", "What is the policy engine responsible for?",
     [r"responsible for the ultimate decision to grant access to a resource"], [["decision"], ["access"]]),
    ("zero-trust", "What does the policy enforcement point do?",
     [r"Policy enforcement point \(PEP\): This system is responsible for enabling, monitoring, and eventually terminating"],
     [["terminating", "monitoring"], ["connections"]]),
    ("zero-trust", "How does an enterprise treat its own network in a zero trust model?",
     [r"an enterprise must assume no implicit trust"], [["no implicit trust", "not implicit"]]),
    # ---- Incident response (SP 800-61r3)
    ("incident", "Which CSF 2.0 Functions cover discovering, containing and recovering from incidents?",
     [r"Detect, Respond, and Recover help organizations discover, manage, prioritize, contain"],
     [["detect"], ["respond"], ["recover"]]),
    ("incident", "Why do lessons learned from incidents matter for cybersecurity risk management?",
     [r"Lessons learned from incident response activities and root cause analysis"],
     [["improve"], ["root cause", "lessons"]]),
    ("incident", "What is the basis of the incident response life cycle proposed in SP 800-61 Rev. 3?",
     [r"proposes a new life cycle model based on CSF 2.0 Functions"], [["csf 2.0"], ["life cycle"]]),
    ("incident", "How many CSF 2.0 Functions does the high-level incident response model use?",
     [r"life cycle model based on the six CSF 2.0 Functions"], [["six"]]),
    # ---- Digital identity (SP 800-63-4)
    ("identity", "How does SP 800-63-4 change IAL1?", [r"repurposing IAL1 as a new assurance level"],
     [["ial1"], ["repurpos", "new assurance level"]]),
    ("identity", "What new authentication options does SP 800-63-4 provide?",
     [r"new options for phishing-resistant authentication"], [["phishing-resistant"]]),
    ("identity", "Can an organization require phishing-resistant authentication at AAL2?",
     [r"restrict users to only phishing-resistant authentication at AAL2"], [["phishing-resistant"], ["aal2"]]),
    ("identity", "What does IAL1 support?", [r"IAL1 supports the real-world existence of the claimed identity"],
     [["real-world existence", "claimed identity"]]),
]

# Questions the library does NOT answer: the system should say so instead of guessing.
# "off" = unrelated to the corpus; "near" = plausible topic, but the answer is not in these documents.
U = [
    ("What is the recipe for a classic margherita pizza?", "off", []),
    ("Who won the 2022 FIFA World Cup?", "off", []),
    ("What is the maximum fine for a GDPR violation?", "near", [r"GDPR|General Data Protection Regulation", r"20 million|4 ?%"]),
    ("Which specific vendor products does NIST certify as zero trust compliant?", "near", [r"certif\w+ .{0,40}vendor", r"vendor.{0,40}certif"]),
    ("What is the required minimum password length under PCI DSS?", "near", [r"PCI DSS"]),
    ("How much does a NIST Cybersecurity Framework consulting engagement cost?", "near", [r"consulting (fee|cost|rate)"]),
    ("What did the HIPAA Security Rule require for audit logging in 1996?", "near", [r"HIPAA Security Rule.{0,80}audit"]),
    ("What GPU hardware is recommended for training the models discussed in the AI RMF?", "near", [r"GPU"]),
]


def main() -> int:
    out, problems = [], []
    for i, (doc, q, evid, groups) in enumerate(A, start=1):
        d = chunks[chunks.doc_id == doc]
        gold = set()
        for pat in evid:
            gold |= set(d[d.text.str.contains(pat, regex=True)].page)
        if not gold:
            problems.append(f"A{i} no gold page for evidence: {evid}")
            continue
        if len(gold) > 6:
            problems.append(f"A{i} gold pages too wide {sorted(gold)} for {q!r}")
        gold_text = " ".join(d[d.page.isin(gold)].text).lower()
        for grp in groups:
            if not any(alt.lower() in gold_text for alt in grp):
                problems.append(f"A{i} must_include {grp} absent from gold pages {sorted(gold)} ({q!r})")
        out.append({"id": f"a{i:02d}", "question": q, "answerable": True, "doc": doc,
                    "gold_pages": sorted(int(p) for p in gold), "must_include": groups,
                    "evidence": evid})
    for j, (q, kind, must_absent) in enumerate(U, start=1):
        # confirm the corpus really lacks an answer (a mention of the topic is fine; the evidence patterns must not hit)
        hits = [pat for pat in must_absent if chunks.text.str.contains(pat, case=False, regex=True).any()]
        if hits and kind == "near":
            problems.append(f"U{j} corpus mentions {hits} - question may be answerable: {q!r}")
        out.append({"id": f"u{j:02d}", "question": q, "answerable": False, "kind": kind})
    for i, item in enumerate(out):  # deterministic dev/test split: ~1 in 4 goes to dev (abstain-threshold tuning)
        item["split"] = "dev" if i % 4 == 0 else "test"
    if problems:
        print("\n".join(problems))
        return 1
    (ROOT / "eval" / "qa_set.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    n_a = sum(x["answerable"] for x in out)
    print(f"wrote {len(out)} questions: {n_a} answerable, {len(out) - n_a} unanswerable; "
          f"dev={sum(x['split'] == 'dev' for x in out)} test={sum(x['split'] == 'test' for x in out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
