"""
Test-data loader + dummy-set builder for the ambiguity-aware eval pipeline.

The real test set format is not yet finalized. This module ships a v1 schema
and a dummy builder so downstream code (inference_loop, run_eval) can develop
and smoke-test without waiting on the real data.

Schema (one JSON object per line):
    {
      "id":                str,
      "query":             str,
      "expected_behavior": "pause" | "clear",
      "gold_clarification": str,          # required iff expected_behavior == "pause"
      "reference_answer":  str,           # required iff expected_behavior == "clear"
      "category":          str,           # free-form tag ("scope_ambiguous", "factual", ...)
      "schema_version":    "v1"
    }
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

SCHEMA_VERSION = "v1"


@dataclass
class TestCase:
    id: str
    query: str
    expected_behavior: str  # "pause" | "clear"
    gold_clarification: str = ""
    reference_answer: str = ""
    category: str = ""
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.expected_behavior not in ("pause", "clear"):
            raise ValueError(f"{self.id}: expected_behavior must be pause|clear")
        if self.expected_behavior == "pause" and not self.gold_clarification:
            raise ValueError(f"{self.id}: pause row missing gold_clarification")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"{self.id}: schema_version {self.schema_version} != {SCHEMA_VERSION}"
            )


def _row_to_testcase(d: dict, idx: int) -> TestCase:
    """Convert one input row to a TestCase. Handles two schemas:

    1. Native v1 (TestCase fields directly) — used by the dummy builder.
    2. Real held-out format with a `type` field:
         clear: {clarified_query, final_response, type: "clear", ...}
         vague: {query, clarified_query, final_response, type: "vague", ...}
       `clarified_query` for vague rows is the disambiguated version the
       simulated user agent will hand back when the model asks.
    """
    if "type" in d and d["type"] in ("clear", "vague"):
        t = d["type"]
        if t == "clear":
            return TestCase(
                id=d.get("id") or f"clear_{idx:03d}",
                query=d.get("clarified_query") or d.get("query") or "",
                expected_behavior="clear",
                gold_clarification="",
                reference_answer=d.get("final_response") or "",
                category=d.get("category", "clear"),
            )
        # vague
        return TestCase(
            id=d.get("id") or f"vague_{idx:03d}",
            query=d.get("query") or "",
            expected_behavior="pause",
            gold_clarification=d.get("clarified_query") or "",
            reference_answer=d.get("final_response") or "",
            category=d.get("category", "scope_ambiguous"),
        )
    # Native v1 row
    return TestCase(**d)


def load_test_set(path: str | Path) -> List[TestCase]:
    """Load a JSONL test set. Accepts either native v1 rows or the real held-out
    format (see `_row_to_testcase`). Supports top-level JSON arrays too."""
    path = Path(path)
    text = path.read_text().strip()
    rows: List[dict] = []
    if text.startswith("["):
        rows = json.loads(text)
    else:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    out: List[TestCase] = []
    for i, d in enumerate(rows, 1):
        tc = _row_to_testcase(d, i)
        tc.validate()
        out.append(tc)
    return out


def iter_general_purpose_prompts(root: Path) -> Iterator[str]:
    """Yield user_prompt strings from BaseAgent/Data/generalpurposedata/train/*/data.jsonl."""
    prefix = "Please gather comprehensive research on the following topic: "
    for subdir in sorted(root.iterdir()):
        jf = subdir / "data.jsonl"
        if not jf.exists():
            continue
        with jf.open() as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                q = (d.get("user_prompt") or "").strip()
                if q.startswith(prefix):
                    q = q[len(prefix):].strip()
                if q:
                    yield q


# 20 clear queries: factual + scoped deep-research.
_CLEAR_QUERIES = [
    ("What is CVE-2021-44228 (Log4Shell), how does the JNDI lookup exploit work, and what were the cascading effects on the software supply chain?",
     "Log4Shell is CVE-2021-44228 in Apache Log4j 2.x (<=2.14.1) allowing JNDI lookups from log messages, leading to RCE via attacker-controlled LDAP/RMI endpoints. Cascading effects: widespread internet-scale scanning, emergency patching across JVM ecosystem, multi-vendor advisories, and acceleration of SBOM adoption.",
     "factual_cve"),
    ("Explain the MOVEit Transfer CVE-2023-34362 SQL injection vulnerability: root cause, exploitation timeline by Cl0p, and how organizations detected ongoing compromise.",
     "CVE-2023-34362 is a SQLi in Progress MOVEit Transfer's web interface leading to authentication bypass and RCE via webshell (LEMURLOOT). Cl0p mass-exploited it starting May 2023; detection relied on scanning for unknown .aspx files in wwwroot and anomalous human2.aspx requests.",
     "factual_cve"),
    ("Compare the root causes and long-term industry impact of the SolarWinds Orion (2020) and XZ Utils (2024) supply-chain attacks.",
     "SolarWinds: nation-state (APT29) inserted SUNBURST backdoor into Orion build pipeline. XZ Utils: multi-year social-engineering by Jia Tan planting obfuscated sshd backdoor in liblzma. Both highlight software-signing gaps; led to EO 14028, SBOM mandates, and scrutiny of single-maintainer OSS.",
     "research_compare"),
    ("Trace the evolution of ransomware-as-a-service from CryptoLocker (2013) through LockBit 3.0 (2023). How did affiliate models and encryption schemes change?",
     "CryptoLocker was monolithic. By CryptoWall/CTB-Locker affiliate programs emerged. REvil/DarkSide introduced 20-30% affiliate splits, double extortion, and data-leak sites. LockBit 3.0 added bug bounty and stricter cryptoschemes (ChaCha20+RSA). Law enforcement disrupted REvil (2021), Hive (2023), LockBit (2024).",
     "research_evolution"),
    ("Write a technical briefing on post-quantum cryptography migration: which NIST-standardized algorithms exist, and how are major cloud providers preparing?",
     "NIST FIPS 203 (ML-KEM/Kyber), FIPS 204 (ML-DSA/Dilithium), FIPS 205 (SLH-DSA/SPHINCS+) standardized Aug 2024. AWS KMS added Kyber hybrid TLS 1.3; Google Chrome enabled ML-KEM-768 in Chrome 124; Cloudflare offers post-quantum TLS. Main pitfalls: larger keys/signatures and downgrade attacks.",
     "research_pqc"),
    ("What is the CVSS v3.1 base score and vector string for CVE-2014-0160 (Heartbleed)?",
     "CVE-2014-0160 (Heartbleed) CVSS v3.1 base score is 7.5 (HIGH), vector AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:N/A:N.",
     "factual_cvss"),
    ("Explain how the EternalBlue (MS17-010) SMBv1 exploit works at a packet level.",
     "EternalBlue exploits a buffer overflow in SMBv1's SRVNET.sys srv!SrvOs2FeaListSizeToNt conversion. A crafted SMB_COM_TRANSACTION2 request with specific FEA list sizes causes heap corruption, enabling kernel-mode RCE. Used by WannaCry and NotPetya.",
     "factual_exploit"),
    ("What is the OWASP API Security Top 10 2023 list?",
     "BOLA, Broken Authentication, BOPLA, Unrestricted Resource Consumption, BFLA, Unrestricted Access to Sensitive Business Flows, SSRF, Security Misconfiguration, Improper Inventory Management, Unsafe Consumption of APIs.",
     "factual_owasp"),
    ("What year was the Stuxnet worm first publicly identified by security researchers?",
     "Stuxnet was first publicly identified in June 2010 by VirusBlokAda (Belarusian firm), later analyzed in depth by Symantec and Kaspersky.",
     "factual_year"),
    ("Which NIST publication defines the Cybersecurity Framework 2.0?",
     "NIST CSWP 29 (Cybersecurity Framework 2.0), published February 2024, adds the 'Govern' function to the original five (Identify, Protect, Detect, Respond, Recover).",
     "factual_standard"),
    ("Summarize the key findings of the 2024 Verizon DBIR regarding credential-based attacks.",
     "The 2024 DBIR reports credentials remain the top initial access vector (~24% of breaches), with stolen credentials used in 77% of basic web-application attacks; MFA reduces risk but phishing-resistant MFA (FIDO2) remains under-deployed.",
     "research_report"),
    ("Describe how Kerberoasting attacks work against Active Directory service accounts.",
     "Attacker requests TGS tickets for SPN-registered service accounts via KRB_TGS_REQ, receives tickets encrypted with the service account's NTLM hash, cracks offline (e.g., hashcat mode 13100). Mitigations: long/random passwords for service accounts, AES-only Kerberos encryption, managed service accounts.",
     "factual_technique"),
    ("What is the CVE identifier and patch status of the 2024 regreSSHion OpenSSH vulnerability?",
     "CVE-2024-6387 (regreSSHion) is a signal-handler race in OpenSSH sshd on glibc Linux versions 8.5p1-9.7p1; patched in OpenSSH 9.8p1 released July 2024.",
     "factual_cve"),
    ("List the five pillars of the NIST Zero Trust Architecture as defined in SP 800-207.",
     "NIST SP 800-207 defines pillars: Identity, Device, Network/Environment, Application Workload, and Data — each governed by continuous verification, least privilege, assume-breach.",
     "factual_standard"),
    ("What is the primary cryptographic primitive used in Signal's double ratchet algorithm?",
     "Signal's double ratchet combines a symmetric-key ratchet (HKDF chain) with a Diffie-Hellman ratchet (X25519). AES-256 in CBC mode with HMAC-SHA256 is used for message encryption (AES-GCM in newer specs).",
     "factual_crypto"),
    ("Summarize the technical cause of the 2024 CrowdStrike Falcon sensor outage.",
     "CrowdStrike Channel File 291 (July 19 2024) contained malformed template instances that caused a null-pointer dereference in Falcon sensor's content interpreter, producing kernel-mode BSOD on ~8.5M Windows hosts. Post-mortem cited insufficient staged rollout and content-validation gaps.",
     "research_incident"),
    ("What is the exact HTTP status code used for rate limiting per RFC 6585?",
     "HTTP 429 Too Many Requests, defined in RFC 6585 (April 2012).",
     "factual_rfc"),
    ("Which symmetric cipher and mode are used by default in TLS 1.3?",
     "TLS 1.3 mandates AEAD ciphers; the three MTI suites use AES-128-GCM, AES-256-GCM, and ChaCha20-Poly1305, each with HKDF-SHA256 or HKDF-SHA384.",
     "factual_crypto"),
    ("What is the main security improvement in HTTP Strict Transport Security (HSTS) as defined in RFC 6797?",
     "RFC 6797 HSTS forces user agents to upgrade all requests to a host to HTTPS for a declared max-age, eliminating SSL-stripping MITM attacks on the first and subsequent visits within the window.",
     "factual_rfc"),
    ("Explain the difference between DMARC p=none, p=quarantine, and p=reject policies.",
     "DMARC p=none is monitor-only (no enforcement, full reports). p=quarantine instructs receivers to spam-folder failing mail. p=reject instructs receivers to block failing mail at SMTP time. Senders typically ramp none→quarantine→reject as alignment coverage improves.",
     "factual_standard"),
]


def build_dummy_set(
    out_path: Path,
    n_clear: int = 20,
    n_vague: int = 20,
    seed: int = 17,
    general_purpose_root: Optional[Path] = None,
    use_rater: bool = False,
) -> List[TestCase]:
    """
    Emit a dummy test set so the eval pipeline is runnable end-to-end.

    Clear queries come from a hand-written bank with reference answers.
    Vague queries sample from BaseAgent/Data/generalpurposedata/train/*/data.jsonl
    (all pre-labeled [PAUSE]). Gold clarifications are heuristic stubs — the eval
    still works with stubs; only judge-of-trajectory quality is affected.
    """
    rng = random.Random(seed)

    if n_clear > len(_CLEAR_QUERIES):
        raise ValueError(f"n_clear={n_clear} exceeds bank size {len(_CLEAR_QUERIES)}")

    clear_bank = _CLEAR_QUERIES[:]
    rng.shuffle(clear_bank)
    chosen_clear = clear_bank[:n_clear]

    if general_purpose_root is None:
        general_purpose_root = Path(
            "/home/soumya/reasoning/BaseAgent/Data/generalpurposedata/train"
        )

    vague_prompts = list(iter_general_purpose_prompts(general_purpose_root))
    rng.shuffle(vague_prompts)
    vague_prompts = vague_prompts[: n_vague * 3]  # oversample, dedupe below
    seen = set()
    dedup = []
    for p in vague_prompts:
        if p.lower() in seen:
            continue
        seen.add(p.lower())
        dedup.append(p)
        if len(dedup) >= n_vague:
            break

    if len(dedup) < n_vague:
        raise RuntimeError(
            f"Only found {len(dedup)} unique vague prompts under {general_purpose_root}"
        )

    cases: List[TestCase] = []

    for i, (q, ref, cat) in enumerate(chosen_clear, 1):
        cases.append(TestCase(
            id=f"clear_{i:03d}",
            query=q,
            expected_behavior="clear",
            reference_answer=ref,
            category=cat,
        ))

    # Heuristic gold clarifications — real data will replace these.
    stub_template = (
        "Assume the user is a technical reader asking about a mainstream, "
        "widely-documented variant of the topic. Default scope: past 12 months, "
        "English-language sources, production/enterprise context."
    )
    for i, q in enumerate(dedup, 1):
        cases.append(TestCase(
            id=f"vague_{i:03d}",
            query=q,
            expected_behavior="pause",
            gold_clarification=stub_template,
            reference_answer="",
            category="scope_ambiguous",
        ))

    for tc in cases:
        tc.validate()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for tc in cases:
            f.write(json.dumps(asdict(tc), ensure_ascii=False) + "\n")

    return cases


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--build-dummy", action="store_true")
    p.add_argument("--out", default="/home/soumya/reasoning/BaseAgent/Data/TestData/ambiguity_eval_v1.jsonl")
    p.add_argument("--n-clear", type=int, default=20)
    p.add_argument("--n-vague", type=int, default=20)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--general-purpose-root", default=None)
    args = p.parse_args()

    if args.build_dummy:
        root = Path(args.general_purpose_root) if args.general_purpose_root else None
        cases = build_dummy_set(
            out_path=Path(args.out),
            n_clear=args.n_clear,
            n_vague=args.n_vague,
            seed=args.seed,
            general_purpose_root=root,
        )
        n_pause = sum(1 for c in cases if c.expected_behavior == "pause")
        n_clear = sum(1 for c in cases if c.expected_behavior == "clear")
        print(f"Wrote {len(cases)} rows to {args.out} ({n_clear} clear / {n_pause} pause)")
        return 0

    print("Nothing to do. Pass --build-dummy.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
