#!/usr/bin/env python3
"""
translation_evaluator.py
------------------------
Evaluates an AI-produced translation against its source document.
Works on .txt and .docx files — no live API or internet access needed.

What it checks:
  1. LENGTH RATIO        — target length vs source length (words & sentences)
  2. KEY TERM COVERAGE   — domain-specific terms present in source but missing in target
  3. UNTRANSLATED SEGMENTS — chunks of target that still look like the source language
  4. SENTENCE ALIGNMENT  — large count mismatch between source and target sentences
  5. NUMBER PRESERVATION — numeric values that appear in source but not in target
  6. NAMED ENTITY CHECK  — proper nouns / capitalized phrases missing from translation

Usage:
    python3 translation_evaluator.py source.txt translation.txt
    python3 translation_evaluator.py source.docx translation.txt --src-lang fi --tgt-lang en
    python3 translation_evaluator.py source.txt translation.txt -o report.txt --csv results.csv

Options:
    --src-lang LANG     Source language hint: fi | en | auto (default: auto)
    --tgt-lang LANG     Target language hint: fi | en | auto (default: auto)
    -o, --output FILE   Save text report to file
    --csv FILE          Export findings to CSV
    -v, --verbose       Print flagged segments in full

Dependencies:
    pip install python-docx --break-system-packages
"""

import re
import sys
import csv
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from collections import Counter
from typing import Optional

try:
    import docx as _docx
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

# ── colours ───────────────────────────────────────────────────────────────────
C = {
    "green":   "\033[92m", "red":     "\033[91m", "yellow": "\033[93m",
    "blue":    "\033[94m", "cyan":    "\033[96m", "bold":   "\033[1m",
    "magenta": "\033[95m", "reset":   "\033[0m",
}
def clr(t, c): return f"{C[c]}{t}{C['reset']}"
DIV  = "=" * 70
DIV2 = "-" * 70

# ── language fingerprints ─────────────────────────────────────────────────────
# High-frequency function words that strongly signal a specific language.
LANG_MARKERS = {
    "fi": ["ja","on","ei","se","että","tai","oli","ovat","myös","kuin",
           "mutta","jos","sekä","kun","tämä","nämä","siitä","niiden"],
    "en": ["the","and","is","are","was","were","of","in","to","that",
           "for","with","this","have","has","not","from","they","their"],
}

def detect_language(text: str) -> str:
    """Return 'fi', 'en', or 'unknown' based on function-word frequency."""
    words = re.sub(r'[^\w\s]', '', text.lower()).split()
    if not words:
        return "unknown"
    scores = {lang: sum(1 for w in words if w in markers) / len(words)
              for lang, markers in LANG_MARKERS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0.04 else "unknown"

# ── domain keyword list (Finnish + English, geopolitics / reports focus) ──────
DOMAIN_TERMS = [
    # Geopolitics / relations
    "nato","eu","un","g7","g20","imf","wto","oecd",
    "alliance","treaty","sanction","pakote","liittouma","sopimus",
    "diplomacy","diplomatia","sovereignty","suvereniteetti",
    # Military / security
    "military","sotilaallinen","defense","puolustus","operation","operaatio",
    "ceasefire","tulitauko","deterrence","pelote","escalation","eskalaatio",
    # Economy
    "gdp","bkt","tariff","tulli","inflation","inflaatio","recession","taantuma",
    "budget","budjetti","export","vienti","import","tuonti",
    # Governance
    "parliament","eduskunta","government","hallitus","legislation","lainsäädäntö",
    "election","vaali","constitution","perustuslaki","policy","politiikka",
]

def find_domain_terms(text: str) -> set:
    tl = text.lower()
    return {t for t in DOMAIN_TERMS if t in tl}

# ── file readers ──────────────────────────────────────────────────────────────
def read_txt(path: Path) -> str:
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode {path}")

def read_docx(path: Path) -> str:
    if not HAS_DOCX:
        raise ImportError("pip install python-docx --break-system-packages")
    doc = _docx.Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)

def read_file(path: Path) -> str:
    return read_docx(path) if path.suffix.lower() == ".docx" else read_txt(path)

# ── text utilities ────────────────────────────────────────────────────────────
def word_count(text: str) -> int:
    return len(text.split())

def get_sentences(text: str) -> list:
    parts = re.split(r'(?<=[.!?])\s+|\n{2,}', text.strip())
    return [s.strip() for s in parts if len(s.strip()) > 8]

def extract_numbers(text: str) -> set:
    return set(re.findall(r'\b\d[\d.,]*\b', text))

def extract_named_entities(text: str) -> set:
    """Capitalized multi-word or single-word tokens (rough NER substitute)."""
    tokens = re.findall(r'\b[A-ZÄÖÅ][a-zäöå]{2,}(?:\s+[A-ZÄÖÅ][a-zäöå]{2,})*\b', text)
    return {t for t in tokens if len(t) > 3}

# ── data model ────────────────────────────────────────────────────────────────
@dataclass
class Finding:
    category:  str
    severity:  str        # low | medium | high
    message:   str
    detail:    str = ""

@dataclass
class EvalResult:
    src_path:    str
    tgt_path:    str
    src_lang:    str
    tgt_lang:    str
    src_words:   int
    tgt_words:   int
    src_sents:   int
    tgt_sents:   int
    length_ratio:   float
    sent_ratio:     float
    term_coverage:  float   # 0–1
    findings:    list = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        """Simple 0–1 health score. Higher = fewer / less severe issues."""
        penalty = sum(
            0.20 if f.severity == "high"   else
            0.10 if f.severity == "medium" else 0.04
            for f in self.findings
        )
        return max(0.0, 1.0 - penalty)

# ── evaluation logic ──────────────────────────────────────────────────────────
def evaluate(src_text: str, tgt_text: str,
             src_path: str, tgt_path: str,
             src_lang: str = "auto", tgt_lang: str = "auto") -> EvalResult:

    # Language detection
    sl = src_lang if src_lang != "auto" else detect_language(src_text)
    tl = tgt_lang if tgt_lang != "auto" else detect_language(tgt_text)

    src_words = word_count(src_text)
    tgt_words = word_count(tgt_text)
    src_sents = get_sentences(src_text)
    tgt_sents = get_sentences(tgt_text)

    lr = tgt_words / src_words if src_words else 0.0
    sr = len(tgt_sents) / len(src_sents) if src_sents else 0.0

    findings = []

    # ── 1. Length ratio ───────────────────────────────────────────────────────
    # Translations typically land between 0.7× and 1.5× source length.
    # Fi→En tends to shrink; En→Fi tends to grow.
    if lr < 0.50:
        findings.append(Finding(
            "LENGTH RATIO", "high",
            f"Target is only {lr:.0%} of source length — likely truncated.",
            f"Source: {src_words} words  →  Target: {tgt_words} words"
        ))
    elif lr < 0.70:
        findings.append(Finding(
            "LENGTH RATIO", "medium",
            f"Target ({lr:.0%} of source) is shorter than typical translation range.",
            f"Source: {src_words} words  →  Target: {tgt_words} words"
        ))
    elif lr > 2.0:
        findings.append(Finding(
            "LENGTH RATIO", "high",
            f"Target is {lr:.0%} of source — possible repeated or padded content.",
            f"Source: {src_words} words  →  Target: {tgt_words} words"
        ))
    elif lr > 1.6:
        findings.append(Finding(
            "LENGTH RATIO", "medium",
            f"Target ({lr:.0%} of source) is longer than typical translation range.",
            f"Source: {src_words} words  →  Target: {tgt_words} words"
        ))

    # ── 2. Sentence count alignment ───────────────────────────────────────────
    if sr < 0.60:
        findings.append(Finding(
            "SENTENCE ALIGNMENT", "medium",
            f"Target has {len(tgt_sents)} sentences vs source {len(src_sents)} — "
            f"possible merging or omissions.",
            f"Sentence ratio: {sr:.2f}"
        ))
    elif sr > 1.8:
        findings.append(Finding(
            "SENTENCE ALIGNMENT", "low",
            f"Target has significantly more sentences ({len(tgt_sents)}) than source "
            f"({len(src_sents)}) — possible splitting or added content.",
            f"Sentence ratio: {sr:.2f}"
        ))

    # ── 3. Key term coverage ──────────────────────────────────────────────────
    src_terms = find_domain_terms(src_text)
    tgt_terms = find_domain_terms(tgt_text)
    if src_terms:
        missing = src_terms - tgt_terms
        coverage = len(tgt_terms & src_terms) / len(src_terms)
        if missing and coverage < 0.60:
            findings.append(Finding(
                "KEY TERM COVERAGE", "high",
                f"Only {coverage:.0%} of domain terms from source appear in target.",
                f"Missing: {', '.join(sorted(missing)[:10])}"
            ))
        elif missing and coverage < 0.80:
            findings.append(Finding(
                "KEY TERM COVERAGE", "medium",
                f"{coverage:.0%} domain term coverage.",
                f"Missing: {', '.join(sorted(missing)[:8])}"
            ))
    else:
        coverage = 1.0  # no terms to check → not penalized

    # ── 4. Untranslated segments ──────────────────────────────────────────────
    # If source and target share the same detected language, flag whole segments
    # that appear verbatim in both (copy-paste / forgotten paragraphs).
    if sl != "unknown" and tl != "unknown" and sl == tl:
        findings.append(Finding(
            "LANGUAGE MISMATCH", "high",
            f"Source and target appear to be the same language ({sl.upper()}).",
            "Expected different languages for a translation pair."
        ))
    else:
        # Sentence-level: flag target sentences whose word overlap with any
        # source sentence is very high (possible untranslated carry-overs).
        src_tokens_per_sent = [
            set(re.sub(r'[^\w]', ' ', s.lower()).split()) for s in src_sents
        ]
        untranslated = []
        for ts in tgt_sents:
            tts = set(re.sub(r'[^\w]', ' ', ts.lower()).split())
            if len(tts) < 5:
                continue
            for ss in src_tokens_per_sent:
                if len(ss) < 5:
                    continue
                overlap = len(tts & ss) / len(tts | ss)
                if overlap > 0.65:
                    untranslated.append(ts)
                    break
        if untranslated:
            sev = "high" if len(untranslated) >= 3 else "medium"
            findings.append(Finding(
                "UNTRANSLATED SEGMENTS", sev,
                f"{len(untranslated)} target sentence(s) look nearly identical to source.",
                "\n      ".join(s[:120] for s in untranslated[:4])
            ))

    # ── 5. Number preservation ────────────────────────────────────────────────
    src_nums = extract_numbers(src_text)
    tgt_nums = extract_numbers(tgt_text)
    missing_nums = src_nums - tgt_nums
    # Filter out very short numbers (1-2 digit years, etc.) to reduce noise
    missing_nums = {n for n in missing_nums if len(n.replace(",","").replace(".","")) >= 3}
    if missing_nums:
        sev = "high" if len(missing_nums) >= 4 else "medium" if len(missing_nums) >= 2 else "low"
        findings.append(Finding(
            "NUMBER PRESERVATION", sev,
            f"{len(missing_nums)} numeric value(s) from source not found in target.",
            f"Missing: {', '.join(sorted(missing_nums)[:8])}"
        ))

    # ── 6. Named entity check ─────────────────────────────────────────────────
    src_ents = extract_named_entities(src_text)
    tgt_ents = extract_named_entities(tgt_text)
    # Entities that appear in source but not target (exact or as a substring)
    tgt_text_lower = tgt_text.lower()
    missing_ents = {e for e in src_ents if e.lower() not in tgt_text_lower and len(e) >= 5}
    # Ignore very common words that get capitalized at sentence start
    STOPWORDS = {"This","That","The","These","Those","There","Their",
                 "Tämä","Nämä","Siitä","Niiden","Myös","Sekä"}
    missing_ents -= STOPWORDS
    if missing_ents and len(missing_ents) <= 20:
        sev = "medium" if len(missing_ents) >= 3 else "low"
        findings.append(Finding(
            "NAMED ENTITY", sev,
            f"{len(missing_ents)} proper noun(s) from source not found in target.",
            f"Missing: {', '.join(sorted(missing_ents)[:8])}"
        ))

    return EvalResult(
        src_path     = src_path,
        tgt_path     = tgt_path,
        src_lang     = sl,
        tgt_lang     = tl,
        src_words    = src_words,
        tgt_words    = tgt_words,
        src_sents    = len(src_sents),
        tgt_sents    = len(tgt_sents),
        length_ratio = round(lr, 3),
        sent_ratio   = round(sr, 3),
        term_coverage= round(coverage, 3),
        findings     = findings,
    )

# ── display ───────────────────────────────────────────────────────────────────
SEV_COLOR = {"high": "red", "medium": "yellow", "low": "cyan"}

def print_result(r: EvalResult, verbose: bool):
    score = r.overall_score
    score_color = "green" if score >= 0.80 else "yellow" if score >= 0.55 else "red"

    print(f"\n{clr(DIV, 'bold')}")
    print(clr("  TRANSLATION EVALUATION", "bold"))
    print(DIV)
    print(f"  Source      : {clr(Path(r.src_path).name, 'bold')}  "
          f"[{clr(r.src_lang.upper(), 'cyan')}]  {r.src_words} words  "
          f"{r.src_sents} sentences")
    print(f"  Target      : {clr(Path(r.tgt_path).name, 'bold')}  "
          f"[{clr(r.tgt_lang.upper(), 'magenta')}]  {r.tgt_words} words  "
          f"{r.tgt_sents} sentences")
    print(f"  Length ratio: {r.length_ratio:.2f}x    "
          f"Sentence ratio: {r.sent_ratio:.2f}x    "
          f"Term coverage: {r.term_coverage:.0%}")
    print(f"  Overall score : {clr(f'{score:.0%}', score_color)}  "
          f"({len(r.findings)} issue(s) found)")

    if not r.findings:
        print(f"\n  {clr('✓ No issues detected.', 'green')}")
        return

    print(f"\n{clr('  FINDINGS', 'bold')}")
    print(DIV2)
    for f in sorted(r.findings, key=lambda x: {"high":0,"medium":1,"low":2}[x.severity]):
        sev_c = SEV_COLOR[f.severity]
        print(f"\n  {clr(f'[{f.severity.upper()}]', sev_c)}  "
              f"{clr(f.category, 'bold')}")
        print(f"  {f.message}")
        if f.detail and verbose:
            for line in f.detail.split("\n"):
                print(f"    {clr(line, 'blue')}")
        elif f.detail:
            short = f.detail.split("\n")[0]
            print(f"    {clr(short[:120], 'blue')}")

# ── text report ───────────────────────────────────────────────────────────────
def build_report(r: EvalResult) -> str:
    lines = [
        DIV,
        "TRANSLATION EVALUATION REPORT",
        DIV, "",
        f"Source : {r.src_path}  [{r.src_lang.upper()}]",
        f"Target : {r.tgt_path}  [{r.tgt_lang.upper()}]",
        f"Words  : {r.src_words} → {r.tgt_words}  (ratio {r.length_ratio:.2f}x)",
        f"Sents  : {r.src_sents} → {r.tgt_sents}  (ratio {r.sent_ratio:.2f}x)",
        f"Term coverage : {r.term_coverage:.0%}",
        f"Overall score : {r.overall_score:.0%}",
        f"Issues found  : {len(r.findings)}", "",
    ]
    for f in r.findings:
        lines += [
            f"[{f.severity.upper()}]  {f.category}",
            f"  {f.message}",
            f"  {f.detail}" if f.detail else "",
            "",
        ]
    return "\n".join(lines)

# ── CSV export ────────────────────────────────────────────────────────────────
def export_csv(r: EvalResult, out: Path):
    rows = []
    for f in r.findings:
        rows.append({
            "severity":     f.severity,
            "category":     f.category,
            "message":      f.message,
            "detail":       f.detail[:300],
            "src_file":     Path(r.src_path).name,
            "tgt_file":     Path(r.tgt_path).name,
            "src_lang":     r.src_lang,
            "tgt_lang":     r.tgt_lang,
            "length_ratio": r.length_ratio,
            "term_coverage":r.term_coverage,
            "overall_score":f"{r.overall_score:.2f}",
        })
    if not rows:
        print(clr("No issues to export — writing summary row.", "yellow"))
        rows = [{"severity":"none","category":"PASS","message":"No issues found",
                 "detail":"","src_file":Path(r.src_path).name,
                 "tgt_file":Path(r.tgt_path).name,
                 "src_lang":r.src_lang,"tgt_lang":r.tgt_lang,
                 "length_ratio":r.length_ratio,"term_coverage":r.term_coverage,
                 "overall_score":f"{r.overall_score:.2f}"}]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(clr(f"\nCSV exported → {out}", "green"))

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate an AI-generated translation against its source document."
    )
    parser.add_argument("source",      help="Source document (.txt / .docx)")
    parser.add_argument("translation", help="Translated document (.txt / .docx)")
    parser.add_argument("--src-lang", default="auto", choices=["fi","en","auto"],
                        help="Source language hint (default: auto-detect)")
    parser.add_argument("--tgt-lang", default="auto", choices=["fi","en","auto"],
                        help="Target language hint (default: auto-detect)")
    parser.add_argument("-o", "--output", help="Save text report to file")
    parser.add_argument("--csv",          help="Export findings to CSV")
    parser.add_argument("-v", "--verbose",action="store_true",
                        help="Print full detail for each finding")
    args = parser.parse_args()

    src_p = Path(args.source)
    tgt_p = Path(args.translation)

    for p in (src_p, tgt_p):
        if not p.exists():
            print(clr(f"Error: not found — {p}", "red")); sys.exit(1)

    print(clr(f"\nReading source      : {src_p.name}", "bold"))
    print(clr(f"Reading translation : {tgt_p.name}", "bold"))

    try:
        src_text = read_file(src_p)
        tgt_text = read_file(tgt_p)
    except Exception as e:
        print(clr(f"Error reading file: {e}", "red")); sys.exit(1)

    result = evaluate(src_text, tgt_text,
                      str(src_p), str(tgt_p),
                      args.src_lang, args.tgt_lang)

    print_result(result, args.verbose)

    if args.output:
        Path(args.output).write_text(build_report(result), encoding="utf-8")
        print(clr(f"\nReport saved → {args.output}", "green"))

    if args.csv:
        export_csv(result, Path(args.csv))

if __name__ == "__main__":
    main()
