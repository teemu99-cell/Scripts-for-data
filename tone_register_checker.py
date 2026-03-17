#!/usr/bin/env python3
"""
tone_register_checker.py
─────────────────────────────────────────────────────────────────────────────
Checks whether an AI-produced translation or summary preserves the *tone and
register* of its source document. Catches cases where a formal institutional
document comes back sounding casual, hedged, or over-inflated.

Register = the level of formality, technicality, and stylistic conventions
appropriate to a document type. A military brief, a legal contract, and a
customer email all use very different registers even if they cover the same topic.

Requires:
    pip install python-docx pymupdf --break-system-packages

Usage:
    python3 tone_register_checker.py source.txt translation.txt
    python3 tone_register_checker.py source.docx ai1.txt ai2.txt \\
        --label "Onsite AI" --label "Gemini" -o report.txt --csv scores.csv -v

Flags:
    --label NAME    Display name per output file, repeatable
    -o FILE         Save report to file
    --csv FILE      Export dimension scores to CSV
    -v              Print example sentences for flagged dimensions

Dimensions scored:
    Formality Level         Formal vs casual vocabulary and phrasing
    Passive Voice Ratio     Formal docs use passive heavily; drift is detectable
    Sentence Complexity     Average clause depth and sentence length
    Hedging / Uncertainty   AI models tend to over-hedge (perhaps, it seems, etc.)
    AI-ism Contamination    Filler phrases that appear in AI output but not source
    Vocabulary Overlap      Domain-specific vocabulary retained from source

Grades: A (88+) · B (75+) · C (60+) · D (45+) · F (<45)
"""

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Word lists
# ─────────────────────────────────────────────────────────────────────────────

FORMAL_MARKERS = {
    "therefore", "furthermore", "subsequently", "accordingly", "notwithstanding",
    "henceforth", "pursuant", "whereby", "aforementioned", "hereinafter",
    "constitute", "establish", "provide", "ensure", "require", "maintain",
    "conduct", "implement", "undertake", "designate", "authorize", "facilitate",
    "initiate", "terminate", "execute", "prohibit", "mandate", "coordinate",
    "assess", "determine", "evaluate", "specify", "indicate", "demonstrate",
}

CASUAL_MARKERS = {
    "okay", "ok", "yeah", "yep", "nope", "gonna", "wanna", "kinda", "sorta",
    "stuff", "things", "get", "got", "lots", "big", "good", "bad", "nice",
    "pretty", "really", "very", "just", "like", "so", "also", "but",
    "anyway", "basically", "literally", "actually", "honestly", "clearly",
}

HEDGE_PHRASES = [
    r"\bit seems?\b", r"\bit appears?\b", r"\bperhaps\b", r"\bpossibly\b",
    r"\bmight\b", r"\bcould be\b", r"\bmay be\b", r"\bone might\b",
    r"\bsome would argue\b", r"\bit is worth noting\b", r"\binterestingly\b",
    r"\bnotably\b", r"\bimportantly\b", r"\bit should be noted\b",
    r"\bin some cases\b", r"\bin many ways\b", r"\bto some extent\b",
]

AI_ISMS = [
    r"\bdelve\b", r"\bin conclusion\b", r"\bto summarize\b", r"\bin summary\b",
    r"\bit is important to note\b", r"\bit is worth noting\b",
    r"\bfurthermore\b.*\badditionally\b",  # stacked connectors
    r"\bin the realm of\b", r"\bin the context of\b", r"\blandscape\b",
    r"\bpivotal\b", r"\bunderscores?\b", r"\bemphasizes?\b", r"\bfostering\b",
    r"\btailored\b", r"\brobust\b", r"\bseamlessly\b", r"\bleverage\b",
    r"\bsynergy\b", r"\bholistic\b", r"\bparadigm\b", r"\bnavigating\b",
    r"\bin today's world\b", r"\bin today's society\b",
]

PASSIVE_PATTERN = re.compile(
    r"\b(is|are|was|were|be|been|being)\s+\w+ed\b", re.IGNORECASE
)


# ─────────────────────────────────────────────────────────────────────────────
# Text extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_text(path: str) -> str:
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".docx":
        try:
            from docx import Document
            return "\n".join(para.text for para in Document(path).paragraphs)
        except ImportError:
            sys.exit("python-docx not installed. Run: pip install python-docx --break-system-packages")
    elif ext == ".pdf":
        try:
            import fitz
            return "\n".join(page.get_text() for page in fitz.open(path))
        except ImportError:
            sys.exit("pymupdf not installed. Run: pip install pymupdf --break-system-packages")
    else:
        for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                return Path(path).read_text(encoding=enc)
            except (UnicodeDecodeError, LookupError):
                continue
    sys.exit(f"Could not decode {path}")


def tokenise(text: str) -> list:
    return re.findall(r"\b[a-zA-Z']+\b", text.lower())


def split_sentences(text: str) -> list:
    text = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text)
            if len(s.strip()) > 10]


# ─────────────────────────────────────────────────────────────────────────────
# Scoring dimensions
# ─────────────────────────────────────────────────────────────────────────────

def score_formality(tokens: list) -> tuple:
    """
    Ratio of formal to casual vocabulary.
    Returns (score_0_100, formal_count, casual_count, casual_examples).
    """
    formal = sum(1 for t in tokens if t in FORMAL_MARKERS)
    casual = sum(1 for t in tokens if t in CASUAL_MARKERS)
    total = formal + casual
    if total == 0:
        return 70.0, 0, 0, []
    ratio = formal / total
    score = round(min(100, ratio * 130), 1)   # 0.77+ formal ratio → 100
    examples = [t for t in tokens if t in CASUAL_MARKERS]
    return score, formal, casual, list(set(examples))[:8]


def score_passive_voice(text: str, sentences: list) -> tuple:
    """
    Passive voice ratio compared to source baseline.
    Returns (raw_ratio_0_1, passive_sentence_count, examples).
    """
    passive_sents = [s for s in sentences if PASSIVE_PATTERN.search(s)]
    ratio = len(passive_sents) / len(sentences) if sentences else 0
    return ratio, len(passive_sents), passive_sents[:3]


def score_sentence_complexity(sentences: list) -> tuple:
    """
    Average word count per sentence and average comma count (clause depth proxy).
    Returns (avg_length, avg_commas).
    """
    if not sentences:
        return 0.0, 0.0
    lengths = [len(s.split()) for s in sentences]
    commas  = [s.count(",") for s in sentences]
    return round(sum(lengths) / len(lengths), 1), round(sum(commas) / len(commas), 2)


def count_hedge_phrases(text: str) -> tuple:
    """
    Count AI-style hedging phrases. Returns (count, matched_phrases).
    """
    found = []
    for pattern in HEDGE_PHRASES:
        matches = re.findall(pattern, text, re.IGNORECASE)
        found.extend(matches)
    return len(found), list(set(found))[:6]


def count_ai_isms(text: str) -> tuple:
    """
    Count AI-characteristic filler phrases. Returns (count, examples).
    """
    found = []
    for pattern in AI_ISMS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        found.extend(matches)
    return len(found), list(set(found))[:6]


def score_vocabulary_overlap(src_tokens: list, tgt_tokens: list) -> float:
    """
    Jaccard similarity of the unique vocabulary sets.
    Measures how much domain-specific vocabulary the output retained.
    Very low overlap often means the AI substituted generic words for
    domain-specific ones — a common register problem.
    """
    # Remove ultra-common stop words before comparing
    STOPS = {
        "the", "a", "an", "of", "in", "to", "and", "or", "is", "are", "was",
        "were", "be", "been", "it", "its", "that", "this", "by", "for", "on",
        "at", "as", "not", "with", "from", "but", "have", "has", "had", "will",
        "would", "shall", "should", "may", "can", "could",
    }
    src_vocab = {t for t in set(src_tokens) if t not in STOPS and len(t) > 3}
    tgt_vocab = {t for t in set(tgt_tokens) if t not in STOPS and len(t) > 3}
    if not src_vocab:
        return 0.0
    intersection = src_vocab & tgt_vocab
    union = src_vocab | tgt_vocab
    return round((len(intersection) / len(union)) * 100, 1)


def build_register_score(src_metrics: dict, tgt_metrics: dict) -> dict:
    """
    Compare target metrics against source and produce per-dimension scores.

    Key insight: we are NOT scoring the target in isolation. We are scoring
    *how similar* the target's register is to the source's register.
    """
    scores = {}

    # ── Formality ────────────────────────────────────────────────────────────
    src_formality = src_metrics["formality_score"]
    tgt_formality = tgt_metrics["formality_score"]
    formality_drift = abs(src_formality - tgt_formality)
    scores["formality"] = round(max(0, 100 - formality_drift * 1.5), 1)

    # ── Passive voice drift ───────────────────────────────────────────────────
    src_passive = src_metrics["passive_ratio"]
    tgt_passive = tgt_metrics["passive_ratio"]
    passive_drift = abs(src_passive - tgt_passive)
    scores["passive_voice"] = round(max(0, 100 - passive_drift * 200), 1)

    # ── Sentence complexity drift ─────────────────────────────────────────────
    src_len, src_commas = src_metrics["avg_sent_len"], src_metrics["avg_commas"]
    tgt_len, tgt_commas = tgt_metrics["avg_sent_len"], tgt_metrics["avg_commas"]
    len_drift    = abs(src_len - tgt_len) / max(src_len, 1)
    comma_drift  = abs(src_commas - tgt_commas) / max(src_commas, 0.1)
    scores["sentence_complexity"] = round(
        max(0, 100 - (len_drift * 40) - (comma_drift * 20)), 1)

    # ── Hedging penalty (absolute — hedging not present in source is bad) ─────
    hedge_per_1k = tgt_metrics["hedge_count"] / max(tgt_metrics["word_count"] / 1000, 0.1)
    scores["hedging"] = round(max(0, 100 - hedge_per_1k * 25), 1)

    # ── AI-ism penalty ────────────────────────────────────────────────────────
    aism_per_1k = tgt_metrics["aism_count"] / max(tgt_metrics["word_count"] / 1000, 0.1)
    scores["ai_isms"] = round(max(0, 100 - aism_per_1k * 30), 1)

    # ── Vocabulary overlap ────────────────────────────────────────────────────
    scores["vocabulary_overlap"] = tgt_metrics["vocab_overlap"]

    # ── Weighted composite ────────────────────────────────────────────────────
    weights = {
        "formality": 0.25,
        "passive_voice": 0.15,
        "sentence_complexity": 0.15,
        "hedging": 0.20,
        "ai_isms": 0.15,
        "vocabulary_overlap": 0.10,
    }
    overall = sum(scores[k] * weights[k] for k in weights)
    scores["overall"] = round(overall, 1)

    return scores


def grade(score: float) -> str:
    if score >= 88: return "A"
    if score >= 75: return "B"
    if score >= 60: return "C"
    if score >= 45: return "D"
    return "F"


# ─────────────────────────────────────────────────────────────────────────────
# Compute metrics for a single document
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(text: str, src_tokens_for_overlap: list = None) -> dict:
    tokens    = tokenise(text)
    sentences = split_sentences(text)
    word_count = len(tokens)

    formality_score, formal_ct, casual_ct, casual_ex = score_formality(tokens)
    passive_ratio, passive_ct, passive_ex = score_passive_voice(text, sentences)
    avg_len, avg_commas = score_sentence_complexity(sentences)
    hedge_count, hedge_ex = count_hedge_phrases(text)
    aism_count, aism_ex = count_ai_isms(text)
    vocab_overlap = (score_vocabulary_overlap(src_tokens_for_overlap, tokens)
                     if src_tokens_for_overlap else 0.0)

    return {
        "word_count":      word_count,
        "sent_count":      len(sentences),
        "formality_score": formality_score,
        "formal_count":    formal_ct,
        "casual_count":    casual_ct,
        "casual_examples": casual_ex,
        "passive_ratio":   passive_ratio,
        "passive_count":   passive_ct,
        "passive_examples":passive_ex,
        "avg_sent_len":    avg_len,
        "avg_commas":      avg_commas,
        "hedge_count":     hedge_count,
        "hedge_examples":  hedge_ex,
        "aism_count":      aism_count,
        "aism_examples":   aism_ex,
        "vocab_overlap":   vocab_overlap,
        "sentences":       sentences,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def format_report(src_path: str, src_metrics: dict,
                  results: list, verbose: bool) -> str:
    W = 70
    lines = []
    lines.append("=" * W)
    lines.append("TONE & REGISTER CONSISTENCY REPORT")
    lines.append(f"Source : {src_path}")
    lines.append("=" * W)

    # Source baseline
    lines.append("")
    lines.append("SOURCE BASELINE")
    lines.append("-" * W)
    lines.append(f"  Words            : {src_metrics['word_count']}")
    lines.append(f"  Sentences        : {src_metrics['sent_count']}")
    lines.append(f"  Formality score  : {src_metrics['formality_score']:.1f}")
    lines.append(f"  Passive ratio    : {src_metrics['passive_ratio']:.0%}")
    lines.append(f"  Avg sent length  : {src_metrics['avg_sent_len']} words")
    lines.append(f"  Avg commas/sent  : {src_metrics['avg_commas']:.2f}")
    lines.append(f"  Hedge phrases    : {src_metrics['hedge_count']}")
    lines.append(f"  AI-isms          : {src_metrics['aism_count']}")

    for res in results:
        m = res["metrics"]
        s = res["scores"]
        lines.append("")
        lines.append(f"OUTPUT: {res['label']}")
        lines.append("-" * W)
        lines.append(f"  Overall Register Score    : {s['overall']:>6.1f} / 100   Grade: {grade(s['overall'])}")
        lines.append(f"  Formality Level           : {s['formality']:>6.1f} / 100"
                     f"  (source: {src_metrics['formality_score']:.1f}, output: {m['formality_score']:.1f})")
        lines.append(f"  Passive Voice Ratio       : {s['passive_voice']:>6.1f} / 100"
                     f"  (source: {src_metrics['passive_ratio']:.0%}, output: {m['passive_ratio']:.0%})")
        lines.append(f"  Sentence Complexity       : {s['sentence_complexity']:>6.1f} / 100"
                     f"  (source avg: {src_metrics['avg_sent_len']} words, output: {m['avg_sent_len']} words)")
        lines.append(f"  Hedging / Uncertainty     : {s['hedging']:>6.1f} / 100"
                     f"  ({m['hedge_count']} hedge phrases found)")
        lines.append(f"  AI-ism Contamination      : {s['ai_isms']:>6.1f} / 100"
                     f"  ({m['aism_count']} AI-isms found)")
        lines.append(f"  Vocabulary Overlap        : {s['vocabulary_overlap']:>6.1f} / 100")

        if verbose:
            if m["casual_examples"]:
                lines.append("")
                lines.append(f"  Casual vocabulary detected: {', '.join(m['casual_examples'])}")
            if m["hedge_examples"]:
                lines.append(f"  Hedge phrases detected    : {', '.join(m['hedge_examples'])}")
            if m["aism_examples"]:
                lines.append(f"  AI-isms detected          : {', '.join(m['aism_examples'])}")
            if m["passive_examples"]:
                lines.append(f"  Passive voice examples:")
                for ex in m["passive_examples"][:2]:
                    snippet = ex[:80] + ("…" if len(ex) > 80 else "")
                    lines.append(f"    · {snippet}")

    lines.append("")
    if len(results) > 1:
        lines.append("RANKING")
        lines.append("-" * W)
        ranked = sorted(results, key=lambda r: r["scores"]["overall"], reverse=True)
        for rank, r in enumerate(ranked, 1):
            lines.append(f"  #{rank}  {r['label']:<40} {r['scores']['overall']:.1f}  ({grade(r['scores']['overall'])})")
    lines.append("")
    lines.append("=" * W)
    return "\n".join(lines)


def write_csv(csv_path: str, src_path: str, results: list):
    fieldnames = ["source_file", "output_file", "overall", "formality",
                  "passive_voice", "sentence_complexity", "hedging",
                  "ai_isms", "vocabulary_overlap", "grade"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            s = res["scores"]
            writer.writerow({
                "source_file":          src_path,
                "output_file":          res["label"],
                "overall":              s["overall"],
                "formality":            s["formality"],
                "passive_voice":        s["passive_voice"],
                "sentence_complexity":  s["sentence_complexity"],
                "hedging":              s["hedging"],
                "ai_isms":              s["ai_isms"],
                "vocabulary_overlap":   s["vocabulary_overlap"],
                "grade":                grade(s["overall"]),
            })


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Check tone and register consistency between source and AI output."
    )
    parser.add_argument("source", help="Source document (.txt / .docx / .pdf)")
    parser.add_argument("outputs", nargs="+",
                        help="One or more AI output files to evaluate")
    parser.add_argument("--label", action="append", dest="labels", default=[],
                        help="Display name per output file, repeatable")
    parser.add_argument("-o", "--output", dest="out_file",
                        help="Save report to file")
    parser.add_argument("--csv", dest="csv_file",
                        help="Export dimension scores to CSV")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print example sentences for flagged dimensions")
    args = parser.parse_args()

    labels = list(args.labels)
    for i in range(len(labels), len(args.outputs)):
        labels.append(Path(args.outputs[i]).name)

    src_text    = extract_text(args.source)
    src_tokens  = tokenise(src_text)
    src_metrics = compute_metrics(src_text)

    results = []
    for out_path, label in zip(args.outputs, labels):
        tgt_text    = extract_text(out_path)
        tgt_metrics = compute_metrics(tgt_text, src_tokens_for_overlap=src_tokens)
        scores      = build_register_score(src_metrics, tgt_metrics)
        results.append({"label": label, "metrics": tgt_metrics, "scores": scores})

    report = format_report(args.source, src_metrics, results, args.verbose)
    print(report)

    if args.out_file:
        Path(args.out_file).write_text(report, encoding="utf-8")
        print(f"Report saved to {args.out_file}")

    if args.csv_file:
        write_csv(args.csv_file, args.source, results)
        print(f"CSV saved to {args.csv_file}")


if __name__ == "__main__":
    main()
