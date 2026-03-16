#!/usr/bin/env python3
"""
readability_scorer.py
---------------------
Scores how natural and fluent an AI-generated text reads, independent of
accuracy. A translation or summary can be technically correct but still
read awkwardly — this script catches that.

Works on any AI output: translations, summaries, session answers.
No source document required (though one can be provided for comparison).

What it measures:
  1. SENTENCE LENGTH VARIETY  — mix of short and long sentences (natural prose varies)
  2. WORD COMPLEXITY          — ratio of long/complex words (overly dense text is harder to read)
  3. PASSIVE VOICE RATIO      — high passive voice is a common sign of awkward AI translation
  4. REPETITION               — same phrases/words repeated too close together
  5. AVG SENTENCE LENGTH      — very long sentences suggest run-ons; very short suggest choppy prose
  6. PARAGRAPH STRUCTURE      — walls of text with no paragraph breaks
  7. FILLER / HEDGE OVERUSE   — excessive AI-isms like "it is worth noting", "it should be mentioned"
  8. PUNCTUATION FLOW         — unusual punctuation patterns that break reading flow

Scores are 0–100 per dimension, overall score with A–F grade.
If a source document is provided, scores are also compared against it as a baseline.

Usage:
    python3 readability_scorer.py translation.txt
    python3 readability_scorer.py ai_output.docx --source original.txt
    python3 readability_scorer.py ai1.txt ai2.txt --label "Onsite AI" --label "Gemini"
    python3 readability_scorer.py output.txt -o report.txt --csv scores.csv -v

Options:
    --source FILE       Source document for baseline comparison (optional)
    --label NAME        Display name per file (repeatable)
    -o, --output FILE   Save text report to file
    --csv FILE          Export scores to CSV
    -v, --verbose       Print examples for each flagged dimension

Dependencies:
    pip install python-docx pymupdf --break-system-packages
"""

import re
import sys
import csv
import math
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

try:
    import fitz
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

# ── colours ───────────────────────────────────────────────────────────────────
C = {
    "green":   "\033[92m", "red":     "\033[91m", "yellow": "\033[93m",
    "blue":    "\033[94m", "cyan":    "\033[96m", "bold":   "\033[1m",
    "magenta": "\033[95m", "reset":   "\033[0m",
}
def clr(t, c): return f"{C[c]}{t}{C['reset']}"
DIV  = "=" * 72
DIV2 = "-" * 72

# ── filler / AI-ism phrases ───────────────────────────────────────────────────
FILLER_PHRASES = [
    "it is worth noting", "it should be noted", "it is important to note",
    "it is worth mentioning", "it should be mentioned", "it is noteworthy",
    "needless to say", "as mentioned above", "as stated above",
    "as previously mentioned", "as noted above", "first and foremost",
    "in conclusion", "to summarize", "in summary", "to sum up",
    "it goes without saying", "as a matter of fact", "the fact of the matter",
    "it can be seen that", "it can be observed", "one can see that",
    "in light of the above", "taking into account", "taking into consideration",
    "with regard to", "with respect to", "in terms of",
    "on the other hand", "on one hand",  # not bad alone but flagged if overused
    "at the end of the day", "all things considered",
]

# ── passive voice indicators (English) ───────────────────────────────────────
PASSIVE_PATTERNS = [
    re.compile(r'\b(?:is|are|was|were|be|been|being)\s+\w+ed\b', re.I),
    re.compile(r'\b(?:is|are|was|were)\s+(?:being\s+)?\w+en\b', re.I),
    re.compile(r'\bhas|have|had\s+been\s+\w+ed\b', re.I),
]

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

def read_pdf(path: Path) -> str:
    if not HAS_PDF:
        raise ImportError("pip install pymupdf --break-system-packages")
    lines = []
    with fitz.open(str(path)) as doc:
        for page in doc:
            for line in page.get_text().splitlines():
                if line.strip():
                    lines.append(line)
    return "\n".join(lines)

def read_file(path: Path) -> str:
    s = path.suffix.lower()
    if s == ".docx": return read_docx(path)
    if s == ".pdf":  return read_pdf(path)
    return read_txt(path)

# ── text utilities ────────────────────────────────────────────────────────────
def get_sentences(text: str) -> list:
    parts = re.split(r'(?<=[.!?])\s+|\n{2,}', text.strip())
    return [s.strip() for s in parts if len(s.strip()) > 5]

def get_paragraphs(text: str) -> list:
    parts = re.split(r'\n{2,}', text.strip())
    return [p.strip() for p in parts if p.strip()]

def word_count(text: str) -> int:
    return len(text.split())

def syllable_count(word: str) -> int:
    """Rough syllable count — counts vowel groups."""
    word = word.lower().strip(".,!?;:'\"")
    count = len(re.findall(r'[aeiouy]+', word))
    if word.endswith('e') and count > 1:
        count -= 1
    return max(1, count)

def is_complex_word(word: str) -> bool:
    """Words with 3+ syllables are considered complex."""
    return syllable_count(word) >= 3

# ── data model ────────────────────────────────────────────────────────────────
@dataclass
class Dimension:
    name:    str
    score:   int
    verdict: str
    detail:  str  = ""
    examples:list = field(default_factory=list)
    raw:     float = 0.0

@dataclass
class ReadabilityResult:
    label:      str
    path:       str
    word_count: int
    dimensions: list = field(default_factory=list)

    @property
    def overall(self) -> int:
        if not self.dimensions: return 0
        return round(sum(d.score for d in self.dimensions) / len(self.dimensions))

    @property
    def grade(self) -> str:
        s = self.overall
        return ("A" if s >= 88 else "B" if s >= 75 else
                "C" if s >= 60 else "D" if s >= 45 else "F")

# ── scoring engine ────────────────────────────────────────────────────────────
def score_readability(text: str, label: str, path: str) -> ReadabilityResult:
    sentences  = get_sentences(text)
    paragraphs = get_paragraphs(text)
    words      = text.split()
    wc         = len(words)

    dimensions = []

    if wc < 20:
        dimensions.append(Dimension(
            "Text Too Short", 50,
            "Text is too short for reliable readability scoring.",
            f"Only {wc} words found.", raw=float(wc)
        ))
        return ReadabilityResult(label=label, path=path,
                                 word_count=wc, dimensions=dimensions)

    sent_lengths = [len(s.split()) for s in sentences]
    avg_sent_len  = sum(sent_lengths) / max(len(sent_lengths), 1)

    # ── 1. Sentence length variety ────────────────────────────────────────────
    # Good prose mixes short and long sentences. We measure the standard
    # deviation of sentence lengths relative to the mean.
    if len(sent_lengths) >= 3:
        variance = sum((l - avg_sent_len) ** 2 for l in sent_lengths) / len(sent_lengths)
        std_dev  = math.sqrt(variance)
        cv       = std_dev / max(avg_sent_len, 1)  # coefficient of variation
    else:
        cv = 0.0

    if cv >= 0.55:
        score, verdict = 100, "Good sentence length variety"
    elif cv >= 0.35:
        score, verdict = 80,  "Reasonable sentence length variety"
    elif cv >= 0.20:
        score, verdict = 58,  "Low sentence variety — prose feels monotonous"
    else:
        score, verdict = 30,  "Very uniform sentence length — mechanical / repetitive feel"
    dimensions.append(Dimension(
        "Sentence Length Variety", score, verdict,
        f"Coefficient of variation: {cv:.2f}  "
        f"(Avg: {avg_sent_len:.0f} words, StdDev: {std_dev:.1f})",
        raw=round(cv, 4)
    ))

    # ── 2. Average sentence length ────────────────────────────────────────────
    # Ideal range: 15–25 words. Under 10 = choppy, over 35 = run-ons.
    if 15 <= avg_sent_len <= 25:
        score, verdict = 100, f"Ideal average sentence length ({avg_sent_len:.0f} words)"
    elif 10 <= avg_sent_len < 15:
        score, verdict = 78,  f"Slightly short average sentence ({avg_sent_len:.0f} words)"
    elif 25 < avg_sent_len <= 35:
        score, verdict = 75,  f"Slightly long average sentence ({avg_sent_len:.0f} words)"
    elif avg_sent_len < 10:
        score, verdict = 45,  f"Very short sentences ({avg_sent_len:.0f} words avg) — choppy prose"
    else:
        score, verdict = 40,  f"Very long sentences ({avg_sent_len:.0f} words avg) — likely run-ons"
    long_sents  = [s for s in sentences if len(s.split()) > 45]
    short_sents = [s for s in sentences if len(s.split()) < 5]
    detail = ""
    examples = []
    if long_sents:
        detail += f"{len(long_sents)} sentence(s) over 45 words.  "
        examples += [s[:100] + "…" for s in long_sents[:2]]
    if short_sents:
        detail += f"{len(short_sents)} sentence(s) under 5 words."
        examples += short_sents[:2]
    dimensions.append(Dimension(
        "Avg Sentence Length", score, verdict, detail, examples, raw=round(avg_sent_len, 2)
    ))

    # ── 3. Word complexity ────────────────────────────────────────────────────
    # Ratio of complex words (3+ syllables). Over 20% is dense.
    content_words = [w for w in words if len(w) > 3 and w.isalpha()]
    if content_words:
        complex_ratio = sum(1 for w in content_words if is_complex_word(w)) / len(content_words)
    else:
        complex_ratio = 0.0

    if complex_ratio <= 0.12:
        score, verdict = 100, f"Low word complexity — accessible language ({complex_ratio:.0%})"
    elif complex_ratio <= 0.18:
        score, verdict = 85,  f"Moderate word complexity ({complex_ratio:.0%})"
    elif complex_ratio <= 0.25:
        score, verdict = 62,  f"High word complexity ({complex_ratio:.0%}) — may feel dense"
    else:
        score, verdict = 35,  f"Very high word complexity ({complex_ratio:.0%}) — difficult to read"
    dimensions.append(Dimension(
        "Word Complexity", score, verdict,
        f"{complex_ratio:.0%} of content words have 3+ syllables.",
        raw=round(complex_ratio, 4)
    ))

    # ── 4. Passive voice ratio ────────────────────────────────────────────────
    passive_hits = []
    for sent in sentences:
        for pat in PASSIVE_PATTERNS:
            matches = pat.findall(sent)
            if matches:
                passive_hits.append(sent)
                break
    passive_ratio = len(passive_hits) / max(len(sentences), 1)

    if passive_ratio <= 0.15:
        score, verdict = 100, f"Low passive voice use ({passive_ratio:.0%} of sentences)"
    elif passive_ratio <= 0.25:
        score, verdict = 80,  f"Moderate passive voice ({passive_ratio:.0%})"
    elif passive_ratio <= 0.40:
        score, verdict = 55,  f"High passive voice ({passive_ratio:.0%}) — common in translated text"
    else:
        score, verdict = 28,  f"Very high passive voice ({passive_ratio:.0%}) — strongly affects fluency"
    dimensions.append(Dimension(
        "Passive Voice Ratio", score, verdict,
        f"{len(passive_hits)} of {len(sentences)} sentences use passive constructions.",
        [s[:100] for s in passive_hits[:3]],
        raw=round(passive_ratio, 4)
    ))

    # ── 5. Repetition ─────────────────────────────────────────────────────────
    # Check for content words that appear multiple times within a short window.
    stopwords = {"the","a","an","and","or","but","in","on","at","to","of",
                 "for","with","by","from","is","are","was","were","it","this",
                 "that","be","been","has","have","had","not","as","so","if"}
    content_tokens = [w.lower() for w in words
                      if w.isalpha() and w.lower() not in stopwords and len(w) > 3]
    window_size = 30
    repetitions = []
    for i, word in enumerate(content_tokens):
        window = content_tokens[max(0, i-window_size):i]
        if window.count(word) >= 2:
            repetitions.append(word)

    rep_ratio = len(set(repetitions)) / max(len(content_tokens), 1)
    if rep_ratio <= 0.03:
        score, verdict = 100, "Low word repetition"
    elif rep_ratio <= 0.07:
        score, verdict = 78,  "Some repetition detected"
    elif rep_ratio <= 0.13:
        score, verdict = 52,  "Noticeable repetition — same words used closely together"
    else:
        score, verdict = 28,  "High repetition — significantly affects reading flow"
    top_reps = [w for w, _ in Counter(repetitions).most_common(5)]
    dimensions.append(Dimension(
        "Repetition", score, verdict,
        f"{len(set(repetitions))} word(s) repeated within 30-word windows.  "
        f"Most repeated: {', '.join(top_reps)}" if top_reps else "",
        raw=round(rep_ratio, 4)
    ))

    # ── 6. Paragraph structure ────────────────────────────────────────────────
    if not paragraphs:
        score, verdict = 50, "No paragraph breaks detected"
        detail = "Entire text is one block."
    else:
        para_lengths = [word_count(p) for p in paragraphs]
        avg_para_len = sum(para_lengths) / len(para_lengths)
        long_paras   = [p for p in paragraphs if word_count(p) > 150]

        if avg_para_len <= 120 and not long_paras:
            score, verdict = 100, f"Good paragraph structure ({len(paragraphs)} paragraphs)"
        elif avg_para_len <= 200:
            score, verdict = 75, f"Moderate paragraph length (avg {avg_para_len:.0f} words)"
        else:
            score, verdict = 45, f"Long paragraphs (avg {avg_para_len:.0f} words) — walls of text"
        detail = (f"{len(paragraphs)} paragraph(s), avg {avg_para_len:.0f} words each.  "
                  f"{len(long_paras)} paragraph(s) over 150 words." if long_paras else
                  f"{len(paragraphs)} paragraph(s), avg {avg_para_len:.0f} words each.")
    dimensions.append(Dimension(
        "Paragraph Structure", score, verdict, detail,
        raw=float(len(paragraphs))
    ))

    # ── 7. Filler / AI-ism overuse ────────────────────────────────────────────
    text_lower = text.lower()
    found_fillers = []
    for phrase in FILLER_PHRASES:
        count = text_lower.count(phrase)
        if count >= 1:
            found_fillers.append((phrase, count))
    found_fillers.sort(key=lambda x: -x[1])

    total_filler = sum(c for _, c in found_fillers)
    filler_density = total_filler / max(len(sentences), 1)

    if not found_fillers:
        score, verdict = 100, "No filler phrases detected"
        detail = ""
    elif filler_density <= 0.05:
        score, verdict = 85, f"Minor filler usage ({len(found_fillers)} unique phrase(s))"
        detail = ", ".join(f"'{p}' ×{c}" for p, c in found_fillers[:3])
    elif filler_density <= 0.15:
        score, verdict = 58, f"Moderate filler overuse ({total_filler} occurrences)"
        detail = ", ".join(f"'{p}' ×{c}" for p, c in found_fillers[:4])
    else:
        score, verdict = 28, f"Heavy filler overuse ({total_filler} occurrences) — affects naturalness"
        detail = ", ".join(f"'{p}' ×{c}" for p, c in found_fillers[:5])
    dimensions.append(Dimension(
        "Filler / AI-isms", score, verdict, detail,
        [p for p, _ in found_fillers[:3]],
        raw=round(filler_density, 4)
    ))

    # ── 8. Punctuation flow ───────────────────────────────────────────────────
    issues = []
    # Double spaces
    double_spaces = len(re.findall(r'  +', text))
    if double_spaces > 3:
        issues.append(f"{double_spaces} double-space(s)")
    # Missing space after punctuation
    missing_space = len(re.findall(r'[.!?,;:][A-Za-z]', text))
    if missing_space > 2:
        issues.append(f"{missing_space} missing space after punctuation")
    # Repeated punctuation
    repeated_punct = len(re.findall(r'[.!?,;:]{3,}', text))
    if repeated_punct > 0:
        issues.append(f"{repeated_punct} repeated punctuation cluster(s)")
    # Unclosed parentheses
    open_p  = text.count("(")
    close_p = text.count(")")
    if abs(open_p - close_p) > 2:
        issues.append(f"Unbalanced parentheses ({open_p} open, {close_p} close)")

    if not issues:
        score, verdict = 100, "Clean punctuation throughout"
        detail = ""
    elif len(issues) == 1:
        score, verdict = 78, "Minor punctuation issue"
        detail = issues[0]
    elif len(issues) == 2:
        score, verdict = 55, "Some punctuation issues"
        detail = " | ".join(issues)
    else:
        score, verdict = 35, "Multiple punctuation issues"
        detail = " | ".join(issues)
    dimensions.append(Dimension(
        "Punctuation Flow", score, verdict, detail,
        raw=float(len(issues))
    ))

    return ReadabilityResult(
        label      = label,
        path       = path,
        word_count = wc,
        dimensions = dimensions,
    )

# ── display ───────────────────────────────────────────────────────────────────
def score_color(s: int) -> str:
    return "green" if s >= 78 else "yellow" if s >= 52 else "red"

def grade_color(g: str) -> str:
    return {"A":"green","B":"green","C":"yellow","D":"yellow","F":"red"}.get(g,"reset")

def bar(score: int, width: int = 20) -> str:
    filled = round(score / 100 * width)
    return "█" * filled + "░" * (width - filled)

def print_result(r: ReadabilityResult, verbose: bool):
    print(f"\n{clr(DIV, 'bold')}")
    print(clr(f"  READABILITY SCORE — {r.label}", "bold"))
    print(DIV)
    print(f"  File       : {clr(Path(r.path).name, 'blue')}  ({r.word_count} words)")
    print()

    for d in r.dimensions:
        sc = d.score
        print(f"  {d.name:<28} {clr(bar(sc), score_color(sc))}  "
              f"{clr(str(sc), score_color(sc)):>6}")
        if d.detail:
            print(f"    {clr(d.detail[:120], 'blue')}")
        if verbose and d.examples:
            for ex in d.examples[:2]:
                print(f"    {clr('▸', 'yellow')} {ex[:110]}")

    print(DIV2)
    ov = r.overall
    g  = r.grade
    print(f"  {'OVERALL':<28} {clr(bar(ov), score_color(ov))}  "
          f"{clr(str(ov), score_color(ov)):>6}   "
          f"Grade: {clr(g, grade_color(g))}")

def print_comparison(results: list):
    if len(results) < 2:
        return
    print(f"\n{clr(DIV, 'bold')}")
    print(clr("  READABILITY COMPARISON", "bold"))
    print(DIV)

    dim_names = [d.name for d in results[0].dimensions]
    labels    = [r.label for r in results]

    # Header
    col = 20
    print(f"  {'Dimension':<28}  " +
          "  ".join(clr(f"{l[:col]:<{col}}", "cyan" if i == 0 else "magenta")
                    for i, l in enumerate(labels)))
    print("  " + "-" * (28 + (col + 4) * len(results)))

    for name in dim_names:
        scores = []
        for r in results:
            d = next((d for d in r.dimensions if d.name == name), None)
            scores.append(d.score if d else 0)
        row = f"  {name:<28}  "
        row += "  ".join(
            clr(f"{s:<{col}}", score_color(s)) for s in scores
        )
        print(row)

    print("  " + "-" * (28 + (col + 4) * len(results)))
    overall_scores = [r.overall for r in results]
    row = f"  {'OVERALL':<28}  "
    row += "  ".join(
        clr(f"{s:<{col}}", score_color(s)) for s in overall_scores
    )
    print(row)

    best = max(results, key=lambda x: x.overall)
    print(f"\n  {clr('Most readable:', 'bold')} {clr(best.label, 'green')} "
          f"({best.overall}/100  Grade {best.grade})")

# ── text report ───────────────────────────────────────────────────────────────
def build_report(results: list) -> str:
    lines = [DIV, "READABILITY SCORER REPORT", DIV, ""]
    for r in results:
        lines += [
            f"File    : {r.path}",
            f"Label   : {r.label}",
            f"Words   : {r.word_count}",
            f"Overall : {r.overall}/100  Grade {r.grade}", "",
        ]
        for d in r.dimensions:
            lines += [
                f"  {d.name}",
                f"    Score  : {d.score}/100",
                f"    Verdict: {d.verdict}",
                f"    Detail : {d.detail}" if d.detail else "",
                "",
            ]
    return "\n".join(lines)

# ── CSV export ────────────────────────────────────────────────────────────────
def export_csv(results: list, out: Path):
    rows = []
    for r in results:
        for d in r.dimensions:
            rows.append({
                "label":   r.label,
                "file":    Path(r.path).name,
                "overall": r.overall,
                "grade":   r.grade,
                "dimension": d.name,
                "score":   d.score,
                "verdict": d.verdict,
                "raw":     d.raw,
                "detail":  d.detail[:200],
            })
    if not rows:
        print(clr("No data to export.", "yellow")); return
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(clr(f"\nCSV exported → {out}", "green"))

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Score the readability and fluency of AI-generated text."
    )
    parser.add_argument("files", nargs="+",
                        help="Text file(s) to score (.txt / .docx / .pdf)")
    parser.add_argument("--source",  default=None,
                        help="Optional source document for baseline comparison")
    parser.add_argument("--label",   nargs="*", default=[], dest="labels",
                        help="Display name per file (one per file)")
    parser.add_argument("-o", "--output", help="Save report to file")
    parser.add_argument("--csv",          help="Export scores to CSV")
    parser.add_argument("-v", "--verbose",action="store_true",
                        help="Print example sentences for flagged dimensions")
    args = parser.parse_args()

    paths = []
    for f in args.files:
        p = Path(f)
        if not p.exists():
            print(clr(f"Warning: not found — {f}", "yellow"))
        else:
            paths.append(p)
    if not paths:
        print(clr("No valid files found.", "red")); sys.exit(1)

    labels = list(args.labels)
    while len(labels) < len(paths):
        labels.append(paths[len(labels)].stem)

    # Optional source baseline
    src_result = None
    if args.source:
        src_p = Path(args.source)
        if src_p.exists():
            print(clr(f"\nScoring source baseline: {src_p.name}", "bold"))
            try:
                src_text = read_file(src_p)
                src_result = score_readability(src_text, f"SOURCE: {src_p.stem}", str(src_p))
                print_result(src_result, args.verbose)
            except Exception as e:
                print(clr(f"  Error loading source: {e}", "red"))
        else:
            print(clr(f"Warning: source not found — {args.source}", "yellow"))

    results = []
    for p, label in zip(paths, labels):
        try:
            text = read_file(p)
        except Exception as e:
            print(clr(f"Error reading {p.name}: {e}", "red")); continue
        if not text.strip():
            print(clr(f"Error: {p.name} is empty.", "red")); continue
        r = score_readability(text, label, str(p))
        results.append(r)
        print_result(r, args.verbose)

    if len(results) > 1:
        print_comparison(results)
    elif src_result and results:
        print_comparison([src_result, results[0]])

    if args.output:
        all_results = ([src_result] if src_result else []) + results
        Path(args.output).write_text(build_report(all_results), encoding="utf-8")
        print(clr(f"\nReport saved → {args.output}", "green"))

    if args.csv:
        all_results = ([src_result] if src_result else []) + results
        export_csv(all_results, Path(args.csv))

if __name__ == "__main__":
    main()
