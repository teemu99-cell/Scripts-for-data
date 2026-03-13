#!/usr/bin/env python3
"""
summary_scorer.py
-----------------
Scores an AI-generated summary against its source document.
Works on .txt and .docx files — no live API or internet access needed.

What it measures:
  1. COMPRESSION RATIO    — summary length relative to source (target: 10–35%)
  2. KEYWORD RETENTION    — top source keywords present in summary
  3. TOPIC COVERAGE       — domain topic areas captured vs source
  4. HALLUCINATION SIGNAL — content in summary absent from source
  5. NUMERIC CONSISTENCY  — numbers in summary that differ from source
  6. SENTENCE QUALITY     — very short or cut-off sentences in summary

Scores are reported on a 0–100 scale per dimension and as an overall grade.

Usage:
    python3 summary_scorer.py source.txt summary.txt
    python3 summary_scorer.py source.docx summary.txt -o score_report.txt
    python3 summary_scorer.py source.txt summary.txt --csv scores.csv -v

Options:
    -o, --output FILE   Save text report to file
    --csv FILE          Export scores and findings to CSV
    -v, --verbose       Print keyword and hallucination detail

Dependencies:
    pip install python-docx --break-system-packages
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

# ── colours ───────────────────────────────────────────────────────────────────
C = {
    "green":   "\033[92m", "red":     "\033[91m", "yellow": "\033[93m",
    "blue":    "\033[94m", "cyan":    "\033[96m", "bold":   "\033[1m",
    "magenta": "\033[95m", "reset":   "\033[0m",
}
def clr(t, c): return f"{C[c]}{t}{C['reset']}"
DIV  = "=" * 70
DIV2 = "-" * 70

# ── stop-words (Finnish + English) ───────────────────────────────────────────
STOPWORDS = {
    # English
    "the","a","an","and","or","but","in","on","at","to","of","for",
    "with","by","from","is","are","was","were","be","been","has","have",
    "had","that","this","which","it","its","they","their","we","our",
    "as","not","no","so","if","than","then","also","more","most","can",
    "will","would","could","should","may","might","very","just","only",
    "one","two","three","all","any","some","such","each","into","about",
    # Finnish
    "ja","tai","on","ei","se","että","oli","ovat","myös","kuin","mutta",
    "jos","sekä","kun","tämä","nämä","siitä","niiden","hän","he","me",
    "te","minä","sinä","jo","vain","sekä","sekä","koko","eri","yli",
    "alle","sekä","koska","jotta","vaikka","siis","sillä","kuitenkin",
}

# ── domain topic groups ───────────────────────────────────────────────────────
TOPIC_GROUPS = {
    "Geopolitics":  ["nato","eu","un","sanctions","alliance","treaty","diplomacy",
                     "pakote","liittouma","sopimus","diplomatia"],
    "Military":     ["military","defense","operation","ceasefire","weapons","troops",
                     "sotilaallinen","puolustus","operaatio","joukot"],
    "Economy":      ["gdp","tariff","inflation","trade","budget","economy","fiscal",
                     "bkt","tulli","inflaatio","kauppa","budjetti","talous"],
    "Governance":   ["parliament","election","legislation","policy","constitution",
                     "eduskunta","vaali","lainsäädäntö","politiikka","perustuslaki"],
    "Climate":      ["climate","energy","emission","renewable","environment",
                     "ilmasto","energia","päästö","uusiutuva","ympäristö"],
}

def detect_topics(text: str) -> set:
    tl = text.lower()
    return {group for group, kws in TOPIC_GROUPS.items()
            if any(kw in tl for kw in kws)}

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

def tokenize(text: str) -> list:
    return re.sub(r'[^\w\s]', '', text.lower()).split()

def content_words(text: str) -> list:
    return [w for w in tokenize(text) if w not in STOPWORDS and len(w) > 2]

def get_sentences(text: str) -> list:
    parts = re.split(r'(?<=[.!?])\s+|\n{2,}', text.strip())
    return [s.strip() for s in parts if len(s.strip()) > 5]

def extract_numbers(text: str) -> set:
    return set(re.findall(r'\b\d[\d.,]*\b', text))

def top_keywords(text: str, n: int = 25) -> list:
    """Return top-n content words by frequency."""
    freq = Counter(content_words(text))
    return [w for w, _ in freq.most_common(n)]

def token_set(text: str) -> set:
    return set(content_words(text))

def jaccard(a: str, b: str) -> float:
    ta, tb = token_set(a), token_set(b)
    if not ta or not tb: return 0.0
    return len(ta & tb) / len(ta | tb)

# ── scoring ───────────────────────────────────────────────────────────────────
@dataclass
class Dimension:
    name:    str
    score:   int          # 0–100
    verdict: str          # short label
    detail:  str = ""

@dataclass
class SummaryScore:
    src_path:     str
    sum_path:     str
    src_words:    int
    sum_words:    int
    compression:  float    # summary / source word ratio
    dimensions:   list = field(default_factory=list)
    hallucinated: list = field(default_factory=list)   # suspect phrases

    @property
    def overall(self) -> int:
        if not self.dimensions: return 0
        return round(sum(d.score for d in self.dimensions) / len(self.dimensions))

    @property
    def grade(self) -> str:
        s = self.overall
        return ("A" if s >= 88 else "B" if s >= 75 else
                "C" if s >= 60 else "D" if s >= 45 else "F")

def score_summary(src_text: str, sum_text: str,
                  src_path: str, sum_path: str) -> SummaryScore:

    src_words = word_count(src_text)
    sum_words = word_count(sum_text)
    compression = sum_words / src_words if src_words else 0.0

    src_sents = get_sentences(src_text)
    sum_sents = get_sentences(sum_text)

    src_kw = top_keywords(src_text, 30)
    sum_kw_set = token_set(sum_text)

    hallucinated = []
    dimensions = []

    # ── 1. Compression ratio ──────────────────────────────────────────────────
    # Ideal: 10–35% of source.
    if 0.10 <= compression <= 0.35:
        score, verdict = 100, "Good compression"
    elif 0.05 <= compression < 0.10:
        score, verdict = 70, "Slightly over-compressed"
    elif 0.35 < compression <= 0.55:
        score, verdict = 75, "Slightly verbose"
    elif compression < 0.05:
        score, verdict = 35, "Extremely compressed — likely incomplete"
    elif 0.55 < compression <= 0.80:
        score, verdict = 50, "Very verbose — more extraction than summary"
    else:
        score, verdict = 20, "Summary nearly as long as source"
    dimensions.append(Dimension(
        "Compression Ratio", score, verdict,
        f"{compression:.1%} of source  ({sum_words} / {src_words} words)"
    ))

    # ── 2. Keyword retention ──────────────────────────────────────────────────
    retained = [kw for kw in src_kw if kw in sum_kw_set]
    retention_rate = len(retained) / len(src_kw) if src_kw else 1.0
    if retention_rate >= 0.70:
        score, verdict = 100, "Excellent keyword retention"
    elif retention_rate >= 0.50:
        score, verdict = 80, "Good keyword retention"
    elif retention_rate >= 0.35:
        score, verdict = 60, "Moderate keyword retention"
    elif retention_rate >= 0.20:
        score, verdict = 40, "Low keyword retention"
    else:
        score, verdict = 20, "Poor keyword retention — key content likely missing"
    missed = [kw for kw in src_kw[:20] if kw not in sum_kw_set]
    dimensions.append(Dimension(
        "Keyword Retention", score, verdict,
        f"{retention_rate:.0%} of top keywords retained.  "
        f"Missing: {', '.join(missed[:8])}" if missed else
        f"{retention_rate:.0%} of top keywords retained."
    ))

    # ── 3. Topic coverage ─────────────────────────────────────────────────────
    src_topics = detect_topics(src_text)
    sum_topics = detect_topics(sum_text)
    if src_topics:
        covered = src_topics & sum_topics
        topic_rate = len(covered) / len(src_topics)
        missing_topics = src_topics - covered
        if topic_rate >= 0.85:
            score, verdict = 100, "All topics covered"
        elif topic_rate >= 0.65:
            score, verdict = 80, "Most topics covered"
        elif topic_rate >= 0.45:
            score, verdict = 55, "Some topics missing"
        else:
            score, verdict = 30, "Major topic gaps"
        detail = (f"Covered: {', '.join(sorted(covered)) or 'none'}  |  "
                  f"Missing: {', '.join(sorted(missing_topics)) or 'none'}")
    else:
        score, verdict, detail = 85, "No domain topics detected", "Possibly non-domain text"
        topic_rate = 1.0
    dimensions.append(Dimension("Topic Coverage", score, verdict, detail))

    # ── 4. Hallucination signal ───────────────────────────────────────────────
    # Flag summary sentences whose content words overlap very little with source.
    src_token_set = token_set(src_text)
    suspect_sents = []
    for ss in sum_sents:
        ss_tokens = token_set(ss)
        if len(ss_tokens) < 5:
            continue
        overlap = len(ss_tokens & src_token_set) / len(ss_tokens)
        if overlap < 0.25:
            suspect_sents.append((ss, round(overlap, 2)))
            hallucinated.append(ss)

    if not suspect_sents:
        score, verdict = 100, "No hallucination signals"
        detail = "All summary sentences traceable to source vocabulary."
    elif len(suspect_sents) == 1:
        score, verdict = 70, "1 suspect sentence"
        detail = f"Low source overlap: '{suspect_sents[0][0][:100]}' (overlap {suspect_sents[0][1]:.0%})"
    elif len(suspect_sents) <= 3:
        score, verdict = 45, f"{len(suspect_sents)} suspect sentences"
        detail = "  |  ".join(f"'{s[:80]}' ({o:.0%})" for s, o in suspect_sents[:3])
    else:
        score, verdict = 20, f"{len(suspect_sents)} suspect sentences — review carefully"
        detail = "  |  ".join(f"'{s[:60]}'" for s, _ in suspect_sents[:4])
    dimensions.append(Dimension("Hallucination Signal", score, verdict, detail))

    # ── 5. Numeric consistency ────────────────────────────────────────────────
    src_nums = extract_numbers(src_text)
    sum_nums = extract_numbers(sum_text)
    # Numbers in summary that do NOT appear in source
    alien_nums = {n for n in sum_nums if n not in src_nums
                  and len(n.replace(",","").replace(".","")) >= 3}
    if not alien_nums:
        score, verdict = 100, "All numbers traceable to source"
        detail = ""
    elif len(alien_nums) <= 2:
        score, verdict = 65, "Minor numeric discrepancy"
        detail = f"Numbers in summary not found in source: {', '.join(sorted(alien_nums))}"
    else:
        score, verdict = 30, "Multiple alien numbers — possible hallucination"
        detail = f"Not in source: {', '.join(sorted(alien_nums)[:8])}"
    dimensions.append(Dimension("Numeric Consistency", score, verdict, detail))

    # ── 6. Sentence quality ───────────────────────────────────────────────────
    short_sents = [s for s in sum_sents if len(s.split()) < 5]
    cut_sents   = [s for s in sum_sents if not re.search(r'[.!?]$', s.rstrip())]
    issues = len(short_sents) + len(cut_sents)
    if issues == 0:
        score, verdict = 100, "Clean sentence structure"
        detail = ""
    elif issues <= 2:
        score, verdict = 75, "Minor sentence quality issues"
        detail = (f"Short (<5 words): {len(short_sents)}  |  "
                  f"Possibly cut off: {len(cut_sents)}")
    else:
        score, verdict = 45, "Multiple sentence quality issues"
        detail = (f"Short sentences: {len(short_sents)}  |  "
                  f"Cut-off sentences: {len(cut_sents)}")
    dimensions.append(Dimension("Sentence Quality", score, verdict, detail))

    return SummaryScore(
        src_path     = src_path,
        sum_path     = sum_path,
        src_words    = src_words,
        sum_words    = sum_words,
        compression  = round(compression, 3),
        dimensions   = dimensions,
        hallucinated = hallucinated,
    )

# ── display ───────────────────────────────────────────────────────────────────
def score_color(s: int) -> str:
    return "green" if s >= 80 else "yellow" if s >= 55 else "red"

def grade_color(g: str) -> str:
    return {"A":"green","B":"green","C":"yellow","D":"yellow","F":"red"}.get(g,"reset")

def bar(score: int, width: int = 20) -> str:
    filled = round(score / 100 * width)
    return "█" * filled + "░" * (width - filled)

def print_score(r: SummaryScore, verbose: bool):
    print(f"\n{clr(DIV, 'bold')}")
    print(clr("  SUMMARY SCORER", "bold"))
    print(DIV)
    print(f"  Source  : {clr(Path(r.src_path).name, 'bold')}  ({r.src_words} words)")
    print(f"  Summary : {clr(Path(r.sum_path).name, 'bold')}  ({r.sum_words} words  "
          f"= {r.compression:.1%} of source)")
    print()

    for d in r.dimensions:
        sc = d.score
        print(f"  {d.name:<25} {clr(bar(sc), score_color(sc))} "
              f"{clr(str(sc), score_color(sc)):>10}   {d.verdict}")
        if d.detail and verbose:
            print(f"    {clr(d.detail[:130], 'blue')}")
        elif d.detail:
            print(f"    {clr(d.detail[:100], 'blue')}")

    print(DIV2)
    g = r.grade
    ov = r.overall
    print(f"  {'Overall':25} {clr(bar(ov), score_color(ov))} "
          f"{clr(str(ov), score_color(ov)):>10}   "
          f"Grade: {clr(g, grade_color(g))}")

    if r.hallucinated and verbose:
        print(f"\n  {clr('Suspect sentences (low source overlap):', 'yellow')}")
        for s in r.hallucinated[:5]:
            print(f"    {clr('▸', 'yellow')} {s[:140]}")

# ── text report ───────────────────────────────────────────────────────────────
def build_report(r: SummaryScore) -> str:
    lines = [
        DIV, "SUMMARY SCORER REPORT", DIV, "",
        f"Source  : {r.src_path}  ({r.src_words} words)",
        f"Summary : {r.sum_path}  ({r.sum_words} words = {r.compression:.1%} of source)",
        f"Overall : {r.overall}/100  Grade: {r.grade}", "",
    ]
    for d in r.dimensions:
        lines += [
            f"  {d.name}",
            f"    Score  : {d.score}/100",
            f"    Verdict: {d.verdict}",
            f"    Detail : {d.detail}" if d.detail else "",
            "",
        ]
    if r.hallucinated:
        lines.append("Suspect sentences:")
        for s in r.hallucinated:
            lines.append(f"  ▸ {s[:200]}")
    return "\n".join(lines)

# ── CSV export ────────────────────────────────────────────────────────────────
def export_csv(r: SummaryScore, out: Path):
    rows = [{
        "src_file":    Path(r.src_path).name,
        "sum_file":    Path(r.sum_path).name,
        "src_words":   r.src_words,
        "sum_words":   r.sum_words,
        "compression": f"{r.compression:.3f}",
        "overall":     r.overall,
        "grade":       r.grade,
        "dimension":   d.name,
        "score":       d.score,
        "verdict":     d.verdict,
        "detail":      d.detail[:300],
    } for d in r.dimensions]

    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(clr(f"\nCSV exported → {out}", "green"))

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Score an AI-generated summary against its source document."
    )
    parser.add_argument("source",  help="Original source document (.txt / .docx)")
    parser.add_argument("summary", help="AI-generated summary (.txt / .docx)")
    parser.add_argument("-o", "--output", help="Save text report to file")
    parser.add_argument("--csv",          help="Export scores to CSV")
    parser.add_argument("-v", "--verbose",action="store_true",
                        help="Print full detail for each dimension")
    args = parser.parse_args()

    src_p = Path(args.source)
    sum_p = Path(args.summary)

    for p in (src_p, sum_p):
        if not p.exists():
            print(clr(f"Error: not found — {p}", "red")); sys.exit(1)

    print(clr(f"\nLoading source  : {src_p.name}", "bold"))
    print(clr(f"Loading summary : {sum_p.name}", "bold"))

    try:
        src_text = read_file(src_p)
        sum_text = read_file(sum_p)
    except Exception as e:
        print(clr(f"Error reading file: {e}", "red")); sys.exit(1)

    if not src_text.strip():
        print(clr("Source document is empty.", "red")); sys.exit(1)
    if not sum_text.strip():
        print(clr("Summary document is empty.", "red")); sys.exit(1)

    result = score_summary(src_text, sum_text, str(src_p), str(sum_p))
    print_score(result, args.verbose)

    if args.output:
        Path(args.output).write_text(build_report(result), encoding="utf-8")
        print(clr(f"\nReport saved → {args.output}", "green"))

    if args.csv:
        export_csv(result, Path(args.csv))

if __name__ == "__main__":
    main()
