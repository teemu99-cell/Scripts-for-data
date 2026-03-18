#!/usr/bin/env python3
"""
translation_benchmark.py
-------------------------
Benchmarks two AI translations (e.g. your onsite AI vs Gemini) against a
confirmed, human-verified "gold standard" translation of the same source text.

Because direct machine evaluation of Russian→English quality is not possible
without NLP models, this script uses the gold standard as the reference:
both AI translations are measured against IT, not against the Russian source.
The Russian original is still used for compression-ratio and entity checks.#!/usr/bin/env python3
"""
translation_benchmark.py
-------------------------
Benchmarks two AI translations (e.g. your onsite AI vs Gemini) against a
confirmed, human-verified "gold standard" translation of the same source text.

Because direct machine evaluation of Russian→English quality is not possible
without NLP models, this script uses the gold standard as the reference:
both AI translations are measured against IT, not against the Russian source.
The Russian original is still used for compression-ratio and entity checks.

What it measures (per AI translation vs gold standard):
  1. LEXICAL SIMILARITY     — word-level overlap with gold (Jaccard + cosine)
  2. SENTENCE ALIGNMENT     — how well sentence count / structure matches gold
  3. KEYWORD RETENTION      — key terms from gold present in the AI translation
  4. NUMERIC CONSISTENCY    — numbers in gold missing from the AI translation
  5. NAMED ENTITY MATCH     — proper nouns from gold missing from AI translation
  6. LENGTH FIDELITY        — word count relative to gold (over/under translation)
  7. UNTRANSLATED RUSSIAN   — Cyrillic characters still present in the AI output
  8. SENTENCE-LEVEL DIVERGENCE — individual sentences that differ most from gold

A final head-to-head verdict shows which AI translation is closer to the gold
standard overall, and where each one wins or loses.

Usage:
    python3 translation_benchmark.py russian.txt gold.txt onsite_ai.txt gemini.txt
    python3 translation_benchmark.py russian.txt gold.txt ai1.txt ai2.txt -v
    python3 translation_benchmark.py russian.txt gold.txt ai1.txt ai2.txt \\
            --label1 "Onsite AI" --label2 "Gemini" -o report.txt --csv results.csv

Options:
    --label1 NAME       Display name for the first AI translation (default: filename)
    --label2 NAME       Display name for the second AI translation (default: filename)
    -o, --output FILE   Save text report to file
    --csv FILE          Export dimension scores to CSV
    -v, --verbose       Print sentence-level divergence detail

Dependencies:
    pip install python-docx pymupdf   (python-docx for .docx, pymupdf for .pdf)
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

# ── stop-words (English — gold and AI translations are assumed to be English) ──
STOPWORDS = {
    "the","a","an","and","or","but","in","on","at","to","of","for","with",
    "by","from","is","are","was","were","be","been","has","have","had",
    "that","this","which","it","its","they","their","we","our","as","not",
    "no","so","if","than","then","also","more","most","can","will","would",
    "could","should","may","might","very","just","only","one","two","three",
    "all","any","some","such","each","into","about","up","out","over","after",
    "before","between","both","through","during","i","he","she","him","her",
}

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
        raise ImportError("pip install python-docx")
    doc = _docx.Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)

def read_pdf(path: Path) -> str:
    if not HAS_PDF:
        raise ImportError("pip install pymupdf")
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
def word_count(text: str) -> int:
    return len(text.split())

def get_sentences(text: str) -> list:
    parts = re.split(r'(?<=[.!?])\s+|\n{2,}', text.strip())
    return [s.strip() for s in parts if len(s.strip()) > 8]

def content_words(text: str) -> list:
    tokens = re.sub(r'[^\w\s]', '', text.lower()).split()
    return [w for w in tokens if w not in STOPWORDS and len(w) > 2]

def token_set(text: str) -> set:
    return set(content_words(text))

def top_keywords(text: str, n: int = 30) -> list:
    return [w for w, _ in Counter(content_words(text)).most_common(n)]

def extract_numbers(text: str) -> set:
    return set(re.findall(r'\b\d[\d.,]*\b', text))

def extract_named_entities(text: str) -> set:
    """Capitalised tokens — rough proxy for proper nouns."""
    tokens = re.findall(
        r'\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*\b', text
    )
    noise = {"The","This","That","These","Those","There","Their",
             "However","Although","According","Therefore","Moreover",
             "Furthermore","Meanwhile","Additionally","Nevertheless"}
    return {t for t in tokens if t not in noise and len(t) >= 4}

def has_cyrillic(text: str) -> bool:
    return bool(re.search(r'[\u0400-\u04FF]', text))

def cyrillic_ratio(text: str) -> float:
    """Fraction of words that contain Cyrillic characters."""
    words = text.split()
    if not words: return 0.0
    cyrillic_words = sum(1 for w in words if re.search(r'[\u0400-\u04FF]', w))
    return cyrillic_words / len(words)

# ── similarity metrics ────────────────────────────────────────────────────────
def jaccard(a: str, b: str) -> float:
    ta, tb = token_set(a), token_set(b)
    if not ta or not tb: return 0.0
    return len(ta & tb) / len(ta | tb)

def cosine_sim(a: str, b: str) -> float:
    def tf(text):
        freq = Counter(content_words(text))
        return freq
    fa, fb = tf(a), tf(b)
    vocab  = set(fa) | set(fb)
    dot    = sum(fa[w] * fb[w] for w in vocab)
    na     = math.sqrt(sum(v**2 for v in fa.values()))
    nb     = math.sqrt(sum(v**2 for v in fb.values()))
    if na == 0 or nb == 0: return 0.0
    return dot / (na * nb)

def lexical_similarity(a: str, b: str) -> float:
    """Average of Jaccard and cosine — more robust than either alone."""
    return (jaccard(a, b) + cosine_sim(a, b)) / 2

def sentence_similarity(s1: str, s2: str) -> float:
    """Quick token-overlap similarity for two sentences."""
    t1, t2 = token_set(s1), token_set(s2)
    if not t1 or not t2: return 0.0
    return len(t1 & t2) / len(t1 | t2)

# ── data model ────────────────────────────────────────────────────────────────
@dataclass
class Dimension:
    name:    str
    score:   int          # 0–100
    verdict: str
    detail:  str = ""
    raw:     float = 0.0  # underlying metric value for CSV

@dataclass
class TranslationResult:
    label:       str
    path:        str
    text:        str
    word_count:  int
    dimensions:  list = field(default_factory=list)
    divergent_sentences: list = field(default_factory=list)  # (gold_sent, ai_sent, sim)

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
def score_against_gold(ai_text: str, gold_text: str,
                       russian_text: str, label: str, path: str) -> TranslationResult:

    ai_words   = word_count(ai_text)
    gold_words = word_count(gold_text)
    gold_sents = get_sentences(gold_text)
    ai_sents   = get_sentences(ai_text)

    dimensions = []

    # ── 1. Lexical similarity to gold ─────────────────────────────────────────
    lex_sim = lexical_similarity(ai_text, gold_text)
    if lex_sim >= 0.55:
        score, verdict = 100, "Very high lexical overlap with gold"
    elif lex_sim >= 0.40:
        score, verdict = 82,  "Good lexical overlap with gold"
    elif lex_sim >= 0.28:
        score, verdict = 63,  "Moderate overlap — notable wording differences"
    elif lex_sim >= 0.16:
        score, verdict = 42,  "Low overlap — significant divergence from gold"
    else:
        score, verdict = 20,  "Very low overlap — translation approach very different"
    dimensions.append(Dimension(
        "Lexical Similarity", score, verdict,
        f"Jaccard: {jaccard(ai_text, gold_text):.3f}   "
        f"Cosine: {cosine_sim(ai_text, gold_text):.3f}   "
        f"Combined: {lex_sim:.3f}",
        raw=round(lex_sim, 4)
    ))

    # ── 2. Sentence count alignment ───────────────────────────────────────────
    sent_ratio = len(ai_sents) / len(gold_sents) if gold_sents else 0
    if 0.85 <= sent_ratio <= 1.20:
        score, verdict = 100, f"Sentence count matches gold closely ({len(ai_sents)} vs {len(gold_sents)})"
    elif 0.70 <= sent_ratio < 0.85:
        score, verdict = 75,  f"Slightly fewer sentences than gold ({len(ai_sents)} vs {len(gold_sents)})"
    elif 1.20 < sent_ratio <= 1.40:
        score, verdict = 75,  f"Slightly more sentences than gold ({len(ai_sents)} vs {len(gold_sents)})"
    elif 0.50 <= sent_ratio < 0.70:
        score, verdict = 50,  f"Noticeably fewer sentences — possible merging/omission ({len(ai_sents)} vs {len(gold_sents)})"
    elif sent_ratio > 1.40:
        score, verdict = 55,  f"Many more sentences than gold — possible splitting ({len(ai_sents)} vs {len(gold_sents)})"
    else:
        score, verdict = 25,  f"Large sentence count gap ({len(ai_sents)} vs {len(gold_sents)})"
    dimensions.append(Dimension(
        "Sentence Alignment", score, verdict,
        f"Ratio: {sent_ratio:.2f}x gold",
        raw=round(sent_ratio, 4)
    ))

    # ── 3. Keyword retention from gold ────────────────────────────────────────
    gold_kw    = top_keywords(gold_text, 30)
    ai_kw_set  = token_set(ai_text)
    retained   = [kw for kw in gold_kw if kw in ai_kw_set]
    retention  = len(retained) / len(gold_kw) if gold_kw else 1.0
    missed_kw  = [kw for kw in gold_kw[:20] if kw not in ai_kw_set]
    if retention >= 0.72:
        score, verdict = 100, f"Excellent keyword retention ({retention:.0%})"
    elif retention >= 0.55:
        score, verdict = 80,  f"Good keyword retention ({retention:.0%})"
    elif retention >= 0.38:
        score, verdict = 58,  f"Moderate keyword retention ({retention:.0%})"
    else:
        score, verdict = 30,  f"Low keyword retention ({retention:.0%}) — key content likely rephrased or missing"
    detail = f"Missing from gold's top keywords: {', '.join(missed_kw[:8])}" if missed_kw else \
             "All top gold keywords present."
    dimensions.append(Dimension(
        "Keyword Retention", score, verdict, detail, raw=round(retention, 4)
    ))

    # ── 4. Numeric consistency with gold ──────────────────────────────────────
    gold_nums = extract_numbers(gold_text)
    ai_nums   = extract_numbers(ai_text)
    # Numbers in gold not found in AI translation
    missing_nums = {n for n in gold_nums if n not in ai_nums
                    and len(n.replace(",","").replace(".","")) >= 3}
    # Numbers in AI not in gold (introduced values)
    alien_nums = {n for n in ai_nums if n not in gold_nums
                  and len(n.replace(",","").replace(".","")) >= 3}
    issues = len(missing_nums) + len(alien_nums)
    if issues == 0:
        score, verdict = 100, "All numbers consistent with gold"
        detail = ""
    elif issues <= 2:
        score, verdict = 72,  "Minor numeric discrepancy"
        detail = (f"Missing from gold: {', '.join(sorted(missing_nums))}   "
                  f"Extra (not in gold): {', '.join(sorted(alien_nums))}")
    elif issues <= 5:
        score, verdict = 45,  "Notable numeric differences from gold"
        detail = (f"Missing: {', '.join(sorted(missing_nums)[:5])}   "
                  f"Extra: {', '.join(sorted(alien_nums)[:5])}")
    else:
        score, verdict = 20,  "Significant numeric divergence from gold"
        detail = (f"Missing: {', '.join(sorted(missing_nums)[:5])}   "
                  f"Extra: {', '.join(sorted(alien_nums)[:5])}")
    dimensions.append(Dimension(
        "Numeric Consistency", score, verdict, detail, raw=float(issues)
    ))

    # ── 5. Named entity match with gold ───────────────────────────────────────
    gold_ents = extract_named_entities(gold_text)
    ai_text_lower = ai_text.lower()
    missing_ents  = {e for e in gold_ents
                     if e.lower() not in ai_text_lower and len(e) >= 5}
    if not gold_ents:
        score, verdict, detail = 85, "No named entities detected in gold", ""
    elif not missing_ents:
        score, verdict = 100, "All named entities from gold present"
        detail = f"Checked {len(gold_ents)} entities."
    else:
        ent_coverage = 1 - len(missing_ents) / len(gold_ents)
        if ent_coverage >= 0.80:
            score, verdict = 78, f"Most entities present ({ent_coverage:.0%} coverage)"
        elif ent_coverage >= 0.60:
            score, verdict = 55, f"Some entities missing ({ent_coverage:.0%} coverage)"
        else:
            score, verdict = 30, f"Many entities missing ({ent_coverage:.0%} coverage)"
        detail = f"Missing: {', '.join(sorted(missing_ents)[:8])}"
    dimensions.append(Dimension(
        "Named Entity Match", score, verdict, detail,
        raw=round(1 - len(missing_ents)/max(len(gold_ents),1), 4)
    ))

    # ── 6. Length fidelity to gold ────────────────────────────────────────────
    length_ratio = ai_words / gold_words if gold_words else 0
    if 0.88 <= length_ratio <= 1.15:
        score, verdict = 100, f"Length closely matches gold ({ai_words} vs {gold_words} words)"
    elif 0.75 <= length_ratio < 0.88:
        score, verdict = 78,  f"Slightly shorter than gold ({ai_words} vs {gold_words} words)"
    elif 1.15 < length_ratio <= 1.35:
        score, verdict = 75,  f"Slightly longer than gold ({ai_words} vs {gold_words} words)"
    elif 0.55 <= length_ratio < 0.75:
        score, verdict = 50,  f"Notably shorter than gold — may be under-translated"
    elif length_ratio > 1.35:
        score, verdict = 55,  f"Notably longer than gold — may be over-translated or padded"
    else:
        score, verdict = 22,  f"Large length gap vs gold ({length_ratio:.2f}x)"
    dimensions.append(Dimension(
        "Length Fidelity", score, verdict,
        f"Ratio: {length_ratio:.2f}x gold",
        raw=round(length_ratio, 4)
    ))

    # ── 7. Untranslated Russian check ─────────────────────────────────────────
    cyr_ratio = cyrillic_ratio(ai_text)
    if cyr_ratio == 0:
        score, verdict = 100, "No Cyrillic characters found — fully translated"
        detail = ""
    elif cyr_ratio < 0.01:
        score, verdict = 75, "Trace Cyrillic detected — likely proper nouns left untranslated"
        detail = f"Cyrillic word ratio: {cyr_ratio:.2%}"
    elif cyr_ratio < 0.05:
        score, verdict = 45, "Some Cyrillic text remains — partial translation"
        detail = f"Cyrillic word ratio: {cyr_ratio:.2%}"
    else:
        score, verdict = 10, "Significant Cyrillic content — large portions untranslated"
        detail = f"Cyrillic word ratio: {cyr_ratio:.2%}"
    dimensions.append(Dimension(
        "Cyrillic / Untranslated", score, verdict, detail, raw=round(cyr_ratio, 4)
    ))

    # ── 8. Sentence-level divergence ──────────────────────────────────────────
    # For each gold sentence, find the closest AI sentence.
    # Flag pairs where best match is still low (translation chose very different wording).
    divergent = []
    for gs in gold_sents:
        if len(gs.split()) < 6:
            continue
        best_sim  = 0.0
        best_sent = ""
        for as_ in ai_sents:
            sim = sentence_similarity(gs, as_)
            if sim > best_sim:
                best_sim  = sim
                best_sent = as_
        if best_sim < 0.20 and best_sent:
            divergent.append((gs, best_sent, round(best_sim, 3)))

    if not divergent:
        score, verdict = 100, "No significantly divergent sentences found"
        detail = ""
    elif len(divergent) <= 2:
        score, verdict = 78,  f"{len(divergent)} sentence(s) differ notably from gold"
        detail = ""
    elif len(divergent) <= 5:
        score, verdict = 55,  f"{len(divergent)} sentences differ notably from gold"
        detail = ""
    else:
        score, verdict = 30,  f"{len(divergent)} sentences diverge significantly — systematic difference"
        detail = ""
    dimensions.append(Dimension(
        "Sentence Divergence", score, verdict, detail, raw=float(len(divergent))
    ))

    return TranslationResult(
        label      = label,
        path       = path,
        text       = ai_text,
        word_count = ai_words,
        dimensions = dimensions,
        divergent_sentences = divergent,
    )

# ── display helpers ───────────────────────────────────────────────────────────
def score_color(s: int) -> str:
    return "green" if s >= 78 else "yellow" if s >= 52 else "red"

def grade_color(g: str) -> str:
    return {"A":"green","B":"green","C":"yellow","D":"yellow","F":"red"}.get(g,"reset")

def bar(score: int, width: int = 18) -> str:
    filled = round(score / 100 * width)
    return "█" * filled + "░" * (width - filled)


# ── display — multi-translation ───────────────────────────────────────────────
def print_results_multi(results: list, gold_path: str, gold_words: int, verbose: bool):
    """Print a full comparison table for any number of AI translations."""

    print(f"\n{clr(DIV, 'bold')}")
    print(clr("  TRANSLATION BENCHMARK — vs GOLD STANDARD", "bold"))
    print(DIV)
    print(f"  Gold standard : {clr(Path(gold_path).name, 'bold')}  ({gold_words} words)")
    for i, r in enumerate(results, 1):
        print(f"  AI-{i:<3}         : {clr(r.label, 'cyan')}  ({r.word_count} words)")
    print()

    # ── per-dimension table ───────────────────────────────────────────────────
    dim_names = [d.name for d in results[0].dimensions]
    col = 14  # column width per AI

    print(clr("  DIMENSION SCORES  (vs gold standard)", "bold"))
    print(DIV2)

    # Header row
    header = f"  {'Dimension':<28}"
    for r in results:
        header += f"  {r.label[:col]:<{col}}"
    header += "  Best"
    print(header)
    print("  " + "-" * (28 + (col + 2) * len(results) + 8))

    for name in dim_names:
        scores = [next(d for d in r.dimensions if d.name == name).score
                  for r in results]
        best_score = max(scores)
        best_label = results[scores.index(best_score)].label[:12]

        row = f"  {name:<28}"
        for s in scores:
            colored = clr(f"{s:<{col}}", score_color(s))
            row += f"  {colored}"
        row += f"  {clr(best_label, 'green')}"
        print(row)

    # Overall row
    print("  " + "-" * (28 + (col + 2) * len(results) + 8))
    overalls = [r.overall for r in results]
    grades   = [r.grade   for r in results]
    best_ov  = max(overalls)
    best_r   = results[overalls.index(best_ov)]

    row = f"  {'OVERALL':<28}"
    for r in results:
        ov  = r.overall
        g   = r.grade
        val = f"{ov}({g})"
        colored = clr(f"{val:<{col}}", score_color(ov))
        row += f"  {colored}"
    row += f"  {clr(best_r.label[:12], 'green')}"
    print(row)

    # ── per-dimension details ─────────────────────────────────────────────────
    print(f"\n{clr('  DIMENSION DETAILS', 'bold')}")
    print(DIV2)
    for name in dim_names:
        print(f"\n  {clr(name, 'bold')}")
        for r in results:
            d = next(d for d in r.dimensions if d.name == name)
            sc = score_color(d.score)
            print(f"    {clr(r.label[:20] + ':', 'cyan'):<24} "
                  f"{clr(str(d.score), sc)}/100 — {d.verdict}")
            if d.detail:
                print(f"    {clr('→', 'blue')} {d.detail[:110]}")

    # ── verbose: divergent sentences ──────────────────────────────────────────
    if verbose:
        for r in results:
            if r.divergent_sentences:
                print(f"\n{clr(f'  DIVERGENT SENTENCES — {r.label}', 'bold')}")
                print(DIV2)
                for gs, as_, sim in r.divergent_sentences[:4]:
                    print(f"\n  {clr('Gold:', 'green')}  {gs[:130]}")
                    print(f"  {clr('AI:  ', 'cyan')}  {as_[:130]}")
                    print(f"  {clr(f'Overlap: {sim:.0%}', 'yellow')}")

    # ── head-to-head verdict ──────────────────────────────────────────────────
    print(f"\n{clr(DIV, 'bold')}")
    print(clr("  VERDICT", "bold"))
    print(DIV)

    # Wins per AI across all dimensions
    for r in results:
        wins = sum(1 for name in dim_names
                   if next(d for d in r.dimensions if d.name == name).score ==
                   max(next(d for d in rx.dimensions if d.name == name).score
                       for rx in results))
        g_col = grade_color(r.grade)
        print(f"  {clr(r.label, 'cyan'):<30}  "
              f"Overall: {clr(str(r.overall), score_color(r.overall))}/100  "
              f"Grade: {clr(r.grade, g_col)}  "
              f"Dimension wins: {wins}/{len(dim_names)}")

    print()
    winner = max(results, key=lambda x: x.overall)
    if len([r for r in results if r.overall == winner.overall]) > 1:
        print(f"  {clr('▶  RESULT: TIE', 'yellow')}")
    else:
        margin = winner.overall - sorted(overalls, reverse=True)[1]
        print(f"  {clr('▶  CLOSEST TO GOLD:', 'bold')} "
              f"{clr(winner.label, 'green')} "
              f"(margin: {margin} pts over second place)")

    # Flag dimensions with large spread
    print()
    for name in dim_names:
        scores = [next(d for d in r.dimensions if d.name == name).score
                  for r in results]
        spread = max(scores) - min(scores)
        if spread >= 20:
            best  = results[scores.index(max(scores))].label
            worst = results[scores.index(min(scores))].label
            print(f"  {clr('▸', 'yellow')} {clr(name, 'bold')} — "
                  f"largest spread ({spread} pts): "
                  f"{clr(best, 'green')} vs {clr(worst, 'red')}")
    print()


# ── text report ───────────────────────────────────────────────────────────────
def build_report_multi(results: list, gold_path: str, gold_words: int) -> str:
    lines = [DIV, "TRANSLATION BENCHMARK REPORT", DIV, "",
             f"Gold standard : {gold_path}  ({gold_words} words)", ""]
    for r in results:
        lines.append(f"  {r.label:<24} {r.word_count} words  "
                     f"Overall {r.overall}/100  Grade {r.grade}")
    lines += ["", "DIMENSION SCORES", "-" * 60]

    dim_names = [d.name for d in results[0].dimensions]
    for name in dim_names:
        lines.append(f"\n  {name}")
        for r in results:
            d = next(d for d in r.dimensions if d.name == name)
            lines.append(f"    {r.label}: {d.score}/100 — {d.verdict}")
            if d.detail:
                lines.append(f"      {d.detail}")

    lines += ["", "VERDICT", "-" * 60]
    winner = max(results, key=lambda x: x.overall)
    for r in results:
        lines.append(f"  {r.label}: {r.overall}/100 ({r.grade})")
    lines.append(f"  Closest to gold: {winner.label}")
    return "\n".join(lines)


# ── CSV export — multi-translation comparison matrix ─────────────────────────
def export_csv_multi(results: list, out: Path):
    """
    Produces a clean comparison matrix:
      - One row per dimension + one OVERALL row
      - One score column per AI translation
      - A 'best' column showing which AI won each dimension
      - A 'spread' column showing the gap between best and worst
    """
    dim_names = [d.name for d in results[0].dimensions]
    labels    = [r.label for r in results]

    fieldnames = ["dimension"] + \
                 [f"{l}_score" for l in labels] + \
                 [f"{l}_verdict" for l in labels] + \
                 ["best", "spread"]

    rows = []
    for name in dim_names:
        row = {"dimension": name}
        scores = []
        for r in results:
            d = next(d for d in r.dimensions if d.name == name)
            row[f"{r.label}_score"]   = d.score
            row[f"{r.label}_verdict"] = d.verdict
            scores.append((d.score, r.label))
        scores.sort(reverse=True)
        row["best"]   = scores[0][1]
        row["spread"] = scores[0][0] - scores[-1][0]
        rows.append(row)

    # Overall summary row
    row = {"dimension": "OVERALL"}
    overalls = []
    for r in results:
        row[f"{r.label}_score"]   = r.overall
        row[f"{r.label}_verdict"] = f"Grade {r.grade}"
        overalls.append((r.overall, r.label))
    overalls.sort(reverse=True)
    row["best"]   = overalls[0][1]
    row["spread"] = overalls[0][0] - overalls[-1][0]
    rows.append(row)

    # Word count row
    row = {"dimension": "Word Count"}
    for r in results:
        row[f"{r.label}_score"]   = r.word_count
        row[f"{r.label}_verdict"] = f"{r.word_count} words"
    row["best"] = ""; row["spread"] = ""
    rows.append(row)

    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(clr(f"\nCSV exported → {out}", "green"))
    print(clr(f"  {len(results)} translations × {len(dim_names)} dimensions + "
              f"overall + word count", "blue"))


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark any number of AI translations against a confirmed "
            "gold-standard translation. Accepts 2 or more AI files."
        )
    )
    parser.add_argument("russian",      nargs="?", default=None,
                        help="Original source (.txt/.docx/.pdf) — "
                             "optional, omit or use --no-source for scanned PDFs")
    parser.add_argument("gold",         help="Confirmed gold-standard translation "
                                             "(.txt/.docx/.pdf)")
    parser.add_argument("translations", nargs="+",
                        help="Two or more AI translation files to benchmark "
                             "(.txt/.docx/.pdf/.odt)")
    parser.add_argument("--labels",     nargs="*", default=[],
                        help="Display names for each translation file "
                             "(one per file, in order)")
    parser.add_argument("--no-source",  action="store_true",
                        help="Skip source file (use for scanned/image PDFs)")
    parser.add_argument("-o","--output", help="Save text report to file")
    parser.add_argument("--csv",         help="Export comparison matrix to CSV")
    parser.add_argument("-v","--verbose",action="store_true",
                        help="Print sentence-level divergence examples")
    args = parser.parse_args()

    use_source = (not args.no_source) and (args.russian is not None)

    # Validate files
    if not Path(args.gold).exists():
        print(clr(f"Error: gold file not found — {args.gold}", "red")); sys.exit(1)
    if use_source and not Path(args.russian).exists():
        print(clr(f"Error: source file not found — {args.russian}", "red")); sys.exit(1)

    tgt_paths = []
    for t in args.translations:
        p = Path(t)
        if not p.exists():
            print(clr(f"Warning: translation not found — {t}", "yellow"))
        else:
            tgt_paths.append(p)

    if len(tgt_paths) < 2:
        print(clr("Error: need at least 2 translation files to benchmark.", "red"))
        sys.exit(1)

    # Labels — pad with filenames if not enough supplied
    labels = list(args.labels)
    while len(labels) < len(tgt_paths):
        labels.append(tgt_paths[len(labels)].stem)

    # Load files
    print(clr(f"\nLoading files…", "bold"))
    try:
        russian_text = read_file(Path(args.russian)) if use_source else ""
        gold_text    = read_file(Path(args.gold))
    except Exception as e:
        print(clr(f"Error reading gold/source: {e}", "red")); sys.exit(1)

    if use_source and not russian_text.strip():
        print(clr("  Warning: source is empty — likely a scanned PDF. "
                  "Re-run with --no-source.", "yellow"))
        russian_text = ""

    if not russian_text.strip():
        print(clr("  Source               skipped (scanned/image PDF)", "yellow"))
    else:
        print(f"  {'Source':<20} {word_count(russian_text):>6} words")
    print(f"  {'Gold standard':<20} {word_count(gold_text):>6} words")

    ai_texts = []
    for p, label in zip(tgt_paths, labels):
        try:
            text = read_file(p)
        except Exception as e:
            print(clr(f"  Error reading {p.name}: {e}", "red")); continue
        if not text.strip():
            print(clr(f"  Error: {p.name} is empty.", "red")); continue
        ai_texts.append((label, str(p), text))
        print(f"  {label:<20} {word_count(text):>6} words")

    if len(ai_texts) < 2:
        print(clr("Not enough valid translation files.", "red")); sys.exit(1)

    # Score all translations
    print(clr(f"\nScoring {len(ai_texts)} translation(s) against gold standard…",
              "bold"))
    results = []
    for label, path, text in ai_texts:
        r = score_against_gold(text, gold_text, russian_text, label, path)
        results.append(r)
        g_col = grade_color(r.grade)
        print(f"  {label:<24} Overall: "
              f"{clr(str(r.overall), score_color(r.overall))}/100  "
              f"Grade: {clr(r.grade, g_col)}")

    # Display full results
    print_results_multi(results, str(args.gold), word_count(gold_text), args.verbose)

    if args.output:
        Path(args.output).write_text(
            build_report_multi(results, str(args.gold), word_count(gold_text)),
            encoding="utf-8"
        )
        print(clr(f"Report saved → {args.output}", "green"))

    if args.csv:
        export_csv_multi(results, Path(args.csv))

if __name__ == "__main__":
    main()

What it measures (per AI translation vs gold standard):
  1. LEXICAL SIMILARITY     — word-level overlap with gold (Jaccard + cosine)
  2. SENTENCE ALIGNMENT     — how well sentence count / structure matches gold
  3. KEYWORD RETENTION      — key terms from gold present in the AI translation
  4. NUMERIC CONSISTENCY    — numbers in gold missing from the AI translation
  5. NAMED ENTITY MATCH     — proper nouns from gold missing from AI translation
  6. LENGTH FIDELITY        — word count relative to gold (over/under translation)
  7. UNTRANSLATED RUSSIAN   — Cyrillic characters still present in the AI output
  8. SENTENCE-LEVEL DIVERGENCE — individual sentences that differ most from gold

A final head-to-head verdict shows which AI translation is closer to the gold
standard overall, and where each one wins or loses.

Usage:
    python3 translation_benchmark.py russian.txt gold.txt onsite_ai.txt gemini.txt
    python3 translation_benchmark.py russian.txt gold.txt ai1.txt ai2.txt -v
    python3 translation_benchmark.py russian.txt gold.txt ai1.txt ai2.txt \\
            --label1 "Onsite AI" --label2 "Gemini" -o report.txt --csv results.csv

Options:
    --label1 NAME       Display name for the first AI translation (default: filename)
    --label2 NAME       Display name for the second AI translation (default: filename)
    -o, --output FILE   Save text report to file
    --csv FILE          Export dimension scores to CSV
    -v, --verbose       Print sentence-level divergence detail

Dependencies:
    pip install python-docx pymupdf   (python-docx for .docx, pymupdf for .pdf)
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

# ── stop-words (English — gold and AI translations are assumed to be English) ──
STOPWORDS = {
    "the","a","an","and","or","but","in","on","at","to","of","for","with",
    "by","from","is","are","was","were","be","been","has","have","had",
    "that","this","which","it","its","they","their","we","our","as","not",
    "no","so","if","than","then","also","more","most","can","will","would",
    "could","should","may","might","very","just","only","one","two","three",
    "all","any","some","such","each","into","about","up","out","over","after",
    "before","between","both","through","during","i","he","she","him","her",
}

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
        raise ImportError("pip install python-docx")
    doc = _docx.Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)

def read_pdf(path: Path) -> str:
    if not HAS_PDF:
        raise ImportError("pip install pymupdf")
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
def word_count(text: str) -> int:
    return len(text.split())

def get_sentences(text: str) -> list:
    parts = re.split(r'(?<=[.!?])\s+|\n{2,}', text.strip())
    return [s.strip() for s in parts if len(s.strip()) > 8]

def content_words(text: str) -> list:
    tokens = re.sub(r'[^\w\s]', '', text.lower()).split()
    return [w for w in tokens if w not in STOPWORDS and len(w) > 2]

def token_set(text: str) -> set:
    return set(content_words(text))

def top_keywords(text: str, n: int = 30) -> list:
    return [w for w, _ in Counter(content_words(text)).most_common(n)]

def extract_numbers(text: str) -> set:
    return set(re.findall(r'\b\d[\d.,]*\b', text))

def extract_named_entities(text: str) -> set:
    """Capitalised tokens — rough proxy for proper nouns."""
    tokens = re.findall(
        r'\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*\b', text
    )
    noise = {"The","This","That","These","Those","There","Their",
             "However","Although","According","Therefore","Moreover",
             "Furthermore","Meanwhile","Additionally","Nevertheless"}
    return {t for t in tokens if t not in noise and len(t) >= 4}

def has_cyrillic(text: str) -> bool:
    return bool(re.search(r'[\u0400-\u04FF]', text))

def cyrillic_ratio(text: str) -> float:
    """Fraction of words that contain Cyrillic characters."""
    words = text.split()
    if not words: return 0.0
    cyrillic_words = sum(1 for w in words if re.search(r'[\u0400-\u04FF]', w))
    return cyrillic_words / len(words)

# ── similarity metrics ────────────────────────────────────────────────────────
def jaccard(a: str, b: str) -> float:
    ta, tb = token_set(a), token_set(b)
    if not ta or not tb: return 0.0
    return len(ta & tb) / len(ta | tb)

def cosine_sim(a: str, b: str) -> float:
    def tf(text):
        freq = Counter(content_words(text))
        return freq
    fa, fb = tf(a), tf(b)
    vocab  = set(fa) | set(fb)
    dot    = sum(fa[w] * fb[w] for w in vocab)
    na     = math.sqrt(sum(v**2 for v in fa.values()))
    nb     = math.sqrt(sum(v**2 for v in fb.values()))
    if na == 0 or nb == 0: return 0.0
    return dot / (na * nb)

def lexical_similarity(a: str, b: str) -> float:
    """Average of Jaccard and cosine — more robust than either alone."""
    return (jaccard(a, b) + cosine_sim(a, b)) / 2

def sentence_similarity(s1: str, s2: str) -> float:
    """Quick token-overlap similarity for two sentences."""
    t1, t2 = token_set(s1), token_set(s2)
    if not t1 or not t2: return 0.0
    return len(t1 & t2) / len(t1 | t2)

# ── data model ────────────────────────────────────────────────────────────────
@dataclass
class Dimension:
    name:    str
    score:   int          # 0–100
    verdict: str
    detail:  str = ""
    raw:     float = 0.0  # underlying metric value for CSV

@dataclass
class TranslationResult:
    label:       str
    path:        str
    text:        str
    word_count:  int
    dimensions:  list = field(default_factory=list)
    divergent_sentences: list = field(default_factory=list)  # (gold_sent, ai_sent, sim)

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
def score_against_gold(ai_text: str, gold_text: str,
                       russian_text: str, label: str, path: str) -> TranslationResult:

    ai_words   = word_count(ai_text)
    gold_words = word_count(gold_text)
    gold_sents = get_sentences(gold_text)
    ai_sents   = get_sentences(ai_text)

    dimensions = []

    # ── 1. Lexical similarity to gold ─────────────────────────────────────────
    lex_sim = lexical_similarity(ai_text, gold_text)
    if lex_sim >= 0.55:
        score, verdict = 100, "Very high lexical overlap with gold"
    elif lex_sim >= 0.40:
        score, verdict = 82,  "Good lexical overlap with gold"
    elif lex_sim >= 0.28:
        score, verdict = 63,  "Moderate overlap — notable wording differences"
    elif lex_sim >= 0.16:
        score, verdict = 42,  "Low overlap — significant divergence from gold"
    else:
        score, verdict = 20,  "Very low overlap — translation approach very different"
    dimensions.append(Dimension(
        "Lexical Similarity", score, verdict,
        f"Jaccard: {jaccard(ai_text, gold_text):.3f}   "
        f"Cosine: {cosine_sim(ai_text, gold_text):.3f}   "
        f"Combined: {lex_sim:.3f}",
        raw=round(lex_sim, 4)
    ))

    # ── 2. Sentence count alignment ───────────────────────────────────────────
    sent_ratio = len(ai_sents) / len(gold_sents) if gold_sents else 0
    if 0.85 <= sent_ratio <= 1.20:
        score, verdict = 100, f"Sentence count matches gold closely ({len(ai_sents)} vs {len(gold_sents)})"
    elif 0.70 <= sent_ratio < 0.85:
        score, verdict = 75,  f"Slightly fewer sentences than gold ({len(ai_sents)} vs {len(gold_sents)})"
    elif 1.20 < sent_ratio <= 1.40:
        score, verdict = 75,  f"Slightly more sentences than gold ({len(ai_sents)} vs {len(gold_sents)})"
    elif 0.50 <= sent_ratio < 0.70:
        score, verdict = 50,  f"Noticeably fewer sentences — possible merging/omission ({len(ai_sents)} vs {len(gold_sents)})"
    elif sent_ratio > 1.40:
        score, verdict = 55,  f"Many more sentences than gold — possible splitting ({len(ai_sents)} vs {len(gold_sents)})"
    else:
        score, verdict = 25,  f"Large sentence count gap ({len(ai_sents)} vs {len(gold_sents)})"
    dimensions.append(Dimension(
        "Sentence Alignment", score, verdict,
        f"Ratio: {sent_ratio:.2f}x gold",
        raw=round(sent_ratio, 4)
    ))

    # ── 3. Keyword retention from gold ────────────────────────────────────────
    gold_kw    = top_keywords(gold_text, 30)
    ai_kw_set  = token_set(ai_text)
    retained   = [kw for kw in gold_kw if kw in ai_kw_set]
    retention  = len(retained) / len(gold_kw) if gold_kw else 1.0
    missed_kw  = [kw for kw in gold_kw[:20] if kw not in ai_kw_set]
    if retention >= 0.72:
        score, verdict = 100, f"Excellent keyword retention ({retention:.0%})"
    elif retention >= 0.55:
        score, verdict = 80,  f"Good keyword retention ({retention:.0%})"
    elif retention >= 0.38:
        score, verdict = 58,  f"Moderate keyword retention ({retention:.0%})"
    else:
        score, verdict = 30,  f"Low keyword retention ({retention:.0%}) — key content likely rephrased or missing"
    detail = f"Missing from gold's top keywords: {', '.join(missed_kw[:8])}" if missed_kw else \
             "All top gold keywords present."
    dimensions.append(Dimension(
        "Keyword Retention", score, verdict, detail, raw=round(retention, 4)
    ))

    # ── 4. Numeric consistency with gold ──────────────────────────────────────
    gold_nums = extract_numbers(gold_text)
    ai_nums   = extract_numbers(ai_text)
    # Numbers in gold not found in AI translation
    missing_nums = {n for n in gold_nums if n not in ai_nums
                    and len(n.replace(",","").replace(".","")) >= 3}
    # Numbers in AI not in gold (introduced values)
    alien_nums = {n for n in ai_nums if n not in gold_nums
                  and len(n.replace(",","").replace(".","")) >= 3}
    issues = len(missing_nums) + len(alien_nums)
    if issues == 0:
        score, verdict = 100, "All numbers consistent with gold"
        detail = ""
    elif issues <= 2:
        score, verdict = 72,  "Minor numeric discrepancy"
        detail = (f"Missing from gold: {', '.join(sorted(missing_nums))}   "
                  f"Extra (not in gold): {', '.join(sorted(alien_nums))}")
    elif issues <= 5:
        score, verdict = 45,  "Notable numeric differences from gold"
        detail = (f"Missing: {', '.join(sorted(missing_nums)[:5])}   "
                  f"Extra: {', '.join(sorted(alien_nums)[:5])}")
    else:
        score, verdict = 20,  "Significant numeric divergence from gold"
        detail = (f"Missing: {', '.join(sorted(missing_nums)[:5])}   "
                  f"Extra: {', '.join(sorted(alien_nums)[:5])}")
    dimensions.append(Dimension(
        "Numeric Consistency", score, verdict, detail, raw=float(issues)
    ))

    # ── 5. Named entity match with gold ───────────────────────────────────────
    gold_ents = extract_named_entities(gold_text)
    ai_text_lower = ai_text.lower()
    missing_ents  = {e for e in gold_ents
                     if e.lower() not in ai_text_lower and len(e) >= 5}
    if not gold_ents:
        score, verdict, detail = 85, "No named entities detected in gold", ""
    elif not missing_ents:
        score, verdict = 100, "All named entities from gold present"
        detail = f"Checked {len(gold_ents)} entities."
    else:
        ent_coverage = 1 - len(missing_ents) / len(gold_ents)
        if ent_coverage >= 0.80:
            score, verdict = 78, f"Most entities present ({ent_coverage:.0%} coverage)"
        elif ent_coverage >= 0.60:
            score, verdict = 55, f"Some entities missing ({ent_coverage:.0%} coverage)"
        else:
            score, verdict = 30, f"Many entities missing ({ent_coverage:.0%} coverage)"
        detail = f"Missing: {', '.join(sorted(missing_ents)[:8])}"
    dimensions.append(Dimension(
        "Named Entity Match", score, verdict, detail,
        raw=round(1 - len(missing_ents)/max(len(gold_ents),1), 4)
    ))

    # ── 6. Length fidelity to gold ────────────────────────────────────────────
    length_ratio = ai_words / gold_words if gold_words else 0
    if 0.88 <= length_ratio <= 1.15:
        score, verdict = 100, f"Length closely matches gold ({ai_words} vs {gold_words} words)"
    elif 0.75 <= length_ratio < 0.88:
        score, verdict = 78,  f"Slightly shorter than gold ({ai_words} vs {gold_words} words)"
    elif 1.15 < length_ratio <= 1.35:
        score, verdict = 75,  f"Slightly longer than gold ({ai_words} vs {gold_words} words)"
    elif 0.55 <= length_ratio < 0.75:
        score, verdict = 50,  f"Notably shorter than gold — may be under-translated"
    elif length_ratio > 1.35:
        score, verdict = 55,  f"Notably longer than gold — may be over-translated or padded"
    else:
        score, verdict = 22,  f"Large length gap vs gold ({length_ratio:.2f}x)"
    dimensions.append(Dimension(
        "Length Fidelity", score, verdict,
        f"Ratio: {length_ratio:.2f}x gold",
        raw=round(length_ratio, 4)
    ))

    # ── 7. Untranslated Russian check ─────────────────────────────────────────
    cyr_ratio = cyrillic_ratio(ai_text)
    if cyr_ratio == 0:
        score, verdict = 100, "No Cyrillic characters found — fully translated"
        detail = ""
    elif cyr_ratio < 0.01:
        score, verdict = 75, "Trace Cyrillic detected — likely proper nouns left untranslated"
        detail = f"Cyrillic word ratio: {cyr_ratio:.2%}"
    elif cyr_ratio < 0.05:
        score, verdict = 45, "Some Cyrillic text remains — partial translation"
        detail = f"Cyrillic word ratio: {cyr_ratio:.2%}"
    else:
        score, verdict = 10, "Significant Cyrillic content — large portions untranslated"
        detail = f"Cyrillic word ratio: {cyr_ratio:.2%}"
    dimensions.append(Dimension(
        "Cyrillic / Untranslated", score, verdict, detail, raw=round(cyr_ratio, 4)
    ))

    # ── 8. Sentence-level divergence ──────────────────────────────────────────
    # For each gold sentence, find the closest AI sentence.
    # Flag pairs where best match is still low (translation chose very different wording).
    divergent = []
    for gs in gold_sents:
        if len(gs.split()) < 6:
            continue
        best_sim  = 0.0
        best_sent = ""
        for as_ in ai_sents:
            sim = sentence_similarity(gs, as_)
            if sim > best_sim:
                best_sim  = sim
                best_sent = as_
        if best_sim < 0.20 and best_sent:
            divergent.append((gs, best_sent, round(best_sim, 3)))

    if not divergent:
        score, verdict = 100, "No significantly divergent sentences found"
        detail = ""
    elif len(divergent) <= 2:
        score, verdict = 78,  f"{len(divergent)} sentence(s) differ notably from gold"
        detail = ""
    elif len(divergent) <= 5:
        score, verdict = 55,  f"{len(divergent)} sentences differ notably from gold"
        detail = ""
    else:
        score, verdict = 30,  f"{len(divergent)} sentences diverge significantly — systematic difference"
        detail = ""
    dimensions.append(Dimension(
        "Sentence Divergence", score, verdict, detail, raw=float(len(divergent))
    ))

    return TranslationResult(
        label      = label,
        path       = path,
        text       = ai_text,
        word_count = ai_words,
        dimensions = dimensions,
        divergent_sentences = divergent,
    )

# ── display helpers ───────────────────────────────────────────────────────────
def score_color(s: int) -> str:
    return "green" if s >= 78 else "yellow" if s >= 52 else "red"

def grade_color(g: str) -> str:
    return {"A":"green","B":"green","C":"yellow","D":"yellow","F":"red"}.get(g,"reset")

def bar(score: int, width: int = 18) -> str:
    filled = round(score / 100 * width)
    return "█" * filled + "░" * (width - filled)

def winner_str(v1, v2, label1, label2, higher_is_better=True) -> str:
    if higher_is_better:
        if v1 > v2:   return clr(label1, "green")
        if v2 > v1:   return clr(label2, "green")
    else:
        if v1 < v2:   return clr(label1, "green")
        if v2 < v1:   return clr(label2, "green")
    return clr("Tie", "yellow")

# ── main display ──────────────────────────────────────────────────────────────
def print_results(r1: TranslationResult, r2: TranslationResult,
                  gold_path: str, gold_words: int, verbose: bool):

    print(f"\n{clr(DIV, 'bold')}")
    print(clr("  TRANSLATION BENCHMARK — vs GOLD STANDARD", "bold"))
    print(DIV)
    print(f"  Gold standard : {clr(Path(gold_path).name, 'bold')}  ({gold_words} words)")
    print(f"  AI-1          : {clr(r1.label, 'cyan')}  ({r1.word_count} words)")
    print(f"  AI-2          : {clr(r2.label, 'magenta')}  ({r2.word_count} words)")
    print()

    # ── per-dimension table ───────────────────────────────────────────────────
    print(clr("  DIMENSION SCORES  (vs gold standard)", "bold"))
    print(DIV2)

    # Header
    col = 24
    print(f"  {'Dimension':<26}  "
          f"{clr(r1.label[:col], 'cyan'):<{col+9}}  "
          f"{clr(r2.label[:col], 'magenta'):<{col+9}}  Winner")
    print("  " + "-" * 70)

    dim_names = [d.name for d in r1.dimensions]
    for name in dim_names:
        d1 = next(d for d in r1.dimensions if d.name == name)
        d2 = next(d for d in r2.dimensions if d.name == name)
        higher = name != "Sentence Divergence"  # lower divergence = better
        w = winner_str(d1.score, d2.score, r1.label[:14], r2.label[:14], higher_is_better=True)

        b1 = clr(bar(d1.score), score_color(d1.score))
        b2 = clr(bar(d2.score), score_color(d2.score))
        s1 = clr(f"{d1.score:3d}", score_color(d1.score))
        s2 = clr(f"{d2.score:3d}", score_color(d2.score))

        print(f"  {name:<26}  {b1} {s1}   {b2} {s2}   {w}")

    # Overall
    print("  " + "-" * 70)
    g1, g2 = r1.grade, r2.grade
    o1, o2 = r1.overall, r2.overall
    b1 = clr(bar(o1), score_color(o1))
    b2 = clr(bar(o2), score_color(o2))
    s1 = clr(f"{o1:3d}", score_color(o1))
    s2 = clr(f"{o2:3d}", score_color(o2))
    grd1 = clr(f"({g1})", grade_color(g1))
    grd2 = clr(f"({g2})", grade_color(g2))
    ow = winner_str(o1, o2, r1.label[:14], r2.label[:14])
    print(f"  {'OVERALL':<26}  {b1} {s1} {grd1}  {b2} {s2} {grd2}  {ow}")

    # ── per-dimension details ─────────────────────────────────────────────────
    print(f"\n{clr('  DIMENSION DETAILS', 'bold')}")
    print(DIV2)
    for name in dim_names:
        d1 = next(d for d in r1.dimensions if d.name == name)
        d2 = next(d for d in r2.dimensions if d.name == name)
        print(f"\n  {clr(name, 'bold')}")
        print(f"    {clr(r1.label + ':', 'cyan'):<20} {d1.verdict}")
        if d1.detail:
            print(f"    {clr('→', 'blue')} {d1.detail[:110]}")
        print(f"    {clr(r2.label + ':', 'magenta'):<20} {d2.verdict}")
        if d2.detail:
            print(f"    {clr('→', 'blue')} {d2.detail[:110]}")

    # ── sentence divergence detail (verbose) ──────────────────────────────────
    if verbose:
        for r in (r1, r2):
            if r.divergent_sentences:
                print(f"\n{clr(f'  DIVERGENT SENTENCES — {r.label}', 'bold')}")
                print(DIV2)
                print(f"  {clr('(Sentences where the AI chose very different wording from gold)', 'yellow')}")
                for gs, as_, sim in r.divergent_sentences[:6]:
                    print(f"\n  {clr('Gold:', 'green')}  {gs[:130]}")
                    print(f"  {clr('AI:  ', 'cyan')}  {as_[:130]}")
                    print(f"  {clr(f'Overlap: {sim:.0%}', 'yellow')}")

    # ── head-to-head verdict ──────────────────────────────────────────────────
    print(f"\n{clr(DIV, 'bold')}")
    print(clr("  HEAD-TO-HEAD VERDICT", "bold"))
    print(DIV)

    wins1 = sum(1 for d1, d2 in zip(r1.dimensions, r2.dimensions)
                if d1.score > d2.score)
    wins2 = sum(1 for d1, d2 in zip(r1.dimensions, r2.dimensions)
                if d2.score > d1.score)
    ties  = len(r1.dimensions) - wins1 - wins2

    print(f"  {clr(r1.label, 'cyan'):<30}  wins {wins1}/{len(r1.dimensions)} dimensions")
    print(f"  {clr(r2.label, 'magenta'):<30}  wins {wins2}/{len(r2.dimensions)} dimensions")
    if ties:
        print(f"  Ties: {ties}")

    print()
    if o1 > o2:
        margin = o1 - o2
        print(f"  {clr('▶  CLOSER TO GOLD:', 'bold')} {clr(r1.label, 'green')} "
              f"(overall {o1} vs {o2}, margin: {margin} pts)")
    elif o2 > o1:
        margin = o2 - o1
        print(f"  {clr('▶  CLOSER TO GOLD:', 'bold')} {clr(r2.label, 'green')} "
              f"(overall {o2} vs {o1}, margin: {margin} pts)")
    else:
        print(f"  {clr('▶  TIED — both translations equally close to gold.', 'yellow')}")

    # Dimension-specific callouts
    print()
    for name in dim_names:
        d1 = next(d for d in r1.dimensions if d.name == name)
        d2 = next(d for d in r2.dimensions if d.name == name)
        gap = abs(d1.score - d2.score)
        if gap >= 15:
            leader = r1.label if d1.score > d2.score else r2.label
            lc = "cyan" if d1.score > d2.score else "magenta"
            print(f"  {clr('▸', 'yellow')} {clr(leader, lc)} leads on "
                  f"{clr(name, 'bold')} by {gap} pts")
    print()

# ── text report ───────────────────────────────────────────────────────────────
def build_report(r1: TranslationResult, r2: TranslationResult,
                 gold_path: str, gold_words: int) -> str:
    lines = [
        DIV, "TRANSLATION BENCHMARK REPORT", DIV, "",
        f"Gold standard : {gold_path}  ({gold_words} words)",
        f"AI-1          : {r1.label}  ({r1.word_count} words)  "
        f"Overall {r1.overall}/100  Grade {r1.grade}",
        f"AI-2          : {r2.label}  ({r2.word_count} words)  "
        f"Overall {r2.overall}/100  Grade {r2.grade}", "",
        "DIMENSION SCORES", "-" * 60,
    ]
    dim_names = [d.name for d in r1.dimensions]
    for name in dim_names:
        d1 = next(d for d in r1.dimensions if d.name == name)
        d2 = next(d for d in r2.dimensions if d.name == name)
        lines += [
            f"  {name}",
            f"    {r1.label}: {d1.score}/100 — {d1.verdict}",
            f"      {d1.detail}" if d1.detail else "",
            f"    {r2.label}: {d2.score}/100 — {d2.verdict}",
            f"      {d2.detail}" if d2.detail else "",
            "",
        ]
    lines += [
        "VERDICT", "-" * 60,
        f"  {r1.label}: {r1.overall}/100 ({r1.grade})",
        f"  {r2.label}: {r2.overall}/100 ({r2.grade})",
    ]
    if r1.overall > r2.overall:
        lines.append(f"  Closer to gold: {r1.label}")
    elif r2.overall > r1.overall:
        lines.append(f"  Closer to gold: {r2.label}")
    else:
        lines.append("  Result: Tied")
    return "\n".join(lines)

# ── CSV export ────────────────────────────────────────────────────────────────
def export_csv(r1: TranslationResult, r2: TranslationResult, out: Path):
    rows = []
    dim_names = [d.name for d in r1.dimensions]
    for name in dim_names:
        d1 = next(d for d in r1.dimensions if d.name == name)
        d2 = next(d for d in r2.dimensions if d.name == name)
        rows.append({
            "dimension":     name,
            f"{r1.label}_score":  d1.score,
            f"{r1.label}_raw":    d1.raw,
            f"{r1.label}_verdict":d1.verdict,
            f"{r2.label}_score":  d2.score,
            f"{r2.label}_raw":    d2.raw,
            f"{r2.label}_verdict":d2.verdict,
            "winner": (r1.label if d1.score > d2.score else
                       r2.label if d2.score > d1.score else "Tie"),
        })
    # Summary row
    rows.append({
        "dimension":          "OVERALL",
        f"{r1.label}_score":  r1.overall,
        f"{r1.label}_raw":    "",
        f"{r1.label}_verdict":f"Grade {r1.grade}",
        f"{r2.label}_score":  r2.overall,
        f"{r2.label}_raw":    "",
        f"{r2.label}_verdict":f"Grade {r2.grade}",
        "winner": (r1.label if r1.overall > r2.overall else
                   r2.label if r2.overall > r1.overall else "Tie"),
    })
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(clr(f"\nCSV exported → {out}", "green"))

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark two AI translations against a confirmed gold-standard "
            "translation of a Russian source text."
        )
    )
    parser.add_argument("russian",     help="Original Russian source text (.txt / .docx)")
    parser.add_argument("gold",        help="Confirmed human/gold-standard translation (.txt / .docx)")
    parser.add_argument("ai1",         help="First AI translation to evaluate (.txt / .docx)")
    parser.add_argument("ai2",         help="Second AI translation to evaluate (.txt / .docx)")
    parser.add_argument("--label1",    default=None,
                        help="Display name for ai1 (default: filename stem)")
    parser.add_argument("--label2",    default=None,
                        help="Display name for ai2 (default: filename stem)")
    parser.add_argument("-o","--output", help="Save text report to file")
    parser.add_argument("--csv",         help="Export dimension scores to CSV")
    parser.add_argument("-v","--verbose",action="store_true",
                        help="Print sentence-level divergence examples")
    args = parser.parse_args()

    paths = {
        "russian": Path(args.russian),
        "gold":    Path(args.gold),
        "ai1":     Path(args.ai1),
        "ai2":     Path(args.ai2),
    }
    for role, p in paths.items():
        if not p.exists():
            print(clr(f"Error: {role} file not found — {p}", "red"))
            sys.exit(1)

    label1 = args.label1 or paths["ai1"].stem
    label2 = args.label2 or paths["ai2"].stem

    print(clr(f"\nLoading files…", "bold"))
    try:
        russian_text = read_file(paths["russian"])
        gold_text    = read_file(paths["gold"])
        ai1_text     = read_file(paths["ai1"])
        ai2_text     = read_file(paths["ai2"])
    except Exception as e:
        print(clr(f"Error reading file: {e}", "red")); sys.exit(1)

    for name, text in [("Russian source", russian_text),
                       ("Gold standard",  gold_text),
                       (label1,           ai1_text),
                       (label2,           ai2_text)]:
        if not text.strip():
            print(clr(f"Error: {name} file is empty.", "red")); sys.exit(1)
        print(f"  {name:<20} {word_count(text):>6} words")

    print(clr("\nScoring translations against gold standard…", "bold"))

    result1 = score_against_gold(ai1_text, gold_text, russian_text, label1, str(paths["ai1"]))
    result2 = score_against_gold(ai2_text, gold_text, russian_text, label2, str(paths["ai2"]))

    print_results(result1, result2, str(paths["gold"]),
                  word_count(gold_text), args.verbose)

    if args.output:
        Path(args.output).write_text(
            build_report(result1, result2, str(paths["gold"]), word_count(gold_text)),
            encoding="utf-8"
        )
        print(clr(f"Report saved → {args.output}", "green"))

    if args.csv:
        export_csv(result1, result2, Path(args.csv))

if __name__ == "__main__":
    main()
