#!/usr/bin/env python3
"""
terminology_checker.py
----------------------
Checks that key terms are translated consistently throughout a document.
Given a source document and one or more translations, it finds terms that
appear multiple times in the source and verifies they are rendered the same
way each time in the translation.

Inconsistent terminology is a common AI translation problem — the same phrase
may be translated as "collective defence" in one paragraph and "joint security"
in another. For official or government documents this matters significantly.

What it checks:
  1. TERM CONSISTENCY   — key multi-word phrases translated differently across occurrences
  2. ANCHOR TERMS       — user-supplied glossary terms (source → expected translation)
  3. PARTIAL COVERAGE   — source term present many times but translation count mismatches
  4. DROPPED TERMS      — source term appears frequently but has no clear equivalent in translation

Usage:
    # Basic check — auto-detects key terms
    python3 terminology_checker.py source.txt translation.txt

    # With a custom glossary (source_term=expected_translation pairs)
    python3 terminology_checker.py source.txt translation.txt \\
            --glossary "collective defence=collective defence" \\
                       "вооружённые силы=armed forces"

    # Compare two translations for terminology consistency
    python3 terminology_checker.py source.txt ai1.txt ai2.txt \\
            --label "Onsite AI" --label "Gemini" -o report.txt --csv terms.csv

Options:
    --glossary TERM=TRANSLATION    Expected term pairs (repeatable)
    --label NAME                   Display name for each translation (repeatable,
                                   must match number of translation files)
    --min-freq N                   Minimum source frequency to track a term (default: 2)
    -o, --output FILE              Save text report to file
    --csv FILE                     Export findings to CSV
    -v, --verbose                  Show all occurrences for each inconsistent term

Dependencies:
    pip install python-docx pymupdf --break-system-packages
"""

import re
import sys
import csv
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from collections import Counter, defaultdict
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

# ── stop-words (English + Russian/Ukrainian common words in Latin) ────────────
STOPWORDS = {
    "the","a","an","and","or","but","in","on","at","to","of","for","with",
    "by","from","is","are","was","were","be","been","has","have","had",
    "that","this","which","it","its","they","their","we","our","as","not",
    "no","so","if","than","then","also","more","most","will","would","can",
    "could","should","may","might","very","just","only","all","any","some",
    "such","each","into","about","up","out","over","after","before","both",
    "through","during","i","he","she","him","her","its","these","those",
    "been","being","do","did","does","done","shall","must","need","used",
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
    return [s.strip() for s in parts if len(s.strip()) > 8]

def clean(text: str) -> str:
    return re.sub(r'[^\w\s\-]', ' ', text.lower())

def extract_ngrams(text: str, n: int) -> list:
    """Extract all n-grams from text, filtering stopwords on edges."""
    words = clean(text).split()
    ngrams = []
    for i in range(len(words) - n + 1):
        gram = words[i:i+n]
        # Skip if first or last word is a stopword or very short
        if gram[0] in STOPWORDS or gram[-1] in STOPWORDS:
            continue
        if any(len(w) < 3 for w in gram):
            continue
        ngrams.append(" ".join(gram))
    return ngrams

def find_term_in_sentences(term: str, sentences: list) -> list:
    """Return list of sentences containing the term (case-insensitive)."""
    tl = term.lower()
    return [s for s in sentences if tl in s.lower()]

def extract_surrounding_words(term: str, text: str, window: int = 4) -> list:
    """
    For each occurrence of term in text, extract the window words that follow it.
    This gives a rough picture of how the term is being used in context.
    """
    pattern = re.compile(re.escape(term), re.IGNORECASE)
    results = []
    words = text.split()
    text_lower = text.lower()
    for m in pattern.finditer(text_lower):
        # find word index roughly
        preceding = text[:m.start()].split()
        following = text[m.end():].split()[:window]
        results.append(" ".join(following))
    return results

# ── term extraction ───────────────────────────────────────────────────────────
def extract_candidate_terms(text: str, min_freq: int = 2) -> dict:
    """
    Extract multi-word terms that appear at least min_freq times.
    Returns {term: count} for bigrams and trigrams.
    """
    candidates = Counter()
    for n in (2, 3):
        candidates.update(extract_ngrams(text, n))
    # Keep only those meeting frequency threshold
    return {t: c for t, c in candidates.items() if c >= min_freq}

# ── translation variant finder ────────────────────────────────────────────────
def find_translation_variants(src_term: str, src_text: str,
                               tgt_text: str, window: int = 5) -> dict:
    """
    For each occurrence of src_term in source, find what appears in the
    corresponding position in the target using sentence-level alignment.
    
    Returns a Counter of candidate translations (bigrams/trigrams following
    a concept anchor) — this is a heuristic, not a true aligner.
    """
    src_sents = get_sentences(src_text)
    tgt_sents = get_sentences(tgt_text)

    # Find source sentences containing the term
    src_hits = find_term_in_sentences(src_term, src_sents)
    if not src_hits:
        return Counter()

    # For each source sentence, find its rough positional equivalent in target
    # and extract candidate n-grams from that target sentence
    candidates = Counter()
    n_src = len(src_sents)
    n_tgt = len(tgt_sents)

    for src_sent in src_hits:
        try:
            idx = src_sents.index(src_sent)
        except ValueError:
            continue
        # Map to proportional position in target
        tgt_idx = min(round(idx * n_tgt / max(n_src, 1)), n_tgt - 1)
        # Search a small window around that position
        search_range = range(max(0, tgt_idx - 2), min(n_tgt, tgt_idx + 3))
        for ti in search_range:
            tgt_sent = tgt_sents[ti]
            for n in (2, 3):
                for gram in extract_ngrams(tgt_sent, n):
                    candidates[gram] += 1

    return candidates

# ── data model ────────────────────────────────────────────────────────────────
@dataclass
class TermFinding:
    term:       str           # source term
    category:   str           # INCONSISTENT | DROPPED | ANCHOR_MISMATCH | FREQUENCY_GAP
    severity:   str           # high | medium | low
    message:    str
    detail:     str = ""
    variants:   list = field(default_factory=list)  # translation variants found

@dataclass
class TermResult:
    label:      str
    path:       str
    findings:   list = field(default_factory=list)
    checked:    int  = 0      # number of terms checked
    passed:     int  = 0

    @property
    def score(self) -> int:
        if self.checked == 0: return 100
        high   = sum(1 for f in self.findings if f.severity == "high")
        medium = sum(1 for f in self.findings if f.severity == "medium")
        penalty = (high * 15) + (medium * 7)
        return max(0, 100 - penalty)

# ── core checker ─────────────────────────────────────────────────────────────
def check_terminology(src_text: str, tgt_text: str,
                      label: str, path: str,
                      glossary: dict,
                      min_freq: int = 2,
                      glossary_only: bool = False) -> TermResult:

    result   = TermResult(label=label, path=path)
    src_sents = get_sentences(src_text)
    tgt_sents = get_sentences(tgt_text)
    tgt_lower = tgt_text.lower()

    # ── 1. Glossary / anchor term checks ─────────────────────────────────────
    for src_term, expected in glossary.items():
        # Count how often source term appears
        src_count = len(find_term_in_sentences(src_term, src_sents))
        if src_count == 0:
            continue
        result.checked += 1
        # Count expected translation occurrences
        exp_count = tgt_lower.count(expected.lower())
        if exp_count == 0:
            result.findings.append(TermFinding(
                term     = src_term,
                category = "ANCHOR_MISMATCH",
                severity = "high",
                message  = f"Expected translation '{expected}' not found in output.",
                detail   = f"Source term '{src_term}' appears {src_count}x in source."
            ))
        elif exp_count < src_count * 0.5:
            result.findings.append(TermFinding(
                term     = src_term,
                category = "ANCHOR_MISMATCH",
                severity = "medium",
                message  = f"Expected '{expected}' found only {exp_count}x "
                           f"but source term appears {src_count}x.",
                detail   = "Possible inconsistent use of the glossary term."
            ))
        else:
            result.passed += 1

    # ── 2. Auto-detected term consistency ────────────────────────────────────
    if glossary_only:
        return result

    candidate_terms = extract_candidate_terms(src_text, min_freq)

    # Sort by frequency descending, check top 40
    top_terms = sorted(candidate_terms.items(), key=lambda x: -x[1])[:40]

    for term, src_freq in top_terms:
        if term in [g.lower() for g in glossary]:
            continue  # already checked above
        result.checked += 1

        # How many times does this exact term appear in target?
        tgt_freq = tgt_lower.count(term)

        # Get translation variants via positional heuristic
        variants = find_translation_variants(term, src_text, tgt_text)
        top_variants = [v for v, _ in variants.most_common(5)
                        if len(v.split()) >= 2]

        if tgt_freq == 0 and src_freq >= 3:
            # Term appears 3+ times in source but never in target
            # (could be translated or Cyrillic source)
            if re.search(r'[\u0400-\u04FF]', term):
                # Cyrillic source term — dropped is expected
                result.passed += 1
                continue
            result.findings.append(TermFinding(
                term     = term,
                category = "DROPPED",
                severity = "medium" if src_freq >= 5 else "low",
                message  = f"'{term}' appears {src_freq}x in source but not in translation.",
                detail   = f"Possible translations: {', '.join(top_variants[:3])}" if top_variants else "",
                variants = top_variants,
            ))

        elif src_freq >= 4 and tgt_freq > 0:
            # Term appears frequently in source — check if target uses multiple
            # different phrasings around where this term should appear
            # Rough proxy: count distinct bigrams near matching positions
            if len(top_variants) >= 3:
                # Multiple different continuations suggest inconsistent translation
                result.findings.append(TermFinding(
                    term     = term,
                    category = "INCONSISTENT",
                    severity = "medium",
                    message  = f"'{term}' ({src_freq}x in source) may be translated "
                               f"inconsistently — {len(top_variants)} different phrasings detected nearby.",
                    detail   = f"Variants seen: {', '.join(top_variants[:4])}",
                    variants = top_variants,
                ))
            else:
                result.passed += 1
        else:
            result.passed += 1

    return result

# ── display ───────────────────────────────────────────────────────────────────
SEV_COLOR = {"high": "red", "medium": "yellow", "low": "cyan"}
CAT_LABEL = {
    "ANCHOR_MISMATCH": "GLOSSARY MISMATCH",
    "DROPPED":         "TERM DROPPED",
    "INCONSISTENT":    "INCONSISTENT TRANSLATION",
    "FREQUENCY_GAP":   "FREQUENCY GAP",
}

def score_color(s: int) -> str:
    return "green" if s >= 80 else "yellow" if s >= 55 else "red"

def print_result(r: TermResult, verbose: bool):
    sc = r.score
    print(f"\n  {clr(r.label, 'bold')}  —  {clr(Path(r.path).name, 'blue')}")
    print(f"  Terms checked : {r.checked}   "
          f"Passed : {clr(str(r.passed), 'green')}   "
          f"Issues : {clr(str(len(r.findings)), 'red' if r.findings else 'green')}")
    print(f"  Terminology score : {clr(str(sc), score_color(sc))}/100")

    if not r.findings:
        print(f"  {clr('✓ No terminology issues detected.', 'green')}")
        return

    order = {"high": 0, "medium": 1, "low": 2}
    for f in sorted(r.findings, key=lambda x: order[x.severity]):
        sev_c = SEV_COLOR[f.severity]
        cat   = CAT_LABEL.get(f.category, f.category)
        print(f"\n    {clr(f'[{f.severity.upper()}]', sev_c)}  "
              f"{clr(cat, 'bold')}")
        print(f"    {f.message}")
        if f.detail:
            print(f"    {clr(f.detail, 'blue')}")
        if verbose and f.variants:
            print(f"    {clr('Variants:', 'yellow')} "
                  f"{', '.join(f.variants[:5])}")

def print_comparison(results: list):
    """Side-by-side score comparison when multiple translations given."""
    if len(results) < 2:
        return
    print(f"\n{clr(DIV, 'bold')}")
    print(clr("  HEAD-TO-HEAD TERMINOLOGY SCORES", "bold"))
    print(DIV)
    for r in results:
        sc = r.score
        bar_w  = 30
        filled = round(sc / 100 * bar_w)
        bar    = "█" * filled + "░" * (bar_w - filled)
        issues = len(r.findings)
        print(f"  {r.label:<24}  {clr(bar, score_color(sc))}  "
              f"{clr(str(sc), score_color(sc))}/100   "
              f"{issues} issue(s)")

    best = max(results, key=lambda x: x.score)
    print(f"\n  {clr('Most consistent terminology:', 'bold')} "
          f"{clr(best.label, 'green')}")

    # Show terms where translations differ from each other
    all_terms = set(f.term for r in results for f in r.findings)
    shared_issues = [t for t in all_terms
                     if sum(1 for r in results
                            if any(f.term == t for f in r.findings)) > 1]
    if shared_issues:
        print(f"\n  {clr('Terms flagged in multiple translations:', 'yellow')}")
        for t in shared_issues[:8]:
            print(f"    • {t}")

# ── text report ───────────────────────────────────────────────────────────────
def build_report(results: list, src_path: str) -> str:
    lines = [DIV, "TERMINOLOGY CHECKER REPORT", DIV, "",
             f"Source: {src_path}", ""]
    for r in results:
        lines += [
            f"Translation: {r.label}  ({r.path})",
            f"  Score: {r.score}/100   Checked: {r.checked}   Issues: {len(r.findings)}",
        ]
        for f in r.findings:
            lines += [
                f"  [{f.severity.upper()}]  {f.category}",
                f"    {f.message}",
                f"    {f.detail}" if f.detail else "",
            ]
        lines.append("")
    return "\n".join(lines)

# ── CSV export ────────────────────────────────────────────────────────────────
def export_csv(results: list, out: Path):
    rows = []
    for r in results:
        for f in r.findings:
            rows.append({
                "translation": r.label,
                "file":        Path(r.path).name,
                "score":       r.score,
                "term":        f.term,
                "category":    f.category,
                "severity":    f.severity,
                "message":     f.message,
                "detail":      f.detail[:200],
                "variants":    "; ".join(f.variants[:5]),
            })
    if not rows:
        print(clr("No issues to export.", "yellow")); return
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(clr(f"\nCSV exported → {out}", "green"))

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Check terminology consistency in AI-generated translations."
    )
    parser.add_argument("source",
                        help="Source document (.txt / .docx / .pdf)")
    parser.add_argument("translations", nargs="+",
                        help="One or more translation files to check")
    parser.add_argument("--glossary", nargs="*", default=[],
                        metavar="TERM=TRANSLATION",
                        help="Expected term pairs, e.g. "
                             "'collective defence=collective defence'")
    parser.add_argument("--label", nargs="*", default=[],
                        dest="labels",
                        help="Display names for translations (one per file)")
    parser.add_argument("--glossary-only", action="store_true",
                        help="Only check glossary terms, skip auto-detection "
                             "(recommended for long documents)")
    parser.add_argument("--min-freq", type=int, default=2,
                        help="Minimum source frequency to track a term (default: 2)")
    parser.add_argument("-o", "--output", help="Save text report to file")
    parser.add_argument("--csv",          help="Export findings to CSV")
    parser.add_argument("-v", "--verbose",action="store_true",
                        help="Show all variants for each flagged term")
    args = parser.parse_args()

    src_p = Path(args.source)
    if not src_p.exists():
        print(clr(f"Error: source not found — {src_p}", "red")); sys.exit(1)

    tgt_paths = []
    for t in args.translations:
        p = Path(t)
        if not p.exists():
            print(clr(f"Warning: not found — {t}", "yellow"))
        else:
            tgt_paths.append(p)
    if not tgt_paths:
        print(clr("No valid translation files found.", "red")); sys.exit(1)

    # Labels
    labels = list(args.labels) if args.labels else []
    while len(labels) < len(tgt_paths):
        labels.append(tgt_paths[len(labels)].stem)

    # Parse glossary
    glossary = {}
    for pair in args.glossary:
        if "=" in pair:
            src_t, tgt_t = pair.split("=", 1)
            glossary[src_t.strip()] = tgt_t.strip()
        else:
            print(clr(f"Warning: ignoring malformed glossary entry '{pair}' "
                      f"(expected format: source_term=target_term)", "yellow"))

    print(clr(f"\nLoading source: {src_p.name}", "bold"))
    try:
        src_text = read_file(src_p)
    except Exception as e:
        print(clr(f"Error: {e}", "red")); sys.exit(1)

    if not src_text.strip():
        print(clr("Error: source file is empty.", "red")); sys.exit(1)

    if glossary:
        print(f"  Glossary: {len(glossary)} term pair(s) loaded")
    if args.glossary_only:
        print(clr("  Mode: glossary-only (auto-detection disabled)", "yellow"))
    print(f"  Min term frequency: {args.min_freq}")

    print(clr(f"\n{DIV}", "bold"))
    print(clr("  TERMINOLOGY CHECKER", "bold"))
    print(DIV)

    results = []
    for p, label in zip(tgt_paths, labels):
        print(f"\n  Loading translation: {clr(p.name, 'cyan')}")
        try:
            tgt_text = read_file(p)
        except Exception as e:
            print(clr(f"  Error: {e}", "red")); continue
        if not tgt_text.strip():
            print(clr(f"  Error: {p.name} is empty.", "red")); continue

        r = check_terminology(src_text, tgt_text, label, str(p),
                              glossary, args.min_freq, args.glossary_only)
        results.append(r)
        print_result(r, args.verbose)

    if len(results) > 1:
        print_comparison(results)

    if args.output:
        Path(args.output).write_text(
            build_report(results, str(src_p)), encoding="utf-8"
        )
        print(clr(f"\nReport saved → {args.output}", "green"))

    if args.csv:
        export_csv(results, Path(args.csv))

if __name__ == "__main__":
    main()
