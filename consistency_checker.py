#!/usr/bin/env python3
"""
consistency_checker.py
-----------------------
Detects when an AI gives contradictory or inconsistent answers to the
same / similar questions across multiple session logs.

Works entirely on saved .txt / .docx files — no live API access needed.

What it checks:
  1. DUPLICATE QUESTIONS   — near-identical questions asked in multiple sessions
  2. ANSWER DRIFT          — same question, meaningfully different answer length/content
  3. CONTRADICTION SIGNALS — opposite sentiment/stance detected between answers
  4. TOPIC STABILITY       — whether the same topics appear in answers to the same question
  5. FACTUAL FLIP DETECTOR — "yes/no", "will/won't", numeric claims that differ

Usage:
    python3 consistency_checker.py session1.txt session2.txt session3.docx
    python3 consistency_checker.py *.txt -o consistency_report.txt
    python3 consistency_checker.py *.txt --csv consistency.csv --threshold 0.55

Options:
    -t, --threshold FLOAT   Similarity threshold 0–1 for "same question" (default 0.60)
    -o, --output FILE       Save text report
    --csv FILE              Export flagged pairs to CSV
    -v, --verbose           Show full answer text for each flagged pair

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
from typing import Optional
from collections import defaultdict

try:
    import docx as _docx
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

# ── colours ───────────────────────────────────────────────────────────────────
C = {
    "green":"\033[92m","red":"\033[91m","yellow":"\033[93m",
    "blue":"\033[94m","cyan":"\033[96m","bold":"\033[1m",
    "magenta":"\033[95m","reset":"\033[0m",
}
def clr(t, c): return f"{C[c]}{t}{C['reset']}"
DIV  = "=" * 70
DIV2 = "-" * 70

# ── sentiment/stance word lists ───────────────────────────────────────────────
POSITIVE_WORDS = [
    "yes","will","supports","agrees","confirmed","positive","beneficial",
    "kyllä","tukee","myönteinen","hyödyllinen","vahvistaa",
    "increases","strengthens","promotes","approves","endorses",
]
NEGATIVE_WORDS = [
    "no","won't","opposes","disagrees","denied","negative","harmful",
    "ei","vastustaa","kielteinen","haitallinen","kiistää",
    "decreases","weakens","undermines","rejects","condemns",
]

def stance_score(text: str) -> float:
    """
    Returns a float:
      > 0  → net positive/affirming language
      < 0  → net negative/opposing language
      = 0  → neutral or balanced
    """
    tl = text.lower()
    pos = sum(1 for w in POSITIVE_WORDS if re.search(r'\b' + re.escape(w) + r'\b', tl))
    neg = sum(1 for w in NEGATIVE_WORDS if re.search(r'\b' + re.escape(w) + r'\b', tl))
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total   # normalised: -1 to +1


# ── data model ────────────────────────────────────────────────────────────────
@dataclass
class Turn:
    session: str
    index:   int
    question: str
    answer:   str
    q_words:  int = 0
    a_words:  int = 0

    def __post_init__(self):
        self.q_words = len(self.question.split())
        self.a_words = len(self.answer.split())

@dataclass
class FlaggedPair:
    q_similarity:   float
    turn_a:         Turn
    turn_b:         Turn
    flags:          list = field(default_factory=list)
    severity:       str  = "low"   # low | medium | high

    @property
    def sessions(self):
        return f"{self.turn_a.session}  ↔  {self.turn_b.session}"


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


# ── session parser ────────────────────────────────────────────────────────────
RE_YOU = re.compile(r'^You:\s*\d{1,2}:\d{2}\s*(?:AM|PM)\s*$', re.I)
RE_AI  = re.compile(r'^(\S[\w\s]{0,30}?)\s+\d{1,2}:\d{2}\s*(?:AM|PM)\s*$', re.I)

def parse_turns(text: str, session_name: str) -> list:
    lines           = text.splitlines()
    turns           = []
    current_speaker = None
    current_block   = []
    pending_q       = None

    def flush(speaker, block):
        nonlocal pending_q
        content = "\n".join(block).strip()
        if not content: return
        if speaker == "user":
            pending_q = content
        elif speaker == "ai" and pending_q is not None:
            turns.append(Turn(
                session  = session_name,
                index    = len(turns) + 1,
                question = pending_q,
                answer   = content,
            ))
            pending_q = None

    for line in lines:
        stripped = line.strip()
        if RE_YOU.match(stripped):
            flush(current_speaker, current_block)
            current_speaker, current_block = "user", []
            continue
        if RE_AI.match(stripped):
            cand = RE_AI.match(stripped).group(1).strip()
            if cand.lower() not in ("you", "raakateksti"):
                flush(current_speaker, current_block)
                current_speaker, current_block = "ai", []
                continue
        if stripped.lower() == "raakateksti:":
            continue
        current_block.append(line)

    flush(current_speaker, current_block)

    if not turns and text.strip():
        turns = [Turn(session=session_name, index=1,
                      question="(full document)", answer=text.strip())]
    return turns

def load_file(path: Path) -> list:
    try:
        raw = read_docx(path) if path.suffix.lower() == ".docx" else read_txt(path)
        return parse_turns(raw, path.stem)
    except Exception as e:
        print(clr(f"  [error] {path.name}: {e}", "red"))
        return []


# ── similarity ────────────────────────────────────────────────────────────────
def tokenize(text: str) -> set:
    """Simple word-token set (lowercased, stripped of punctuation)."""
    return set(re.sub(r'[^\w\s]', '', text.lower()).split())

def jaccard(a: str, b: str) -> float:
    ta, tb = tokenize(a), tokenize(b)
    if not ta or not tb: return 0.0
    return len(ta & tb) / len(ta | tb)

def cosine_tfidf_approx(a: str, b: str) -> float:
    """
    Approximate cosine similarity using term-frequency overlap.
    Good enough for short question strings without heavy deps.
    """
    def tf(text):
        words = re.sub(r'[^\w\s]', '', text.lower()).split()
        freq = defaultdict(int)
        for w in words: freq[w] += 1
        return freq

    fa, fb = tf(a), tf(b)
    vocab  = set(fa) | set(fb)
    dot    = sum(fa[w] * fb[w] for w in vocab)
    norm_a = math.sqrt(sum(v**2 for v in fa.values()))
    norm_b = math.sqrt(sum(v**2 for v in fb.values()))
    if norm_a == 0 or norm_b == 0: return 0.0
    return dot / (norm_a * norm_b)

def question_similarity(a: str, b: str) -> float:
    """Average of Jaccard and cosine — robust to paraphrasing."""
    return (jaccard(a, b) + cosine_tfidf_approx(a, b)) / 2


# ── numeric claim extractor ───────────────────────────────────────────────────
RE_NUM = re.compile(r'\b(\d{1,3}(?:[.,]\d{1,3})*(?:\s*%|\s*billion|\s*million|\s*miljard|\s*prosentt)?)\b')

def extract_numbers(text: str) -> set:
    """Extract numeric values / percentages from text."""
    nums = set()
    for m in RE_NUM.finditer(text):
        raw = m.group(1).replace(",", "").replace(" ", "").lower()
        nums.add(raw)
    return nums


# ── yes/no flip detector ──────────────────────────────────────────────────────
YES_PAT = re.compile(r'\b(yes|kyllä|indeed|confirmed|will|does|is true|totta)\b', re.I)
NO_PAT  = re.compile(r'\b(no|ei|does not|will not|won\'t|is not|false|väärin|kiistää)\b', re.I)

def yn_stance(text: str) -> Optional[str]:
    has_yes = bool(YES_PAT.search(text[:400]))
    has_no  = bool(NO_PAT.search(text[:400]))
    if has_yes and not has_no: return "yes"
    if has_no  and not has_yes: return "no"
    return None


# ── flag generator ────────────────────────────────────────────────────────────
def analyse_pair(ta: Turn, tb: Turn, q_sim: float) -> FlaggedPair:
    flags = []

    # 1. Answer length drift
    ratio = max(ta.a_words, tb.a_words) / max(min(ta.a_words, tb.a_words), 1)
    if ratio >= 3.0:
        flags.append(f"LENGTH DRIFT ×{ratio:.1f}  ({ta.a_words} vs {tb.a_words} words)")
    elif ratio >= 1.8:
        flags.append(f"Length difference ×{ratio:.1f}  ({ta.a_words} vs {tb.a_words} words)")

    # 2. Stance/sentiment flip
    sa = stance_score(ta.answer)
    sb = stance_score(tb.answer)
    if sa * sb < 0 and abs(sa - sb) > 0.3:
        flags.append(f"STANCE FLIP  ({ta.session}: {sa:+.2f}  vs  {tb.session}: {sb:+.2f})")

    # 3. Yes/No flip
    yn_a = yn_stance(ta.answer)
    yn_b = yn_stance(tb.answer)
    if yn_a and yn_b and yn_a != yn_b:
        flags.append(f"YES/NO FLIP  ({ta.session}: '{yn_a}'  vs  {tb.session}: '{yn_b}')")

    # 4. Numeric claim disagreement
    nums_a = extract_numbers(ta.answer)
    nums_b = extract_numbers(tb.answer)
    diff_nums = (nums_a | nums_b) - (nums_a & nums_b)
    if diff_nums and len(diff_nums) <= 6:
        flags.append(f"NUMERIC DIFF  values unique to one answer: {', '.join(sorted(diff_nums)[:5])}")

    # 5. Content similarity (answers to the same question are suspiciously different)
    ans_sim = question_similarity(ta.answer, tb.answer)
    if q_sim >= 0.7 and ans_sim < 0.15:
        flags.append(f"LOW ANSWER OVERLAP  (answer similarity: {ans_sim:.2f})")

    # Severity
    high_signals = sum(1 for f in flags if any(w in f for w in ("FLIP","DRIFT")))
    severity = "high" if high_signals >= 2 else "medium" if flags else "low"

    return FlaggedPair(
        q_similarity = round(q_sim, 3),
        turn_a = ta,
        turn_b = tb,
        flags  = flags,
        severity = severity,
    )


# ── main comparison logic ─────────────────────────────────────────────────────
def check_consistency(all_turns: list, threshold: float) -> list:
    """
    Compare every question in every session against every question in
    every other session. Return flagged pairs.
    """
    # group by session
    by_session = defaultdict(list)
    for t in all_turns:
        by_session[t.session].append(t)

    session_names = list(by_session.keys())
    flagged = []

    for i in range(len(session_names)):
        for j in range(i + 1, len(session_names)):
            s_a = by_session[session_names[i]]
            s_b = by_session[session_names[j]]

            for ta in s_a:
                for tb in s_b:
                    sim = question_similarity(ta.question, tb.question)
                    if sim >= threshold:
                        pair = analyse_pair(ta, tb, sim)
                        if pair.flags:   # only keep pairs with at least one flag
                            flagged.append(pair)

    # de-duplicate (keep highest similarity if same pair found multiple ways)
    seen = set()
    unique = []
    for p in sorted(flagged, key=lambda x: -x.q_similarity):
        key = tuple(sorted([
            (p.turn_a.session, p.turn_a.index),
            (p.turn_b.session, p.turn_b.index),
        ]))
        if key not in seen:
            seen.add(key)
            unique.append(p)

    return unique


# ── display ───────────────────────────────────────────────────────────────────
SEV_COLOR = {"high":"red","medium":"yellow","low":"cyan"}

def print_pair(p: FlaggedPair, verbose: bool):
    sev_c = SEV_COLOR.get(p.severity, "reset")
    print(f"\n  {clr(f'[{p.severity.upper()}]', sev_c)}  Q-similarity: {clr(str(p.q_similarity), 'bold')}")
    print(f"  Sessions : {clr(p.sessions, 'bold')}")
    print(f"  {clr('Q (A):', 'cyan')} {p.turn_a.question[:110]}")
    print(f"  {clr('Q (B):', 'magenta')} {p.turn_b.question[:110]}")
    for f in p.flags:
        print(f"  {clr('▸', sev_c)} {f}")
    if verbose:
        print(f"\n  {clr('Answer A:', 'cyan')}")
        print(f"    {p.turn_a.answer[:400]}")
        print(f"\n  {clr('Answer B:', 'magenta')}")
        print(f"    {p.turn_b.answer[:400]}")
    print(f"  {DIV2}")

def print_summary(flagged: list, sessions: list, threshold: float):
    high   = sum(1 for p in flagged if p.severity == "high")
    medium = sum(1 for p in flagged if p.severity == "medium")
    low    = sum(1 for p in flagged if p.severity == "low")

    print(f"\n{clr(DIV, 'bold')}")
    print(clr("  CONSISTENCY CHECK SUMMARY", "bold"))
    print(DIV)
    print(f"  Sessions checked : {clr(str(len(sessions)), 'bold')}")
    print(f"  Q-sim threshold  : {threshold}")
    print(f"  Flagged pairs    : {clr(str(len(flagged)), 'bold')}")
    print(f"    {clr('HIGH',   'red')}   : {high}")
    print(f"    {clr('MEDIUM', 'yellow')}: {medium}")
    print(f"    {clr('LOW',    'cyan')}   : {low}")

    if not flagged:
        print(f"\n  {clr('✓ No inconsistencies detected at this threshold.', 'green')}")
    else:
        print(f"\n  {clr('Flagged pairs by type:', 'bold')}")
        type_counts = defaultdict(int)
        for p in flagged:
            for f in p.flags:
                ftype = f.split()[0]
                type_counts[ftype] += 1
        for ftype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"    {ftype:<30} {count}")


# ── text report ───────────────────────────────────────────────────────────────
def build_report(flagged: list, sessions: list, threshold: float) -> str:
    lines = [DIV, "CONSISTENCY CHECK REPORT", DIV, "",
             f"Sessions: {', '.join(sessions)}",
             f"Threshold: {threshold}",
             f"Flagged pairs: {len(flagged)}", ""]
    for p in flagged:
        lines += [
            f"[{p.severity.upper()}]  Q-sim: {p.q_similarity}",
            f"  Sessions: {p.sessions}",
            f"  Q-A: {p.turn_a.question[:100]}",
            f"  Q-B: {p.turn_b.question[:100]}",
        ]
        for f in p.flags:
            lines.append(f"  ▸ {f}")
        lines.append("")
    return "\n".join(lines)


# ── CSV export ────────────────────────────────────────────────────────────────
def export_csv(flagged: list, out: Path):
    rows = []
    for p in flagged:
        rows.append({
            "severity":      p.severity,
            "q_similarity":  p.q_similarity,
            "session_a":     p.turn_a.session,
            "turn_a":        p.turn_a.index,
            "session_b":     p.turn_b.session,
            "turn_b":        p.turn_b.index,
            "question_a":    p.turn_a.question[:200],
            "question_b":    p.turn_b.question[:200],
            "flags":         " | ".join(p.flags),
            "answer_a":      p.turn_a.answer[:300],
            "answer_b":      p.turn_b.answer[:300],
        })
    if not rows:
        print(clr("No flagged pairs to export.", "yellow")); return
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(clr(f"\nCSV exported → {out}", "green"))


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Detect inconsistencies in AI answers across session logs."
    )
    parser.add_argument("files", nargs="+", help="Session log files (.txt / .docx)")
    parser.add_argument("-t", "--threshold", type=float, default=0.60,
                        help="Question similarity threshold 0–1 (default: 0.60)")
    parser.add_argument("-o", "--output", help="Save text report to file")
    parser.add_argument("--csv",           help="Export flagged pairs to CSV")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print full answer text for flagged pairs")
    args = parser.parse_args()

    paths = [Path(f) for f in args.files if Path(f).exists()]
    missing = [f for f in args.files if not Path(f).exists()]
    for m in missing:
        print(clr(f"[warn] Not found: {m}", "yellow"))

    if len(paths) < 2:
        print(clr("Need at least 2 session files to compare.", "red")); sys.exit(1)

    print(clr(f"\nLoading {len(paths)} session file(s)…", "bold"))
    all_turns = []
    for p in paths:
        turns = load_file(p)
        print(f"  {p.name}: {len(turns)} turns")
        all_turns.extend(turns)

    if not all_turns:
        print(clr("No Q&A turns found.", "red")); sys.exit(1)

    print(clr(f"\nChecking consistency (threshold={args.threshold})…", "bold"))
    flagged = check_consistency(all_turns, args.threshold)

    session_names = list(dict.fromkeys(t.session for t in all_turns))
    print_summary(flagged, session_names, args.threshold)

    if flagged:
        print(f"\n{clr(DIV, 'bold')}")
        print(clr("  FLAGGED PAIRS", "bold"))
        print(DIV)
        # Sort: high → medium → low
        order = {"high": 0, "medium": 1, "low": 2}
        for p in sorted(flagged, key=lambda x: (order.get(x.severity,3), -x.q_similarity)):
            print_pair(p, args.verbose)

    if args.output:
        Path(args.output).write_text(
            build_report(flagged, session_names, args.threshold), encoding="utf-8"
        )
        print(clr(f"\nReport saved → {args.output}", "green"))

    if args.csv:
        export_csv(flagged, Path(args.csv))


if __name__ == "__main__":
    main()
