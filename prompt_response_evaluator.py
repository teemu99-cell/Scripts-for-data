#!/usr/bin/env python3
"""
prompt_response_evaluator.py
-----------------------------
Given a question (prompt) and an AI response, scores how well the response
actually addresses what was asked. Works on session log files or individual
prompt/response pairs.

This complements the translation-side tools by evaluating the Q&A session
quality — did the AI stay on topic, answer completely, and not pad its response?

What it scores:
  1. RELEVANCE          — how much of the response content relates to the question
  2. COMPLETENESS       — whether the question's key aspects were addressed
  3. CONCISENESS        — penalises padding, repetition, and over-length responses
  4. DIRECTNESS         — does the AI actually answer, or hedge and deflect?
  5. TOPIC CONSISTENCY  — does the response stay on the topic of the question?
  6. QUESTION COVERAGE  — for multi-part questions, how many parts were addressed

Can process:
  - A session log file (.txt / .docx) — evaluates every Q&A turn
  - A prompt file + response file pair (--prompt / --response flags)

Usage:
    # Evaluate all turns in a session log
    python3 prompt_response_evaluator.py session.txt

    # Evaluate a single prompt/response pair from files
    python3 prompt_response_evaluator.py --prompt question.txt --response answer.txt

    # Compare two AI sessions
    python3 prompt_response_evaluator.py session_ai1.txt session_ai2.txt \\
            --label "Onsite AI" --label "Gemini" -o report.txt --csv scores.csv

Options:
    --prompt FILE       Evaluate a single prompt from a file
    --response FILE     Response file to evaluate against --prompt
    --label NAME        Display name per session file (repeatable)
    --min-words N       Minimum response words to score (default: 10)
    -o, --output FILE   Save report to file
    --csv FILE          Export turn-level scores to CSV
    -v, --verbose       Show scoring detail for each turn

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

# ── deflection / non-answer phrases ──────────────────────────────────────────
DEFLECTION_PHRASES = [
    "i cannot", "i can't", "i don't have", "i am unable",
    "i'm unable", "i do not have access", "as an ai",
    "as a language model", "i'm just an ai", "i don't know",
    "i cannot provide", "i'm not able to", "it is not possible for me",
    "unfortunately i", "i apologize", "i'm sorry but",
    "that is outside", "beyond my capabilities", "i have no information",
]

# ── padding / filler phrases ──────────────────────────────────────────────────
PADDING_PHRASES = [
    "it is worth noting that", "it should be noted that",
    "it is important to note", "as mentioned above",
    "as previously stated", "in conclusion", "to summarize",
    "thank you for your question", "great question",
    "that's a great question", "certainly", "of course",
    "absolutely", "sure", "i'd be happy to",
    "let me explain", "let me elaborate",
    "without further ado", "that being said",
]

# ── stop-words ────────────────────────────────────────────────────────────────
STOPWORDS = {
    "the","a","an","and","or","but","in","on","at","to","of","for","with",
    "by","from","is","are","was","were","be","been","has","have","had",
    "that","this","which","it","its","they","their","we","our","as","not",
    "no","so","if","than","then","also","more","most","will","would","can",
    "could","should","may","might","very","just","only","all","any","some",
    "such","each","into","about","i","he","she","him","her","do","did","does",
    "what","when","where","who","how","why","which",
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

# ── session parser ────────────────────────────────────────────────────────────
RE_YOU = re.compile(r'^You:\s*\d{1,2}:\d{2}\s*(?:AM|PM)\s*$', re.I)
RE_AI  = re.compile(r'^(\S[\w\s]{0,30}?)\s+\d{1,2}:\d{2}\s*(?:AM|PM)\s*$', re.I)

def parse_turns(text: str, session_name: str) -> list:
    lines = text.splitlines()
    turns = []
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
            turns.append({"session": session_name,
                          "index": len(turns) + 1,
                          "question": pending_q,
                          "answer": content})
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
        turns = [{"session": session_name, "index": 1,
                  "question": "(document)", "answer": text.strip()}]
    return turns

# ── text utilities ────────────────────────────────────────────────────────────
def content_words(text: str) -> set:
    tokens = re.sub(r'[^\w\s]', '', text.lower()).split()
    return {w for w in tokens if w not in STOPWORDS and len(w) > 2}

def extract_question_parts(question: str) -> list:
    """Split a multi-part question into individual sub-questions."""
    parts = re.split(r'\?\s+(?=[A-Z])|;\s+|\band\s+(?=\w+\s+(?:is|are|was|were|do|does|can|will|how|why|what|when|where|who))',
                     question)
    return [p.strip() for p in parts if len(p.strip()) > 5]

def count_question_marks(text: str) -> int:
    return text.count("?")

# ── scoring engine ────────────────────────────────────────────────────────────
@dataclass
class TurnScore:
    session:    str
    index:      int
    question:   str
    answer:     str
    relevance:  int
    completeness: int
    conciseness:  int
    directness:   int
    consistency:  int
    coverage:     int

    @property
    def overall(self) -> int:
        return round((self.relevance + self.completeness + self.conciseness +
                      self.directness + self.consistency + self.coverage) / 6)

    @property
    def grade(self) -> str:
        s = self.overall
        return ("A" if s >= 88 else "B" if s >= 75 else
                "C" if s >= 60 else "D" if s >= 45 else "F")

def score_turn(question: str, answer: str,
               session: str, index: int) -> TurnScore:

    q_words = content_words(question)
    a_words = content_words(answer)
    ans_lower = answer.lower()
    ans_word_count = len(answer.split())

    # ── 1. Relevance ──────────────────────────────────────────────────────────
    if not q_words or not a_words:
        relevance = 50
    else:
        overlap = len(q_words & a_words) / len(q_words)
        # Also check if answer expands on question topic (uses related words)
        q_list = list(q_words)
        expansion = sum(1 for qw in q_list
                        if any(qw[:4] in aw for aw in a_words
                               if len(qw) >= 4))
        exp_ratio = expansion / max(len(q_list), 1)
        combined  = (overlap * 0.6) + (exp_ratio * 0.4)
        if combined >= 0.45:   relevance = 100
        elif combined >= 0.30: relevance = 82
        elif combined >= 0.18: relevance = 62
        elif combined >= 0.08: relevance = 40
        else:                  relevance = 18

    # ── 2. Completeness ───────────────────────────────────────────────────────
    q_parts = extract_question_parts(question)
    if len(q_parts) <= 1:
        # Single question — check answer length as proxy
        if ans_word_count >= 50:    completeness = 100
        elif ans_word_count >= 25:  completeness = 80
        elif ans_word_count >= 10:  completeness = 55
        else:                       completeness = 25
    else:
        # Multi-part — check how many parts are addressed
        addressed = 0
        for part in q_parts:
            part_words = content_words(part)
            if part_words & a_words:
                addressed += 1
        ratio = addressed / len(q_parts)
        if ratio >= 0.85:   completeness = 100
        elif ratio >= 0.65: completeness = 78
        elif ratio >= 0.45: completeness = 55
        else:               completeness = 28

    # ── 3. Conciseness ────────────────────────────────────────────────────────
    # Penalise padding phrases and extreme over-length
    padding_count = sum(1 for p in PADDING_PHRASES if p in ans_lower)
    # Expected response length: ~3–5x question length is normal
    q_wc  = len(question.split())
    ratio = ans_word_count / max(q_wc, 1)

    if padding_count == 0 and ratio <= 15:
        conciseness = 100
    elif padding_count <= 1 and ratio <= 20:
        conciseness = 82
    elif padding_count <= 2 or ratio <= 30:
        conciseness = 60
    elif padding_count <= 4 or ratio <= 50:
        conciseness = 38
    else:
        conciseness = 18

    # ── 4. Directness ─────────────────────────────────────────────────────────
    deflection_count = sum(1 for p in DEFLECTION_PHRASES if p in ans_lower)
    # Also check if answer starts with the topic or with meta-commentary
    first_words = answer.lower().split()[:8]
    meta_start  = any(w in first_words for w in
                      ["certainly","absolutely","sure","great","thank",
                       "of course","i'd","let me","i will","i'll"])

    if deflection_count == 0 and not meta_start:
        directness = 100
    elif deflection_count == 0 and meta_start:
        directness = 78
    elif deflection_count == 1:
        directness = 50
    elif deflection_count == 2:
        directness = 28
    else:
        directness = 10

    # ── 5. Topic consistency ──────────────────────────────────────────────────
    # Check if the answer drifts to unrelated content by splitting into halves
    # and comparing overlap with the question across first and second half.
    mid = len(answer) // 2
    first_half  = content_words(answer[:mid])
    second_half = content_words(answer[mid:])

    if not first_half or not second_half:
        consistency = 85
    else:
        # Both halves should relate to question
        fh_overlap = len(q_words & first_half)  / max(len(q_words), 1)
        sh_overlap = len(q_words & second_half) / max(len(q_words), 1)
        # Also check halves relate to each other
        halves_sim = len(first_half & second_half) / len(first_half | second_half)

        if halves_sim >= 0.25 and sh_overlap >= fh_overlap * 0.5:
            consistency = 100
        elif halves_sim >= 0.15:
            consistency = 78
        elif halves_sim >= 0.08:
            consistency = 52
        else:
            consistency = 28

    # ── 6. Question coverage ──────────────────────────────────────────────────
    # Count explicit question marks in the original question
    q_marks = count_question_marks(question)
    if q_marks <= 1:
        coverage = 100  # single question — coverage checked via completeness
    else:
        # Multi-question: rough check of how many distinct question topics answered
        q_sentences = [s for s in re.split(r'\?', question) if s.strip()]
        covered = 0
        for qs in q_sentences:
            qs_words = content_words(qs)
            if qs_words and len(qs_words & a_words) >= 1:
                covered += 1
        ratio = covered / max(len(q_sentences), 1)
        if ratio >= 0.85:   coverage = 100
        elif ratio >= 0.65: coverage = 75
        elif ratio >= 0.40: coverage = 48
        else:               coverage = 22

    return TurnScore(
        session      = session,
        index        = index,
        question     = question,
        answer       = answer,
        relevance    = relevance,
        completeness = completeness,
        conciseness  = conciseness,
        directness   = directness,
        consistency  = consistency,
        coverage     = coverage,
    )

# ── session aggregation ───────────────────────────────────────────────────────
@dataclass
class SessionResult:
    label:       str
    path:        str
    turn_scores: list = field(default_factory=list)

    @property
    def avg_overall(self) -> int:
        if not self.turn_scores: return 0
        return round(sum(t.overall for t in self.turn_scores) / len(self.turn_scores))

    @property
    def grade(self) -> str:
        s = self.avg_overall
        return ("A" if s >= 88 else "B" if s >= 75 else
                "C" if s >= 60 else "D" if s >= 45 else "F")

    def avg_dim(self, dim: str) -> int:
        vals = [getattr(t, dim) for t in self.turn_scores]
        return round(sum(vals) / max(len(vals), 1))

# ── display ───────────────────────────────────────────────────────────────────
DIMS = ["relevance","completeness","conciseness","directness","consistency","coverage"]
DIM_LABELS = {
    "relevance":    "Relevance",
    "completeness": "Completeness",
    "conciseness":  "Conciseness",
    "directness":   "Directness",
    "consistency":  "Topic Consistency",
    "coverage":     "Question Coverage",
}

def score_color(s: int) -> str:
    return "green" if s >= 78 else "yellow" if s >= 52 else "red"

def grade_color(g: str) -> str:
    return {"A":"green","B":"green","C":"yellow","D":"yellow","F":"red"}.get(g,"reset")

def bar(score: int, width: int = 18) -> str:
    filled = round(score / 100 * width)
    return "█" * filled + "░" * (width - filled)

def print_session_result(r: SessionResult, verbose: bool):
    print(f"\n{clr(DIV, 'bold')}")
    print(clr(f"  PROMPT-RESPONSE EVALUATION — {r.label}", "bold"))
    print(DIV)
    print(f"  File   : {clr(Path(r.path).name, 'blue')}")
    print(f"  Turns  : {len(r.turn_scores)}")
    print()

    # Dimension averages
    for dim in DIMS:
        avg = r.avg_dim(dim)
        label = DIM_LABELS[dim]
        print(f"  {label:<24} {clr(bar(avg), score_color(avg))}  "
              f"{clr(str(avg), score_color(avg)):>5}")

    print(DIV2)
    ov = r.avg_overall
    g  = r.grade
    print(f"  {'OVERALL (avg)':<24} {clr(bar(ov), score_color(ov))}  "
          f"{clr(str(ov), score_color(ov)):>5}   Grade: {clr(g, grade_color(g))}")

    if verbose:
        print(f"\n  {clr('PER-TURN SCORES', 'bold')}")
        print(DIV2)
        print(f"  {'#':<4} {'Q (preview)':<40} {'Rel':>4} {'Cmp':>4} "
              f"{'Con':>4} {'Dir':>4} {'Ovr':>4} {'Grd':>4}")
        print("  " + "-" * 68)
        for t in r.turn_scores:
            q_prev = t.question[:38].replace('\n', ' ')
            g_col  = grade_color(t.grade)
            print(f"  {t.index:<4} {q_prev:<40} "
                  f"{clr(str(t.relevance):>4, score_color(t.relevance))} "
                  f"{clr(str(t.completeness):>4, score_color(t.completeness))} "
                  f"{clr(str(t.conciseness):>4, score_color(t.conciseness))} "
                  f"{clr(str(t.directness):>4, score_color(t.directness))} "
                  f"{clr(str(t.overall):>4, score_color(t.overall))} "
                  f"{clr(t.grade:>4, g_col)}")

        # Flag worst turns
        worst = sorted(r.turn_scores, key=lambda x: x.overall)[:3]
        if worst and worst[0].overall < 60:
            print(f"\n  {clr('Lowest scoring turns:', 'yellow')}")
            for t in worst:
                if t.overall < 60:
                    print(f"    Turn #{t.index}  Overall: {clr(str(t.overall),'red')}  "
                          f"Q: {t.question[:80]}")

def print_comparison(results: list):
    if len(results) < 2:
        return
    print(f"\n{clr(DIV, 'bold')}")
    print(clr("  SESSION COMPARISON", "bold"))
    print(DIV)

    col = 20
    labels = [r.label for r in results]
    print(f"  {'Dimension':<26}  " +
          "  ".join(clr(f"{l[:col]:<{col}}", "cyan" if i == 0 else "magenta")
                    for i, l in enumerate(labels)))
    print("  " + "-" * (26 + (col + 4) * len(results)))

    for dim in DIMS:
        avgs = [r.avg_dim(dim) for r in results]
        row  = f"  {DIM_LABELS[dim]:<26}  "
        row += "  ".join(clr(f"{a:<{col}}", score_color(a)) for a in avgs)
        print(row)

    print("  " + "-" * (26 + (col + 4) * len(results)))
    ovrls = [r.avg_overall for r in results]
    row   = f"  {'OVERALL':<26}  "
    row  += "  ".join(clr(f"{o:<{col}}", score_color(o)) for o in ovrls)
    print(row)

    best = max(results, key=lambda x: x.avg_overall)
    print(f"\n  {clr('Best prompt-response quality:', 'bold')} "
          f"{clr(best.label, 'green')} ({best.avg_overall}/100  Grade {best.grade})")

# ── text report ───────────────────────────────────────────────────────────────
def build_report(results: list) -> str:
    lines = [DIV, "PROMPT-RESPONSE EVALUATOR REPORT", DIV, ""]
    for r in results:
        lines += [
            f"Session : {r.label}  ({r.path})",
            f"Turns   : {len(r.turn_scores)}",
            f"Overall : {r.avg_overall}/100  Grade {r.grade}", "",
        ]
        for dim in DIMS:
            lines.append(f"  {DIM_LABELS[dim]:<24} {r.avg_dim(dim)}/100")
        lines.append("")
        for t in r.turn_scores:
            lines += [
                f"  Turn #{t.index}  Overall: {t.overall}/100 ({t.grade})",
                f"    Q: {t.question[:100]}",
                f"    Rel:{t.relevance} Cmp:{t.completeness} "
                f"Con:{t.conciseness} Dir:{t.directness}",
                "",
            ]
    return "\n".join(lines)

# ── CSV export ────────────────────────────────────────────────────────────────
def export_csv(results: list, out: Path):
    rows = []
    for r in results:
        for t in r.turn_scores:
            rows.append({
                "session":      r.label,
                "file":         Path(r.path).name,
                "turn":         t.index,
                "overall":      t.overall,
                "grade":        t.grade,
                "relevance":    t.relevance,
                "completeness": t.completeness,
                "conciseness":  t.conciseness,
                "directness":   t.directness,
                "consistency":  t.consistency,
                "coverage":     t.coverage,
                "question":     t.question[:200],
                "answer_words": len(t.answer.split()),
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
        description="Evaluate how well AI responses address the questions asked."
    )
    parser.add_argument("files", nargs="*",
                        help="Session log files (.txt / .docx / .pdf)")
    parser.add_argument("--prompt",   default=None,
                        help="Single prompt file (use with --response)")
    parser.add_argument("--response", default=None,
                        help="Single response file (use with --prompt)")
    parser.add_argument("--label",   nargs="*", default=[], dest="labels",
                        help="Display name per session file")
    parser.add_argument("--min-words", type=int, default=10,
                        help="Minimum response words to score (default: 10)")
    parser.add_argument("-o", "--output", help="Save report to file")
    parser.add_argument("--csv",          help="Export turn scores to CSV")
    parser.add_argument("-v", "--verbose",action="store_true",
                        help="Print per-turn score table and worst turns")
    args = parser.parse_args()

    results = []

    # ── Single prompt/response pair mode ─────────────────────────────────────
    if args.prompt and args.response:
        pp = Path(args.prompt)
        rp = Path(args.response)
        for p in (pp, rp):
            if not p.exists():
                print(clr(f"Error: not found — {p}", "red")); sys.exit(1)
        try:
            q_text = read_file(pp)
            a_text = read_file(rp)
        except Exception as e:
            print(clr(f"Error: {e}", "red")); sys.exit(1)

        ts = score_turn(q_text.strip(), a_text.strip(), pp.stem, 1)
        sr = SessionResult(label=pp.stem, path=str(pp), turn_scores=[ts])
        results.append(sr)

    # ── Session file mode ─────────────────────────────────────────────────────
    paths = [Path(f) for f in args.files if Path(f).exists()]
    missing = [f for f in args.files if not Path(f).exists()]
    for m in missing:
        print(clr(f"Warning: not found — {m}", "yellow"))

    labels = list(args.labels)
    while len(labels) < len(paths):
        labels.append(paths[len(labels)].stem)

    for p, label in zip(paths, labels):
        try:
            text = read_file(p)
        except Exception as e:
            print(clr(f"Error reading {p.name}: {e}", "red")); continue

        turns = parse_turns(text, label)
        if not turns:
            print(clr(f"  No Q&A turns found in {p.name}", "yellow")); continue

        turn_scores = []
        for t in turns:
            if len(t["answer"].split()) < args.min_words:
                continue
            ts = score_turn(t["question"], t["answer"], label, t["index"])
            turn_scores.append(ts)

        if not turn_scores:
            print(clr(f"  No scoreable turns in {p.name}", "yellow")); continue

        sr = SessionResult(label=label, path=str(p), turn_scores=turn_scores)
        results.append(sr)

    if not results:
        print(clr("No results to display. Check your input files.", "red"))
        sys.exit(1)

    for r in results:
        print_session_result(r, args.verbose)

    if len(results) > 1:
        print_comparison(results)

    if args.output:
        Path(args.output).write_text(build_report(results), encoding="utf-8")
        print(clr(f"\nReport saved → {args.output}", "green"))

    if args.csv:
        export_csv(results, Path(args.csv))

if __name__ == "__main__":
    main()
