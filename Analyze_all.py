#!/usr/bin/env python3
"""
analyze_ai_sessions.py
----------------------
Analyzes AI chatbot test session logs (WatsonX, DocuAI, etc.)
from .txt and .docx files.

Extracts:
  - Questions asked by the user
  - AI responses
  - Per-session statistics (turn count, response length, topics)
  - Side-by-side comparison when multiple files are given

Usage:
    python3 analyze_ai_sessions.py file1.docx file2.txt [options]

Options:
    -o, --output FILE     Save report to a text file
    --csv FILE            Export turn-level data to CSV
    -v, --verbose         Print full Q&A turns
    --lang FI|EN          Language hint for topic detection (default: auto)

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
    import docx
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

# ── colours ───────────────────────────────────────────────────────────────────
C = {"green":"\033[92m","red":"\033[91m","yellow":"\033[93m",
     "blue":"\033[94m","cyan":"\033[96m","bold":"\033[1m","reset":"\033[0m"}
def clr(t, c): return f"{C[c]}{t}{C['reset']}"
DIV = "=" * 64

# ── topic keyword lists (Finnish + English) ───────────────────────────────────
TOPIC_KEYWORDS = {
    "China / Kiina":        ["kiina","china","prc","taiwan","indo-pacific"],
    "Russia / Venäjä":      ["venäjä","russia","ukraina","ukraine","nato"],
    "Economy / Talous":     ["talous","economy","tariff","talouspolitiikka","imf","world bank","maailmanpankki"],
    "Military / Sotilaat":  ["sotilaall","military","operaatio","operation","ase","weapon"],
    "Democracy / Demo.":    ["demokratia","democracy","demokraat","values","arvot"],
    "Europe / Eurooppa":    ["eurooppa","europe","eu","european union"],
    "Climate / Ilmasto":    ["ilmasto","climate","energia","energy","ympäristö"],
    "Allies / Liittolaiset":["liittolai","allies","alliance","partnership","yhteistyö"],
}

def detect_topics(text: str) -> list[str]:
    text_l = text.lower()
    return [topic for topic, kws in TOPIC_KEYWORDS.items()
            if any(kw in text_l for kw in kws)]

# ── data model ────────────────────────────────────────────────────────────────
@dataclass
class Turn:
    index: int
    question: str
    answer: str
    q_words: int = 0
    a_words: int = 0
    topics: list = field(default_factory=list)

    def __post_init__(self):
        self.q_words = len(self.question.split())
        self.a_words = len(self.answer.split())
        self.topics  = detect_topics(self.question + " " + self.answer)

@dataclass
class Session:
    path: str
    file_type: str
    ai_name: str = "AI"
    turns: list = field(default_factory=list)
    raw_text: str = ""
    error: Optional[str] = None

# ── extractors ────────────────────────────────────────────────────────────────
def read_txt(path: Path) -> str:
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode {path}")

def read_docx(path: Path) -> str:
    if not HAS_DOCX:
        raise ImportError("Run: pip install python-docx --break-system-packages")
    doc = docx.Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)

# ── session parser ────────────────────────────────────────────────────────────
# Patterns for: "You: HH:MM AM/PM"  and  "<ainame> HH:MM AM/PM"
RE_YOU  = re.compile(r'^You:\s*\d{1,2}:\d{2}\s*(?:AM|PM)\s*$', re.I)
RE_AI   = re.compile(r'^(\S[\w\s]{0,30}?)\s+\d{1,2}:\d{2}\s*(?:AM|PM)\s*$', re.I)
# Finnish/English question markers (lines ending in ?)
RE_Q    = re.compile(r'\?')

def detect_ai_name(lines: list[str]) -> str:
    """Try to detect the AI assistant's name from the log."""
    for line in lines:
        m = RE_AI.match(line.strip())
        if m:
            name = m.group(1).strip()
            if name.lower() not in ("you", "raakateksti", ""):
                return name
    return "AI"

def parse_turns(text: str) -> tuple[str, list[Turn]]:
    """
    Parse chat log into (ai_name, [Turn]).
    Supports two formats:
      1. Timestamp-based: "You: 08:12 AM" / "watsonx 08:12 AM"
      2. Heading-based:   lines that are clearly questions followed by answer blocks
    """
    lines = text.splitlines()
    ai_name = detect_ai_name(lines)

    turns = []
    current_speaker = None   # "user" | "ai"
    current_block   = []
    pending_q       = None

    def flush(speaker, block):
        nonlocal pending_q, turns
        content = "\n".join(block).strip()
        if not content:
            return
        if speaker == "user":
            pending_q = content
        elif speaker == "ai" and pending_q is not None:
            turns.append(Turn(
                index    = len(turns) + 1,
                question = pending_q,
                answer   = content,
            ))
            pending_q = None

    for line in lines:
        stripped = line.strip()

        if RE_YOU.match(stripped):
            flush(current_speaker, current_block)
            current_speaker = "user"
            current_block   = []
            continue

        if RE_AI.match(stripped):
            cand = RE_AI.match(stripped).group(1).strip()
            if cand.lower() not in ("you", "raakateksti"):
                flush(current_speaker, current_block)
                current_speaker = "ai"
                current_block   = []
                continue

        if stripped.lower() == "raakateksti:":
            continue

        current_block.append(line)

    flush(current_speaker, current_block)

    # Fallback: if no turns parsed, treat the whole text as one AI answer
    if not turns:
        turns = [Turn(index=1, question="(document summary / yhteenveto)", answer=text.strip())]

    return ai_name, turns

# ── analysis ──────────────────────────────────────────────────────────────────
def analyze_file(path: Path) -> Session:
    s = Session(path=str(path), file_type=path.suffix.lower())
    try:
        raw = read_docx(path) if path.suffix.lower() == ".docx" else read_txt(path)
        s.raw_text  = raw
        s.ai_name, s.turns = parse_turns(raw)
    except Exception as e:
        s.error = str(e)
    return s

def session_stats(s: Session) -> dict:
    if not s.turns:
        return {}
    a_words = [t.a_words for t in s.turns]
    q_words = [t.q_words for t in s.turns]
    all_topics = [topic for t in s.turns for topic in t.topics]
    return {
        "turns":          len(s.turns),
        "avg_q_words":    sum(q_words)  / len(q_words),
        "avg_a_words":    sum(a_words)  / len(a_words),
        "total_a_words":  sum(a_words),
        "longest_answer": max(a_words),
        "shortest_answer":min(a_words),
        "top_topics":     Counter(all_topics).most_common(5),
    }

# ── display ───────────────────────────────────────────────────────────────────
def print_session(s: Session, verbose: bool):
    print(f"\n{clr(Path(s.path).name, 'bold')}  [{clr(s.file_type,'blue')}]  AI: {clr(s.ai_name,'cyan')}")
    if s.error:
        print(f"  {clr('ERROR: ' + s.error, 'red')}"); return

    st = session_stats(s)
    if not st:
        print(f"  {clr('No Q&A turns detected.','yellow')}"); return

    print(f"  Turns          : {clr(str(st['turns']), 'bold')}")
    print(f"  Avg Q length   : {st['avg_q_words']:.0f} words")
    avg_a_str = f"{st['avg_a_words']:.0f} words"
    print(f"  Avg A length   : {clr(avg_a_str,'green')}")
    print(f"  Total AI words : {st['total_a_words']}")
    print(f"  Longest answer : {st['longest_answer']} words")
    if st["top_topics"]:
        topics_str = ", ".join(f"{t} ({n})" for t, n in st["top_topics"])
        print(f"  Top topics     : {clr(topics_str, 'yellow')}")

    if verbose:
        print()
        for t in s.turns:
            print(f"  {clr(f'Q{t.index}:', 'cyan')} {t.question[:120]}")
            print(f"  {clr('A:', 'green')} {t.answer[:300]}")
            if t.topics:
                print(f"       topics: {', '.join(t.topics)}")
            print()

def print_comparison(sessions: list[Session]):
    """Side-by-side stats comparison for multiple files."""
    valid = [s for s in sessions if not s.error and s.turns]
    if len(valid) < 2:
        return
    print(f"\n{clr(DIV, 'bold')}")
    print(clr("SIDE-BY-SIDE COMPARISON", "bold"))
    print(DIV)

    stats = [(s, session_stats(s)) for s in valid]
    labels = [Path(s.path).stem[:28] for s in valid]
    col = 28

    def row(label, vals):
        print(f"  {label:<22} " + "  |  ".join(f"{str(v):<{col}}" for v in vals))

    print("  " + " " * 22 + "  ".join(clr(f"{l:<{col}}", "bold") for l in labels))
    print("  " + "-" * (22 + (col + 5) * len(valid)))
    row("AI name",        [s.ai_name for s, _ in stats])
    row("Turns",          [st["turns"] for _, st in stats])
    row("Avg A (words)",  [f"{st['avg_a_words']:.0f}" for _, st in stats])
    row("Total AI words", [st["total_a_words"] for _, st in stats])
    row("Longest answer", [st["longest_answer"] for _, st in stats])

    # Common vs unique topics
    topic_sets = [set(t for turn in s.turns for t in turn.topics) for s in valid]
    common = topic_sets[0].intersection(*topic_sets[1:])
    if common:
        print(f"\n  {clr('Shared topics:', 'green')} {', '.join(sorted(common))}")
    for i, (s, ts) in enumerate(zip(valid, topic_sets)):
        unique = ts - common
        if unique:
            print(f"  {clr(f'Only in {Path(s.path).stem[:20]}:', 'yellow')} {', '.join(sorted(unique))}")

# ── CSV export ────────────────────────────────────────────────────────────────
def export_csv(sessions: list[Session], out: Path):
    rows = []
    for s in sessions:
        for t in s.turns:
            rows.append({
                "file":     Path(s.path).name,
                "ai_name":  s.ai_name,
                "turn":     t.index,
                "q_words":  t.q_words,
                "a_words":  t.a_words,
                "topics":   "; ".join(t.topics),
                "question": t.question[:200],
                "answer":   t.answer[:500],
            })
    if not rows:
        print(clr("No data to export.", "yellow")); return
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(clr(f"\nCSV exported → {out}", "green"))

# ── text report ───────────────────────────────────────────────────────────────
def build_report(sessions: list[Session]) -> str:
    lines = [DIV, "AI SESSION ANALYSIS REPORT", DIV, ""]
    for s in sessions:
        lines.append(f"File : {s.path}")
        lines.append(f"AI   : {s.ai_name}")
        if s.error:
            lines.append(f"Error: {s.error}")
        else:
            st = session_stats(s)
            lines += [
                f"  Turns          : {st.get('turns',0)}",
                f"  Avg Q length   : {st.get('avg_q_words',0):.0f} words",
                f"  Avg A length   : {st.get('avg_a_words',0):.0f} words",
                f"  Total AI words : {st.get('total_a_words',0)}",
                f"  Top topics     : {', '.join(t for t,_ in st.get('top_topics',[]))}",
            ]
        lines.append("")
    return "\n".join(lines)

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Analyze AI chatbot test session logs (.txt / .docx)"
    )
    parser.add_argument("files", nargs="+", help="Session log files to analyze")
    parser.add_argument("-o", "--output", help="Save text report to file")
    parser.add_argument("--csv",          help="Export turn data to CSV")
    parser.add_argument("-v", "--verbose",action="store_true", help="Print Q&A turns")
    args = parser.parse_args()

    paths = []
    for f in args.files:
        p = Path(f)
        if not p.exists(): print(clr(f"Not found: {f}", "red")); continue
        paths.append(p)
    if not paths:
        print(clr("No valid files.", "red")); sys.exit(1)

    print(clr(f"\nAnalyzing {len(paths)} session file(s)...", "bold"))
    print(DIV)

    sessions = [analyze_file(p) for p in paths]
    for s in sessions:
        print_session(s, args.verbose)

    if len(sessions) > 1:
        print_comparison(sessions)

    if args.output:
        Path(args.output).write_text(build_report(sessions), encoding="utf-8")
        print(clr(f"\nReport saved → {args.output}", "green"))
    if args.csv:
        export_csv(sessions, Path(args.csv))

if __name__ == "__main__":
    main()