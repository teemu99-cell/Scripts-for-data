#!/usr/bin/env python3
"""
hallucination_detector.py
--------------------------
Scans AI session logs for assertive factual claims and cross-references
them against any provided source/reference documents. Flags claims that
appear confident but are absent from — or contradict — the source material.

Works entirely on saved text — no live API or internet access needed.

What it flags:
  1. UNSUPPORTED CLAIMS    — assertive statement not found in source docs
  2. NUMERIC MISMATCH      — number in AI answer differs from same number in source
  3. ENTITY MISMATCH       — named entity in AI answer not present in source
  4. HEDGING DEFICIT       — confident claim with zero epistemic hedging
  5. CONTRADICTION SIGNAL  — claim directly negates a sentence in the source

Usage:
    # Scan session for suspicious claims (no source docs — flags high-confidence claims)
    python3 hallucination_detector.py session.txt

    # Cross-reference against source documents
    python3 hallucination_detector.py session.txt --sources brief.txt report.docx

    # Multiple sessions + sources, save CSV
    python3 hallucination_detector.py s1.txt s2.txt --sources ref.docx -o flags.txt --csv flags.csv

Options:
    --sources FILE [FILE …]  Reference/source documents to check against
    -o, --output FILE        Save text report
    --csv FILE               Export flagged claims to CSV
    -v, --verbose            Show surrounding context for each flag
    --min-confidence FLOAT   Min confidence score to include (0–1, default 0.40)

Dependencies:
    pip install python-docx --break-system-packages   (for .docx files)
"""

import re
import sys
import csv
import math
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional

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

# ── assertive claim patterns ──────────────────────────────────────────────────
# Patterns that signal a confident factual assertion
ASSERT_PATTERNS = [
    # Direct statements: "X is Y", "X was Y", "X are Y"
    re.compile(r'\b(\w[\w\s]{3,50}?)\s+(?:is|are|was|were|has|have|had|will be)\s+(.{10,120}?)[.!]', re.I),
    # "According to X, Y" — still a factual claim if no hedging
    re.compile(r'according to\s+[\w\s]{3,40},\s+(.{15,120}?)[.!]', re.I),
    # "X confirmed/stated/said that Y"
    re.compile(r'\b\w[\w\s]{2,30}\s+(?:confirmed|stated|announced|reported|declared)\s+(?:that\s+)?(.{15,120}?)[.!]', re.I),
    # "In [year], X did Y"
    re.compile(r'\bIn\s+(?:19|20)\d{2},?\s+(.{15,120}?)[.!]', re.I),
    # "X% of / X billion / X million"
    re.compile(r'\b\d[\d.,]*\s*(?:%|percent|billion|million|trillion|miljard|prosenttia)\s+(?:of\s+)?[\w\s]{5,60}[.!,]', re.I),
]

# Hedging phrases — when present, claim is softer and lower priority
HEDGE_PHRASES = [
    "may","might","could","possibly","perhaps","likely","arguably",
    "appears to","seems to","suggests","is believed","is thought",
    "according to some","some analysts","it is unclear","not certain",
    "saattaa","ehkä","mahdollisesti","ilmeisesti","arvellaan","epäselvää",
]

# Contradiction signal pairs
CONTRA_PAIRS = [
    (re.compile(r'\bincreas', re.I), re.compile(r'\bdecreas|declin|fell|reduced', re.I)),
    (re.compile(r'\bsupport', re.I), re.compile(r'\boppos|reject|against\b', re.I)),
    (re.compile(r'\bwill\b', re.I),  re.compile(r'\bwill not|won\'t\b', re.I)),
    (re.compile(r'\bconfirmed\b', re.I), re.compile(r'\bdenied\b', re.I)),
    (re.compile(r'\byes\b', re.I),   re.compile(r'\bno\b', re.I)),
]


# ── data model ────────────────────────────────────────────────────────────────
@dataclass
class Claim:
    session:    str
    turn_index: int
    sentence:   str
    claim_text: str
    confidence: float   # 0–1  (how assertive / unsupported)
    flags:      list = field(default_factory=list)
    severity:   str  = "low"

@dataclass
class Turn:
    session:  str
    index:    int
    question: str
    answer:   str


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


# ── source document indexing ──────────────────────────────────────────────────
def get_sentences(text: str) -> list:
    parts = re.split(r'(?<=[.!?])\s+|\n', text)
    return [s.strip() for s in parts if len(s.strip()) > 10]

def tokenize(text: str) -> set:
    return set(re.sub(r'[^\w]', ' ', text.lower()).split())

def build_source_index(source_texts: list) -> dict:
    """Build a word → [sentence] index for fast lookup."""
    index = defaultdict(list)
    all_sentences = []
    for text in source_texts:
        for sent in get_sentences(text):
            tokens = tokenize(sent)
            for tok in tokens:
                if len(tok) > 3:
                    index[tok].append(sent)
            all_sentences.append(sent)
    return {"index": index, "sentences": all_sentences, "full": " ".join(source_texts)}

def source_support_score(claim: str, source_index: dict) -> float:
    """
    0.0 = not found at all in sources
    1.0 = well-supported by sources
    """
    if not source_index:
        return 0.5   # no sources — neutral score
    tokens = tokenize(claim)
    if not tokens:
        return 0.5
    found = sum(1 for tok in tokens if tok in source_index["index"] and len(tok) > 3)
    return found / max(len([t for t in tokens if len(t) > 3]), 1)

def extract_numbers(text: str) -> dict:
    """Return {normalised_number: raw_string} from text."""
    RE_N = re.compile(r'\b(\d{1,3}(?:[.,]\d{1,3})*)\s*(%|percent|billion|million|trillion)?', re.I)
    out = {}
    for m in RE_N.finditer(text):
        raw = m.group(0).strip()
        norm = m.group(1).replace(",", "").replace(".", "")
        out[norm] = raw
    return out

def extract_named_entities(text: str) -> set:
    """
    Very lightweight named-entity approximation:
    capitalised words (not at sentence start) that are 4+ chars.
    """
    words = text.split()
    entities = set()
    for i, w in enumerate(words):
        clean = re.sub(r'[^\w]', '', w)
        if i > 0 and clean and clean[0].isupper() and len(clean) >= 4:
            entities.add(clean.lower())
    return entities


# ── confidence scorer ─────────────────────────────────────────────────────────
def confidence_score(sentence: str) -> float:
    """
    How assertive/overconfident is this sentence?
    High = assertive, no hedging, short sentence (easier to be wrong).
    Returns 0–1.
    """
    sl = sentence.lower()
    hedge_count = sum(1 for h in HEDGE_PHRASES if h in sl)
    has_number  = bool(re.search(r'\b\d', sentence))
    has_named   = bool(re.search(r'\b[A-Z][a-z]{3,}', sentence))
    word_count  = len(sentence.split())

    # Start with 1, reduce for hedging
    base = 1.0
    base -= hedge_count * 0.25
    base  = max(0.1, base)

    # Numbers and named entities raise stakes (more specific = more falsifiable)
    if has_number:  base = min(1.0, base + 0.15)
    if has_named:   base = min(1.0, base + 0.10)

    # Very short sentences are often non-claims
    if word_count < 6:
        base *= 0.5

    return round(base, 3)


# ── claim extractor ───────────────────────────────────────────────────────────
def extract_claims(answer: str, session: str, turn_index: int) -> list:
    """Extract assertive factual claims from an AI answer."""
    sentences = get_sentences(answer)
    claims = []

    for sent in sentences:
        matched = False
        for pat in ASSERT_PATTERNS:
            if pat.search(sent):
                matched = True
                break
        if not matched:
            continue

        conf = confidence_score(sent)
        if conf < 0.15:
            continue

        claims.append(Claim(
            session    = session,
            turn_index = turn_index,
            sentence   = sent,
            claim_text = sent,
            confidence = conf,
        ))

    return claims


# ── flag analyser ─────────────────────────────────────────────────────────────
def analyse_claim(claim: Claim, source_index: dict) -> Claim:
    sl  = source_index
    flags = []

    support = source_support_score(claim.claim_text, sl)

    if sl:
        # 1. Unsupported claim
        if support < 0.20 and claim.confidence > 0.5:
            flags.append(f"UNSUPPORTED  (source overlap: {support:.0%})")

        # 2. Numeric mismatch
        ai_nums  = extract_numbers(claim.claim_text)
        src_nums = extract_numbers(sl.get("full", ""))
        # Numbers that appear in AI answer but differ from source numbers
        for n, raw in ai_nums.items():
            if n and n not in src_nums and len(n) >= 2:
                flags.append(f"NUMERIC NOT IN SOURCE  '{raw}'")

        # 3. Named entity not in source
        ai_ents  = extract_named_entities(claim.claim_text)
        src_text = sl.get("full", "").lower()
        missing_ents = [e for e in ai_ents if e not in src_text and len(e) >= 5]
        if missing_ents and len(missing_ents) <= 4:
            flags.append(f"ENTITY NOT IN SOURCE  {', '.join(missing_ents[:3])}")

        # 4. Contradiction detection
        src_sents = sl.get("sentences", [])
        for src_sent in src_sents:
            for pos_pat, neg_pat in CONTRA_PAIRS:
                ai_pos  = bool(pos_pat.search(claim.claim_text))
                ai_neg  = bool(neg_pat.search(claim.claim_text))
                src_pos = bool(pos_pat.search(src_sent))
                src_neg = bool(neg_pat.search(src_sent))
                # AI says "X increases", source says "X decreases" for same topic
                tok_overlap = len(tokenize(claim.claim_text) & tokenize(src_sent))
                if tok_overlap >= 3:
                    if (ai_pos and src_neg) or (ai_neg and src_pos):
                        flags.append(f"POSSIBLE CONTRADICTION  vs: '{src_sent[:80]}…'")
                        break

    # 5. Hedging deficit (no sources needed — pure text analysis)
    hedge_count = sum(1 for h in HEDGE_PHRASES if h in claim.claim_text.lower())
    if claim.confidence >= 0.80 and hedge_count == 0:
        flags.append(f"HIGH CONFIDENCE, NO HEDGING  (conf: {claim.confidence:.2f})")

    claim.flags    = flags
    sev_score      = len(flags) + (2 if claim.confidence > 0.8 else 0)
    claim.severity = "high" if sev_score >= 3 else "medium" if sev_score >= 1 else "low"
    return claim


# ── display ───────────────────────────────────────────────────────────────────
SEV_COLOR = {"high":"red","medium":"yellow","low":"cyan"}

def print_claim(c: Claim, verbose: bool):
    sev_c = SEV_COLOR.get(c.severity, "reset")
    print(f"\n  {clr(f'[{c.severity.upper()}]', sev_c)}  "
          f"Session: {clr(c.session, 'bold')}  Turn #{c.turn_index}  "
          f"Confidence: {clr(str(c.confidence), 'bold')}")
    print(f"  {clr('Claim:', 'cyan')} {c.sentence[:160]}")
    for f in c.flags:
        print(f"  {clr('▸', sev_c)} {f}")
    print(f"  {DIV2}")

def print_summary(all_claims: list, flagged: list, sessions: list, has_sources: bool):
    high   = sum(1 for c in flagged if c.severity == "high")
    medium = sum(1 for c in flagged if c.severity == "medium")
    low    = sum(1 for c in flagged if c.severity == "low")

    print(f"\n{clr(DIV, 'bold')}")
    print(clr("  HALLUCINATION DETECTOR — SUMMARY", "bold"))
    print(DIV)
    print(f"  Sessions scanned   : {len(sessions)}")
    print(f"  Claims extracted   : {clr(str(len(all_claims)), 'bold')}")
    print(f"  Flagged claims     : {clr(str(len(flagged)), 'bold')}")
    print(f"    {clr('HIGH',   'red')}   : {high}")
    print(f"    {clr('MEDIUM', 'yellow')}: {medium}")
    print(f"    {clr('LOW',    'cyan')}   : {low}")
    if not has_sources:
        print(f"\n  {clr('ℹ No source documents provided.', 'blue')}")
        print(f"    Only hedging-deficit and pattern-based flags are active.")
        print(f"    Add --sources <file> to enable cross-reference checks.")

    if flagged:
        flag_types = defaultdict(int)
        for c in flagged:
            for f in c.flags:
                flag_types[f.split()[0]] += 1
        print(f"\n  {clr('Flag type breakdown:', 'bold')}")
        for ft, n in sorted(flag_types.items(), key=lambda x: -x[1]):
            print(f"    {ft:<35} {n}")
    else:
        print(f"\n  {clr('✓ No suspicious claims flagged.', 'green')}")


# ── text report ───────────────────────────────────────────────────────────────
def build_report(flagged: list, sessions: list) -> str:
    lines = [DIV, "HALLUCINATION DETECTOR REPORT", DIV, "",
             f"Sessions: {', '.join(sessions)}",
             f"Flagged claims: {len(flagged)}", ""]
    for c in flagged:
        lines += [
            f"[{c.severity.upper()}]  {c.session}  Turn #{c.turn_index}  conf={c.confidence}",
            f"  Claim: {c.sentence[:200]}",
        ]
        for f in c.flags:
            lines.append(f"  ▸ {f}")
        lines.append("")
    return "\n".join(lines)


# ── CSV export ────────────────────────────────────────────────────────────────
def export_csv(flagged: list, out: Path):
    rows = [{
        "severity":   c.severity,
        "confidence": c.confidence,
        "session":    c.session,
        "turn":       c.turn_index,
        "claim":      c.sentence[:300],
        "flags":      " | ".join(c.flags),
    } for c in flagged]

    if not rows:
        print(clr("No flagged claims to export.", "yellow")); return
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(clr(f"\nCSV exported → {out}", "green"))


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Detect potentially hallucinated claims in AI session logs."
    )
    parser.add_argument("files", nargs="+",
                        help="Session log files (.txt / .docx) to scan")
    parser.add_argument("--sources", nargs="*", default=[],
                        help="Reference/source documents to cross-check against")
    parser.add_argument("-o", "--output",     help="Save text report to file")
    parser.add_argument("--csv",              help="Export flagged claims to CSV")
    parser.add_argument("-v", "--verbose",    action="store_true",
                        help="Show full claim context")
    parser.add_argument("--min-confidence",   type=float, default=0.40,
                        help="Minimum confidence to show (default: 0.40)")
    args = parser.parse_args()

    # Load session files
    session_paths = [Path(f) for f in args.files if Path(f).exists()]
    for f in args.files:
        if not Path(f).exists(): print(clr(f"[warn] Not found: {f}", "yellow"))
    if not session_paths:
        print(clr("No valid session files found.", "red")); sys.exit(1)

    print(clr(f"\nLoading {len(session_paths)} session file(s)…", "bold"))
    all_turns = []
    for p in session_paths:
        try:
            raw   = read_file(p)
            turns = parse_turns(raw, p.stem)
            print(f"  {p.name}: {len(turns)} turns")
            all_turns.extend(turns)
        except Exception as e:
            print(clr(f"  [error] {p.name}: {e}", "red"))

    # Load source documents
    source_texts = []
    if args.sources:
        print(clr(f"\nLoading {len(args.sources)} source document(s)…", "bold"))
        for f in args.sources:
            p = Path(f)
            if not p.exists():
                print(clr(f"  [warn] Source not found: {f}", "yellow")); continue
            try:
                source_texts.append(read_file(p))
                print(f"  {p.name}: loaded")
            except Exception as e:
                print(clr(f"  [error] {p.name}: {e}", "red"))

    source_index = build_source_index(source_texts)

    # Extract and analyse claims
    print(clr("\nExtracting claims…", "bold"))
    all_claims = []
    for turn in all_turns:
        claims = extract_claims(turn.answer, turn.session, turn.index)
        for c in claims:
            analyse_claim(c, source_index)
        all_claims.extend(claims)

    print(f"  {len(all_claims)} assertive claims found across all sessions")

    # Filter to flagged + min confidence
    flagged = [c for c in all_claims if c.flags and c.confidence >= args.min_confidence]

    session_names = list(dict.fromkeys(t.session for t in all_turns))
    print_summary(all_claims, flagged, session_names, bool(source_texts))

    if flagged:
        print(f"\n{clr(DIV, 'bold')}")
        print(clr("  FLAGGED CLAIMS", "bold"))
        print(DIV)
        order = {"high": 0, "medium": 1, "low": 2}
        for c in sorted(flagged, key=lambda x: (order.get(x.severity, 3), -x.confidence)):
            print_claim(c, args.verbose)

    if args.output:
        Path(args.output).write_text(
            build_report(flagged, session_names), encoding="utf-8"
        )
        print(clr(f"\nReport saved → {args.output}", "green"))

    if args.csv:
        export_csv(flagged, Path(args.csv))


if __name__ == "__main__":
    main()
