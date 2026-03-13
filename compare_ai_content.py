#!/usr/bin/env python3
"""
compare_ai_content.py
---------------------
Content-level comparison of AI session outputs (.txt, .docx, .pdf).
Instead of line-by-line diff, this compares WHAT each AI said:
  - Which topics each AI covered
  - Key points per topic
  - What one AI mentioned that the other missed
  - Agreement / contradiction detection
  - Overall depth and structure score

Usage:
    python3 compare_ai_content.py file1.docx file2.txt
    python3 compare_ai_content.py file1.txt file2.txt -o report.txt
    python3 compare_ai_content.py *.txt *.docx --csv results.csv

Dependencies:
    pip install python-docx pymupdf --break-system-packages
"""

import re
import sys
import csv
import argparse
from pathlib import Path
from collections import defaultdict

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
C = {"green":"\033[92m","red":"\033[91m","yellow":"\033[93m",
     "blue":"\033[94m","cyan":"\033[96m","bold":"\033[1m",
     "magenta":"\033[95m","reset":"\033[0m"}
def clr(t, c): return f"{C[c]}{t}{C['reset']}"
DIV  = "=" * 68
DIV2 = "-" * 68

# ══════════════════════════════════════════════════════════════
# EDIT THIS SECTION to match your document domain/subject area.
#
# FORMAT:
#   "Category Name": ["keyword1", "keyword2", "keyword3"],
#
# RULES:
#   - All keywords MUST be lowercase (the script lowercases text before matching)
#   - Partial words are fine: "sotilaall" matches "sotilaallinen", "sotilaallista" etc.
#   - Add both Finnish AND English keywords if your files mix languages
#   - More keywords = fewer missed matches, so add synonyms generously
#   - Each category needs at least 2-3 keywords to be reliable
#   - Category names can be anything, they are only used for display
#
# EXAMPLES FOR OTHER DOMAINS:
#   Healthcare:  ["hospital","sairaala","patient","potilas","treatment","hoito"]
#   Climate:     ["climate","ilmasto","emission","päästö","renewable","uusiutuva"]
#   Finance:     ["budget","budjetti","revenue","tulo","profit","voitto"]
#
# TIP: Run the script once on your files, then check which categories
#      show as "missing" — those are hints that you need more keywords.
# ══════════════════════════════════════════════════════════════
# ── topic taxonomy (Finnish + English) ───────────────────────────────────────
TOPICS = {
    "Russia / Venäjä":         ["venäjä","russia","ukraina","ukraine","nato","hyökkäys"],
    "China / Kiina":           ["kiina","china","prc","taiwan","indo-pacific","indo"],
    "Economy / Talous":        ["talous","economy","tariff","talouspolitiikka","imf",
                                "maailmanpankki","world bank","infrastruktuuri","infrastructure",
                                "kilpailukyky","rahoituslaitokset"],
    "Military / Sotilaalliset":["sotilaall","military","operaatio","operation","ase",
                                "weapon","puolustus","defense","voima","hammer","midnight"],
    "Democracy / Demokratia":  ["demokratia","democracy","demokraat","values","arvot",
                                "poliittinen","political"],
    "Europe / Eurooppa":       ["eurooppa","europe","eu ","european union","euroopan unioni"],
    "Allies / Liittolaiset":   ["liittolai","allies","alliance","partnership",
                                "yhteistyö","cooperation"],
    "Development / Kehitys":   ["kehitystyö","development","humanitaarinen","humanitarian",
                                "apu","aid","uudistum"],
    "Iran":                    ["iran","ydin","nuclear","nukleaarinen"],
    "Trade / Kauppa":          ["tariff","tariffi","kauppa","trade","teollisuus","industry"],
    "Leadership / Johtajuus":  ["johtajuus","leadership","johtaja","presidentti","hallinto",
                                "administration","trump","biden"],
}

# ── sentence-level signals ────────────────────────────────────────────────────
AGREE_WORDS    = ["molemmat","both","yhteiset","sama","similarly","likewise","myös","also",
                  "korostavat","painottavat","emphasize"]
CONTRAST_WORDS = ["ero","difference","toisaalta","however","whereas","kun taas","mutta",
                  "differ","unlike","ei käsittele","does not","kritisoi","criticize"]
UNIQUE_MARKERS = ["ainoastaan","only","yksinomaan","exclusively","erityisesti","uniquely",
                  "ei käsittele","does not mention","vain","solely"]

# ── file reading ──────────────────────────────────────────────────────────────
def read_txt(path):
    for enc in ("utf-8","latin-1","cp1252"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode {path}")

def read_docx(path):
    if not HAS_DOCX:
        raise ImportError("pip install python-docx --break-system-packages")
    doc = _docx.Document(str(path))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(parts)

def read_pdf(path):
    if not HAS_PDF:
        raise ImportError("pip install pymupdf --break-system-packages")
    with fitz.open(str(path)) as doc:
        return "\n".join(page.get_text() for page in doc)

def read_file(path):
    s = path.suffix.lower()
    if s == ".docx": return read_docx(path)
    if s == ".pdf":  return read_pdf(path)
    return read_txt(path)

# ── content extraction ────────────────────────────────────────────────────────
def extract_ai_answer(text):
    """Strip chat metadata, keep only the AI's answer text."""
    # Remove timestamp lines like "You: 08:12 AM" / "watsonx 08:12 AM"
    text = re.sub(r'^.*?\d{1,2}:\d{2}\s*(?:AM|PM).*$', '', text, flags=re.MULTILINE)
    # Remove "Raakateksti:" header
    text = re.sub(r'^Raakateksti:.*$', '', text, flags=re.MULTILINE)
    # Remove markdown bold/italic markers
    text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)
    # Remove markdown headers
    text = re.sub(r'^#{1,4}\s*', '', text, flags=re.MULTILINE)
    # Collapse whitespace
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return text

def get_sentences(text):
    """Split text into sentences."""
    parts = re.split(r'(?<=[.!?])\s+|\n', text)
    return [s.strip() for s in parts if len(s.strip()) > 15]

def detect_topics_in_text(text):
    """Return dict of topic -> list of sentences mentioning it."""
    tl = text.lower()
    sentences = get_sentences(text)
    result = defaultdict(list)
    for topic, keywords in TOPICS.items():
        for sent in sentences:
            sl = sent.lower()
            if any(kw in sl for kw in keywords):
                result[topic].append(sent)
    return dict(result)

def score_structure(text):
    """
    Heuristic quality scores:
      - depth:     how many distinct topics covered
      - detail:    average sentence length (proxy for elaboration)
      - structure: presence of headers/lists (organised response)
      - contrast:  how much the AI explicitly compared things
    """
    sentences = get_sentences(text)
    topics_found = detect_topics_in_text(text)
    tl = text.lower()

    depth     = len(topics_found)
    detail    = (sum(len(s.split()) for s in sentences) / max(len(sentences),1))
    structure = min(10, text.count("\n") // 3)   # more line breaks = more structure
    contrast  = sum(1 for w in CONTRAST_WORDS if w in tl)
    agreement = sum(1 for w in AGREE_WORDS    if w in tl)

    # Normalise to 0-10
    depth_score    = min(10, depth * 1.2)
    detail_score   = min(10, detail / 4)
    contrast_score = min(10, contrast * 1.5)

    overall = round((depth_score + detail_score + structure + contrast_score) / 4, 1)
    return {
        "topics_covered": depth,
        "avg_sent_words":  round(detail, 1),
        "structure_score": structure,
        "contrast_score":  contrast,
        "agreement_refs":  agreement,
        "overall_score":   overall,
    }

# ── comparison logic ──────────────────────────────────────────────────────────
def compare_content(text1, text2, name1, name2):
    topics1 = detect_topics_in_text(text1)
    topics2 = detect_topics_in_text(text2)
    scores1 = score_structure(text1)
    scores2 = score_structure(text2)

    all_topics = sorted(set(topics1) | set(topics2))
    only_in_1  = sorted(set(topics1) - set(topics2))
    only_in_2  = sorted(set(topics2) - set(topics1))
    shared     = sorted(set(topics1) & set(topics2))

    return {
        "name1": name1, "name2": name2,
        "topics1": topics1, "topics2": topics2,
        "scores1": scores1, "scores2": scores2,
        "all_topics": all_topics,
        "shared": shared,
        "only_in_1": only_in_1,
        "only_in_2": only_in_2,
    }

# ── display ───────────────────────────────────────────────────────────────────
def bar(score, width=20):
    filled = int(round(score / 10 * width))
    color  = "green" if score >= 7 else "yellow" if score >= 4 else "red"
    return clr("█" * filled + "░" * (width - filled), color) + f"  {score}/10"

def print_report(r, verbose=False):
    n1, n2 = r["name1"], r["name2"]
    s1, s2 = r["scores1"], r["scores2"]

    print(f"\n{clr(DIV, 'bold')}")
    print(clr("  AI CONTENT COMPARISON REPORT", "bold"))
    print(clr(DIV, 'bold'))
    print(f"  {clr(n1, 'cyan')}  vs  {clr(n2, 'magenta')}\n")

    # ── scores table ──────────────────────────────────────────────────────────
    print(clr("  QUALITY SCORES", "bold"))
    print(DIV2)
    metrics = [
        ("Overall score",    "overall_score"),
        ("Topics covered",   "topics_covered"),
        ("Avg words/sentence","avg_sent_words"),
        ("Structure",        "structure_score"),
        ("Contrast/comparison","contrast_score"),
        ("Agreement refs",   "agreement_refs"),
    ]
    col = 26
    print(f"  {'Metric':<24}  {clr(n1[:col], 'cyan'):<{col+10}}  {clr(n2[:col], 'magenta')}")
    print("  " + "-" * 72)
    for label, key in metrics:
        v1, v2 = s1[key], s2[key]
        # highlight winner
        if isinstance(v1, float) or key == "overall_score":
            w1 = clr(str(v1), "green") if v1 >= v2 else clr(str(v1), "red")
            w2 = clr(str(v2), "green") if v2 >= v1 else clr(str(v2), "red")
        else:
            w1, w2 = str(v1), str(v2)
        print(f"  {label:<24}  {w1:<30}  {w2}")

    # ── topic coverage ────────────────────────────────────────────────────────
    print(f"\n{clr('  TOPIC COVERAGE', 'bold')}")
    print(DIV2)
    all_t = r["all_topics"]
    col1 = 28
    print(f"  {'Topic':<26}  {clr(n1[:col1],'cyan'):<14}  {clr(n2[:col1],'magenta')}")
    print("  " + "-" * 60)
    for topic in all_t:
        in1 = topic in r["topics1"]
        in2 = topic in r["topics2"]
        mark1 = clr("  ✓  covered", "green") if in1 else clr("  ✗  missing", "red")
        mark2 = clr("  ✓  covered", "green") if in2 else clr("  ✗  missing", "red")
        print(f"  {topic:<26}  {mark1:<24}  {mark2}")

    # ── unique topics ─────────────────────────────────────────────────────────
    if r["only_in_1"]:
        print(f"\n  {clr('Only in ' + n1 + ':', 'cyan')}")
        for t in r["only_in_1"]:
            print(f"    • {t}")
    if r["only_in_2"]:
        print(f"\n  {clr('Only in ' + n2 + ':', 'magenta')}")
        for t in r["only_in_2"]:
            print(f"    • {t}")
    if r["shared"]:
        print(f"\n  {clr('Covered by both:', 'green')}")
        for t in r["shared"]:
            print(f"    • {t}")

    # ── verbose: key sentences per shared topic ───────────────────────────────
    if verbose and r["shared"]:
        print(f"\n{clr('  KEY SENTENCES PER SHARED TOPIC', 'bold')}")
        print(DIV2)
        for topic in r["shared"]:
            print(f"\n  {clr(topic, 'yellow')}")
            s1_sents = r["topics1"][topic][:2]
            s2_sents = r["topics2"][topic][:2]
            print(f"  {clr(n1 + ':', 'cyan')}")
            for s in s1_sents:
                print(f"    {s[:120]}")
            print(f"  {clr(n2 + ':', 'magenta')}")
            for s in s2_sents:
                print(f"    {s[:120]}")

    # ── verdict ───────────────────────────────────────────────────────────────
    print(f"\n{clr('  VERDICT', 'bold')}")
    print(DIV2)
    o1, o2 = s1["overall_score"], s2["overall_score"]
    winner = n1 if o1 > o2 else n2 if o2 > o1 else None
    if winner:
        print(f"  {clr('Higher overall score:', 'bold')} {clr(winner, 'green')}")
    else:
        print(f"  {clr('Both files scored equally.', 'green')}")

    t1, t2 = s1["topics_covered"], s2["topics_covered"]
    broader = n1 if t1 > t2 else n2 if t2 > t1 else None
    if broader:
        print(f"  {clr('Broader topic coverage:', 'bold')} {clr(broader, 'green')} "
              f"({max(t1,t2)} vs {min(t1,t2)} topics)")

    c1, c2 = s1["contrast_score"], s2["contrast_score"]
    more_contrast = n1 if c1 > c2 else n2 if c2 > c1 else None
    if more_contrast:
        print(f"  {clr('More explicit comparisons:', 'bold')} {clr(more_contrast, 'green')}")

    print(f"\n  {clr('Shared topics:', 'bold')} {len(r['shared'])}  |  "
          f"{clr('Unique to ' + n1 + ':', 'bold')} {len(r['only_in_1'])}  |  "
          f"{clr('Unique to ' + n2 + ':', 'bold')} {len(r['only_in_2'])}")
    print()

# ── text report ───────────────────────────────────────────────────────────────
def build_text_report(r):
    lines = [DIV, "AI CONTENT COMPARISON REPORT", DIV, "",
             f"File 1: {r['name1']}", f"File 2: {r['name2']}", ""]
    lines += ["SCORES", "-"*40]
    for k, v in r["scores1"].items():
        lines.append(f"  {k:<26} {r['name1']}: {v}  |  {r['name2']}: {r['scores2'][k]}")
    lines += ["", "TOPIC COVERAGE", "-"*40]
    for t in r["all_topics"]:
        in1 = "✓" if t in r["topics1"] else "✗"
        in2 = "✓" if t in r["topics2"] else "✗"
        lines.append(f"  {t:<30}  {r['name1']}: {in1}   {r['name2']}: {in2}")
    lines += ["", "UNIQUE TOPICS", "-"*40]
    for t in r["only_in_1"]:
        lines.append(f"  Only in {r['name1']}: {t}")
    for t in r["only_in_2"]:
        lines.append(f"  Only in {r['name2']}: {t}")
    return "\n".join(lines)

# ── CSV export ────────────────────────────────────────────────────────────────
def export_csv(r, out):
    rows = []
    for t in r["all_topics"]:
        rows.append({
            "topic":        t,
            r["name1"]:     "yes" if t in r["topics1"] else "no",
            r["name2"]:     "yes" if t in r["topics2"] else "no",
            "shared":       "yes" if t in r["shared"] else "no",
        })
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(clr(f"\nCSV exported → {out}", "green"))

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Content-level comparison of AI session outputs."
    )
    parser.add_argument("file1")
    parser.add_argument("file2")
    parser.add_argument("-o", "--output", help="Save text report")
    parser.add_argument("--csv",          help="Export topic matrix to CSV")
    parser.add_argument("-v", "--verbose",action="store_true",
                        help="Show key sentences per shared topic")
    args = parser.parse_args()

    p1, p2 = Path(args.file1), Path(args.file2)
    for p in (p1, p2):
        if not p.exists():
            print(clr(f"Error: '{p}' not found.", "red")); sys.exit(1)

    try:
        raw1 = read_file(p1)
        raw2 = read_file(p2)
    except (ImportError, ValueError) as e:
        print(clr(str(e), "red")); sys.exit(1)

    text1 = extract_ai_answer(raw1)
    text2 = extract_ai_answer(raw2)

    result = compare_content(text1, text2, p1.stem, p2.stem)
    print_report(result, verbose=args.verbose)

    if args.output:
        Path(args.output).write_text(build_text_report(result), encoding="utf-8")
        print(clr(f"Report saved → {args.output}", "green"))
    if args.csv:
        export_csv(result, Path(args.csv))

if __name__ == "__main__":
    main()
