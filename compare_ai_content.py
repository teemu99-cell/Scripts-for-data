#!/usr/bin/env python3
"""
compare_ai_content.py
---------------------
Content-level comparison of two AI session outputs (.txt, .docx, .pdf).
Instead of line-by-line diff, this compares WHAT each AI said:

  - Auto-extracted topics from the actual documents (no hardcoded keyword list)
  - Specificity score  — named references, section numbers, document names cited
  - Claim density      — distinct factual assertions per 100 words
  - Structure quality  — whether headers/sections have substantive content
  - Depth score        — how much unique information each response adds
  - Contrast/agreement detection
  - Optional custom topic taxonomy (--topics-file) for domain-specific use

Usage:
    python3 compare_ai_content.py file1.docx file2.txt
    python3 compare_ai_content.py file1.txt file2.txt -o report.txt --csv results.csv -v
    python3 compare_ai_content.py file1.txt file2.txt --topics-file my_topics.json

Topics file format (JSON):
    {
      "Category Name": ["keyword1", "keyword2"],
      "Military": ["doctrine", "operations", "annex"]
    }

Dependencies:
    pip install python-docx pymupdf --break-system-packages
"""

import re
import sys
import csv
import json
import math
import argparse
from pathlib import Path
from collections import defaultdict, Counter

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

# ── stop-words ────────────────────────────────────────────────────────────────
STOPWORDS = {
    "the","a","an","and","or","but","in","on","at","to","of","for","with",
    "by","from","is","are","was","were","be","been","has","have","had",
    "that","this","which","it","its","they","their","we","our","as","not",
    "no","so","if","than","then","also","more","most","will","would","can",
    "could","should","may","might","very","just","only","all","any","some",
    "such","each","into","about","both","these","those","there","here",
    "does","did","do","its","than","then","i","he","she","him","her",
    "first","second","third","one","two","three","four","five",
    "document","documents","file","files","text","content","section",
    "includes","include","included","provides","provided","provide",
    "based","following","specific","general","similar","different",
    "information","details","detail","note","notes",
}

# ── fallback topic taxonomy ───────────────────────────────────────────────────
FALLBACK_TOPICS = {
    "Doctrine / Structure":  ["doctrine","annex","publication","edition","version",
                               "hierarchy","framework","protocol","standard","nato",
                               "allied","joint","appendix","supplement","chapter"],
    "Operations":            ["operation","operational","mission","deployment","force",
                               "campaign","manoeuvre","command","control","execute",
                               "conduct","objective","task","tactical","strategic"],
    "Russia / Conflict":     ["russia","ukraine","venäjä","ukraina","conflict",
                               "war","invasion","sanctions","nato","alliance"],
    "Economy / Trade":       ["economy","trade","tariff","gdp","budget","fiscal",
                               "talous","kauppa","inflation","sanction"],
    "Governance / Policy":   ["policy","governance","legislation","government",
                               "parliament","regulation","directive","law","legal",
                               "authority","jurisdiction","compliance"],
    "Language / Translation":["finnish","english","language","translation","national",
                               "addition","audience","suomi","finnish-specific"],
    "Technology":            ["technology","cyber","digital","information","system",
                               "network","software","data","communication","platform"],
    "Climate / Environment": ["climate","environment","energy","emission","renewable",
                               "sustainability","carbon","green"],
}

# ── specificity indicators ────────────────────────────────────────────────────
RE_SECTION_REF   = re.compile(r'\b(?:section|annex|chapter|paragraph|appendix|'
                               r'part|article|clause|table|figure)\s+[\w\d\.]+', re.I)
RE_EDITION_VER   = re.compile(r'\bedition\s+[a-z]\b|\bversion\s+\d|\bv\d\b', re.I)
RE_DATE_REF      = re.compile(r'\b(?:january|february|march|april|may|june|july|'
                               r'august|september|october|november|december)\s+\d{4}\b'
                               r'|\b\d{4}\b', re.I)
RE_ORG_REF       = re.compile(r'\b(?:nato|nso|un|eu|otan|stanag|ajp|atp|allied)\b', re.I)
RE_DOC_NAME      = re.compile(r'\b[A-Z]{2,}[-_]\d{2,}|AJP-\d+|ATP-\d+', re.I)
RE_URL           = re.compile(r'https?://\S+|www\.\S+')
RE_QUOTED        = re.compile(r'"[^"]{4,60}"')
RE_NUMBERED_LIST = re.compile(r'^\s*\d+[\.\)]\s+\S', re.M)
RE_BULLET_LIST   = re.compile(r'^\s*[-\u2022*]\s+\S', re.M)
RE_BOLD_HEADER   = re.compile(r'\*\*[^*]{3,50}\*\*|^#{1,4}\s+\S', re.M)
RE_CLAIM         = re.compile(
    r'\b(?:is|are|was|were|has|have|had|includes?|provides?|contains?|'
    r'covers?|focuses?|describes?|explains?|states?|notes?|indicates?|'
    r'differs?|unlike|whereas|while|both|neither|only)\b', re.I)

# ── contrast / agreement words ────────────────────────────────────────────────
CONTRAST_WORDS = ["however","whereas","unlike","differ","different","while",
                  "contrast","instead","rather","but","although","though",
                  "on the other hand","toisaalta","ero","mutta","kun taas"]
AGREE_WORDS    = ["both","similarly","likewise","also","same","shared",
                  "molemmat","samoin","yhteiset","myös","sekä"]

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
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

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
    text = re.sub(r'^.*?\d{1,2}:\d{2}\s*(?:AM|PM).*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^Raakateksti:.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'###\s*(USER|ASSISTANT)\s*', '', text, flags=re.I)
    text = re.sub(r'^\*{1,3}(USER|ASSISTANT)\*{1,3}\s*$', '', text, flags=re.MULTILINE|re.I)
    text = re.sub(r'\*\*(USER|ASSISTANT)\*\*', '', text, flags=re.I)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return text

def get_sentences(text):
    parts = re.split(r'(?<=[.!?])\s+|\n', text)
    return [s.strip() for s in parts if len(s.strip()) > 12]

def content_tokens(text):
    return [w for w in re.sub(r'[^\w\s]','',text.lower()).split()
            if w not in STOPWORDS and len(w) > 2]

# ── auto topic extraction ─────────────────────────────────────────────────────
def extract_auto_topics(text1, text2, top_n=10):
    def ngrams(text, n):
        tokens = content_tokens(text)
        return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    grams1 = Counter(ngrams(text1,2) + ngrams(text1,3))
    grams2 = Counter(ngrams(text2,2) + ngrams(text2,3))
    all_g  = Counter()
    all_g.update(grams1); all_g.update(grams2)

    scored = []
    for gram, total in all_g.items():
        if total < 2 or len(gram) < 6: continue
        c1, c2 = grams1.get(gram,0), grams2.get(gram,0)
        scored.append((gram, total * (1 + abs(c1-c2)*0.3), c1, c2))
    scored.sort(key=lambda x: -x[1])

    topics, seen_words = {}, set()
    for gram, _, c1, c2 in scored[:top_n*3]:
        words = set(gram.split())
        if words & seen_words and len(words & seen_words) >= len(words)-1:
            continue
        topics[gram.title()] = [gram] + [w for w in gram.split() if len(w)>3]
        seen_words |= words
        if len(topics) >= top_n: break
    return topics

# ── topic detection ───────────────────────────────────────────────────────────
def detect_topics_in_text(text, taxonomy):
    tl = text.lower()
    sentences = get_sentences(text)
    result = defaultdict(list)
    for topic, keywords in taxonomy.items():
        for sent in sentences:
            if any(kw in sent.lower() for kw in keywords):
                result[topic].append(sent)
    return dict(result)

# ── scoring ───────────────────────────────────────────────────────────────────
def score_specificity(text):
    sr = len(RE_SECTION_REF.findall(text))
    er = len(RE_EDITION_VER.findall(text))
    dr = len(RE_DATE_REF.findall(text))
    or_ = len(RE_ORG_REF.findall(text))
    dn = len(RE_DOC_NAME.findall(text))
    ur = len(RE_URL.findall(text))
    qu = len(RE_QUOTED.findall(text))
    raw = sr*2.0 + er*1.5 + dr*0.5 + or_*0.8 + dn*1.5 + ur*2.0 + qu*1.0
    return {"section_refs":sr,"edition_refs":er,"date_refs":dr,"org_refs":or_,
            "doc_names":dn,"urls":ur,"quoted":qu,
            "specificity_score": min(10.0, round(raw,1))}

def score_claim_density(text):
    sentences = get_sentences(text)
    words = len(text.split())
    claims = [s for s in sentences if RE_CLAIM.search(s)]
    seen, unique = set(), []
    for c in claims:
        key = frozenset(content_tokens(c)[:6])
        if key not in seen:
            seen.add(key); unique.append(c)
    density = (len(unique)/max(words,1))*100
    return {"claim_count":len(unique),"total_words":words,
            "claim_density":round(density,2),
            "claim_score":min(10.0,round(density*1.5,1))}

def score_structure_quality(text):
    words = len(text.split())
    nl = len(RE_NUMBERED_LIST.findall(text))
    bl = len(RE_BULLET_LIST.findall(text))
    bh = len(RE_BOLD_HEADER.findall(text))
    lb = text.count("\n")
    elements = nl + bl + bh
    substance = min(10.0, (words/max(elements,1))/15) if elements>0 else (3.0 if words>100 else 1.0)
    presence  = min(10.0, elements*1.2 + (lb/max(words,1)*100))
    return {"numbered_points":nl,"bullet_points":bl,"bold_headers":bh,
            "structural_elements":elements,
            "words_per_element":round(words/max(elements,1),1),
            "structure_score":round((presence+substance)/2,1)}

def score_depth(text, other_text):
    ts = set(content_tokens(text))
    to = set(content_tokens(other_text))
    unique = ts - to
    vr = len(ts)/max(len(text.split()),1)
    sentences = get_sentences(text)
    asw = sum(len(s.split()) for s in sentences)/max(len(sentences),1)
    return {"unique_tokens":len(unique),"shared_tokens":len(ts&to),
            "uniqueness_ratio":round(len(unique)/max(len(ts),1),3),
            "vocab_richness":round(vr,3),"avg_sent_words":round(asw,1),
            "depth_score":min(10.0,round(
                (len(unique)/10)*0.4 + (vr*10)*0.3 + (asw/3)*0.3, 1))}

def score_response(text, other_text):
    sp = score_specificity(text)
    cl = score_claim_density(text)
    st = score_structure_quality(text)
    dp = score_depth(text, other_text)
    tl = text.lower()
    ca = {"contrast_refs": sum(1 for w in CONTRAST_WORDS if w in tl),
          "agreement_refs": sum(1 for w in AGREE_WORDS if w in tl)}
    overall = round(
        sp["specificity_score"]*0.30 + cl["claim_score"]*0.25 +
        st["structure_score"]*0.20  + dp["depth_score"]*0.15 +
        min(10.0, ca["contrast_refs"]*1.5)*0.10, 1)
    return {**sp, **cl, **st, **dp, **ca, "overall_score": overall}

# ── comparison ────────────────────────────────────────────────────────────────
def compare_content(text1, text2, name1, name2, taxonomy):
    topics1 = detect_topics_in_text(text1, taxonomy)
    topics2 = detect_topics_in_text(text2, taxonomy)
    scores1 = score_response(text1, text2)
    scores2 = score_response(text2, text1)
    all_t   = sorted(set(topics1)|set(topics2))
    return {"name1":name1,"name2":name2,
            "topics1":topics1,"topics2":topics2,
            "scores1":scores1,"scores2":scores2,
            "all_topics":all_t,
            "shared":    sorted(set(topics1)&set(topics2)),
            "only_in_1": sorted(set(topics1)-set(topics2)),
            "only_in_2": sorted(set(topics2)-set(topics1))}

# ── display ───────────────────────────────────────────────────────────────────
def bar(score, width=18):
    filled = int(round(score/10*width))
    color  = "green" if score>=7 else "yellow" if score>=4 else "red"
    return clr("█"*filled+"░"*(width-filled), color)+f"  {score}/10"

def print_report(r, verbose=False):
    n1, n2 = r["name1"], r["name2"]
    s1, s2 = r["scores1"], r["scores2"]

    print(f"\n{clr(DIV,'bold')}")
    print(clr("  AI CONTENT COMPARISON REPORT","bold"))
    print(clr(DIV,'bold'))
    print(f"  {clr(n1,'cyan')}  vs  {clr(n2,'magenta')}\n")

    print(clr("  QUALITY SCORES","bold"))
    print(DIV2)
    metrics = [
        ("Overall score",      "overall_score"),
        ("Specificity",        "specificity_score"),
        ("Claim density",      "claim_score"),
        ("Structure quality",  "structure_score"),
        ("Depth / uniqueness", "depth_score"),
        ("Contrast refs",      "contrast_refs"),
        ("Agreement refs",     "agreement_refs"),
    ]
    col = 26
    print(f"  {'Metric':<24}  {clr(n1[:col],'cyan'):<{col+10}}  {clr(n2[:col],'magenta')}")
    print("  "+"-"*72)
    for label, key in metrics:
        v1, v2 = s1.get(key,0), s2.get(key,0)
        w1 = clr(str(v1),"green") if float(v1)>=float(v2) else clr(str(v1),"red")
        w2 = clr(str(v2),"green") if float(v2)>=float(v1) else clr(str(v2),"red")
        print(f"  {label:<24}  {w1:<30}  {w2}")

    if verbose:
        detail = [("  Section refs","section_refs"),("  Edition refs","edition_refs"),
                  ("  Doc names","doc_names"),("  URLs","urls"),("  Quotes","quoted"),
                  ("  Unique tokens","unique_tokens"),("  Claims","claim_count"),
                  ("  Words total","total_words"),("  Avg sent (words)","avg_sent_words"),
                  ("  Struct elements","structural_elements"),("  Words/element","words_per_element")]
        print(f"\n  {clr('DETAIL BREAKDOWN','bold')}")
        print("  "+"-"*72)
        for label, key in detail:
            print(f"  {label:<24}  {str(s1.get(key,0)):<30}  {s2.get(key,0)}")

    print(f"\n{clr('  TOPIC COVERAGE','bold')}")
    print(DIV2)
    if not r["all_topics"]:
        print(f"  {clr('No topics detected.','yellow')}")
    else:
        print(f"  {'Topic':<30}  {clr(n1[:20],'cyan'):<14}  {clr(n2[:20],'magenta')}")
        print("  "+"-"*64)
        for t in r["all_topics"]:
            m1 = clr("  ✓  covered","green") if t in r["topics1"] else clr("  ✗  missing","red")
            m2 = clr("  ✓  covered","green") if t in r["topics2"] else clr("  ✗  missing","red")
            print(f"  {t:<30}  {m1:<24}  {m2}")

    for side, label, lc in [(r["only_in_1"],n1,"cyan"),(r["only_in_2"],n2,"magenta")]:
        if side:
            print(f"\n  {clr('Only in '+label+':',lc)}")
            for t in side: print(f"    • {t}")
    if r["shared"]:
        print(f"\n  {clr('Covered by both:','green')}")
        for t in r["shared"]: print(f"    • {t}")

    if verbose and r["shared"]:
        print(f"\n{clr('  KEY SENTENCES PER SHARED TOPIC','bold')}")
        print(DIV2)
        for topic in r["shared"]:
            print(f"\n  {clr(topic,'yellow')}")
            for s in r["topics1"].get(topic,[])[:2]:
                print(f"  {clr(n1+':','cyan')} {s[:130]}")
            for s in r["topics2"].get(topic,[])[:2]:
                print(f"  {clr(n2+':','magenta')} {s[:130]}")

    print(f"\n{clr('  VERDICT','bold')}")
    print(DIV2)
    o1, o2 = s1["overall_score"], s2["overall_score"]
    if o1>o2:   print(f"  {clr('Higher overall score:','bold')} {clr(n1,'green')} ({o1} vs {o2})")
    elif o2>o1: print(f"  {clr('Higher overall score:','bold')} {clr(n2,'green')} ({o2} vs {o1})")
    else:       print(f"  {clr('Both scored equally.','green')} ({o1})")

    for label, key in metrics[1:]:
        v1, v2 = float(s1.get(key,0)), float(s2.get(key,0))
        if abs(v1-v2) >= 2.0:
            leader = n1 if v1>v2 else n2
            lc     = "cyan" if v1>v2 else "magenta"
            print(f"  {clr('▸','yellow')} {clr(leader,lc)} leads on "
                  f"{clr(label.strip(),'bold')} by {abs(v1-v2):.1f} pts")

    t1c, t2c = len(r["topics1"]), len(r["topics2"])
    if t1c != t2c:
        broader = n1 if t1c>t2c else n2
        lc = "cyan" if t1c>t2c else "magenta"
        print(f"  {clr('▸','yellow')} {clr(broader,lc)} covers more topics "
              f"({max(t1c,t2c)} vs {min(t1c,t2c)})")
    print()

# ── text report ───────────────────────────────────────────────────────────────
def build_text_report(r):
    s1, s2 = r["scores1"], r["scores2"]
    lines = [DIV,"AI CONTENT COMPARISON REPORT",DIV,"",
             f"File 1: {r['name1']}",f"File 2: {r['name2']}","","SCORES","-"*40]
    for k in ("overall_score","specificity_score","claim_score","structure_score",
              "depth_score","contrast_refs","claim_count","total_words","avg_sent_words"):
        lines.append(f"  {k:<30} {r['name1']}: {s1.get(k,0)}  |  {r['name2']}: {s2.get(k,0)}")
    lines += ["","TOPIC COVERAGE","-"*40]
    for t in r["all_topics"]:
        i1 = "✓" if t in r["topics1"] else "✗"
        i2 = "✓" if t in r["topics2"] else "✗"
        lines.append(f"  {t:<34}  {r['name1']}: {i1}   {r['name2']}: {i2}")
    lines += ["","UNIQUE TOPICS","-"*40]
    for t in r["only_in_1"]: lines.append(f"  Only in {r['name1']}: {t}")
    for t in r["only_in_2"]: lines.append(f"  Only in {r['name2']}: {t}")
    return "\n".join(lines)

# ── CSV export ────────────────────────────────────────────────────────────────
def export_csv(r, out):
    rows = []
    for t in r["all_topics"]:
        rows.append({"topic":t, r["name1"]:"yes" if t in r["topics1"] else "no",
                     r["name2"]:"yes" if t in r["topics2"] else "no",
                     "shared":"yes" if t in r["shared"] else "no"})
    for k in ("overall_score","specificity_score","claim_score","structure_score",
              "depth_score","claim_count","total_words"):
        rows.append({"topic":f"SCORE:{k}", r["name1"]:r["scores1"].get(k,0),
                     r["name2"]:r["scores2"].get(k,0),"shared":""})
    with open(out,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(clr(f"\nCSV exported → {out}","green"))

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Content-level comparison of two AI session outputs.")
    parser.add_argument("file1")
    parser.add_argument("file2")
    parser.add_argument("-o","--output",      help="Save text report")
    parser.add_argument("--csv",              help="Export topic matrix + scores to CSV")
    parser.add_argument("-v","--verbose",     action="store_true",
                        help="Show detail breakdown and key sentences per topic")
    parser.add_argument("--topics-file",      default=None,
                        help='JSON file with custom topic taxonomy: {"Cat":["kw1","kw2"]}')
    parser.add_argument("--no-auto-topics",   action="store_true",
                        help="Disable auto-topic extraction, use taxonomy only")
    args = parser.parse_args()

    p1, p2 = Path(args.file1), Path(args.file2)
    for p in (p1,p2):
        if not p.exists():
            print(clr(f"Error: '{p}' not found.","red")); sys.exit(1)

    try:
        raw1, raw2 = read_file(p1), read_file(p2)
    except (ImportError, ValueError) as e:
        print(clr(str(e),"red")); sys.exit(1)

    text1 = extract_ai_answer(raw1)
    text2 = extract_ai_answer(raw2)

    # Build taxonomy
    taxonomy = {}
    if args.topics_file:
        tp = Path(args.topics_file)
        if tp.exists():
            try:
                taxonomy = json.loads(tp.read_text(encoding="utf-8"))
                print(clr(f"Loaded {len(taxonomy)} topic categories from {tp.name}","cyan"))
            except Exception as e:
                print(clr(f"Warning: could not load topics file — {e}","yellow"))
        else:
            print(clr(f"Warning: topics file not found — {tp}","yellow"))

    if not taxonomy:
        taxonomy = FALLBACK_TOPICS.copy()

    if not args.no_auto_topics:
        auto = extract_auto_topics(text1, text2, top_n=10)
        taxonomy.update(auto)
        print(clr(f"Auto-extracted {len(auto)} topic clusters from document content.","cyan"))

    result = compare_content(text1, text2, p1.stem, p2.stem, taxonomy)
    print_report(result, verbose=args.verbose)

    if args.output:
        Path(args.output).write_text(build_text_report(result), encoding="utf-8")
        print(clr(f"Report saved → {args.output}","green"))
    if args.csv:
        export_csv(result, Path(args.csv))

if __name__ == "__main__":
    main()
