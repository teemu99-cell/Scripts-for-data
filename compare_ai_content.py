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
    # English
    "the","a","an","and","or","but","in","on","at","to","of","for","with",
    "by","from","is","are","was","were","be","been","has","have","had",
    "that","this","which","it","its","they","their","we","our","as","not",
    "no","so","if","than","then","also","more","most","will","would","can",
    "could","should","may","might","very","just","only","all","any","some",
    "such","each","into","about","both","these","those","there","here",
    "does","did","do","its","than","then","i","he","she","him","her",
    "first","second","third","one","two","three","four","five",
    "document","documents","file","files","text","content","section",
    "language","translation","audience","publication","structure","details",
    "keskittyy","enemmän","roolia","rooli","asemaa","asema","johtajuutta",
    "table","contents","highlight","adapted","differences","similarities",
    "kanssa","välillä","kautta","avulla","osalta","mukaan","lisäksi",
    "näiden","näillä","näitä","näihin","niitä","niillä","niiden",
    "käsittely","käsittelyä","käsittelyssä","kritisoi","korostaa",
    "painottaa","painottavat","korostavat","pyrkivät","pyrkii",
    "arvojen","arvoilla","arvoista","arvoja","arvo",
    "principles","priorities","strategic","approach","policy",
    "includes","include","included","provides","provided","provide",
    "based","following","specific","general","similar","different",
    "information","details","detail","note","notes","while","whereas",
    "however","although","whereas","therefore","furthermore","moreover",
    "additionally","meanwhile","nevertheless","nonetheless",
    # Finnish discourse words and connectors
    "ja","tai","on","ei","se","että","oli","ovat","myös","kuin","mutta",
    "jos","sekä","kun","tämä","nämä","siitä","niiden","hän","he","me",
    "te","jo","vain","koko","eri","yli","alle","koska","jotta","vaikka",
    "siis","sillä","kuitenkin","joten","sekä","joka","jotka","jonka",
    "joita","joiden","jossa","jolla","jolle","jolta","josta","johon",
    "kun taas","taas","kun","sama","myös","lisäksi","erityisesti",
    "toisaalta","esimerkiksi","mukaan","välillä","osalta","suhteen",
    # Finnish document-comparison discourse phrases (should never be topics)
    "dokumentti","dokumenttia","dokumentin","dokumenttien","dokumentit",
    "aineisto","aineistoa","aineiston","tässä","tässä on","joitain",
    "joitakin","molemmat","molemmissa","molempien","molempia",
    "ensimmäinen","toinen","kolmas","ensimmäisessä","toisessa",
    "ensimmäistä","toista","vuoden","vuodelta","vuodesta","vuotta",
    "strategia","strategiaa","strategian","strategiat","strategioita",
    "dokumentti käsittelee","dokumentti kertoo","dokumentti sisältää",
    "hallinto","hallinnon","hallintoa","presidentti","presidentin",
    # Year references (not topics)
    "2022","2025","2024","2023","vuoden 2022","vuoden 2025",
}

# Additional phrases to filter from auto-extracted topics
AUTO_TOPIC_BLOCKLIST = {
    # Finnish discourse fragments
    "kun taas","toinen dokumentti","molemmat strategiat","aineisto 2022",
    "vuoden 2025","vuoden 2022","tässä on","dokumentti käsittelee",
    "dokumentti kertoo","joitakin tärkeimpiä","tärkeimpiä eroja",
    "eroja yhtäläisyyksiä","yhtäläisyyksiä molemmat","molemmat dokumentit",
    "dokumentit korostavat","korostavat yhdysvaltain","painottavat talouden",
    "talouden vahvistamista","vahvistamista innovaatioiden",
    "sotilaallisen voiman","voiman roolia","roolia niiden",
    "niiden saavuttamisessa","heijastavat hallintojen","hallintojen prioriteetteja",
    "keskittyy enemmän","enemmän globaaleihin","globaaleihin haasteisiin",
    "globaalien haasteiden","haasteiden ratkaisemiseen","ratkaisemiseen yhteistyötä",
    "roolia yhdysvaltain","roolia yhdysvalloiden","yhdysvalloiden asemaa",
    "yhdysvaltain johtajuutta","johtajuutta globaalissa","globaalissa politiikassa",
    "kiinan venäjän","venäjän kanssa","kanssa globaalien",
    "donald trumpin","trumpin hallintoa","hallintoa sen","sen ideologiaa",
    "biden hallintoa","bidenin hallinto",
    # English document-structure fragments
    "table contents","table of","of contents","language audience",
    "language translation","content structure","publication details",
    "additional information","key differences","following differences",
    "these differences","both documents","first document","second document",
    "one document","other document","document focuses","document provides",
    "document includes","document mentions","document describes",
    "highlight how","adapted for","different audiences",
    # Comparison meta-language (not topics)
    "more detailed","more focused","more comprehensive",
    "key differences","main differences","notable differences",
    "similarities differences","common themes","shared themes",
    # More Finnish fragments
    "näillä dokumenteilla","näiden dokumenttien","dokumenteilla on",
    "liittolaisten kanssa","kanssa vahvistamaan","kanssa ratkaisemiseen",
    "käsittely kritisoi","kritisoi voimakkaasti","voimakkaasti kiinaa",
    "venäjän hyökkäykseen","hyökkäykseen ukrainaan","ukrainaan liittyen",
    "demokraattisten arvojen","arvojen puolustamista","puolustamista globaalien",
    "principles priorities","key principles","strategic priorities",
    "näiden erojen","erojen taustalla","taustalla voidaan",
    "tärkeimmät erot","pääasiallinen ero","keskeinen ero",
    "principles and","and priorities","based on",
}

# ── curated topic taxonomy ────────────────────────────────────────────────────
# Fixed list — auto-extraction is disabled by default for Finnish/mixed content.
# Add --no-auto-topics flag or edit this list to customise for your documents.
FALLBACK_TOPICS = {
    # Geopolitics & security
    "Russia / Conflict":      ["russia","ukraine","venäjä","ukraina","conflict",
                                "war","invasion","hyökkäys","sota","ukrainan"],
    "China / Kiina":          ["china","kiina","chinese","prc","taiwan","indo-pacific",
                                "kiinan","beijing"],
    "National Security":      ["national security","kansallinen turvallisuus",
                                "turvallisuusstrategia","security strategy",
                                "kansallinen","turvallisuus"],
    "Military / Operations":  ["military","operation","sotilaallinen","operaatio",
                                "armed forces","asevoimat","defense","puolustus",
                                "midnight hammer","weapon","ase","troops","joukot"],
    "Alliances / NATO":       ["nato","alliance","liittolainen","liittouma",
                                "allies","partnership","yhteistyö","aukus","quad"],
    # Politics & governance
    "Democracy / Values":     ["democracy","demokratia","democratic","demokraattinen",
                                "freedom","vapaus","human rights","ihmisoikeus",
                                "values","arvot","liberal","liberaali"],
    "America First":          ["america first","unilateral","transactional",
                                "sovereignty","suvereniteetti","border","raja",
                                "nationalism","nationalismi","trump corollary"],
    "Multilateralism":        ["multilateral","monenkeskinen","rules-based",
                                "international order","coalition","koalitio",
                                "diplomacy","diplomatia","global cooperation"],
    # Economics
    "Economy / Trade":        ["economy","trade","tariff","tulli","gdp","bkt",
                                "talous","kauppa","inflation","inflaatio",
                                "sanction","pakote","budget","budjetti",
                                "energy","energia","infrastructure","infrastruktuuri"],
    # Other domains
    "Technology / Cyber":     ["technology","cyber","digital","teknologia",
                                "kyber","artificial intelligence","tekoäly",
                                "space","avaruus","innovation","innovaatio"],
    "Climate / Environment":  ["climate","ilmasto","environment","ympäristö",
                                "energy transition","clean energy","emission","päästö"],
    # NATO doctrine (for AJP documents)
    "Doctrine / Structure":   ["doctrine","doktriini","annex","liite","edition",
                                "versio","hierarchy","hierarkia","publication",
                                "julkaisu","capstone","keystone","ajp","allied joint"],
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
        # Skip if any word in the gram is a pure stopword or it's in the blocklist
        if gram in AUTO_TOPIC_BLOCKLIST: continue
        if all(w in STOPWORDS for w in gram.split()): continue
        # Skip if gram looks like a sentence fragment (verb-heavy)
        frag_words = {"käsittelee","kertoo","sisältää","korostaa","painottaa",
                      "edustaa","viittaa","julkaistu","julkaisee","pyrkii",
                      "provides","describes","focuses","includes","represents"}
        if any(w in frag_words for w in gram.split()): continue
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


# ── multi-file scoring ────────────────────────────────────────────────────────
def score_all_files(texts: list, names: list, taxonomy: dict) -> list:
    """
    Score each file against the combined pool of all other files.
    Returns a list of dicts with name, scores, and topics.
    """
    results = []
    for i, (name, text) in enumerate(zip(names, texts)):
        # "other" = all texts except this one combined
        other_texts = [t for j, t in enumerate(texts) if j != i]
        other_combined = " ".join(other_texts)
        scores = score_response(text, other_combined)
        topics = detect_topics_in_text(text, taxonomy)
        results.append({
            "name":   name,
            "scores": scores,
            "topics": topics,
            "text":   text,
        })
    return results


# ── leaderboard display ───────────────────────────────────────────────────────
def print_leaderboard(results: list):
    sorted_r = sorted(results, key=lambda x: -x["scores"]["overall_score"])

    print(f"\n{clr(DIV, 'bold')}")
    print(clr("  AI CONTENT COMPARISON — LEADERBOARD", "bold"))
    print(DIV)

    score_keys = ["overall_score","specificity_score","claim_score",
                  "structure_score","depth_score"]
    labels     = ["Overall","Specificity","Claim Density","Structure","Depth"]
    col = 10

    # Header
    h = f"  {'File':<32}"
    for l in labels:
        h += f"  {l[:col]:<{col}}"
    print(h)
    print("  " + "-" * (32 + (col+2)*len(labels)))

    for r in sorted_r:
        row = f"  {r['name'][:30]:<32}"
        for key in score_keys:
            v  = r["scores"].get(key, 0)
            vc = "green" if float(v)>=7 else "yellow" if float(v)>=4 else "red"
            row += f"  {clr(str(v)[:col], vc):<{col+9}}"
        print(row)

    print()
    winner = sorted_r[0]
    loser  = sorted_r[-1]
    print(f"  {clr('▶  HIGHEST OVERALL:', 'bold')} "
          f"{clr(winner['name'], 'green')} ({winner['scores']['overall_score']}/10)")
    if len(sorted_r) > 1:
        print(f"  {clr('▶  LOWEST OVERALL: ', 'bold')} "
              f"{clr(loser['name'],  'red')}  ({loser['scores']['overall_score']}/10)")

    # Dimension winners
    print(f"\n  {clr('DIMENSION WINNERS', 'bold')}")
    print("  " + "-" * 50)
    for key, label in zip(score_keys[1:], labels[1:]):
        vals = [(r["scores"].get(key,0), r["name"]) for r in results]
        best_v, best_n = max(vals, key=lambda x: float(x[0]))
        worst_v, worst_n = min(vals, key=lambda x: float(x[0]))
        spread = round(float(best_v) - float(worst_v), 1)
        sp_str = clr(f"  (spread: {spread})", "yellow") if spread >= 2 else ""
        print(f"  {label:<20} {clr(best_n[:28], 'green')}{sp_str}")

    # Topic coverage summary
    all_topics = sorted({t for r in results for t in r["topics"]})
    if all_topics:
        print(f"\n  {clr('TOPIC COVERAGE', 'bold')}")
        print("  " + "-" * 50)
        for topic in all_topics:
            covers = [r["name"][:16] for r in results if topic in r["topics"]]
            ratio  = len(covers) / len(results)
            bar_c  = "green" if ratio >= 0.7 else "yellow" if ratio >= 0.4 else "red"
            print(f"  {topic:<30} {clr(f'{len(covers)}/{len(results)} files', bar_c)}")
    print()


# ── multi-file CSV export ─────────────────────────────────────────────────────
def export_csv_multi(results: list, out: Path):
    score_keys = ["overall_score","specificity_score","claim_score",
                  "structure_score","depth_score","contrast_refs",
                  "claim_count","total_words","avg_sent_words",
                  "section_refs","doc_names","urls"]

    all_topics = sorted({t for r in results for t in r["topics"]})

    fieldnames = (["file"] +
                  score_keys +
                  [f"topic:{t[:30]}" for t in all_topics] +
                  ["best_dimension","worst_dimension","topics_covered"])

    rows = []
    for r in sorted(results, key=lambda x: -x["scores"]["overall_score"]):
        row = {"file": r["name"]}
        dim_scores = {}
        for k in score_keys:
            v = r["scores"].get(k, 0)
            row[k] = v
            if k.endswith("_score"):
                dim_scores[k] = float(v)
        for t in all_topics:
            row[f"topic:{t[:30]}"] = "yes" if t in r["topics"] else "no"
        if dim_scores:
            row["best_dimension"]  = max(dim_scores, key=dim_scores.get)
            row["worst_dimension"] = min(dim_scores, key=dim_scores.get)
        row["topics_covered"] = len(r["topics"])
        rows.append(row)

    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)
    print(clr(f"\nCSV exported → {out}", "green"))
    print(clr(f"  {len(results)} files × {len(score_keys)} scores + "
              f"{len(all_topics)} topics, sorted by overall score", "blue"))


# ── text report ───────────────────────────────────────────────────────────────
def build_text_report_multi(results: list) -> str:
    lines = [DIV, "AI CONTENT COMPARISON REPORT", DIV, ""]
    sorted_r = sorted(results, key=lambda x: -x["scores"]["overall_score"])
    for r in sorted_r:
        s = r["scores"]
        lines += [
            f"File    : {r['name']}",
            f"Overall : {s.get('overall_score',0)}/10",
            f"  Specificity : {s.get('specificity_score',0)}",
            f"  Claim Density: {s.get('claim_score',0)}",
            f"  Structure   : {s.get('structure_score',0)}",
            f"  Depth       : {s.get('depth_score',0)}",
            f"  Topics      : {', '.join(sorted(r['topics'])[:6])}",
            "",
        ]
    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Content-level comparison of multiple AI outputs.")
    parser.add_argument("files", nargs="+",
                        help="Two or more AI output files to compare "
                             "(.txt / .docx / .pdf)")
    parser.add_argument("-o","--output",      help="Save text report")
    parser.add_argument("--csv",              help="Export scores + topics to CSV")
    parser.add_argument("-v","--verbose",     action="store_true",
                        help="Show detail breakdown per file")
    parser.add_argument("--topics-file",      default=None,
                        help='JSON file with custom topic taxonomy: {"Cat":["kw1","kw2"]}')
    parser.add_argument("--label", action="append", default=[],
                        dest="labels",
                        help="Short display label per file (repeat once per file, "
                             "in same order as files)")
    parser.add_argument("--no-auto-topics",   action="store_true",
                        help="(deprecated, auto-topics now off by default)")
    parser.add_argument("--auto-topics",      action="store_true",
                        dest="auto_topics",
                        help="Enable auto-topic extraction from document content "
                             "(may produce noisy results with Finnish text)")
    args = parser.parse_args()

    # Validate and load files
    paths = []
    for f in args.files:
        p = Path(f)
        if not p.exists():
            print(clr(f"Warning: not found — {p}", "yellow"))
        else:
            paths.append(p)

    if len(paths) < 2:
        print(clr("Error: need at least 2 files to compare.", "red")); sys.exit(1)

    print(clr(f"\nLoading {len(paths)} file(s)…", "bold"))
    texts, names = [], []
    for p in paths:
        try:
            raw  = read_file(p)
            text = extract_ai_answer(raw)
            if not text.strip():
                print(clr(f"  Warning: {p.name} is empty — skipping.", "yellow"))
                continue
            texts.append(text)
            names.append(p.stem)  # overridden by --label below
            print(f"  {p.name:<50} {len(text.split()):>6} words")
        except Exception as e:
            print(clr(f"  Error reading {p.name}: {e}", "red"))

    if len(texts) < 2:
        print(clr("Not enough valid files.", "red")); sys.exit(1)

    # Apply --label overrides
    if args.labels:
        for i, lbl in enumerate(args.labels):
            if i < len(names):
                names[i] = lbl

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
        # Auto-extract disabled by default for Finnish/mixed content
        # Enable with --auto-topics flag
        pass
    if args.auto_topics:
        combined1 = " ".join(texts[:len(texts)//2])
        combined2 = " ".join(texts[len(texts)//2:])
        auto = extract_auto_topics(combined1, combined2, top_n=10)
        taxonomy.update(auto)
        print(clr(f"Auto-extracted {len(auto)} topic clusters from content.", "cyan"))

    # Score all files
    print(clr(f"\nScoring {len(texts)} file(s)…", "bold"))
    results = score_all_files(texts, names, taxonomy)

    # Print individual scores if verbose, otherwise just leaderboard
    if args.verbose:
        for r in results:
            s = r["scores"]
            print(f"\n  {clr(r['name'], 'bold')}")
            for k, l in [("overall_score","Overall"),("specificity_score","Specificity"),
                         ("claim_score","Claim Density"),("structure_score","Structure"),
                         ("depth_score","Depth")]:
                v  = s.get(k,0)
                vc = "green" if float(v)>=7 else "yellow" if float(v)>=4 else "red"
                print(f"    {l:<20} {clr(str(v), vc)}")
            if r["topics"]:
                print(f"    {'Topics':<20} {', '.join(sorted(r['topics'])[:5])}")

    print_leaderboard(results)

    if args.output:
        Path(args.output).write_text(build_text_report_multi(results), encoding="utf-8")
        print(clr(f"Report saved → {args.output}", "green"))

    if args.csv:
        export_csv_multi(results, Path(args.csv))

if __name__ == "__main__":
    main()
