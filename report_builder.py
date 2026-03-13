#!/usr/bin/env python3
"""
report_builder.py
-----------------
Aggregates output from analyze_ai_sessions.py (CSV + raw files) into a
polished, self-contained HTML report with charts and tables.

Accepts:
  - CSV files exported from analyze_ai_sessions.py  (--csv)
  - Raw session files (.txt / .docx)                (positional args)
  - Both together

Produces:
  - ai_report.html  (self-contained, no external dependencies)

Usage:
    python3 report_builder.py session1.txt session2.docx
    python3 report_builder.py --csv turns.csv
    python3 report_builder.py session1.txt session2.txt --csv turns.csv -o my_report.html
    python3 report_builder.py *.txt --title "WatsonX Evaluation June 2025"

Dependencies:
    pip install python-docx --break-system-packages   (only for .docx input)
"""

import re
import sys
import csv
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

try:
    import docx as _docx
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

# ── topic keywords (mirrors analyze_ai_sessions.py) ──────────────────────────
TOPIC_KEYWORDS = {
    "China / Kiina":         ["kiina","china","prc","taiwan","indo-pacific"],
    "Russia / Venäjä":       ["venäjä","russia","ukraina","ukraine","nato"],
    "Economy / Talous":      ["talous","economy","tariff","talouspolitiikka","imf","world bank"],
    "Military / Sotilaat":   ["sotilaall","military","operaatio","operation","ase","weapon"],
    "Democracy / Demo.":     ["demokratia","democracy","demokraat","values","arvot"],
    "Europe / Eurooppa":     ["eurooppa","europe","eu","european union"],
    "Climate / Ilmasto":     ["ilmasto","climate","energia","energy","ympäristö"],
    "Allies / Liittolaiset": ["liittolai","allies","alliance","partnership","yhteistyö"],
    "Iran":                  ["iran","nuclear","nukleaarinen","ydin"],
    "Trade / Kauppa":        ["tariff","tariffi","kauppa","trade","teollisuus"],
}

def detect_topics(text: str) -> list:
    tl = text.lower()
    return [t for t, kws in TOPIC_KEYWORDS.items() if any(k in tl for k in kws)]


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
    name: str
    ai_name: str = "AI"
    turns: list = field(default_factory=list)
    error: Optional[str] = None

    @property
    def avg_a_words(self):
        if not self.turns: return 0
        return sum(t.a_words for t in self.turns) / len(self.turns)

    @property
    def total_a_words(self):
        return sum(t.a_words for t in self.turns)

    @property
    def topic_counter(self):
        return Counter(topic for t in self.turns for topic in t.topics)


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
        raise ImportError("Run: pip install python-docx --break-system-packages")
    doc = _docx.Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


# ── session parser (same logic as analyze_ai_sessions.py) ────────────────────
RE_YOU = re.compile(r'^You:\s*\d{1,2}:\d{2}\s*(?:AM|PM)\s*$', re.I)
RE_AI  = re.compile(r'^(\S[\w\s]{0,30}?)\s+\d{1,2}:\d{2}\s*(?:AM|PM)\s*$', re.I)

def detect_ai_name(lines):
    for line in lines:
        m = RE_AI.match(line.strip())
        if m:
            name = m.group(1).strip()
            if name.lower() not in ("you", "raakateksti", ""):
                return name
    return "AI"

def parse_session(text: str, name: str) -> Session:
    lines    = text.splitlines()
    ai_name  = detect_ai_name(lines)
    session  = Session(name=name, ai_name=ai_name)

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
            session.turns.append(Turn(
                index    = len(session.turns) + 1,
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

    if not session.turns:
        session.turns = [Turn(index=1, question="(full document)", answer=text.strip())]

    return session

def load_raw_file(path: Path) -> Session:
    try:
        raw = read_docx(path) if path.suffix.lower() == ".docx" else read_txt(path)
        return parse_session(raw, path.stem)
    except Exception as e:
        s = Session(name=path.stem)
        s.error = str(e)
        return s

def load_csv(path: Path) -> list:
    """Load a CSV exported from analyze_ai_sessions.py into Session objects."""
    sessions = {}
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = row.get("file", "unknown")
                if key not in sessions:
                    sessions[key] = Session(
                        name    = Path(key).stem,
                        ai_name = row.get("ai_name", "AI"),
                    )
                sessions[key].turns.append(Turn(
                    index    = int(row.get("turn", 0)),
                    question = row.get("question", ""),
                    answer   = row.get("answer", ""),
                ))
    except Exception as e:
        print(f"[warn] Could not load CSV {path}: {e}")
    return list(sessions.values())


# ── chart helpers (returns inline SVG-safe JSON for Chart.js) ─────────────────
def palette(n):
    """Generate n distinct colours."""
    base = [
        "#4e79a7","#f28e2b","#e15759","#76b7b2","#59a14f",
        "#edc948","#b07aa1","#ff9da7","#9c755f","#bab0ac",
    ]
    return [base[i % len(base)] for i in range(n)]


# ── HTML builder ──────────────────────────────────────────────────────────────
def build_html(sessions: list, title: str) -> str:
    valid = [s for s in sessions if not s.error and s.turns]
    names = [s.name for s in valid]
    colors = palette(len(valid))

    # ── data for charts ───────────────────────────────────────────────────────

    # 1. Response length per session (avg words)
    avg_lengths = [round(s.avg_a_words, 1) for s in valid]

    # 2. Turn count per session
    turn_counts = [len(s.turns) for s in valid]

    # 3. Topic coverage heatmap data
    all_topics = sorted({t for s in valid for t in s.topic_counter})
    topic_matrix = [[s.topic_counter.get(t, 0) for t in all_topics] for s in valid]

    # 4. Answer length distribution (all sessions combined, bucketed)
    bucket_size = 50
    all_a_words = [t.a_words for s in valid for t in s.turns]
    if all_a_words:
        max_w = max(all_a_words)
        buckets = list(range(0, max_w + bucket_size, bucket_size))
        dist_labels = [f"{b}-{b+bucket_size-1}" for b in buckets[:-1]]
        dist_counts = [sum(1 for w in all_a_words if b <= w < b+bucket_size)
                       for b in buckets[:-1]]
    else:
        dist_labels, dist_counts = [], []

    # ── per-session detail rows ───────────────────────────────────────────────
    session_rows_html = ""
    for s, col in zip(valid, colors):
        top5 = s.topic_counter.most_common(5)
        topics_str = ", ".join(f"{t} ({n})" for t, n in top5) or "—"
        session_rows_html += f"""
        <tr>
          <td><span class="dot" style="background:{col}"></span>{s.name}</td>
          <td>{s.ai_name}</td>
          <td>{len(s.turns)}</td>
          <td>{s.avg_a_words:.0f}</td>
          <td>{s.total_a_words:,}</td>
          <td>{max((t.a_words for t in s.turns), default=0)}</td>
          <td class="topics-cell">{topics_str}</td>
        </tr>"""

    # ── turn-level table (first 60 turns across all sessions) ─────────────────
    turn_rows_html = ""
    count = 0
    for s, col in zip(valid, colors):
        for t in s.turns:
            if count >= 60: break
            topics_str = ", ".join(t.topics) if t.topics else "—"
            q_short = (t.question[:90] + "…") if len(t.question) > 90 else t.question
            a_short = (t.answer[:140]   + "…") if len(t.answer)   > 140 else t.answer
            turn_rows_html += f"""
            <tr>
              <td><span class="dot" style="background:{col}"></span>{s.name}</td>
              <td>{t.index}</td>
              <td class="q-cell">{q_short}</td>
              <td class="a-cell">{a_short}</td>
              <td>{t.a_words}</td>
              <td>{topics_str}</td>
            </tr>"""
            count += 1

    more_msg = f"<p class='more-note'>Showing first 60 of {sum(len(s.turns) for s in valid)} turns. Export full CSV with analyze_ai_sessions.py --csv.</p>" if sum(len(s.turns) for s in valid) > 60 else ""

    # ── error section ─────────────────────────────────────────────────────────
    errors_html = ""
    for s in sessions:
        if s.error:
            errors_html += f"<li><code>{s.name}</code>: {s.error}</li>"
    if errors_html:
        errors_html = f"<section class='error-section'><h2>⚠ Load Errors</h2><ul>{errors_html}</ul></section>"

    # ── JSON blobs for Chart.js ────────────────────────────────────────────────
    avg_len_json    = json.dumps({"labels": names, "data": avg_lengths, "colors": colors})
    turn_count_json = json.dumps({"labels": names, "data": turn_counts, "colors": colors})
    dist_json       = json.dumps({"labels": dist_labels, "data": dist_counts})
    topic_json      = json.dumps({
        "sessions": names,
        "topics":   all_topics,
        "matrix":   topic_matrix,
        "colors":   colors,
    })

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  :root {{
    --bg:#f8f9fb; --card:#ffffff; --border:#e2e6ea;
    --text:#212529; --muted:#6c757d; --accent:#4e79a7;
    --green:#28a745; --red:#dc3545; --yellow:#ffc107;
  }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family:'Segoe UI',system-ui,sans-serif; background:var(--bg); color:var(--text); font-size:14px; }}
  header {{ background:var(--accent); color:#fff; padding:28px 40px; }}
  header h1 {{ font-size:1.6rem; font-weight:700; }}
  header p  {{ opacity:.85; margin-top:4px; font-size:.9rem; }}
  main {{ max-width:1280px; margin:0 auto; padding:32px 24px; }}
  h2 {{ font-size:1.1rem; font-weight:600; margin-bottom:16px; color:var(--text); border-left:4px solid var(--accent); padding-left:10px; }}
  section {{ margin-bottom:40px; }}
  .grid-2 {{ display:grid; grid-template-columns:1fr 1fr; gap:24px; }}
  .grid-3 {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px; }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:8px; padding:20px; }}
  .card canvas {{ max-height:280px; }}
  .stat-card {{ text-align:center; }}
  .stat-card .val {{ font-size:2rem; font-weight:700; color:var(--accent); }}
  .stat-card .lbl {{ color:var(--muted); font-size:.85rem; margin-top:4px; }}
  table {{ width:100%; border-collapse:collapse; font-size:.85rem; }}
  th {{ background:#f1f3f5; text-align:left; padding:9px 12px; font-weight:600; border-bottom:2px solid var(--border); }}
  td {{ padding:8px 12px; border-bottom:1px solid var(--border); vertical-align:top; }}
  tr:hover td {{ background:#f8f9fb; }}
  .dot {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; vertical-align:middle; }}
  .topics-cell {{ font-size:.78rem; color:var(--muted); }}
  .q-cell {{ color:#444; max-width:220px; }}
  .a-cell {{ max-width:300px; }}
  .more-note {{ color:var(--muted); font-size:.82rem; margin-top:10px; font-style:italic; }}
  .error-section {{ background:#fff5f5; border:1px solid #ffcccc; border-radius:8px; padding:20px; }}
  .error-section h2 {{ color:var(--red); border-color:var(--red); }}
  footer {{ text-align:center; padding:24px; color:var(--muted); font-size:.8rem; border-top:1px solid var(--border); margin-top:20px; }}
  @media(max-width:768px) {{ .grid-2,.grid-3 {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<header>
  <h1>📊 {title}</h1>
  <p>Generated {generated} &nbsp;·&nbsp; {len(valid)} session(s) &nbsp;·&nbsp; {sum(len(s.turns) for s in valid)} total turns</p>
</header>
<main>

  <!-- KPI cards -->
  <section>
    <div class="grid-3">
      <div class="card stat-card">
        <div class="val">{len(valid)}</div>
        <div class="lbl">Sessions analysed</div>
      </div>
      <div class="card stat-card">
        <div class="val">{sum(len(s.turns) for s in valid)}</div>
        <div class="lbl">Total Q&amp;A turns</div>
      </div>
      <div class="card stat-card">
        <div class="val">{round(sum(s.avg_a_words for s in valid)/max(len(valid),1))}</div>
        <div class="lbl">Avg response length (words)</div>
      </div>
    </div>
  </section>

  <!-- Charts row 1 -->
  <section>
    <div class="grid-2">
      <div class="card">
        <h2>Average Response Length</h2>
        <canvas id="avgLenChart"></canvas>
      </div>
      <div class="card">
        <h2>Turn Count per Session</h2>
        <canvas id="turnChart"></canvas>
      </div>
    </div>
  </section>

  <!-- Charts row 2 -->
  <section>
    <div class="grid-2">
      <div class="card">
        <h2>Answer Length Distribution (all sessions)</h2>
        <canvas id="distChart"></canvas>
      </div>
      <div class="card">
        <h2>Topic Coverage per Session</h2>
        <canvas id="topicChart"></canvas>
      </div>
    </div>
  </section>

  <!-- Session summary table -->
  <section>
    <h2>Session Summary</h2>
    <div class="card">
      <table>
        <thead><tr>
          <th>Session</th><th>AI</th><th>Turns</th>
          <th>Avg A (words)</th><th>Total AI words</th>
          <th>Longest A</th><th>Top Topics</th>
        </tr></thead>
        <tbody>{session_rows_html}</tbody>
      </table>
    </div>
  </section>

  <!-- Turn-level table -->
  <section>
    <h2>Turn-Level Detail</h2>
    <div class="card">
      <table>
        <thead><tr>
          <th>Session</th><th>#</th><th>Question</th>
          <th>Answer (preview)</th><th>Words</th><th>Topics</th>
        </tr></thead>
        <tbody>{turn_rows_html}</tbody>
      </table>
      {more_msg}
    </div>
  </section>

  {errors_html}

</main>
<footer>AI Session Analysis Report &nbsp;·&nbsp; report_builder.py</footer>

<script>
(function() {{
  const avgLen    = {avg_len_json};
  const turnData  = {turn_count_json};
  const distData  = {dist_json};
  const topicData = {topic_json};

  // ── avg response length (bar) ──────────────────────────────────────────────
  new Chart(document.getElementById('avgLenChart'), {{
    type: 'bar',
    data: {{ labels: avgLen.labels, datasets: [{{
      label: 'Avg words', data: avgLen.data,
      backgroundColor: avgLen.colors, borderRadius: 5,
    }}]}},
    options: {{ plugins:{{legend:{{display:false}}}}, scales:{{ y:{{ beginAtZero:true }} }} }}
  }});

  // ── turn count (bar) ───────────────────────────────────────────────────────
  new Chart(document.getElementById('turnChart'), {{
    type: 'bar',
    data: {{ labels: turnData.labels, datasets: [{{
      label: 'Turns', data: turnData.data,
      backgroundColor: turnData.colors, borderRadius: 5,
    }}]}},
    options: {{ plugins:{{legend:{{display:false}}}}, scales:{{ y:{{ beginAtZero:true }} }} }}
  }});

  // ── answer length distribution (bar) ──────────────────────────────────────
  new Chart(document.getElementById('distChart'), {{
    type: 'bar',
    data: {{ labels: distData.labels, datasets: [{{
      label: 'Answers', data: distData.data,
      backgroundColor: '#4e79a7aa', borderRadius: 4,
    }}]}},
    options: {{ plugins:{{legend:{{display:false}}}},
               scales:{{ x:{{title:{{display:true,text:'Word count range'}}}},
                         y:{{beginAtZero:true,title:{{display:true,text:'# answers'}}}} }} }}
  }});

  // ── topic coverage (grouped bar) ──────────────────────────────────────────
  if (topicData.topics.length > 0 && topicData.sessions.length > 0) {{
    const datasets = topicData.sessions.map((name, i) => ({{
      label: name,
      data: topicData.topics.map((_, j) => topicData.matrix[i][j]),
      backgroundColor: topicData.colors[i],
      borderRadius: 3,
    }}));
    new Chart(document.getElementById('topicChart'), {{
      type: 'bar',
      data: {{ labels: topicData.topics, datasets }},
      options: {{
        scales: {{
          x: {{ ticks:{{ maxRotation:35, font:{{size:10}} }} }},
          y: {{ beginAtZero:true, title:{{display:true,text:'mentions'}} }}
        }},
        plugins: {{ legend:{{ position:'bottom', labels:{{boxWidth:12,font:{{size:11}}}} }} }}
      }}
    }});
  }}
}})();
</script>
</body>
</html>"""


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Build an HTML analysis report from AI session logs or CSVs."
    )
    parser.add_argument("files", nargs="*",
                        help="Raw session files (.txt / .docx)")
    parser.add_argument("--csv", action="append", default=[],
                        help="CSV files exported from analyze_ai_sessions.py (repeatable)")
    parser.add_argument("-o", "--output", default="ai_report.html",
                        help="Output HTML file (default: ai_report.html)")
    parser.add_argument("--title", default="AI Session Analysis Report",
                        help="Report title")
    args = parser.parse_args()

    sessions = []

    # Load raw files
    for f in args.files:
        p = Path(f)
        if not p.exists():
            print(f"[warn] Not found: {f}"); continue
        print(f"  Loading {p.name} …")
        sessions.append(load_raw_file(p))

    # Load CSVs
    for c in args.csv:
        p = Path(c)
        if not p.exists():
            print(f"[warn] CSV not found: {c}"); continue
        print(f"  Loading CSV {p.name} …")
        sessions.extend(load_csv(p))

    if not sessions:
        print("No input files provided. Use positional args for .txt/.docx or --csv for CSV files.")
        parser.print_help()
        sys.exit(1)

    print(f"\n  Building report for {len(sessions)} session(s)…")
    html = build_html(sessions, args.title)

    out = Path(args.output)
    out.write_text(html, encoding="utf-8")
    print(f"\n  ✓ Report saved → {out}  ({len(html)//1024} KB)")
    print(f"    Open in any browser: file://{out.resolve()}")


if __name__ == "__main__":
    main()
