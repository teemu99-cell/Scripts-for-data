#!/usr/bin/env python3
"""
dashboard_builder.py
─────────────────────────────────────────────────────────────────────────────
Reads CSV output files from any of the toolkit scripts and builds a single
self-contained HTML dashboard with interactive charts.

The dashboard is a single .html file with no external dependencies — all
JavaScript (Chart.js) is embedded inline, so it works offline and can be
shared with non-technical stakeholders by just sending the file.

Requires:
    No extra packages — uses only the Python standard library.

Usage:
    # Feed any mix of CSV files from the toolkit
    python3 dashboard_builder.py scores1.csv scores2.csv -o dashboard.html

    # With a title
    python3 dashboard_builder.py *.csv -o report.html --title "AI Eval - Week 12"

    # Auto-detect all CSVs in a folder
    python3 dashboard_builder.py --folder ./results -o dashboard.html

Flags:
    -o / --output FILE      Output HTML file (default: dashboard.html)
    --title TEXT            Dashboard title (default: "AI Evaluation Dashboard")
    --folder DIR            Load all .csv files from a directory
    -v                      Print detected CSV types and table summaries

Supported CSV schemas (auto-detected):
    translation_benchmark   label1/label2, all dimension columns
    summary_scorer          source_file, output_file, overall, grade, dimensions
    readability_scorer      file / label, overall, grade, dimensions
    tone_register_checker   source_file, output_file, overall, grade, dimensions
    semantic_similarity     source_file, output_file, cosine_similarity (sentence-level)
    prompt_response         session_file / label, overall, grade, dimensions
    generic                 any CSV with an "overall" or "score" column
"""

import argparse
import csv
import json
import sys
import os
from pathlib import Path
from collections import defaultdict
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# CSV loading & schema detection
# ─────────────────────────────────────────────────────────────────────────────

DIMENSION_COLS = {
    # translation_benchmark
    "lexical_similarity", "sentence_alignment", "keyword_retention",
    "numeric_consistency", "named_entity_match", "length_fidelity",
    "sentence_divergence",
    # summary_scorer
    "compression_ratio", "keyword_retention", "topic_coverage",
    "hallucination_signal", "sentence_quality",
    # readability_scorer
    "sentence_length_variety", "avg_sentence_length", "word_complexity",
    "passive_voice_ratio", "repetition", "paragraph_structure",
    "filler_ai_isms", "punctuation_flow",
    # tone_register_checker
    "formality", "passive_voice", "sentence_complexity",
    "hedging", "ai_isms", "vocabulary_overlap",
    # prompt_response_evaluator
    "relevance", "completeness", "conciseness", "directness",
    "topic_consistency", "question_coverage",
    # compare_ai_content
    "specificity", "claim_density", "structure_quality", "depth",
}

SCORE_LIKE = {"overall", "score", "total", "grade"}


def read_csv(path: str) -> list:
    rows = []
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with open(path, newline="", encoding=enc) as f:
                rows = list(csv.DictReader(f))
            break
        except (UnicodeDecodeError, LookupError):
            continue
    return rows


def detect_schema(rows: list, headers: list) -> str:
    h = {c.lower().strip() for c in headers}
    if "cosine_similarity" in h:
        return "semantic_similarity"
    if "label1" in h or "label2" in h:
        return "translation_benchmark"
    if "src_sentence_idx" in h:
        return "semantic_similarity"
    if "relevance" in h and "completeness" in h:
        return "prompt_response"
    if "formality" in h and "hedging" in h:
        return "tone_register"
    if "compression_ratio" in h or "hallucination_signal" in h:
        return "summary_scorer"
    if "sentence_length_variety" in h or "word_complexity" in h:
        return "readability"
    if "overall" in h or "score" in h:
        return "generic"
    return "unknown"


def safe_float(val) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def extract_score_series(rows: list, schema: str, headers: list) -> dict:
    """
    Returns a dict:
        {
          "labels":     [str],            # x-axis labels (file names / model names)
          "overall":    [float],           # overall scores
          "dimensions": {dim_name: [float]}  # per-dimension scores
        }
    """
    h_lower = {c.lower().strip(): c for c in headers}
    out = {"labels": [], "overall": [], "dimensions": defaultdict(list)}

    if schema == "semantic_similarity":
        # Sentence-level CSV — aggregate by output file
        by_file = defaultdict(list)
        for row in rows:
            key = row.get("output_file", row.get("label", "?"))
            sim = safe_float(row.get("cosine_similarity", 0))
            by_file[key].append(sim)
        for fname, sims in by_file.items():
            out["labels"].append(Path(fname).name)
            out["overall"].append(round(sum(sims) / len(sims) * 100, 1))
        return out

    label_col = next(
        (h_lower[c] for c in ("label", "output_file", "file", "label1", "session_file")
         if c in h_lower), None
    )
    overall_col = next(
        (h_lower[c] for c in ("overall", "score", "total") if c in h_lower), None
    )
    dim_cols = [c for c in headers
                if c.lower().strip() in DIMENSION_COLS and c.lower().strip() != "overall"]

    seen = set()
    for row in rows:
        label = Path(row.get(label_col, "?")).name if label_col else "?"
        if label in seen:
            continue
        seen.add(label)
        out["labels"].append(label)
        out["overall"].append(safe_float(row.get(overall_col, 0)) if overall_col else 0)
        for dc in dim_cols:
            out["dimensions"][dc].append(safe_float(row.get(dc, 0)))

    return out


# ─────────────────────────────────────────────────────────────────────────────
# HTML generation
# ─────────────────────────────────────────────────────────────────────────────

CHART_COLORS = [
    "rgba(99,  179, 237, 0.85)",
    "rgba(154, 117, 234, 0.85)",
    "rgba(72,  187, 120, 0.85)",
    "rgba(246, 173,  85, 0.85)",
    "rgba(252, 129, 129, 0.85)",
    "rgba(129, 230, 217, 0.85)",
]

BORDER_COLORS = [c.replace("0.85", "1") for c in CHART_COLORS]


def make_bar_chart_js(canvas_id: str, labels: list, datasets: list,
                      title: str, y_max: int = 100) -> str:
    return f"""
new Chart(document.getElementById('{canvas_id}'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(labels)},
    datasets: {json.dumps(datasets)}
  }},
  options: {{
    responsive: true,
    plugins: {{
      legend: {{ display: {str(len(datasets) > 1).lower()} }},
      title: {{ display: true, text: {json.dumps(title)}, font: {{ size: 14 }} }}
    }},
    scales: {{
      y: {{ min: 0, max: {y_max}, ticks: {{ stepSize: 20 }} }},
      x: {{ ticks: {{ maxRotation: 35, minRotation: 20 }} }}
    }}
  }}
}});
"""


def make_radar_chart_js(canvas_id: str, labels: list, datasets: list, title: str) -> str:
    return f"""
new Chart(document.getElementById('{canvas_id}'), {{
  type: 'radar',
  data: {{
    labels: {json.dumps(labels)},
    datasets: {json.dumps(datasets)}
  }},
  options: {{
    responsive: true,
    plugins: {{
      legend: {{ display: {str(len(datasets) > 1).lower()} }},
      title: {{ display: true, text: {json.dumps(title)}, font: {{ size: 14 }} }}
    }},
    scales: {{
      r: {{ min: 0, max: 100, ticks: {{ stepSize: 20, backdropColor: 'transparent' }} }}
    }}
  }}
}});
"""


def grade(score: float) -> str:
    if score >= 88: return "A"
    if score >= 75: return "B"
    if score >= 60: return "C"
    if score >= 45: return "D"
    return "F"


def grade_color(g: str) -> str:
    return {"A": "#38a169", "B": "#3182ce", "C": "#d69e2e",
            "D": "#dd6b20", "F": "#e53e3e"}.get(g, "#718096")


def build_summary_cards(all_series: list) -> str:
    cards = []
    for series in all_series:
        for i, label in enumerate(series["labels"]):
            ov = series["overall"][i] if i < len(series["overall"]) else 0
            g  = grade(ov)
            gc = grade_color(g)
            cards.append(f"""
      <div class="card">
        <div class="card-label">{label}</div>
        <div class="card-score">{ov:.1f}</div>
        <div class="card-grade" style="color:{gc}">{g}</div>
        <div class="card-source">{series['source_name']}</div>
      </div>""")
    return "\n".join(cards)


def build_html(title: str, panels: list, summary_cards_html: str,
               generated: str) -> str:
    scripts = "\n".join(p["script"] for p in panels)
    canvases = ""
    for p in panels:
        canvases += f"""
    <div class="panel">
      <canvas id="{p['canvas_id']}"></canvas>
    </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  *, *::before, *::after {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f0f4f8;
    color: #2d3748;
    margin: 0;
    padding: 0 0 40px;
  }}
  header {{
    background: linear-gradient(135deg, #2b4c7e 0%, #567ebb 100%);
    color: white;
    padding: 28px 40px 22px;
  }}
  header h1 {{ margin: 0 0 4px; font-size: 1.7rem; font-weight: 700; }}
  header p  {{ margin: 0; opacity: 0.8; font-size: 0.85rem; }}
  .cards {{
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
    padding: 28px 40px 0;
  }}
  .card {{
    background: white;
    border-radius: 10px;
    padding: 16px 20px;
    min-width: 140px;
    box-shadow: 0 1px 4px rgba(0,0,0,.09);
    text-align: center;
  }}
  .card-label  {{ font-size: 0.72rem; color: #718096; word-break: break-all; margin-bottom: 4px; }}
  .card-score  {{ font-size: 2rem;   font-weight: 700; line-height: 1; }}
  .card-grade  {{ font-size: 1.2rem; font-weight: 600; margin-top: 2px; }}
  .card-source {{ font-size: 0.65rem; color: #a0aec0; margin-top: 6px; }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(460px, 1fr));
    gap: 20px;
    padding: 24px 40px 0;
  }}
  .panel {{
    background: white;
    border-radius: 10px;
    padding: 20px 24px;
    box-shadow: 0 1px 4px rgba(0,0,0,.09);
  }}
  .panel canvas {{ width: 100% !important; }}
  footer {{
    text-align: center;
    color: #a0aec0;
    font-size: 0.75rem;
    margin-top: 32px;
  }}
</style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <p>Generated {generated} · AI Evaluation Toolkit</p>
</header>

<div class="cards">
{summary_cards_html}
</div>

<div class="grid">
{canvases}
</div>

<footer>Built with dashboard_builder.py</footer>

<script>
{scripts}
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build an HTML dashboard from toolkit CSV output files."
    )
    parser.add_argument("csvfiles", nargs="*",
                        help="CSV files from any toolkit script")
    parser.add_argument("--folder", dest="folder",
                        help="Load all .csv files from a directory")
    parser.add_argument("-o", "--output", dest="out_file",
                        default="dashboard.html",
                        help="Output HTML file (default: dashboard.html)")
    parser.add_argument("--title", default="AI Evaluation Dashboard",
                        help="Dashboard title")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print detected CSV types and summaries")
    args = parser.parse_args()

    # Collect CSV paths
    csv_paths = list(args.csvfiles)
    if args.folder:
        folder = Path(args.folder)
        if not folder.is_dir():
            sys.exit(f"Folder not found: {args.folder}")
        csv_paths += [str(p) for p in sorted(folder.glob("*.csv"))]

    if not csv_paths:
        sys.exit("No CSV files provided. Use positional args or --folder.")

    # ── Process each CSV ──────────────────────────────────────────────────────
    panels = []
    all_series = []
    canvas_counter = [0]

    def next_canvas():
        canvas_counter[0] += 1
        return f"chart_{canvas_counter[0]}"

    for csv_path in csv_paths:
        if not Path(csv_path).exists():
            print(f"WARNING: {csv_path} not found, skipping.")
            continue

        rows = read_csv(csv_path)
        if not rows:
            print(f"WARNING: {csv_path} is empty, skipping.")
            continue

        headers = list(rows[0].keys())
        schema  = detect_schema(rows, headers)
        source_name = Path(csv_path).stem

        if args.verbose:
            print(f"  {csv_path} → schema: {schema}, rows: {len(rows)}")

        series = extract_score_series(rows, schema, headers)
        series["source_name"] = source_name
        all_series.append(series)

        labels  = series["labels"]
        overall = series["overall"]
        dims    = series["dimensions"]

        if not labels:
            continue

        # Overall scores bar chart
        cid = next_canvas()
        color_idx = (canvas_counter[0] - 1) % len(CHART_COLORS)
        datasets = [{
            "label":           "Overall Score",
            "data":            overall,
            "backgroundColor": CHART_COLORS[color_idx],
            "borderColor":     BORDER_COLORS[color_idx],
            "borderWidth":     1,
        }]
        panels.append({
            "canvas_id": cid,
            "script":    make_bar_chart_js(cid, labels, datasets,
                                           f"{source_name} — Overall Scores"),
        })

        # Dimension radar chart (if dimensions exist and ≤2 output files)
        if dims and len(labels) <= 4:
            dim_names = [d.replace("_", " ").title() for d in dims]
            radar_datasets = []
            for i, label in enumerate(labels):
                vals = [dims[d][i] if i < len(dims[d]) else 0
                        for d in dims]
                ci = i % len(CHART_COLORS)
                radar_datasets.append({
                    "label":           label,
                    "data":            vals,
                    "backgroundColor": CHART_COLORS[ci].replace("0.85", "0.25"),
                    "borderColor":     BORDER_COLORS[ci],
                    "borderWidth":     2,
                    "pointBackgroundColor": BORDER_COLORS[ci],
                })
            cid2 = next_canvas()
            panels.append({
                "canvas_id": cid2,
                "script":    make_radar_chart_js(
                    cid2, dim_names, radar_datasets,
                    f"{source_name} — Dimensions Breakdown"),
            })

        # Dimension grouped bar chart (if >4 outputs, radar gets crowded)
        if dims and len(labels) > 4:
            dim_names  = list(dims.keys())
            dim_labels = [d.replace("_", " ").title() for d in dim_names]
            bar_datasets = []
            for i, label in enumerate(labels[:6]):   # cap at 6 for readability
                ci = i % len(CHART_COLORS)
                bar_datasets.append({
                    "label":           label,
                    "data":            [dims[d][i] if i < len(dims[d]) else 0
                                        for d in dim_names],
                    "backgroundColor": CHART_COLORS[ci],
                    "borderColor":     BORDER_COLORS[ci],
                    "borderWidth":     1,
                })
            cid3 = next_canvas()
            panels.append({
                "canvas_id": cid3,
                "script":    make_bar_chart_js(
                    cid3, dim_labels, bar_datasets,
                    f"{source_name} — Dimensions by Output"),
            })

    if not panels:
        sys.exit("No usable data found in the provided CSV files.")

    # ── Aggregate overall scores across ALL CSVs ──────────────────────────────
    agg_labels, agg_scores = [], []
    for s in all_series:
        for label, ov in zip(s["labels"], s["overall"]):
            key = f"{label} ({s['source_name']})"
            agg_labels.append(key)
            agg_scores.append(ov)

    if len(agg_labels) > 1:
        cid_agg = next_canvas()
        agg_colors = [CHART_COLORS[i % len(CHART_COLORS)] for i in range(len(agg_labels))]
        agg_dataset = [{
            "label":           "Overall Score",
            "data":            agg_scores,
            "backgroundColor": agg_colors,
            "borderColor":     [c.replace("0.85", "1") for c in agg_colors],
            "borderWidth":     1,
        }]
        panels.insert(0, {
            "canvas_id": cid_agg,
            "script":    make_bar_chart_js(cid_agg, agg_labels, agg_dataset,
                                           "All Outputs — Overall Scores"),
        })

    # ── Build and write HTML ──────────────────────────────────────────────────
    cards_html = build_summary_cards(all_series)
    generated  = datetime.now().strftime("%Y-%m-%d %H:%M")
    html       = build_html(args.title, panels, cards_html, generated)

    out_path = Path(args.out_file)
    out_path.write_text(html, encoding="utf-8")
    print(f"Dashboard written to {out_path}  ({len(panels)} charts, "
          f"{sum(len(s['labels']) for s in all_series)} output files)")


if __name__ == "__main__":
    main()
