#!/usr/bin/env python3
"""
results_aggregator.py
---------------------
Combines CSVs from all analysis scripts into one master comparison table.
One row per AI model, columns for every metric across all task types.

Reads:
  --translation FILE    CSV from translation_benchmark.py
  --summary FILE        CSV from summary_scorer.py
  --comparison FILE     CSV from compare_ai_content.py
  --readability FILE    CSV from readability_scorer.py (repeatable for multiple tasks)

Produces:
  - Terminal leaderboard ranked by overall average
  - Single master CSV with one row per model

Usage:
    python3 results_aggregator.py \\
        --translation NSS_benchmark.csv \\
        --summary NatoFI_summary_scores.csv \\
        --comparison TrumpBiden_comparison_scores.csv \\
        --readability readability_translations.csv \\
        --readability readability_summaries.csv \\
        --readability readability_comparisons_trumpbiden.csv \\
        --csv master_results.csv

Options:
    --translation FILE    Translation benchmark CSV (translation_benchmark.py)
    --summary FILE        Summary scorer CSV (summary_scorer.py)
    --comparison FILE     Content comparison CSV (compare_ai_content.py)
    --readability FILE    Readability scorer CSV — repeatable (readability_scorer.py)
    --csv FILE            Output master CSV (default: master_results.csv)
    -v, --verbose         Print detailed per-model breakdown
"""

import re
import sys
import csv
import argparse
from pathlib import Path
from collections import defaultdict

# ── colours ───────────────────────────────────────────────────────────────────
C = {
    "green":   "\033[92m", "red":     "\033[91m", "yellow": "\033[93m",
    "blue":    "\033[94m", "cyan":    "\033[96m", "bold":   "\033[1m",
    "magenta": "\033[95m", "reset":   "\033[0m",
}
def clr(t, c): return f"{C[c]}{t}{C['reset']}"
DIV  = "=" * 72
DIV2 = "-" * 72

# ── helpers ───────────────────────────────────────────────────────────────────
def read_csv(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def safe_float(val) -> float:
    """Convert a value to float safely, return None if not possible."""
    if val is None or str(val).strip() == "":
        return None
    try:
        return float(str(val).replace("%","").strip())
    except (ValueError, TypeError):
        return None

def score_color(s: float) -> str:
    if s is None: return "reset"
    return "green" if s >= 75 else "yellow" if s >= 50 else "red"

def grade_color(g: str) -> str:
    return {"A":"green","B":"green","C":"yellow","D":"yellow","F":"red"}.get(g,"reset")

def bar(score: float, width: int = 16, max_val: float = 100) -> str:
    if score is None: return "░" * width
    filled = round(score / max_val * width)
    filled = max(0, min(filled, width))
    return "█" * filled + "░" * (width - filled)

# ── parsers for each CSV format ───────────────────────────────────────────────

def parse_translation_csv(path: Path) -> dict:
    """
    translation_benchmark.py CSV:
    - Rows are dimensions, columns are model scores
    - Special rows: OVERALL, Word Count
    - Column format: {label}_score, {label}_verdict
    """
    rows = read_csv(path)
    if not rows:
        return {}

    # Find model labels from column names (anything ending in _score)
    score_cols = [k for k in rows[0].keys() if k.endswith("_score")]
    labels = [c[:-6] for c in score_cols]  # strip "_score"

    results = {}
    for label in labels:
        results[label] = {"task": "translation", "source_csv": path.name}

    for row in rows:
        dim = row.get("dimension", "").strip()
        for label in labels:
            val = safe_float(row.get(f"{label}_score"))
            if dim == "OVERALL":
                results[label]["translation_overall"] = val
                verdict = row.get(f"{label}_verdict", "")
                grade = verdict.replace("Grade ", "").strip() if "Grade" in verdict else ""
                results[label]["translation_grade"] = grade
            elif dim == "Word Count":
                results[label]["translation_word_count"] = val
            elif dim and val is not None:
                key = f"trans_{dim.lower().replace(' ','_').replace('/','')}"
                results[label][key] = val

    return results


def parse_summary_csv(path: Path) -> dict:
    """
    summary_scorer.py CSV:
    - One row per model
    - Columns: summary_file, src_words, sum_words, compression,
               overall, grade, + dimension columns
    """
    rows = read_csv(path)
    results = {}
    for row in rows:
        raw = row.get("summary_file", "unknown")
        # Strip extension if present
        label = Path(raw).stem if '.' in raw else raw
        entry = {
            "task": "summary",
            "source_csv": path.name,
            "summary_overall": safe_float(row.get("overall")),
            "summary_grade":   row.get("grade", ""),
            "summary_words":   safe_float(row.get("sum_words")),
            "summary_compression": row.get("compression", ""),
        }
        # Dimension scores
        dim_keys = [k for k in row.keys()
                    if k not in ("summary_file","src_words","sum_words",
                                 "compression","overall","grade",
                                 "best_dimension","worst_dimension")]
        for k in dim_keys:
            val = safe_float(row.get(k))
            if val is not None:
                entry[f"summ_{k}"] = val
        results[label] = entry
    return results


def parse_comparison_csv(path: Path) -> dict:
    """
    compare_ai_content.py CSV:
    - One row per model
    - Columns: file, overall_score, specificity_score, claim_score,
               structure_score, depth_score, + topic columns
    """
    rows = read_csv(path)
    results = {}
    for row in rows:
        label = Path(row.get("file", "unknown")).stem
        entry = {
            "task": "comparison",
            "source_csv": path.name,
            "comparison_overall":     safe_float(row.get("overall_score")),
            "comparison_specificity": safe_float(row.get("specificity_score")),
            "comparison_claims":      safe_float(row.get("claim_score")),
            "comparison_structure":   safe_float(row.get("structure_score")),
            "comparison_depth":       safe_float(row.get("depth_score")),
            "comparison_words":       safe_float(row.get("total_words")),
            "comparison_topics":      safe_float(row.get("topics_covered")),
        }
        results[label] = entry
    return results


def parse_readability_csv(path: Path) -> dict:
    """
    readability_scorer.py CSV:
    - One row per dimension per model
    - Columns: label, file, overall, grade, dimension, score, verdict, raw, detail
    """
    rows = read_csv(path)
    results = {}

    # Group by label/file
    for row in rows:
        label = row.get("label") or Path(row.get("file","unknown")).stem
        if label not in results:
            results[label] = {
                "task": "readability",
                "source_csv": path.name,
                "readability_overall": safe_float(row.get("overall")),
                "readability_grade":   row.get("grade", ""),
            }
        dim  = row.get("dimension", "").strip()
        val  = safe_float(row.get("score"))
        if dim and val is not None:
            key = f"read_{dim.lower().replace(' ','_').replace('/','')}"
            results[label][key] = val

    return results


# ── merge all results ─────────────────────────────────────────────────────────
def merge_results(all_dicts: list) -> dict:
    """
    Merge multiple {label: data} dicts into one master dict.
    Uses fuzzy label matching to handle minor naming differences.
    """
    master = {}

    def normalise(label: str) -> str:
        """Lowercase, remove common suffixes/prefixes for matching."""
        l = label.lower().strip()
        for pat in [r'\d+$', r'_\d+$', r'käännös\d*$', r'tiivistelmä\d*$',
                    r'vertailu$', r'01$', r'02$']:
            l = re.sub(pat, '', l).strip('_- ')
        return l

    for d in all_dicts:
        for label, data in d.items():
            # Try exact match first
            if label in master:
                master[label].update(data)
                continue
            # Try normalised match
            norm = normalise(label)
            matched = None
            for existing in master:
                if normalise(existing) == norm:
                    matched = existing
                    break
            if matched:
                master[matched].update(data)
            else:
                master[label] = dict(data)

    return master


# ── calculate overall average ─────────────────────────────────────────────────
def calc_master_overall(entry: dict) -> float:
    """Average of available task-level overall scores, normalised to 0-100."""
    scores = []
    # Translation overall (0-100)
    v = safe_float(entry.get("translation_overall"))
    if v is not None: scores.append(v)
    # Summary overall (0-100)
    v = safe_float(entry.get("summary_overall"))
    if v is not None: scores.append(v)
    # Comparison overall (0-10 → normalise to 0-100)
    v = safe_float(entry.get("comparison_overall"))
    if v is not None: scores.append(v * 10)
    # Readability excluded from master score

    return round(sum(scores) / len(scores), 1) if scores else 0.0


# ── display ───────────────────────────────────────────────────────────────────
def task_coverage(entry: dict) -> int:
    """Count how many tasks have data for this model."""
    count = 0
    if safe_float(entry.get("translation_overall")) is not None: count += 1
    if safe_float(entry.get("summary_overall"))     is not None: count += 1
    if safe_float(entry.get("comparison_overall"))  is not None: count += 1
    # Readability excluded from task coverage count
    return count

def print_model_row(label, entry, col, verbose):
    overall   = calc_master_overall(entry)
    trans     = safe_float(entry.get("translation_overall"))
    summ      = safe_float(entry.get("summary_overall"))
    comp      = safe_float(entry.get("comparison_overall"))
    comp_norm = comp * 10 if comp is not None else None

    def fmt(v):
        if v is None: return clr(f"{'N/A':>{col}}", "blue")
        return clr(f"{v:>{col}.0f}", score_color(v))

    print(f"  {label[:24]:<26}  "
          f"{clr(bar(overall), score_color(overall))} "
          f"{clr(f'{overall:>5.1f}', score_color(overall))}  "
          f"{fmt(trans)}  {fmt(summ)}  {fmt(comp_norm)}")

    if verbose:
        tasks_with_data = task_coverage(entry)
        if tasks_with_data < 3:
            print(f"    {clr(f'⚠  Only {tasks_with_data}/4 tasks — limited confidence', 'yellow')}")
        if trans is not None:
            tg = entry.get("translation_grade","")
            print(f"    {'Translation':<20} Grade: {clr(tg, grade_color(tg))}  "
                  f"Words: {entry.get('translation_word_count','?')}")
        if summ is not None:
            sg = entry.get("summary_grade","")
            print(f"    {'Summary':<20} Grade: {clr(sg, grade_color(sg))}  "
                  f"Compression: {entry.get('summary_compression','?')}")
        if comp is not None:
            print(f"    {'Comparison':<20} "
                  f"Specificity: {entry.get('comparison_specificity','?')}  "
                  f"Depth: {entry.get('comparison_depth','?')}")
        # Readability excluded

def print_leaderboard(master: dict, verbose: bool):
    col = 10
    header = (f"  {'Model':<26}  {'Overall':>{col}}  {'Transl':>{col}}  "
              f"{'Summary':>{col}}  {'Compare':>{col}}")

    tier1 = {l: e for l, e in master.items() if task_coverage(e) == 3}
    tier2 = {l: e for l, e in master.items() if task_coverage(e) == 2}
    tier3 = {l: e for l, e in master.items() if task_coverage(e) == 1}

    ranked1 = sorted(tier1.items(), key=lambda x: -calc_master_overall(x[1]))
    ranked2 = sorted(tier2.items(), key=lambda x: -calc_master_overall(x[1]))
    ranked3 = sorted(tier3.items(), key=lambda x: -calc_master_overall(x[1]))

    # ── Tier 1 ────────────────────────────────────────────────────────────────
    print(f"\n{clr(DIV, 'bold')}")
    print(clr("  TIER 1 — FULL RANKING  (3 tasks)", "bold"))
    print(DIV)
    if ranked1:
        print(header)
        print("  " + "-" * 72)
        for label, entry in ranked1:
            print_model_row(label, entry, col, verbose)
        print(f"\n  {clr('▶  BEST OVERALL:', 'bold')} "
              f"{clr(ranked1[0][0], 'green')} "
              f"({calc_master_overall(ranked1[0][1]):.1f}/100)")
    else:
        print(f"  {clr('No models with 3+ tasks.', 'yellow')}")

    # ── Tier 2 ────────────────────────────────────────────────────────────────
    if ranked2:
        print(f"\n{clr(DIV, 'bold')}")
        print(clr("  TIER 2 — LIMITED DATA  (2 tasks)  ⚠ treat with caution", "yellow"))
        print(DIV)
        print(header)
        print("  " + "-" * 72)
        for label, entry in ranked2:
            print_model_row(label, entry, col, verbose)

    # ── Tier 3 ────────────────────────────────────────────────────────────────
    if ranked3:
        print(f"\n{clr(DIV, 'bold')}")
        print(clr("  TIER 3 — INSUFFICIENT DATA  (1 task)  ✗ not comparable", "red"))
        print(DIV)
        print(header)
        print("  " + "-" * 72)
        for label, entry in ranked3:
            print_model_row(label, entry, col, verbose)
        print(f"\n  {clr('Run more tasks on these models to include them in ranking.', 'yellow')}")

    # ── Task winners (Tier 1 only) ────────────────────────────────────────────
    print(f"\n{clr(DIV, 'bold')}")
    print(clr("  TASK WINNERS  (Tier 1 only)", "bold"))
    print(DIV2)
    for task, key in [("Translation","translation_overall"),
                      ("Summary","summary_overall"),
                      ("Comparison","comparison_overall")]:
        scores = [(safe_float(e.get(key)), l) for l, e in tier1.items()
                  if safe_float(e.get(key)) is not None]
        if not scores: continue
        scores.sort(reverse=True)
        best_v, best_l   = scores[0]
        worst_v, worst_l = scores[-1]
        spread = round(best_v - worst_v, 1) if len(scores) > 1 else 0
        print(f"  {task:<16} Best: {clr(best_l[:22], 'green'):<30} "
              f"Worst: {clr(worst_l[:22], 'red'):<30} Spread: {spread}")
    print()

# ── CSV export ────────────────────────────────────────────────────────────────
def export_master_csv(master: dict, out: Path):
    """
    One row per model, all metrics as columns.
    Sorted by master overall score descending.
    """
    # Collect all unique keys across all models
    all_keys = set()
    for entry in master.values():
        all_keys.update(entry.keys())

    # Define column order
    priority = [
        "master_overall",
        "translation_overall", "translation_grade", "translation_word_count",
        "trans_lexical_similarity", "trans_sentence_alignment",
        "trans_keyword_retention", "trans_numeric_consistency",
        "trans_named_entity_match", "trans_length_fidelity",
        "trans_cyrillicuntranslated", "trans_sentence_divergence",
        "summary_overall", "summary_grade", "summary_words", "summary_compression",
        "summ_compression_ratio", "summ_keyword_retention", "summ_topic_coverage",
        "summ_hallucination_signal", "summ_numeric_consistency", "summ_sentence_quality",
        "comparison_overall", "comparison_specificity", "comparison_claims",
        "comparison_structure", "comparison_depth", "comparison_words", "comparison_topics",
        # Readability columns excluded
    ]
    # Add any remaining keys not in priority list
    remaining = sorted(k for k in all_keys
                       if k not in priority and k not in ("task","source_csv"))
    fieldnames = ["model"] + priority + remaining

    rows = []
    for label, entry in master.items():
        row = {"model": label, "master_overall": calc_master_overall(entry)}
        for k in priority + remaining:
            row[k] = entry.get(k, "")
        rows.append(row)

    rows.sort(key=lambda x: -(safe_float(x.get("master_overall")) or 0))

    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(clr(f"Master CSV exported → {out}", "green"))
    print(clr(f"  {len(rows)} models × {len(fieldnames)-1} metrics", "blue"))


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Aggregate all analysis CSVs into one master comparison table."
    )
    parser.add_argument("--translation",  default=None,
                        help="CSV from translation_benchmark.py")
    parser.add_argument("--summary",      default=None,
                        help="CSV from summary_scorer.py")
    parser.add_argument("--comparison",   default=None,
                        help="CSV from compare_ai_content.py")
    # --readability removed — readability excluded from aggregation
    parser.add_argument("--csv",          default="master_results.csv",
                        help="Output master CSV (default: master_results.csv)")
    parser.add_argument("--names-file",   default=None, dest="names_file",
                        help='JSON file mapping canonical model names to aliases '
                             '(format: {"ModelName": ["alias1", "alias2"]})')
    parser.add_argument("-v","--verbose", action="store_true",
                        help="Print per-model task breakdown")
    args = parser.parse_args()

    if not any([args.translation, args.summary,
                args.comparison]):
        print(clr("Error: provide at least one input CSV.", "red"))
        parser.print_help(); sys.exit(1)

    all_dicts = []

    if args.translation:
        p = Path(args.translation)
        if not p.exists():
            print(clr(f"Warning: not found — {p}", "yellow"))
        else:
            d = parse_translation_csv(p)
            print(clr(f"Translation CSV loaded: {len(d)} models", "cyan"))
            all_dicts.append(d)

    if args.summary:
        p = Path(args.summary)
        if not p.exists():
            print(clr(f"Warning: not found — {p}", "yellow"))
        else:
            d = parse_summary_csv(p)
            print(clr(f"Summary CSV loaded: {len(d)} models", "cyan"))
            all_dicts.append(d)

    if args.comparison:
        p = Path(args.comparison)
        if not p.exists():
            print(clr(f"Warning: not found — {p}", "yellow"))
        else:
            d = parse_comparison_csv(p)
            print(clr(f"Comparison CSV loaded: {len(d)} models", "cyan"))
            all_dicts.append(d)

    # Readability loading removed

    if not all_dicts:
        print(clr("No valid CSVs loaded.", "red")); sys.exit(1)

    # ── Load and apply name mapping ───────────────────────────────────────────
    name_map = {}  # lowercase alias -> canonical name
    if args.names_file:
        import json
        p = Path(args.names_file)
        if not p.exists():
            print(clr(f"Warning: names file not found — {p}", "yellow"))
        else:
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
                for canonical, aliases in raw.items():
                    for alias in aliases:
                        name_map[alias.lower().strip()] = canonical
                print(clr(f"Name mapping: {len(raw)} canonical models, "
                          f"{len(name_map)} aliases loaded", "cyan"))
            except Exception as e:
                print(clr(f"Warning: could not load names file — {e}", "yellow"))

    if name_map:
        mapped = []
        merged_count = 0
        for d in all_dicts:
            new_d = {}
            for label, data in d.items():
                canonical = name_map.get(label.lower().strip(), label)
                if canonical != label:
                    merged_count += 1
                if canonical in new_d:
                    new_d[canonical].update(data)
                else:
                    new_d[canonical] = dict(data)
            mapped.append(new_d)
        all_dicts = mapped
        print(clr(f"  {merged_count} label(s) resolved to canonical names", "cyan"))

    print(clr(f"\nMerging results…", "bold"))
    master = merge_results(all_dicts)
    print(f"  {len(master)} unique models found across all CSVs")

    print_leaderboard(master, args.verbose)
    export_master_csv(master, Path(args.csv))


if __name__ == "__main__":
    main()