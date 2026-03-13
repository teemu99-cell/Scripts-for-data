#!/usr/bin/env python3
"""
batch_runner.py
---------------
Orchestrates running your AI analysis toolkit across a folder of .txt / .docx
files and collects all results into a single aggregated CSV and/or report.

Supported script modes
  analyze    → runs analyze_all.py        on every session file
  compare    → runs compare_files.py      on every pair of files in the folder
  halluc     → runs hallucination_detector.py on every session file
  consist    → runs consistency_checker.py   across all session files (single run)
  translate  → runs translation_evaluator.py on SOURCE→TRANSLATION pairs
               (files must be named *_source.txt + *_translation.txt or similar —
                see --src-suffix / --tgt-suffix flags)
  summarize  → runs summary_scorer.py    on SOURCE→SUMMARY pairs
               (files must be named *_source.txt + *_summary.txt or similar)
  all        → runs analyze + halluc + consist in sequence

Usage:
    # Analyze every .txt session file in a folder
    python3 batch_runner.py ./sessions --mode analyze

    # Check hallucinations in all files, collect one combined CSV
    python3 batch_runner.py ./sessions --mode halluc --csv results/halluc.csv

    # Evaluate translation pairs (source_*.txt  →  translation_*.txt)
    python3 batch_runner.py ./translations --mode translate \\
            --src-suffix _source --tgt-suffix _translation

    # Run everything and save report
    python3 batch_runner.py ./sessions --mode all -o batch_report.txt

    # Dry-run — show what would run, don't execute
    python3 batch_runner.py ./sessions --mode analyze --dry-run

Options:
    --mode MODE           Which script(s) to run (default: analyze)
    --ext EXT             File extension to process: txt | docx | both (default: txt)
    --scripts-dir DIR     Folder containing the analysis scripts (default: same folder
                          as batch_runner.py)
    --src-suffix SUFFIX   Filename suffix identifying source files for translate/summarize
                          (default: _source)
    --tgt-suffix SUFFIX   Filename suffix identifying target files for translate/summarize
                          (default: _translation  or  _summary  depending on mode)
    --sources FILE [..]   Reference documents forwarded to hallucination_detector
    --csv FILE            Path for aggregated CSV output
    -o, --output FILE     Path for aggregated text report
    -v, --verbose         Pass --verbose to every child script
    --dry-run             Print commands without executing them
    --python PYTHON       Python interpreter to use (default: python3)

No extra dependencies beyond the standard library.
"""

import sys
import csv
import argparse
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field

# ── colours ───────────────────────────────────────────────────────────────────
C = {
    "green":   "\033[92m", "red":     "\033[91m", "yellow": "\033[93m",
    "blue":    "\033[94m", "cyan":    "\033[96m", "bold":   "\033[1m",
    "magenta": "\033[95m", "reset":   "\033[0m",
}
def clr(t, c): return f"{C[c]}{t}{C['reset']}"
DIV  = "=" * 70
DIV2 = "-" * 70

# ── data model ────────────────────────────────────────────────────────────────
@dataclass
class RunResult:
    mode:       str
    files:      list      # list of Path objects passed to the script
    script:     str
    returncode: int
    stdout:     str
    stderr:     str
    duration_s: float

    @property
    def ok(self) -> bool:
        return self.returncode == 0

# ── script registry ───────────────────────────────────────────────────────────
SCRIPT_MAP = {
    "analyze":   "Analyze_all.py",
    "compare":   "compare_files.py",
    "halluc":    "hallucination_detector.py",
    "consist":   "consistency_checker.py",
    "translate": "translation_evaluator.py",
    "summarize": "summary_scorer.py",
}

# ── helpers ───────────────────────────────────────────────────────────────────
def find_script(name: str, scripts_dir: Path) -> Optional_Path := None:
    """Locate a script by name, checking scripts_dir then PATH."""
    candidate = scripts_dir / name
    if candidate.exists():
        return candidate
    found = shutil.which(name)
    return Path(found) if found else None

# The above syntax trick doesn't work in Python — define the function normally:
def find_script(name: str, scripts_dir: Path):
    candidate = scripts_dir / name
    if candidate.exists():
        return candidate
    found = shutil.which(name)
    return Path(found) if found else None

def collect_files(folder: Path, ext: str) -> list:
    exts = [".txt", ".docx"] if ext == "both" else [f".{ext}"]
    files = []
    for e in exts:
        files.extend(sorted(folder.glob(f"*{e}")))
    return files

def pair_files(files: list, src_suffix: str, tgt_suffix: str) -> list:
    """
    Match source files to target files by stripping their suffix and
    re-attaching the other. Returns [(src_path, tgt_path), ...].
    """
    pairs = []
    src_files = [f for f in files
                 if f.stem.endswith(src_suffix) or src_suffix in f.stem]
    for src in src_files:
        # Try to find the matching target
        stem_base = src.stem
        if stem_base.endswith(src_suffix):
            stem_base = stem_base[: -len(src_suffix)]
        # Try same extension first, then other extension
        for ext in (src.suffix, ".txt", ".docx"):
            tgt = src.parent / f"{stem_base}{tgt_suffix}{ext}"
            if tgt.exists():
                pairs.append((src, tgt))
                break
    return pairs

def run_command(cmd: list, dry_run: bool) -> RunResult:
    import time
    script_name = Path(cmd[1]).name if len(cmd) > 1 else "?"
    file_args   = [Path(a).name for a in cmd[2:] if not a.startswith("-")]

    if dry_run:
        print(f"  {clr('[DRY RUN]', 'yellow')}  {' '.join(str(c) for c in cmd)}")
        return RunResult(
            mode="dry-run", files=[], script=script_name,
            returncode=0, stdout="", stderr="", duration_s=0.0
        )

    t0 = time.time()
    try:
        proc = subprocess.run(
            [str(c) for c in cmd],
            capture_output=True, text=True, encoding="utf-8"
        )
        duration = time.time() - t0
        return RunResult(
            mode       = script_name,
            files      = file_args,
            script     = script_name,
            returncode = proc.returncode,
            stdout     = proc.stdout,
            stderr     = proc.stderr,
            duration_s = round(duration, 2),
        )
    except Exception as e:
        duration = time.time() - t0
        return RunResult(
            mode="error", files=file_args, script=script_name,
            returncode=-1, stdout="", stderr=str(e), duration_s=round(duration,2)
        )

# ── mode runners ──────────────────────────────────────────────────────────────
def run_per_file(script_path: Path, files: list, python: str,
                 extra_args: list, dry_run: bool) -> list:
    """Run the script once per file."""
    results = []
    for f in files:
        cmd = [python, str(script_path), str(f)] + extra_args
        r = run_command(cmd, dry_run)
        results.append(r)
        status = clr("✓", "green") if r.ok else clr("✗", "red")
        print(f"  {status}  {f.name:<40}  {r.duration_s:.1f}s")
        if not r.ok and r.stderr.strip():
            print(f"      {clr(r.stderr.strip()[:120], 'red')}")
    return results

def run_per_pair(script_path: Path, pairs: list, python: str,
                 extra_args: list, dry_run: bool) -> list:
    """Run the script once per source↔target pair."""
    results = []
    for src, tgt in pairs:
        cmd = [python, str(script_path), str(src), str(tgt)] + extra_args
        r = run_command(cmd, dry_run)
        results.append(r)
        status = clr("✓", "green") if r.ok else clr("✗", "red")
        print(f"  {status}  {src.name}  →  {tgt.name}  ({r.duration_s:.1f}s)")
        if not r.ok and r.stderr.strip():
            print(f"      {clr(r.stderr.strip()[:120], 'red')}")
    return results

def run_all_files_once(script_path: Path, files: list, python: str,
                       extra_args: list, dry_run: bool) -> list:
    """Run the script once with all files as arguments (e.g. consistency_checker)."""
    cmd = [python, str(script_path)] + [str(f) for f in files] + extra_args
    r = run_command(cmd, dry_run)
    status = clr("✓", "green") if r.ok else clr("✗", "red")
    names = ", ".join(f.name for f in files[:4])
    if len(files) > 4:
        names += f" (+{len(files)-4} more)"
    print(f"  {status}  [{names}]  ({r.duration_s:.1f}s)")
    if not r.ok and r.stderr.strip():
        print(f"      {clr(r.stderr.strip()[:120], 'red')}")
    return [r]

def run_pairs_from_folder(script_path: Path, files: list, python: str,
                          extra_args: list, dry_run: bool) -> list:
    """Run compare_files.py on every consecutive pair in the folder."""
    results = []
    file_list = list(files)
    if len(file_list) < 2:
        print(clr("  Need at least 2 files for comparison mode.", "yellow"))
        return results
    for i in range(len(file_list) - 1):
        f1, f2 = file_list[i], file_list[i + 1]
        cmd = [python, str(script_path), str(f1), str(f2)] + extra_args
        r = run_command(cmd, dry_run)
        results.append(r)
        status = clr("✓", "green") if r.ok else clr("✗", "red")
        print(f"  {status}  {f1.name}  ↔  {f2.name}  ({r.duration_s:.1f}s)")
        if not r.ok and r.stderr.strip():
            print(f"      {clr(r.stderr.strip()[:120], 'red')}")
    return results

# ── aggregated report & CSV ───────────────────────────────────────────────────
def build_report(all_results: list, modes: list, folder: Path,
                 started: str) -> str:
    lines = [
        DIV,
        "BATCH RUNNER — AGGREGATED REPORT",
        DIV, "",
        f"Folder   : {folder}",
        f"Modes    : {', '.join(modes)}",
        f"Started  : {started}",
        f"Total    : {len(all_results)} run(s)",
        f"Passed   : {sum(1 for r in all_results if r.ok)}",
        f"Failed   : {sum(1 for r in all_results if not r.ok)}",
        "",
    ]
    for r in all_results:
        fnames = ", ".join(str(f) for f in r.files) if r.files else "(batch)"
        lines += [
            f"Script : {r.script}",
            f"Files  : {fnames}",
            f"Status : {'PASS' if r.ok else 'FAIL'}  ({r.duration_s}s)",
        ]
        if r.stdout.strip():
            for line in r.stdout.strip().splitlines()[:30]:
                lines.append(f"  {line}")
        if r.stderr.strip() and not r.ok:
            lines.append(f"  STDERR: {r.stderr.strip()[:200]}")
        lines.append("")
    return "\n".join(lines)

def export_csv(all_results: list, out: Path):
    rows = [{
        "script":     r.script,
        "files":      "; ".join(str(f) for f in r.files),
        "status":     "pass" if r.ok else "fail",
        "returncode": r.returncode,
        "duration_s": r.duration_s,
        "stdout_lines": r.stdout.count("\n"),
        "stderr_preview": r.stderr.strip()[:200],
    } for r in all_results]

    if not rows:
        print(clr("No results to export.", "yellow")); return
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(clr(f"\nAggregated CSV → {out}", "green"))

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Batch-run the AI analysis toolkit across a folder of files."
    )
    parser.add_argument("folder",
                        help="Folder containing .txt / .docx session files")
    parser.add_argument("--mode", default="analyze",
                        choices=["analyze","compare","halluc","consist",
                                 "translate","summarize","all"],
                        help="Which analysis to run (default: analyze)")
    parser.add_argument("--ext", default="txt",
                        choices=["txt","docx","both"],
                        help="File extension to process (default: txt)")
    parser.add_argument("--scripts-dir", default=None,
                        help="Folder containing the analysis scripts "
                             "(default: same directory as batch_runner.py)")
    parser.add_argument("--src-suffix", default="_source",
                        help="Stem suffix for source files in translate/summarize "
                             "(default: _source)")
    parser.add_argument("--tgt-suffix", default=None,
                        help="Stem suffix for target files (default: _translation "
                             "for translate, _summary for summarize)")
    parser.add_argument("--sources", nargs="*", default=[],
                        help="Reference docs forwarded to hallucination_detector")
    parser.add_argument("-o", "--output",   help="Save aggregated report to file")
    parser.add_argument("--csv",            help="Save aggregated run log to CSV")
    parser.add_argument("-v", "--verbose",  action="store_true",
                        help="Pass --verbose to child scripts")
    parser.add_argument("--dry-run",        action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--python",         default="python3",
                        help="Python interpreter (default: python3)")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(clr(f"Error: '{folder}' is not a directory.", "red")); sys.exit(1)

    scripts_dir = Path(args.scripts_dir) if args.scripts_dir \
                  else Path(__file__).parent.resolve()

    started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(clr(f"\n{DIV}", "bold"))
    print(clr("  BATCH RUNNER", "bold"))
    print(DIV)
    print(f"  Folder      : {folder.resolve()}")
    print(f"  Mode        : {clr(args.mode, 'cyan')}")
    print(f"  Scripts dir : {scripts_dir}")
    print(f"  Started     : {started}")
    print(DIV2)

    files = collect_files(folder, args.ext)
    if not files:
        print(clr(f"No .{args.ext} files found in {folder}", "yellow")); sys.exit(0)
    print(f"  Found {len(files)} file(s): {', '.join(f.name for f in files[:6])}"
          + (" ..." if len(files) > 6 else ""))
    print()

    modes_to_run = (["analyze","halluc","consist"]
                    if args.mode == "all" else [args.mode])

    verbose_flag = ["--verbose"] if args.verbose else []
    all_results  = []

    for mode in modes_to_run:
        script_name = SCRIPT_MAP[mode]
        script_path = find_script(script_name, scripts_dir)

        if script_path is None:
            print(clr(f"  [SKIP] {script_name} not found in {scripts_dir}", "yellow"))
            continue

        print(clr(f"── {mode.upper()}  ({script_name})", "bold"))

        if mode == "analyze":
            results = run_per_file(script_path, files, args.python,
                                   verbose_flag, args.dry_run)

        elif mode == "compare":
            results = run_pairs_from_folder(script_path, files, args.python,
                                            verbose_flag, args.dry_run)

        elif mode == "halluc":
            src_extra = []
            for s in args.sources:
                if Path(s).exists():
                    src_extra += ["--sources", s]
                else:
                    print(clr(f"  [warn] source not found: {s}", "yellow"))
            results = run_per_file(script_path, files, args.python,
                                   src_extra + verbose_flag, args.dry_run)

        elif mode == "consist":
            if len(files) < 2:
                print(clr("  Need ≥2 files for consistency check.", "yellow"))
                results = []
            else:
                results = run_all_files_once(script_path, files, args.python,
                                             verbose_flag, args.dry_run)

        elif mode in ("translate", "summarize"):
            tgt_suffix = args.tgt_suffix or (
                "_translation" if mode == "translate" else "_summary"
            )
            pairs = pair_files(files, args.src_suffix, tgt_suffix)
            if not pairs:
                print(clr(
                    f"  No matching pairs found. "
                    f"Expected files ending with '{args.src_suffix}' and '{tgt_suffix}'.",
                    "yellow"
                ))
                results = []
            else:
                results = run_per_pair(script_path, pairs, args.python,
                                       verbose_flag, args.dry_run)

        else:
            results = []

        all_results.extend(results)
        print()

    # ── summary ───────────────────────────────────────────────────────────────
    passed = sum(1 for r in all_results if r.ok)
    failed = sum(1 for r in all_results if not r.ok)
    total  = len(all_results)
    total_time = sum(r.duration_s for r in all_results)

    print(clr(DIV, "bold"))
    print(clr("  BATCH SUMMARY", "bold"))
    print(DIV)
    print(f"  Total runs : {total}")
    print(f"  Passed     : {clr(str(passed), 'green')}")
    print(f"  Failed     : {clr(str(failed), 'red') if failed else clr('0', 'green')}")
    print(f"  Total time : {total_time:.1f}s")

    if all_results and not args.dry_run:
        print(f"\n  {clr('Stdout preview from last run:', 'cyan')}")
        for line in all_results[-1].stdout.splitlines()[-12:]:
            print(f"    {line}")

    if args.output:
        report = build_report(all_results, modes_to_run, folder, started)
        Path(args.output).write_text(report, encoding="utf-8")
        print(clr(f"\nReport saved → {args.output}", "green"))

    if args.csv:
        export_csv(all_results, Path(args.csv))

    sys.exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    main()
