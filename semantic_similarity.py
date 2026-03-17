#!/usr/bin/env python3
"""
semantic_similarity.py
─────────────────────────────────────────────────────────────────────────────
Measures how well an AI output (translation or summary) preserves the *meaning*
of its source document using sentence embeddings.

Unlike keyword/lexical tools that count shared words, this script encodes every
sentence into a high-dimensional vector and compares them by cosine similarity —
so "The vehicle was damaged" and "The car sustained harm" score as near-identical
even though they share no words.

Requires:
    pip install sentence-transformers python-docx pymupdf --break-system-packages

Usage:
    python3 semantic_similarity.py source.txt translation.txt
    python3 semantic_similarity.py source.docx summary.txt -v
    python3 semantic_similarity.py source.txt ai1.txt ai2.txt \\
        --label "Onsite AI" --label "Gemini" -o report.txt --csv scores.csv
    python3 semantic_similarity.py source.txt translation.txt --mode translation
    python3 semantic_similarity.py source.txt summary.txt    --mode summary

Flags:
    --mode MODE         Evaluation mode: translation, summary, or auto (default: auto)
    --label NAME        Display name per output file, repeatable
    --threshold FLOAT   Cosine similarity floor for "good" sentence match (default: 0.75)
    -o FILE             Save report to file
    --csv FILE          Export sentence-level scores to CSV
    -v                  Print per-sentence similarity table and flag weak matches

Grades: A (88+) · B (75+) · C (60+) · D (45+) · F (<45)

Dimensions scored:
    Overall Semantic Score   Weighted composite of all dimensions
    Mean Sentence Similarity Average cosine similarity across best-matched sentence pairs
    Coverage                 % of source sentences with at least one good match in output
    Depth Alignment          How evenly meaning is distributed (vs. front/back loading)
    Coherence                Internal semantic consistency of the output itself
"""

import argparse
import csv
import sys
import os
import re
import warnings
from pathlib import Path

# ── Suppress noisy warnings from transformers / torch ────────────────────────
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ─────────────────────────────────────────────────────────────────────────────
# Text extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_text(path: str) -> str:
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".docx":
        try:
            from docx import Document
            doc = Document(path)
            return "\n".join(para.text for para in doc.paragraphs)
        except ImportError:
            sys.exit("python-docx not installed. Run: pip install python-docx --break-system-packages")
    elif ext == ".pdf":
        try:
            import fitz
            doc = fitz.open(path)
            return "\n".join(page.get_text() for page in doc)
        except ImportError:
            sys.exit("pymupdf not installed. Run: pip install pymupdf --break-system-packages")
    else:
        for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                return p.read_text(encoding=enc)
            except (UnicodeDecodeError, LookupError):
                continue
        sys.exit(f"Could not decode {path}")


def split_sentences(text: str) -> list:
    text = re.sub(r"\s+", " ", text).strip()
    raw = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in raw if len(s.strip()) > 15]


# ─────────────────────────────────────────────────────────────────────────────
# Embedding helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit(
            "sentence-transformers not installed.\n"
            "Run: pip install sentence-transformers --break-system-packages"
        )
    # paraphrase-multilingual is compact, fast, and handles Finnish/English well
    print("Loading embedding model (first run downloads ~400 MB) …", flush=True)
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def cosine(a, b):
    import numpy as np
    a, b = np.array(a), np.array(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def embed_sentences(model, sentences: list):
    if not sentences:
        return []
    return model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)


# ─────────────────────────────────────────────────────────────────────────────
# Scoring dimensions
# ─────────────────────────────────────────────────────────────────────────────

def score_mean_similarity(src_embs, tgt_embs) -> tuple:
    """
    For each source sentence, find its best-matching target sentence.
    Returns (mean_score_0_100, list_of_(src_idx, best_tgt_idx, sim) tuples).

    This is the core metric: it answers "on average, how well does each source
    sentence have a meaning-equivalent somewhere in the output?"
    """
    import numpy as np
    pairs = []
    for i, se in enumerate(src_embs):
        sims = [cosine(se, te) for te in tgt_embs]
        best_j = int(np.argmax(sims))
        pairs.append((i, best_j, sims[best_j]))
    mean_sim = float(np.mean([p[2] for p in pairs]))
    return round(mean_sim * 100, 1), pairs


def score_coverage(pairs: list, threshold: float) -> float:
    """
    % of source sentences that have at least one target match above threshold.
    Penalises outputs that cover only part of the source meaning.
    """
    covered = sum(1 for _, _, sim in pairs if sim >= threshold)
    return round((covered / len(pairs)) * 100, 1) if pairs else 0.0


def score_depth_alignment(pairs: list, src_len: int, tgt_len: int) -> float:
    """
    Measures whether meaning is spread evenly across the output.
    A perfectly aligned translation maps source positions proportionally to
    target positions. Large jumps = content was reordered or skipped.
    Score is 100 minus the mean absolute deviation of normalised positions.
    """
    import numpy as np
    if src_len == 0 or tgt_len == 0:
        return 0.0
    deviations = []
    for i, j, _ in pairs:
        src_pos = i / src_len
        tgt_pos = j / tgt_len
        deviations.append(abs(src_pos - tgt_pos))
    mean_dev = float(np.mean(deviations))
    return round(max(0.0, (1 - mean_dev * 2)) * 100, 1)


def score_coherence(tgt_embs) -> float:
    """
    Measures internal semantic consistency of the output by computing cosine
    similarity between consecutive sentence pairs. Very low coherence scores
    often indicate machine-translated text that is locally correct but globally
    disconnected, or summaries that jump between unrelated topics.
    """
    import numpy as np
    if len(tgt_embs) < 2:
        return 100.0
    sims = [cosine(tgt_embs[i], tgt_embs[i + 1]) for i in range(len(tgt_embs) - 1)]
    mean_sim = float(np.mean(sims))
    # Typical coherent prose sits around 0.5–0.7 cosine; normalise to 0–100
    normalised = min(1.0, mean_sim / 0.65)
    return round(normalised * 100, 1)


def composite_score(mean_sim: float, coverage: float,
                    depth: float, coherence: float, mode: str) -> float:
    """
    Weighted composite. Translation weights coverage and depth heavily;
    summary weights coverage less (intentional compression is fine).
    """
    if mode == "summary":
        weights = {"mean": 0.45, "coverage": 0.20, "depth": 0.15, "coherence": 0.20}
    else:  # translation or auto
        weights = {"mean": 0.40, "coverage": 0.30, "depth": 0.18, "coherence": 0.12}

    score = (mean_sim * weights["mean"] +
             coverage * weights["coverage"] +
             depth * weights["depth"] +
             coherence * weights["coherence"])
    return round(score, 1)


def grade(score: float) -> str:
    if score >= 88: return "A"
    if score >= 75: return "B"
    if score >= 60: return "C"
    if score >= 45: return "D"
    return "F"


def detect_mode(src_sents: int, tgt_sents: int) -> str:
    ratio = tgt_sents / src_sents if src_sents else 1
    return "summary" if ratio < 0.6 else "translation"


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def format_report(src_path: str, results: list, mode: str) -> str:
    lines = []
    W = 70
    lines.append("=" * W)
    lines.append("SEMANTIC SIMILARITY REPORT")
    lines.append(f"Source : {src_path}")
    lines.append(f"Mode   : {mode}")
    lines.append("=" * W)

    for res in results:
        lines.append("")
        lines.append(f"Output : {res['label']}")
        lines.append("-" * W)
        lines.append(f"  Overall Semantic Score  : {res['overall']:>6.1f} / 100   Grade: {grade(res['overall'])}")
        lines.append(f"  Mean Sentence Similarity: {res['mean_sim']:>6.1f} / 100")
        lines.append(f"  Coverage                : {res['coverage']:>6.1f} / 100")
        lines.append(f"  Depth Alignment         : {res['depth']:>6.1f} / 100")
        lines.append(f"  Coherence               : {res['coherence']:>6.1f} / 100")
        lines.append("")

        # Weak sentences
        weak = [(i, j, s) for i, j, s in res["pairs"] if s < res["threshold"]]
        lines.append(f"  Sentences below threshold ({res['threshold']:.2f}): {len(weak)} / {len(res['pairs'])}")

        if res.get("verbose") and weak:
            lines.append("")
            lines.append("  Weakly-matched source sentences (meaning may be lost):")
            src_sents = res["src_sents"]
            for i, j, s in weak[:10]:
                snippet = src_sents[i][:90] + ("…" if len(src_sents[i]) > 90 else "")
                lines.append(f"    [{s:.2f}] {snippet}")
            if len(weak) > 10:
                lines.append(f"    … and {len(weak) - 10} more.")

    lines.append("")
    if len(results) > 1:
        lines.append("RANKING")
        lines.append("-" * W)
        ranked = sorted(results, key=lambda r: r["overall"], reverse=True)
        for rank, r in enumerate(ranked, 1):
            lines.append(f"  #{rank}  {r['label']:<40} {r['overall']:.1f}  ({grade(r['overall'])})")
    lines.append("")
    lines.append("=" * W)
    return "\n".join(lines)


def write_csv(csv_path: str, src_path: str, results: list):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source_file", "output_file", "src_sentence_idx",
                         "tgt_sentence_idx", "cosine_similarity",
                         "src_sentence_snippet"])
        for res in results:
            src_sents = res["src_sents"]
            for i, j, sim in res["pairs"]:
                snippet = src_sents[i][:80].replace("\n", " ")
                writer.writerow([src_path, res["label"], i, j,
                                 f"{sim:.4f}", snippet])


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Score meaning preservation using sentence embeddings."
    )
    parser.add_argument("source", help="Source document (.txt / .docx / .pdf)")
    parser.add_argument("outputs", nargs="+",
                        help="One or more AI output files to evaluate")
    parser.add_argument("--mode", choices=["translation", "summary", "auto"],
                        default="auto",
                        help="Evaluation mode (default: auto-detect)")
    parser.add_argument("--label", action="append", dest="labels", default=[],
                        help="Display name per output file, repeatable")
    parser.add_argument("--threshold", type=float, default=0.75,
                        help="Cosine similarity floor for 'good' match (default: 0.75)")
    parser.add_argument("-o", "--output", dest="out_file",
                        help="Save report to file")
    parser.add_argument("--csv", dest="csv_file",
                        help="Export sentence-level scores to CSV")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print per-sentence similarity table")
    args = parser.parse_args()

    # Pad labels if fewer were supplied than output files
    labels = list(args.labels)
    for i in range(len(labels), len(args.outputs)):
        labels.append(Path(args.outputs[i]).name)

    # ── Load texts ────────────────────────────────────────────────────────────
    src_text = extract_text(args.source)
    src_sents = split_sentences(src_text)
    if not src_sents:
        sys.exit("Source document yielded no usable sentences.")

    # ── Load model ────────────────────────────────────────────────────────────
    model = load_model()
    src_embs = embed_sentences(model, src_sents)

    # ── Determine mode ────────────────────────────────────────────────────────
    results = []
    for out_path, label in zip(args.outputs, labels):
        tgt_text = extract_text(out_path)
        tgt_sents = split_sentences(tgt_text)
        if not tgt_sents:
            print(f"WARNING: {out_path} yielded no usable sentences, skipping.")
            continue

        tgt_embs = embed_sentences(model, tgt_sents)

        mode = args.mode
        if mode == "auto":
            mode = detect_mode(len(src_sents), len(tgt_sents))

        mean_sim, pairs = score_mean_similarity(src_embs, tgt_embs)
        coverage       = score_coverage(pairs, args.threshold)
        depth          = score_depth_alignment(pairs, len(src_sents), len(tgt_sents))
        coherence      = score_coherence(tgt_embs)
        overall        = composite_score(mean_sim, coverage, depth, coherence, mode)

        results.append({
            "label":     label,
            "mean_sim":  mean_sim,
            "coverage":  coverage,
            "depth":     depth,
            "coherence": coherence,
            "overall":   overall,
            "pairs":     pairs,
            "src_sents": src_sents,
            "threshold": args.threshold,
            "verbose":   args.verbose,
        })

    if not results:
        sys.exit("No output files could be processed.")

    # Use last detected mode for report header
    mode = args.mode if args.mode != "auto" else detect_mode(
        len(src_sents), len(split_sentences(extract_text(args.outputs[-1]))))

    report = format_report(args.source, results, mode)
    print(report)

    if args.out_file:
        Path(args.out_file).write_text(report, encoding="utf-8")
        print(f"Report saved to {args.out_file}")

    if args.csv_file:
        write_csv(args.csv_file, args.source, results)
        print(f"CSV saved to {args.csv_file}")


if __name__ == "__main__":
    main()
