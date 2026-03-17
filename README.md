# AI Analysis Toolkit

Scripts for testing and evaluating AI session outputs, translations, and summaries.  
All scripts work on `.txt` and `.docx` files unless noted otherwise.

---

## Installation

```bash
pip install python-docx pymupdf pandas --break-system-packages
```
On Windows omit `--break-system-packages`:
```
python -m pip install python-docx pymupdf pandas
```

For `semantic_similarity.py` only:
```bash
pip install sentence-transformers --break-system-packages
```

---

## Scripts

### 1. `Analyze_all.py`
Analyzes AI chatbot session logs. Extracts questions and answers, calculates per-session statistics (turn count, response length, topics), and compares multiple sessions side by side.

```bash
python3 Analyze_all.py session1.txt session2.docx
python3 Analyze_all.py session1.txt session2.txt -o report.txt --csv data.csv -v
```

| Flag | Description |
|------|-------------|
| `-o FILE` | Save report to file |
| `--csv FILE` | Export turn-level data to CSV |
| `-v` | Print full Q&A turns |

---

### 2. `compare_files.py`
Line-by-line diff of two files. Shows what was added, removed, or changed between them.

```bash
python3 compare_files.py file1.txt file2.docx
python3 compare_files.py file1.pdf file2.txt -m side-by-side
```

| Flag | Description |
|------|-------------|
| `-m` | Output mode: `unified` (default), `side-by-side`, or `html` |
| `-c N` | Number of context lines (default: 3) |
| `-e ENCODING` | Force encoding for .txt files |

---

### 3. `compare_ai_content.py`
Content-level comparison of two AI outputs — not line diffs, but *what* each AI said. Scores specificity, claim density, structure quality, and depth. Topics are auto-extracted from the actual document content rather than a hardcoded keyword list, so it works correctly regardless of domain.

```bash
python3 compare_ai_content.py file1.txt file2.txt
python3 compare_ai_content.py file1.docx file2.txt -o report.txt --csv out.csv -v
python3 compare_ai_content.py file1.txt file2.txt --topics-file nato_topics.json
```

**Scores reported:** Overall · Specificity · Claim Density · Structure Quality · Depth / Uniqueness · Contrast refs · Agreement refs

| Flag | Description |
|------|-------------|
| `-o FILE` | Save report to file |
| `--csv FILE` | Export topic matrix and scores to CSV |
| `-v` | Show detail breakdown and key sentences per shared topic |
| `--topics-file FILE` | JSON file with custom topic taxonomy for domain-specific detection |
| `--no-auto-topics` | Disable auto-topic extraction, use taxonomy only |

**Topics file format (`nato_topics.json`):**
```json
{
  "Doctrine": ["doctrine", "annex", "edition", "nato"],
  "Operations": ["operational", "mission", "command"]
}
```

---

### 4. `consistency_checker.py`
Detects when an AI gives contradictory or inconsistent answers to the same question across multiple sessions. Flags answer drift, yes/no flips, and numeric disagreements.

```bash
python3 consistency_checker.py session1.txt session2.txt session3.txt
python3 consistency_checker.py *.txt --threshold 0.55 -o report.txt --csv out.csv
```

| Flag | Description |
|------|-------------|
| `-t / --threshold` | Similarity threshold for "same question" 0–1 (default: 0.60) |
| `-o FILE` | Save report to file |
| `--csv FILE` | Export flagged pairs to CSV |
| `-v` | Print full answer text for flagged pairs |

---

### 5. `hallucination_detector.py`
Scans AI session logs for assertive factual claims and cross-references them against source documents. Flags unsupported claims, numeric mismatches, missing entities, and contradictions.

```bash
# Without source documents — flags high-confidence claims only
python3 hallucination_detector.py session.txt

# With source documents for cross-referencing
python3 hallucination_detector.py session.txt --sources brief.txt report.docx
python3 hallucination_detector.py s1.txt s2.txt --sources ref.docx -o flags.txt --csv flags.csv
```

| Flag | Description |
|------|-------------|
| `--sources FILE(s)` | Reference documents to check claims against |
| `--min-confidence` | Minimum confidence score to show (default: 0.40) |
| `-o FILE` | Save report to file |
| `--csv FILE` | Export flagged claims to CSV |
| `-v` | Show full context for each flagged claim |

---

### 6. `report_builder.py`
Builds formatted summary reports from session data.

```bash
python3 report_builder.py session1.txt session2.txt -o final_report.txt
```

---

### 7. `translation_evaluator.py`
Evaluates an AI-produced translation against its source document. Checks length ratio, sentence alignment, key term coverage, untranslated segments, number preservation, and named entities.

```bash
python3 translation_evaluator.py source.txt translation.txt
python3 translation_evaluator.py source.txt translation.txt --src-lang fi --tgt-lang en -v
python3 translation_evaluator.py source.docx translation.txt -o report.txt --csv out.csv
```

| Flag | Description |
|------|-------------|
| `--src-lang` | Source language hint: `fi`, `en`, or `auto` (default: auto) |
| `--tgt-lang` | Target language hint: `fi`, `en`, or `auto` (default: auto) |
| `-o FILE` | Save report to file |
| `--csv FILE` | Export findings to CSV |
| `-v` | Print full detail for each finding |

---

### 8. `summary_scorer.py`
Scores an AI-generated summary against its source document across 6 dimensions. Produces a 0–100 score and an A–F grade.

```bash
python3 summary_scorer.py source.txt summary.txt
python3 summary_scorer.py source.docx summary.txt -v --csv scores.csv -o report.txt
```

**Dimensions scored:** Compression Ratio · Keyword Retention · Topic Coverage · Hallucination Signal · Numeric Consistency · Sentence Quality

| Flag | Description |
|------|-------------|
| `-o FILE` | Save report to file |
| `--csv FILE` | Export scores to CSV |
| `-v` | Print full detail for each dimension |

**Grades:** A (88+) · B (75+) · C (60+) · D (45+) · F (<45)

---

### 9. `batch_runner.py`
Runs any of the above scripts across an entire folder of files at once. Collects all results into one report and CSV.

```bash
# Analyze all session files in a folder
python3 batch_runner.py ./sessions --mode analyze

# Run everything at once
python3 batch_runner.py ./sessions --mode all -o report.txt --csv log.csv

# Hallucination check with reference documents
python3 batch_runner.py ./sessions --mode halluc --sources reference.txt

# Translation pairs (files must be named base_source.txt + base_translation.txt)
python3 batch_runner.py ./translations --mode translate --src-suffix _source --tgt-suffix _translation

# Preview commands without running
python3 batch_runner.py ./sessions --mode all --dry-run
```

**Available modes:** `analyze` · `compare` · `halluc` · `consist` · `translate` · `summarize` · `all`

| Flag | Description |
|------|-------------|
| `--mode MODE` | Which script to run (default: analyze) |
| `--ext` | File types to process: `txt`, `docx`, or `both` (default: txt) |
| `--src-suffix` | Filename suffix for source files in translate/summarize (default: `_source`) |
| `--tgt-suffix` | Filename suffix for target files (default: `_translation` or `_summary`) |
| `--sources` | Reference docs forwarded to hallucination_detector |
| `--scripts-dir` | Folder containing the scripts if not in the same directory |
| `--dry-run` | Print commands without executing |
| `-o FILE` | Save aggregated report to file |
| `--csv FILE` | Save run log to CSV |
| `-v` | Pass --verbose to all child scripts |

---

### 10. `translation_benchmark.py`
Benchmarks two AI translations against a confirmed gold-standard translation. Scores each AI translation across 8 dimensions and produces a head-to-head verdict.

**File order always:** `russian` → `gold` → `ai1` → `ai2`

```bash
# Standard usage
python3 translation_benchmark.py russian.pdf gold.pdf ai1.odt ai2.docx

# With labels and output files
python3 translation_benchmark.py russian.pdf gold.pdf ai1.odt ai2.docx \
    --label1 "Gemini" --label2 "Onsite AI" -o report.txt --csv scores.csv -v

# If Russian source is a scanned PDF (no selectable text)
python3 translation_benchmark.py --no-source gold.pdf ai1.odt ai2.docx \
    --label1 "Gemini" --label2 "Onsite AI"
```

**Dimensions scored:** Lexical Similarity · Sentence Alignment · Keyword Retention · Numeric Consistency · Named Entity Match · Length Fidelity · Cyrillic/Untranslated · Sentence Divergence

| Flag | Description |
|------|-------------|
| `--label1 NAME` | Display name for ai1 (default: filename) |
| `--label2 NAME` | Display name for ai2 (default: filename) |
| `--no-source` | Skip Russian source file (use for scanned/image PDFs) |
| `-o FILE` | Save report to file |
| `--csv FILE` | Export dimension scores to CSV |
| `-v` | Print sentence-level divergence examples |

**Grades:** A (88+) · B (75+) · C (60+) · D (45+) · F (<45)

---

### 11. `terminology_checker.py`
Checks that key terms are translated consistently throughout a document. Detects cases where the same source term is rendered differently across occurrences. Supports a custom glossary of expected term pairs and can compare two translations side by side.

```bash
# Auto-detect terms
python3 terminology_checker.py source.pdf translation.txt

# With a glossary and two translations compared
python3 terminology_checker.py source.pdf ai1.docx ai2.txt \
    --label "Onsite AI" --label "Gemini" \
    --glossary "collective defence=collective defence" "armed forces=armed forces" \
    -o report.txt --csv terms.csv -v
```

| Flag | Description |
|------|-------------|
| `--glossary TERM=TRANSLATION` | Expected term pairs, repeatable |
| `--label NAME` | Display name per translation file, repeatable |
| `--min-freq N` | Minimum source frequency to track a term (default: 2) |
| `-o FILE` | Save report to file |
| `--csv FILE` | Export findings to CSV |
| `-v` | Show all detected variants for each flagged term |

---

### 12. `readability_scorer.py`
Scores how natural and fluent AI-generated text reads, independent of accuracy. Useful for catching translations or summaries that are technically correct but awkward to read.

```bash
python3 readability_scorer.py translation.txt
python3 readability_scorer.py ai1.txt ai2.txt --label "Onsite AI" --label "Gemini" -v
python3 readability_scorer.py output.txt --source original.txt -o report.txt --csv scores.csv
```

**Dimensions scored:** Sentence Length Variety · Avg Sentence Length · Word Complexity · Passive Voice Ratio · Repetition · Paragraph Structure · Filler/AI-isms · Punctuation Flow

| Flag | Description |
|------|-------------|
| `--source FILE` | Optional source document for baseline comparison |
| `--label NAME` | Display name per file, repeatable |
| `-o FILE` | Save report to file |
| `--csv FILE` | Export scores to CSV |
| `-v` | Print example sentences for flagged dimensions |

**Grades:** A (88+) · B (75+) · C (60+) · D (45+) · F (<45)

---

### 13. `prompt_response_evaluator.py`
Scores how well AI responses address the questions asked. Works on session log files or individual prompt/response file pairs. Useful for evaluating Q&A session quality alongside the translation tools.

```bash
# Evaluate all turns in a session log
python3 prompt_response_evaluator.py session.txt -v

# Compare two AI sessions
python3 prompt_response_evaluator.py session_ai1.txt session_ai2.txt \
    --label "Onsite AI" --label "Gemini" -o report.txt --csv scores.csv

# Single prompt/response pair from files
python3 prompt_response_evaluator.py --prompt question.txt --response answer.txt
```

**Dimensions scored:** Relevance · Completeness · Conciseness · Directness · Topic Consistency · Question Coverage

| Flag | Description |
|------|-------------|
| `--prompt FILE` | Single prompt file (use with --response) |
| `--response FILE` | Single response file (use with --prompt) |
| `--label NAME` | Display name per session file, repeatable |
| `--min-words N` | Minimum response words to score (default: 10) |
| `-o FILE` | Save report to file |
| `--csv FILE` | Export turn-level scores to CSV |
| `-v` | Print per-turn score table and flag worst turns |

**Grades:** A (88+) · B (75+) · C (60+) · D (45+) · F (<45)

---

### 14. `semantic_similarity.py`
Measures how well an AI output (translation or summary) preserves the *meaning* of its source document using sentence embeddings. Unlike keyword-based tools that count shared words, this script encodes every sentence into a vector and compares by cosine similarity — so semantically equivalent sentences score as near-identical even when they share no words.

> **Requires:** `pip install sentence-transformers --break-system-packages`  
> The model (`paraphrase-multilingual-MiniLM-L12-v2`) downloads ~400 MB on first run and is then cached locally. Handles Finnish/English pairs well.

```bash
python3 semantic_similarity.py source.txt translation.txt
python3 semantic_similarity.py source.docx summary.txt -v
python3 semantic_similarity.py source.txt ai1.txt ai2.txt \
    --label "Onsite AI" --label "Gemini" -o report.txt --csv scores.csv
python3 semantic_similarity.py source.txt translation.txt --mode translation
python3 semantic_similarity.py source.txt summary.txt --mode summary
```

**Dimensions scored:** Overall Semantic Score · Mean Sentence Similarity · Coverage · Depth Alignment · Coherence

| Flag | Description |
|------|-------------|
| `--mode MODE` | Evaluation mode: `translation`, `summary`, or `auto` (default: auto-detect) |
| `--label NAME` | Display name per output file, repeatable |
| `--threshold FLOAT` | Cosine similarity floor for a "good" sentence match (default: 0.75) |
| `-o FILE` | Save report to file |
| `--csv FILE` | Export sentence-level scores to CSV |
| `-v` | Print per-sentence similarity table and flag weak matches |

**Grades:** A (88+) · B (75+) · C (60+) · D (45+) · F (<45)

---

### 15. `tone_register_checker.py`
Checks whether an AI-produced translation or summary preserves the *tone and register* of its source document. Catches cases where a formal institutional document comes back sounding casual, over-hedged, or contaminated with AI filler phrases. Scores the output *relative to the source* — if the source is already informal, an informal output is not penalised.

```bash
python3 tone_register_checker.py source.txt translation.txt
python3 tone_register_checker.py source.docx ai1.txt ai2.txt \
    --label "Onsite AI" --label "Gemini" -o report.txt --csv scores.csv -v
```

**Dimensions scored:** Formality Level · Passive Voice Ratio · Sentence Complexity · Hedging / Uncertainty · AI-ism Contamination · Vocabulary Overlap

| Flag | Description |
|------|-------------|
| `--label NAME` | Display name per output file, repeatable |
| `-o FILE` | Save report to file |
| `--csv FILE` | Export dimension scores to CSV |
| `-v` | Print example sentences for flagged dimensions |

**Grades:** A (88+) · B (75+) · C (60+) · D (45+) · F (<45)

---

### 16. `dashboard_builder.py`
Reads CSV output files from any of the toolkit scripts and builds a single self-contained HTML dashboard with interactive charts. No external dependencies — the output is one `.html` file that works offline and can be shared with non-technical stakeholders by simply sending the file.

Auto-detects the schema of each CSV (translation benchmark, summary scorer, readability, tone register, semantic similarity, prompt response, or generic) and generates the appropriate chart types automatically.

```bash
# Feed any mix of CSV files from the toolkit
python3 dashboard_builder.py scores1.csv scores2.csv -o dashboard.html

# With a custom title
python3 dashboard_builder.py semantic.csv tone.csv \
    --title "AI Eval - Week 12" -o report.html

# Auto-detect all CSVs in a folder
python3 dashboard_builder.py --folder ./results -o dashboard.html
```

**Typical end-to-end workflow:**
```bash
python3 semantic_similarity.py source.txt ai1.txt ai2.txt \
    --label "Onsite AI" --label "Gemini" --csv semantic.csv

python3 tone_register_checker.py source.txt ai1.txt ai2.txt \
    --label "Onsite AI" --label "Gemini" --csv tone.csv

python3 dashboard_builder.py semantic.csv tone.csv \
    --title "Week 12 Evaluation" -o dashboard.html
```

| Flag | Description |
|------|-------------|
| `--title TEXT` | Dashboard title (default: "AI Evaluation Dashboard") |
| `--folder DIR` | Load all `.csv` files from a directory |
| `-o FILE` | Output HTML file (default: `dashboard.html`) |
| `-v` | Print detected CSV types and table summaries |

---

### 17. `spreadsheet_to_sql.py`
Converts CSV and spreadsheet files into SQL statements for database import. Generates CREATE TABLE and INSERT statements compatible with SQLite or PostgreSQL. Useful for loading test data, survey results, or AI evaluation datasets into databases for further analysis.

> **Requires:** `pip install pandas --break-system-packages`

```bash
# Generate SQL file for SQLite
python3 spreadsheet_to_sql.py data.csv --dialect sqlite --output data.sql

# Load directly into a SQLite database
python3 spreadsheet_to_sql.py data.csv --dialect sqlite --db mydatabase.db

# Generate PostgreSQL-compatible SQL
python3 spreadsheet_to_sql.py survey_results.csv --dialect postgres --output import.sql

# Both file output and database load
python3 spreadsheet_to_sql.py data.csv --dialect sqlite --output backup.sql --db live.db
```

**Important:** The CSV file must have a proper header row with column names. If your CSV has an incomplete header (e.g., `,,column_name,,`), the script will auto-generate placeholder names like `unnamed__0`. Fix the header row first for best results.

| Flag | Description |
|------|-------------|
| `--dialect` / `-d` | **Required.** SQL dialect: `sqlite` or `postgres` |
| `--output` | Save SQL statements to a file |
| `--db` | Database file to create/update (SQLite) or connection string (PostgreSQL) |
| `input` | CSV file to convert (positional argument) |

**Note:** You must provide at least one of `--output` or `--db`. Providing both will generate the SQL file *and* execute it against the database.

**Common issues:**
- **Missing headers:** If columns show as `unnamed__N`, your CSV header row is incomplete. Add proper column names to the first row.
- **Empty rows:** The script reads all rows in the CSV. Remove empty rows from your source file before conversion.
- **Encoding:** For non-ASCII characters, ensure your CSV is UTF-8 encoded.

---

## Quick Reference

| Script | Input | Purpose |
|--------|-------|---------|
| `Analyze_all.py` | Session logs | Session stats and topic analysis |
| `compare_files.py` | 2 files | Line-by-line diff |
| `compare_ai_content.py` | 2 AI outputs | Specificity, claim density, structure and depth comparison |
| `consistency_checker.py` | 2+ session logs | Detect contradictory answers |
| `hallucination_detector.py` | Session log + optional sources | Flag unsupported claims |
| `report_builder.py` | Session logs | Build formatted reports |
| `translation_evaluator.py` | Source + translation | Evaluate one translation |
| `summary_scorer.py` | Source + summary | Score an AI summary |
| `batch_runner.py` | Folder of files | Run any script across a whole folder |
| `translation_benchmark.py` | Source + gold + 2 AI translations | Compare two AI translations head-to-head |
| `terminology_checker.py` | Source + 1–2 translations | Check consistent use of key terms |
| `readability_scorer.py` | 1+ AI output files | Score fluency and natural reading flow |
| `prompt_response_evaluator.py` | Session logs or prompt/response pair | Score how well responses address questions |
| `semantic_similarity.py` | Source + 1+ AI outputs | Score meaning preservation using sentence embeddings |
| `tone_register_checker.py` | Source + 1+ AI outputs | Check tone and register consistency against source |
| `dashboard_builder.py` | CSV files from any script | Build an interactive HTML dashboard from results |
| `spreadsheet_to_sql.py` | CSV file | Convert spreadsheet data to SQL statements |
