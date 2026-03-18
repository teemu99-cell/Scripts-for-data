"""
spreadsheet_to_sql.py
─────────────────────
Converts .csv / .xlsx files into SQL (PostgreSQL or SQLite).

Features:
  - Auto-detects column data types (INTEGER, FLOAT, TIMESTAMP, TEXT fallback)
  - Infers PRIMARY KEY from the first fully-unique, non-null column
  - Infers CHECK constraints on low-cardinality TEXT columns (enums/categoricals)
  - Detects constant columns and extracts them into a separate _constants table
  - Multi-sheet .xlsx support — each sheet becomes its own table
  - Outputs a .sql file, writes directly to a DB, or both
  - Supports both PostgreSQL and SQLite dialects

Usage examples:
  # Generate a .sql file (SQLite dialect)
  python spreadsheet_to_sql.py data.xlsx --dialect sqlite --output out.sql

  # Generate a .sql file (PostgreSQL dialect)
  python spreadsheet_to_sql.py data.csv --dialect postgres --output out.sql

  # Write directly to a SQLite database file
  python spreadsheet_to_sql.py data.xlsx --dialect sqlite --db mydb.sqlite3

  # Write directly to a PostgreSQL database
  python spreadsheet_to_sql.py data.xlsx --dialect postgres --db "postgresql://user:pass@localhost:5432/mydb"

  # Do both (generate .sql AND write to DB)
  python spreadsheet_to_sql.py data.xlsx --dialect sqlite --output out.sql --db mydb.sqlite3

Dependencies:
  pip install pandas openpyxl psycopg2-binary numpy
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# 1. UTILITIES
# ─────────────────────────────────────────────

def sanitize_name(name: str) -> str:
    """
    Convert any string into a safe SQL identifier.
    e.g. "First Name!" -> "first_name_"
         "123abc"      -> "_123abc"
    """
    name = str(name).strip().lower()
    name = re.sub(r'\s+', '_', name)
    name = re.sub(r'[^a-z0-9_]', '_', name)
    if name and name[0].isdigit():
        name = '_' + name
    return name or 'unnamed'


def to_python(v):
    """
    Convert a value to a native Python type safe for DB drivers.
    Numpy scalars (np.int64, np.float64, etc.) are not guaranteed
    to be accepted by sqlite3 or psycopg2, so we convert explicitly.
    """
    if pd.isna(v):
        return None
    if hasattr(v, 'item'):  # catches any numpy scalar
        return v.item()
    return v


# ─────────────────────────────────────────────
# 2. TYPE INFERENCE & SCHEMA ANALYSIS
# ─────────────────────────────────────────────

def infer_sql_type(series: pd.Series, dialect: str) -> str:
    """
    Inspect a pandas Series and return the best-fit SQL type.

    Detection order:
      1. INTEGER   - all non-null values are whole numbers within safe int range
      2. FLOAT     - all non-null values are numeric (but not all whole)
      3. TIMESTAMP - all non-null values parse as dates/datetimes
      4. TEXT      - fallback for everything else

    Dialect differences:
      - SQLite  uses REAL instead of DOUBLE PRECISION
      - Postgres uses DOUBLE PRECISION for floats, TIMESTAMP for datetimes
      - SQLite has no native TIMESTAMP; dates stored as ISO-8601 TEXT
    """
    col = series.dropna()

    if col.empty:
        return 'TEXT'

    # --- Try INTEGER ---
    try:
        converted = pd.to_numeric(col, errors='raise')
        # Use modulo to check whole numbers instead of unsafe float == int cast.
        # Also guard against values outside the safe integer range for float64
        # (beyond 2**53, floats lose integer precision and can silently corrupt).
        if np.isfinite(converted).all() and (converted % 1 == 0).all():
            if converted.abs().max() < 2**53:
                return 'INTEGER'
        return 'REAL' if dialect == 'sqlite' else 'DOUBLE PRECISION'
    except (ValueError, TypeError):
        pass

    # --- Try TIMESTAMP ---
    # Note: infer_datetime_format was removed in pandas 2.0, so it is omitted.
    try:
        pd.to_datetime(col, errors='raise')
        return 'TEXT' if dialect == 'sqlite' else 'TIMESTAMP'
    except (ValueError, TypeError):
        pass

    # --- Fallback: TEXT ---
    return 'TEXT'


def infer_primary_key(df: pd.DataFrame, clean_cols: list) -> str | None:
    """
    Returns the sanitized name of the best PRIMARY KEY candidate, or None.

    Only checks the first column — PK columns are almost always placed first
    by convention. A column qualifies if it has no NULLs and no duplicates.

    This is a heuristic: it can produce false positives (a column that is
    coincidentally unique) or false negatives (a dirty ID column with one
    duplicate). Always review the output SQL before running it.
    """
    if not clean_cols:
        return None

    first_col_orig = df.columns[0]
    series = df[first_col_orig]

    has_no_nulls = series.notna().all()
    is_unique    = series.nunique() == len(series)

    if has_no_nulls and is_unique:
        return clean_cols[0]

    return None


def infer_check_constraint(series: pd.Series) -> str | None:
    """
    If a TEXT column looks like an enum/categorical, return a CHECK IN(...)
    constraint string. Returns None if the column looks like free text.

    A column is treated as categorical if:
      - It has 10 or fewer distinct non-null values, AND
      - Fewer than 50% of rows are unique (rules out coincidentally short text)

    Example return value: "IN ('Air', 'Land', 'Sea', 'Unknown')"
    """
    col = series.dropna()
    if col.empty:
        return None

    unique_vals  = col.unique()
    unique_count = len(unique_vals)

    if unique_count <= 10 and (unique_count / len(col)) < 0.5:
        # Escape any single quotes inside the values themselves
        quoted = ', '.join(
            f"'{str(v).replace(chr(39), chr(39) + chr(39))}'"
            for v in sorted(unique_vals)
        )
        return f"IN ({quoted})"

    return None


def detect_constant_columns(df: pd.DataFrame, clean_cols: list) -> list:
    """
    Returns a list of (clean_col_name, orig_col_name, constant_value) tuples
    for every column where all non-null values are identical.

    These columns carry no per-row information and are better stored in a
    separate single-row constants table to avoid repeating the same value
    across every row (a normalization violation).
    """
    constants = []
    for clean, orig in zip(clean_cols, df.columns):
        series = df[orig].dropna()
        if not series.empty and series.nunique() == 1:
            constants.append((clean, orig, series.iloc[0]))
    return constants


# ─────────────────────────────────────────────
# 3. FILE LOADING
# ─────────────────────────────────────────────

def load_file(filepath: str) -> dict:
    """
    Load a .csv or .xlsx file.
    Returns a dict mapping table_name -> DataFrame.
    """
    path = Path(filepath)

    # Provide a clean error instead of a confusing pandas/IO crash.
    if not path.exists():
        sys.exit(f"ERROR: Input file not found: '{filepath}'")

    ext = path.suffix.lower()
    tables = {}

    if ext == '.csv':
        df = pd.read_csv(filepath)
        table_name = sanitize_name(path.stem)
        tables[table_name] = df
        print(f"  + Loaded CSV  -> table '{table_name}' ({len(df)} rows, {len(df.columns)} cols)")

    elif ext in ('.xlsx', '.xls'):
        xls = pd.ExcelFile(filepath)
        seen_names = {}
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            table_name = sanitize_name(sheet_name)

            # Two sheets can sanitize to the same name (e.g. "My Sheet" and "My-Sheet"
            # both become "my_sheet"). Deduplicate with a numeric suffix.
            if table_name in seen_names:
                seen_names[table_name] += 1
                table_name = f"{table_name}_{seen_names[table_name]}"
            else:
                seen_names[table_name] = 0

            tables[table_name] = df
            print(f"  + Loaded sheet '{sheet_name}' -> table '{table_name}' ({len(df)} rows, {len(df.columns)} cols)")

    else:
        sys.exit(f"ERROR: Unsupported file type: '{ext}'. Use .csv or .xlsx")

    return tables


# ─────────────────────────────────────────────
# 4. SQL GENERATION
# ─────────────────────────────────────────────

def escape_value(val, col_type: str, dialect: str) -> str:
    """Format a single Python value as a SQL literal."""
    if pd.isna(val):
        return 'NULL'
    if col_type in ('TEXT', 'TIMESTAMP'):
        escaped = str(val).replace("'", "''")
        return f"'{escaped}'"
    if col_type == 'INTEGER':
        return str(int(val))
    if col_type in ('REAL', 'DOUBLE PRECISION'):
        return str(float(val))
    # Raise explicitly so unknown types surface immediately rather than
    # silently falling through to a potentially wrong float conversion.
    raise ValueError(f"Unknown col_type: '{col_type}'")


def generate_sql_for_table(table_name: str, df: pd.DataFrame, dialect: str) -> str:
    """
    Build DROP + CREATE + INSERT SQL for one table.

    Also handles:
      - PRIMARY KEY on the first unique/non-null column
      - CHECK constraints on low-cardinality TEXT columns
      - Constant column extraction into a separate _constants table
    """
    lines      = []
    clean_cols = [sanitize_name(c) for c in df.columns]
    orig_cols  = list(df.columns)
    col_types  = {
        clean: infer_sql_type(df[orig], dialect)
        for clean, orig in zip(clean_cols, orig_cols)
    }

    # --- Schema analysis ---
    pk_col             = infer_primary_key(df, clean_cols)
    constant_cols      = detect_constant_columns(df, clean_cols)
    constant_col_names = {c[0] for c in constant_cols}

    if pk_col:
        print(f"    ~ PRIMARY KEY inferred: '{pk_col}'")
    for col, _, _ in constant_cols:
        print(f"    ~ Constant column detected: '{col}' -> extracting to '{table_name}_constants'")

    # ── Constants table ──────────────────────────────────────────────────────
    if constant_cols:
        const_table = f"{table_name}_constants"
        lines.append(
            f'-- Constant columns from "{table_name}" extracted here.\n'
            f'-- Every row in "{table_name}" shared the same value for these columns,\n'
            f'-- so storing them once here avoids repeating them {len(df)} times.'
        )
        lines.append(f'DROP TABLE IF EXISTS "{const_table}";')

        const_defs = ',\n    '.join(
            f'"{col}" {col_types[col]}'
            for col, _, _ in constant_cols
        )
        lines.append(f'CREATE TABLE "{const_table}" (\n    {const_defs}\n);')

        const_col_list = ', '.join(f'"{col}"' for col, _, _ in constant_cols)
        const_vals     = ', '.join(
            escape_value(val, col_types[col], dialect)
            for col, _, val in constant_cols
        )
        lines.append(f'INSERT INTO "{const_table}" ({const_col_list}) VALUES ({const_vals});')

    # ── Main table ───────────────────────────────────────────────────────────
    lines.append(f'DROP TABLE IF EXISTS "{table_name}";')

    col_def_parts = []
    for clean, orig in zip(clean_cols, orig_cols):
        if clean in constant_col_names:
            continue  # already in constants table

        type_str  = col_types[clean]
        def_parts = [f'"{clean}"', type_str]

        if clean == pk_col:
            def_parts.append('PRIMARY KEY')
        elif type_str == 'TEXT':
            check = infer_check_constraint(df[orig])
            if check:
                def_parts.append(f'CHECK("{clean}" {check})')

        col_def_parts.append(' '.join(def_parts))

    col_defs = ',\n    '.join(col_def_parts)
    # No IF NOT EXISTS — the DROP above guarantees the table is gone.
    lines.append(f'CREATE TABLE "{table_name}" (\n    {col_defs}\n);')

    # Only insert non-constant columns
    active_cols = [
        (clean, orig)
        for clean, orig in zip(clean_cols, orig_cols)
        if clean not in constant_col_names
    ]
    col_list = ', '.join(f'"{c}"' for c, _ in active_cols)

    if dialect == 'postgres':
        value_rows = []
        for _, row in df.iterrows():
            vals = ', '.join(
                escape_value(row[orig], col_types[clean], dialect)
                for clean, orig in active_cols
            )
            value_rows.append(f'    ({vals})')
        if value_rows:
            lines.append(
                f'INSERT INTO "{table_name}" ({col_list}) VALUES\n'
                + ',\n'.join(value_rows) + ';'
            )
    else:
        for _, row in df.iterrows():
            vals = ', '.join(
                escape_value(row[orig], col_types[clean], dialect)
                for clean, orig in active_cols
            )
            lines.append(f'INSERT INTO "{table_name}" ({col_list}) VALUES ({vals});')

    return '\n\n'.join(lines)


def generate_full_sql(tables: dict, dialect: str) -> str:
    """Combine SQL for all tables into one string."""
    header = f"-- Generated by spreadsheet_to_sql.py\n-- Dialect: {dialect.upper()}\n"
    blocks = [generate_sql_for_table(name, df, dialect) for name, df in tables.items()]
    separator = '\n\n-- ─────────────────────────────────────────\n\n'
    return header + '\n\n' + separator.join(blocks) + '\n'


# ─────────────────────────────────────────────
# 5. OUTPUT WRITERS
# ─────────────────────────────────────────────

def write_sql_file(sql: str, output_path: str):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(sql)
    print(f"  + SQL file written -> {output_path}")


def _write_to_db(tables: dict, conn, cursor, placeholder: str, dialect: str):
    """
    Shared insert logic for both SQLite and PostgreSQL.

    Applies the same schema analysis as the SQL generator:
      - PRIMARY KEY on first unique/non-null column
      - CHECK constraints on low-cardinality TEXT columns
      - Constant columns extracted into a separate _constants table

    Uses executemany() instead of a per-row execute() loop for better
    performance on large datasets.

    The caller wraps this in 'with conn' so the entire operation is a
    single transaction — any failure rolls back all changes automatically.
    """
    for table_name, df in tables.items():
        clean_cols = [sanitize_name(c) for c in df.columns]
        orig_cols  = list(df.columns)
        col_types  = {
            clean: infer_sql_type(df[orig], dialect)
            for clean, orig in zip(clean_cols, orig_cols)
        }

        # --- Schema analysis ---
        pk_col             = infer_primary_key(df, clean_cols)
        constant_cols      = detect_constant_columns(df, clean_cols)
        constant_col_names = {c[0] for c in constant_cols}

        # ── Constants table ──────────────────────────────────────────────────
        if constant_cols:
            const_table = f"{table_name}_constants"
            cursor.execute(f'DROP TABLE IF EXISTS "{const_table}"')
            const_defs = ', '.join(
                f'"{col}" {col_types[col]}'
                for col, _, _ in constant_cols
            )
            cursor.execute(f'CREATE TABLE "{const_table}" ({const_defs})')

            const_col_list     = ', '.join(f'"{col}"' for col, _, _ in constant_cols)
            const_placeholders = ', '.join([placeholder] * len(constant_cols))
            const_vals         = tuple(to_python(val) for _, _, val in constant_cols)
            cursor.execute(
                f'INSERT INTO "{const_table}" ({const_col_list}) '
                f'VALUES ({const_placeholders})',
                const_vals
            )
            print(f"  + Written constants table -> '{const_table}'")

        # ── Main table ───────────────────────────────────────────────────────
        col_def_parts = []
        for clean, orig in zip(clean_cols, orig_cols):
            if clean in constant_col_names:
                continue

            type_str  = col_types[clean]
            def_parts = [f'"{clean}"', type_str]

            if clean == pk_col:
                def_parts.append('PRIMARY KEY')
            elif type_str == 'TEXT':
                check = infer_check_constraint(df[orig])
                if check:
                    def_parts.append(f'CHECK("{clean}" {check})')

            col_def_parts.append(' '.join(def_parts))

        cursor.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        col_defs = ', '.join(col_def_parts)
        # No IF NOT EXISTS — DROP above guarantees the table is gone.
        cursor.execute(f'CREATE TABLE "{table_name}" ({col_defs})')

        # Only insert non-constant columns
        active_cols = [
            (clean, orig)
            for clean, orig in zip(clean_cols, orig_cols)
            if clean not in constant_col_names
        ]
        col_list         = ', '.join(f'"{c}"' for c, _ in active_cols)
        placeholders_str = ', '.join([placeholder] * len(active_cols))

        rows = [
            tuple(to_python(row[orig]) for _, orig in active_cols)
            for _, row in df.iterrows()
        ]
        cursor.executemany(
            f'INSERT INTO "{table_name}" ({col_list}) VALUES ({placeholders_str})',
            rows
        )

        print(f"  + Written to DB -> table '{table_name}'")


def write_to_sqlite(tables: dict, db_path: str, dialect: str):
    import sqlite3
    conn = sqlite3.connect(db_path)
    try:
        with conn:  # handles BEGIN / COMMIT / ROLLBACK as a single transaction
            _write_to_db(tables, conn, conn.cursor(), '?', dialect)
    except Exception as e:
        sys.exit(f"ERROR writing to SQLite: {e}")
    finally:
        conn.close()
    print(f"  + SQLite DB written -> '{db_path}'")


def write_to_postgres(tables: dict, conn_string: str, dialect: str):
    try:
        import psycopg2
    except ImportError:
        sys.exit("ERROR: psycopg2 not installed. Run: pip install psycopg2-binary")

    try:
        conn = psycopg2.connect(conn_string)
    except Exception as e:
        sys.exit(f"ERROR connecting to PostgreSQL: {e}")

    try:
        with conn:  # handles BEGIN / COMMIT / ROLLBACK as a single transaction
            _write_to_db(tables, conn, conn.cursor(), '%s', dialect)
    except Exception as e:
        sys.exit(f"ERROR writing to PostgreSQL: {e}")
    finally:
        conn.close()
    print(f"  + PostgreSQL DB written.")


# ─────────────────────────────────────────────
# 6. CLI ENTRY POINT
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert .csv/.xlsx spreadsheets to SQL (PostgreSQL or SQLite)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('input', help='Path to input .csv or .xlsx file')
    parser.add_argument(
        '--dialect', '-d',
        choices=['sqlite', 'postgres'],
        required=True,
        help='SQL dialect: sqlite or postgres'
    )
    parser.add_argument('--output', '-o', help='Output .sql file path (optional)')
    parser.add_argument(
        '--db',
        help=(
            'Database connection target (optional).\n'
            '  SQLite:   path to .sqlite3 / .db file\n'
            '  Postgres: postgresql://user:pass@host:port/dbname'
        )
    )
    args = parser.parse_args()

    if not args.output and not args.db:
        parser.error('Provide at least one of --output or --db')

    return args


def main():
    args = parse_args()

    print(f"\nLoading: {args.input}")
    tables = load_file(args.input)

    print(f"\nGenerating SQL ({args.dialect.upper()} dialect)...")
    sql = generate_full_sql(tables, args.dialect)

    if args.output:
        print(f"\nWriting .sql file...")
        write_sql_file(sql, args.output)

    if args.db:
        print(f"\nWriting to database...")
        if args.dialect == 'sqlite':
            write_to_sqlite(tables, args.db, args.dialect)
        else:
            write_to_postgres(tables, args.db, args.dialect)

    print("\nDone!\n")


if __name__ == '__main__':
    main()
