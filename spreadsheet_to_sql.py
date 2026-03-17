"""
spreadsheet_to_sql.py
─────────────────────
Converts .csv / .xlsx files into SQL (PostgreSQL or SQLite).

Features:
  - Auto-detects column data types (INTEGER, FLOAT, TIMESTAMP, TEXT fallback)
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
  pip install pandas openpyxl psycopg2-binary
"""

import argparse
import os
import re
import sys
from pathlib import Path

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


# ─────────────────────────────────────────────
# 2. TYPE INFERENCE
# ─────────────────────────────────────────────

def infer_sql_type(series: pd.Series, dialect: str) -> str:
    """
    Inspect a pandas Series and return the best-fit SQL type.

    Detection order:
      1. INTEGER   - all non-null values are whole numbers
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
        if (converted == converted.astype('int64')).all():
            return 'INTEGER'
        return 'REAL' if dialect == 'sqlite' else 'DOUBLE PRECISION'
    except (ValueError, TypeError):
        pass

    # --- Try TIMESTAMP ---
    try:
        pd.to_datetime(col, errors='raise', infer_datetime_format=True)
        return 'TEXT' if dialect == 'sqlite' else 'TIMESTAMP'
    except (ValueError, TypeError):
        pass

    # --- Fallback: TEXT ---
    return 'TEXT'


# ─────────────────────────────────────────────
# 3. FILE LOADING
# ─────────────────────────────────────────────

def load_file(filepath: str) -> dict:
    """
    Load a .csv or .xlsx file.
    Returns a dict mapping table_name -> DataFrame.
    """
    path = Path(filepath)
    ext = path.suffix.lower()
    tables = {}

    if ext == '.csv':
        df = pd.read_csv(filepath)
        table_name = sanitize_name(path.stem)
        tables[table_name] = df
        print(f"  + Loaded CSV  -> table '{table_name}' ({len(df)} rows, {len(df.columns)} cols)")

    elif ext in ('.xlsx', '.xls'):
        xls = pd.ExcelFile(filepath)
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            table_name = sanitize_name(sheet_name)
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
    return str(float(val))


def generate_sql_for_table(table_name: str, df: pd.DataFrame, dialect: str) -> str:
    """Build DROP + CREATE + INSERT SQL for one table."""
    lines = []
    clean_cols = [sanitize_name(c) for c in df.columns]
    col_types = {
        clean: infer_sql_type(df[orig], dialect)
        for clean, orig in zip(clean_cols, df.columns)
    }

    lines.append(f'DROP TABLE IF EXISTS "{table_name}";')

    col_defs = ',\n    '.join(f'"{col}" {col_types[col]}' for col in clean_cols)
    lines.append(f'CREATE TABLE IF NOT EXISTS "{table_name}" (\n    {col_defs}\n);')

    col_list = ', '.join(f'"{c}"' for c in clean_cols)

    if dialect == 'postgres':
        value_rows = []
        for _, row in df.iterrows():
            vals = ', '.join(
                escape_value(row[orig], col_types[clean], dialect)
                for clean, orig in zip(clean_cols, df.columns)
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
                for clean, orig in zip(clean_cols, df.columns)
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


def write_to_sqlite(tables: dict, db_path: str, dialect: str):
    import sqlite3
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for table_name, df in tables.items():
        clean_cols = [sanitize_name(c) for c in df.columns]
        col_types = {
            clean: infer_sql_type(df[orig], dialect)
            for clean, orig in zip(clean_cols, df.columns)
        }
        df = df.copy()
        df.columns = clean_cols

        cursor.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        col_defs = ', '.join(f'"{c}" {col_types[c]}' for c in clean_cols)
        cursor.execute(f'CREATE TABLE IF NOT EXISTS "{table_name}" ({col_defs})')

        placeholders = ', '.join(['?'] * len(clean_cols))
        col_list = ', '.join(f'"{c}"' for c in clean_cols)
        for _, row in df.iterrows():
            cursor.execute(
                f'INSERT INTO "{table_name}" ({col_list}) VALUES ({placeholders})',
                tuple(None if pd.isna(v) else v for v in row)
            )
        conn.commit()
        print(f"  + Written to SQLite -> table '{table_name}' in '{db_path}'")

    conn.close()


def write_to_postgres(tables: dict, conn_string: str, dialect: str):
    try:
        import psycopg2
    except ImportError:
        sys.exit("ERROR: psycopg2 not installed. Run: pip install psycopg2-binary")

    conn = psycopg2.connect(conn_string)
    cursor = conn.cursor()

    for table_name, df in tables.items():
        clean_cols = [sanitize_name(c) for c in df.columns]
        col_types = {
            clean: infer_sql_type(df[orig], dialect)
            for clean, orig in zip(clean_cols, df.columns)
        }
        df = df.copy()
        df.columns = clean_cols

        cursor.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        col_defs = ', '.join(f'"{c}" {col_types[c]}' for c in clean_cols)
        cursor.execute(f'CREATE TABLE IF NOT EXISTS "{table_name}" ({col_defs})')

        placeholders = ', '.join(['%s'] * len(clean_cols))
        col_list = ', '.join(f'"{c}"' for c in clean_cols)
        for _, row in df.iterrows():
            cursor.execute(
                f'INSERT INTO "{table_name}" ({col_list}) VALUES ({placeholders})',
                tuple(None if pd.isna(v) else v for v in row)
            )
        conn.commit()
        print(f"  + Written to PostgreSQL -> table '{table_name}'")

    cursor.close()
    conn.close()


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