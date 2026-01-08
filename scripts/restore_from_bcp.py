"""
Restore ConcordDb_v5 into a target SQL Server using bcp exports and schema_cache.json.
Creates basic tables from schema_cache (columns, types, nullability) then bulk imports .bcp files.
"""
import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

import pymssql


TYPE_MAP = {
    "int": "INT",
    "bigint": "BIGINT",
    "smallint": "SMALLINT",
    "tinyint": "TINYINT",
    "bit": "BIT",
    "decimal": "DECIMAL",
    "numeric": "NUMERIC",
    "float": "FLOAT",
    "real": "REAL",
    "money": "MONEY",
    "smallmoney": "SMALLMONEY",
    "datetime": "DATETIME",
    "datetime2": "DATETIME2",
    "smalldatetime": "SMALLDATETIME",
    "date": "DATE",
    "time": "TIME",
    "datetimeoffset": "DATETIMEOFFSET",
    "uniqueidentifier": "UNIQUEIDENTIFIER",
    "binary": "BINARY",
    "varbinary": "VARBINARY",
    "image": "VARBINARY(MAX)",
    "text": "VARCHAR(MAX)",
    "ntext": "NVARCHAR(MAX)",
    "xml": "XML",
}


def map_type(col: dict) -> str:
    raw_type = (col.get("type") or "").lower()
    max_len = col.get("max_length")

    if raw_type in ("varchar", "nvarchar", "char", "nchar"):
        if max_len in (None, "", -1):
            return f"{raw_type.upper()}(MAX)"
        if isinstance(max_len, str) and max_len.isdigit():
            max_len = int(max_len)
        try:
            max_len_int = int(max_len)
        except Exception:
            max_len_int = 255
        if max_len_int <= 0:
            return f"{raw_type.upper()}(MAX)"
        return f"{raw_type.upper()}({max_len_int})"

    if raw_type in ("decimal", "numeric"):
        # Not always present in schema_cache; default to (18,2)
        precision = col.get("precision", 18)
        scale = col.get("scale", 2)
        try:
            precision = int(precision)
        except Exception:
            precision = 18
        try:
            scale = int(scale)
        except Exception:
            scale = 2
        return f"{raw_type.upper()}({precision},{scale})"

    if raw_type in ("varbinary", "binary"):
        if max_len in (None, "", -1):
            return "VARBINARY(MAX)"
        try:
            max_len_int = int(max_len)
        except Exception:
            max_len_int = 255
        if max_len_int <= 0:
            return "VARBINARY(MAX)"
        return f"{raw_type.upper()}({max_len_int})"

    mapped = TYPE_MAP.get(raw_type)
    if mapped:
        return mapped
    # Fallback
    return raw_type.upper() if raw_type else "NVARCHAR(MAX)"


def quote_ident(name: str) -> str:
    # Wrap with brackets, escaping ending bracket
    safe = name.replace("]", "]]")
    return f"[{safe}]"


def load_schema(schema_path: Path):
    data = json.loads(schema_path.read_text(encoding="utf-8"))
    if "tables" in data:
        return data["tables"]
    # flat format fallback
    return data


def create_tables(conn, tables: dict):
    cursor = conn.cursor()
    for full_name, info in tables.items():
        try:
            schema = info.get("schema") or full_name.split(".")[0] if "." in full_name else "dbo"
            table = info.get("name") or full_name.split(".")[-1]
            cols = info.get("columns", [])
            if not cols:
                print(f"Skipping {full_name}: no columns")
                continue

            col_defs = []
            for col in cols:
                col_name = col.get("name") or col.get("column_name")
                if not col_name:
                    continue
                col_type = map_type(col)
                nullable = col.get("nullable", True)
                null_sql = "NULL" if nullable else "NOT NULL"
                col_defs.append(f"{quote_ident(col_name)} {col_type} {null_sql}")

            if not col_defs:
                print(f"Skipping {full_name}: could not map columns")
                continue

            create_sql = f"IF OBJECT_ID('{schema}.{table}', 'U') IS NULL CREATE TABLE {quote_ident(schema)}.{quote_ident(table)} ({', '.join(col_defs)});"
            cursor.execute(create_sql)
            conn.commit()
            print(f"Created table {schema}.{table}")
        except Exception as exc:
            print(f"Failed to create {full_name}: {exc}")
            conn.rollback()


def bcp_import_all(server, port, database, user, password, bcp_dir: Path):
    files = sorted(bcp_dir.glob("*.bcp"))
    for f in files:
        # file name pattern: schema_table.bcp (schema.table from export script)
        base = f.stem
        parts = base.split("_", 1)
        if len(parts) == 2:
            schema = parts[0]
            table = parts[1]
        else:
            schema = "dbo"
            table = base
        table_qualified = f"{schema}.{table}"
        cmd = [
            "bcp",
            table_qualified,
            "in",
            str(f),
            "-S",
            f"{server},{port}",
            "-d",
            database,
            "-U",
            user,
            "-P",
            password,
            "-n",
            "-b",
            "50000",
            "-q",
        ]
        print(f"BCP IN {table_qualified} from {f.name}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"BCP failed for {table_qualified}: {proc.stdout}\n{proc.stderr}")
        else:
            print(proc.stdout.strip())


def main():
    parser = argparse.ArgumentParser(description="Restore SQL Server DB from bcp exports and schema cache.")
    parser.add_argument("--schema", default="db-dump/schema/schema_cache.json")
    parser.add_argument("--bcp-dir", default="db-dump/data")
    parser.add_argument("--server", default="localhost")
    parser.add_argument("--port", type=int, default=14333)
    parser.add_argument("--database", default="ConcordDb_v5")
    parser.add_argument("--user", default="sa")
    parser.add_argument("--password", required=True)
    args = parser.parse_args()

    schema_path = Path(args.schema)
    bcp_dir = Path(args.bcp_dir)

    tables = load_schema(schema_path)

    print(f"Connecting to {args.server},{args.port} db={args.database} as {args.user}")
    conn = pymssql.connect(
        server=args.server,
        port=args.port,
        user=args.user,
        password=args.password,
        database=args.database,
        login_timeout=10,
        timeout=60,
    )

    create_tables(conn, tables)
    conn.close()

    bcp_import_all(args.server, args.port, args.database, args.user, args.password, bcp_dir)


if __name__ == "__main__":
    main()
