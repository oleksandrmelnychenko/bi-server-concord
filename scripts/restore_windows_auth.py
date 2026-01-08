"""
Restore ConcordDb_v5 using Windows Authentication.
Creates tables from schema_cache.json and imports .bcp files.
"""
import json
import subprocess
from pathlib import Path
import pyodbc

SCHEMA_PATH = Path("db-dump/schema/schema_cache.json")
BCP_DIR = Path("db-dump/data")
SERVER = "localhost"
DATABASE = "ConcordDb_v5"

TYPE_MAP = {
    "int": "INT", "bigint": "BIGINT", "smallint": "SMALLINT", "tinyint": "TINYINT",
    "bit": "BIT", "decimal": "DECIMAL", "numeric": "NUMERIC", "float": "FLOAT",
    "real": "REAL", "money": "MONEY", "smallmoney": "SMALLMONEY",
    "datetime": "DATETIME", "datetime2": "DATETIME2", "smalldatetime": "SMALLDATETIME",
    "date": "DATE", "time": "TIME", "datetimeoffset": "DATETIMEOFFSET",
    "uniqueidentifier": "UNIQUEIDENTIFIER", "binary": "BINARY", "varbinary": "VARBINARY",
    "image": "VARBINARY(MAX)", "text": "VARCHAR(MAX)", "ntext": "NVARCHAR(MAX)", "xml": "XML",
}

def map_type(col: dict) -> str:
    raw_type = (col.get("type") or "").lower()
    max_len = col.get("max_length")

    if raw_type in ("varchar", "nvarchar", "char", "nchar"):
        if max_len in (None, "", -1):
            return f"{raw_type.upper()}(MAX)"
        try:
            max_len_int = int(max_len)
        except:
            max_len_int = 255
        if max_len_int <= 0:
            return f"{raw_type.upper()}(MAX)"
        return f"{raw_type.upper()}({max_len_int})"

    if raw_type in ("decimal", "numeric"):
        precision = col.get("precision", 18)
        scale = col.get("scale", 2)
        try:
            precision, scale = int(precision), int(scale)
        except:
            precision, scale = 18, 2
        return f"{raw_type.upper()}({precision},{scale})"

    if raw_type in ("varbinary", "binary"):
        if max_len in (None, "", -1):
            return "VARBINARY(MAX)"
        try:
            max_len_int = int(max_len)
        except:
            max_len_int = 255
        if max_len_int <= 0:
            return "VARBINARY(MAX)"
        return f"{raw_type.upper()}({max_len_int})"

    return TYPE_MAP.get(raw_type, raw_type.upper() if raw_type else "NVARCHAR(MAX)")

def quote_ident(name: str) -> str:
    return f"[{name.replace(']', ']]')}]"

def load_schema():
    data = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return data.get("tables", data)

def create_tables(cursor, tables: dict):
    for full_name, info in tables.items():
        try:
            schema = info.get("schema") or (full_name.split(".")[0] if "." in full_name else "dbo")
            table = info.get("name") or full_name.split(".")[-1]
            cols = info.get("columns", [])
            if not cols:
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
                continue

            create_sql = f"""
            IF OBJECT_ID('{schema}.{table}', 'U') IS NULL
            CREATE TABLE {quote_ident(schema)}.{quote_ident(table)} ({', '.join(col_defs)});
            """
            cursor.execute(create_sql)
            cursor.commit()
            print(f"Created: {schema}.{table}")
        except Exception as e:
            print(f"Failed {full_name}: {e}")

def bcp_import_all():
    files = sorted(BCP_DIR.glob("*.bcp"))
    for f in files:
        base = f.stem
        parts = base.split("_", 1)
        schema = parts[0] if len(parts) == 2 else "dbo"
        table = parts[1] if len(parts) == 2 else base
        table_qualified = f"{schema}.{table}"

        cmd = [
            "bcp", table_qualified, "in", str(f),
            "-S", SERVER, "-d", DATABASE,
            "-T",  # Windows Auth
            "-n", "-b", "50000", "-q"
        ]
        print(f"Importing {table_qualified}...")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"  FAILED: {proc.stderr[:200]}")
        else:
            # Extract row count from output
            for line in proc.stdout.split('\n'):
                if 'rows copied' in line:
                    print(f"  {line.strip()}")
                    break

def main():
    print(f"Connecting to {SERVER}/{DATABASE} with Windows Auth...")
    conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={SERVER};DATABASE={DATABASE};Trusted_Connection=yes;"
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()

    print("Loading schema...")
    tables = load_schema()
    print(f"Found {len(tables)} tables")

    print("\nCreating tables...")
    create_tables(cursor, tables)

    conn.close()

    print("\nImporting data via BCP...")
    bcp_import_all()

    print("\nRestore complete!")

if __name__ == "__main__":
    main()
