#!/usr/bin/env python
# -*- coding: utf-8 -*-
import requests
import sys
sys.stdout.reconfigure(encoding='utf-8')

queries = [
    "Залишки товарів по всіх складах з загальною вартістю",
    "Маржинальність по категоріях товарів",
]

for q in queries:
    print(f"Query: {q}")
    try:
        r = requests.post('http://127.0.0.1:8001/query', json={'question': q}, timeout=180)
        d = r.json()
        print(f"  Success: {d.get('success')}")
        print(f"  Rows: {d.get('row_count')}")
        sql = d.get("sql", "")
        if sql:
            print(f"  SQL: {sql[:200]}...")
        if d.get("error"):
            print(f"  Error: {d.get('error')}")
    except Exception as e:
        print(f"  Exception: {e}")
    print()
