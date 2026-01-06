#!/usr/bin/env python
# -*- coding: utf-8 -*-
import requests, sys, time
sys.stdout.reconfigure(encoding='utf-8')

# Test the exact failing queries from test_30
queries = [
    ('[11]', 'Маржинальність по категоріях товарів'),
    ('[25]', 'Залишки товарів по всіх складах з загальною вартістю'),
]

for num, q in queries:
    print(f'{num} {q}')
    start = time.time()
    try:
        r = requests.post('http://127.0.0.1:8000/query', json={'question': q}, timeout=180)
        d = r.json()
        elapsed = time.time()-start
        status = "OK" if d.get("success") else "FAIL"
        rows = d.get("row_count", 0)
        print(f'    {status} ({elapsed:.1f}s) - {rows} rows')
        if d.get("success"):
            print(f'    SQL: {d.get("sql", "")[:150]}...')
        else:
            err = str(d.get("error", ""))[:100]
            print(f'    Error: {err}')
    except Exception as e:
        print(f'    Exception: {str(e)[:80]}')
    print()
