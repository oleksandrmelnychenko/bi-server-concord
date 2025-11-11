#!/usr/bin/env python3
import ast
import sys
from pathlib import Path
from typing import List, Tuple

def remove_redundant_comments(filepath: Path) -> bool:
    with open(filepath, 'r', encoding='utf-8') as f:
        original_lines = f.readlines()

    try:
        tree = ast.parse(''.join(original_lines), filename=str(filepath))
    except SyntaxError as e:
        print(f"✗ Syntax error in {filepath}: {e}")
        return False

    docstring_ranges = set()

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module, ast.AsyncFunctionDef)):
            docstring = ast.get_docstring(node, clean=False)
            if docstring and hasattr(node, 'body') and node.body:
                first_stmt = node.body[0]
                if isinstance(first_stmt, ast.Expr) and isinstance(first_stmt.value, ast.Constant):
                    start_line = first_stmt.lineno - 1
                    end_line = first_stmt.end_lineno - 1 if hasattr(first_stmt, 'end_lineno') else start_line

                    for line_num in range(start_line, end_line + 1):
                        docstring_ranges.add(line_num)

    cleaned_lines = []
    for i, line in enumerate(original_lines):
        stripped = line.lstrip()

        if i == 0 and stripped.startswith('#!'):
            cleaned_lines.append(line)
            continue

        if stripped.startswith('# -*- coding:') or stripped.startswith('# coding:'):
            cleaned_lines.append(line)
            continue

        if i in docstring_ranges:
            continue

        if '#' in line:
            before_hash = line[:line.index('#')]

            if '"' not in before_hash and "'" not in before_hash:
                line = before_hash.rstrip() + '\n'

        if line.strip() or (cleaned_lines and cleaned_lines[-1].strip()):
            cleaned_lines.append(line)

    while cleaned_lines and not cleaned_lines[-1].strip():
        cleaned_lines.pop()

    if cleaned_lines and not cleaned_lines[-1].endswith('\n'):
        cleaned_lines[-1] += '\n'

    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(cleaned_lines)

    removed_count = len(original_lines) - len(cleaned_lines)
    print(f"✓ {filepath.name}: removed {removed_count} lines ({len(docstring_ranges)} docstring lines)")
    return True

if __name__ == '__main__':
    files = [
        'scripts/forecasting/core/pattern_analyzer.py',
        'scripts/forecasting/core/customer_predictor.py',
        'scripts/forecasting/core/product_aggregator.py',
        'scripts/forecasting/core/forecast_engine.py',
        'scripts/improved_hybrid_recommender_v32.py',
        'scripts/improved_hybrid_recommender_v31.py',
        'scripts/improved_hybrid_recommender.py',
        'scripts/datetime_utils.py',
        'scripts/redis_helper.py',
        'scripts/forecasting/forecast_worker.py',
        'scripts/weekly_recommendation_worker.py',
        'workers/weekly_recommendation_worker.py',
        'api/main.py',
        'api/models/forecast_schemas.py',
        'api/models/recommendation_schemas.py',
        'api/routes/recommendations.py',
        'api/db_pool.py',
        'scripts/forecasting/test_worker.py',
        'scripts/forecasting/test_rfm_accuracy.py',
        'scripts/forecasting/trigger_relearn.py',
        'scripts/forecasting/__init__.py',
        'scripts/forecasting/core/__init__.py',
    ]

    base_path = Path('/Users/oleksandrmelnychenko/Projects/Concord-BI-Server')

    success_count = 0
    for filepath in files:
        full_path = base_path / filepath
        if full_path.exists():
            if remove_redundant_comments(full_path):
                success_count += 1
        else:
            print(f"✗ Not found: {filepath}")

    print(f"\n✅ Successfully processed {success_count}/{len(files)} files")
