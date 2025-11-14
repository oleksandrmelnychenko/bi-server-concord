#!/usr/bin/env python3
import re
import sys
from pathlib import Path

def remove_comments_from_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cleaned_lines = []
    in_multiline_string = False
    multiline_quote = None
    skip_next_blank = False

    for i, line in enumerate(lines):
        stripped = line.lstrip()

        if i == 0 and stripped.startswith('#!'):
            cleaned_lines.append(line)
            continue

        if not in_multiline_string:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                multiline_quote = stripped[:3]
                if stripped.count(multiline_quote) >= 2 and len(stripped.strip()) > len(multiline_quote):
                    skip_next_blank = True
                    continue
                in_multiline_string = True
                continue

            if '#' in line and not any(q in line[:line.index('#')] for q in ['"', "'"]):
                line = line[:line.index('#')].rstrip() + '\n'

            if line.strip():
                cleaned_lines.append(line)
                skip_next_blank = False
            elif not skip_next_blank:
                cleaned_lines.append(line)
        else:
            if multiline_quote in line:
                in_multiline_string = False
                multiline_quote = None
                skip_next_blank = True
            continue

    while cleaned_lines and not cleaned_lines[-1].strip():
        cleaned_lines.pop()

    if cleaned_lines and not cleaned_lines[-1].endswith('\n'):
        cleaned_lines[-1] += '\n'

    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(cleaned_lines)

    print(f"✓ Cleaned: {filepath}")

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

    for filepath in files:
        full_path = Path('/Users/oleksandrmelnychenko/Projects/Concord-BI-Server') / filepath
        if full_path.exists():
            remove_comments_from_file(full_path)
        else:
            print(f"✗ Not found: {filepath}")

    print(f"\n✅ Processed {len(files)} files")
