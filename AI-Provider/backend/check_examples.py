#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Check specific training examples."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from training_data.retriever import QueryExampleRetriever

r = QueryExampleRetriever()

# Search for stock examples
results = r.find_similar("Залишки товарів по всіх складах з загальною вартістю", top_k=5, min_score=0.0)

print("Stock examples found:")
for i, res in enumerate(results):
    print(f"\n{i+1}. Score: {res.get('similarity_score', 0):.3f}")
    print(f"   ID: {res.get('id', 'N/A')}")
    print(f"   Question UK: {res.get('question_uk', 'N/A')}")
    print(f"   SQL: {res.get('sql', 'N/A')}")
    print(f"   Notes: {res.get('notes', 'N/A')}")

# Search for margin examples
print("\n\n=== Margin examples ===")
results2 = r.find_similar("Маржинальність по категоріях товарів", top_k=5, min_score=0.0)
for i, res in enumerate(results2):
    print(f"\n{i+1}. Score: {res.get('similarity_score', 0):.3f}")
    print(f"   ID: {res.get('id', 'N/A')}")
    print(f"   Question UK: {res.get('question_uk', 'N/A')}")
    print(f"   SQL: {res.get('sql', 'N/A')[:150]}...")
