#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Check if specific examples are in the FAISS index."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from training_data.retriever import QueryExampleRetriever

r = QueryExampleRetriever()
print(f"Total examples: {r.get_stats()['total_examples']}")

# Search for our training examples
test_queries = [
    "Маржинальність по категоріях товарів",
    "Залишки товарів по всіх складах",
    "регіони зросли продажі",
]

for q in test_queries:
    print(f"\nQuery: {q}")
    results = r.find_similar(q, top_k=3, min_score=0.0)  # No min score to see all
    if results:
        for i, res in enumerate(results):
            score = res.get('similarity_score', 0)
            question = res.get('question_uk', res.get('question_en', ''))[:60]
            sql = res.get('sql', '')[:80]
            print(f"  {i+1}. [{score:.2f}] {question}...")
            print(f"     SQL: {sql}...")
    else:
        print("  No results found!")
