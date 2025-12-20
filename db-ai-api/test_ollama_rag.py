"""Quick test for Ollama with Query Examples RAG."""
import os
import sys

# Set environment variables for testing (minimal config)
os.environ.setdefault("DB_NAME", "ConcordDb_v5")
os.environ.setdefault("DB_USER", "test")
os.environ.setdefault("DB_PASSWORD", "test")

from training_data import QueryExampleRetriever
import ollama

def test_rag_prompt():
    """Test that RAG retrieves correct examples and builds good prompts."""

    retriever = QueryExampleRetriever()

    if not retriever.is_available():
        print("ERROR: Query examples not available!")
        return

    print("=" * 60)
    print("TESTING OLLAMA + QUERY EXAMPLES RAG")
    print("=" * 60)

    stats = retriever.get_stats()
    print(f"\nLoaded {stats['total_examples']} query examples")
    print(f"Domains: {list(stats['domains'].keys())}")

    # Test queries
    test_queries = [
        "Show top 10 products by sales",
        "Покажи топ клієнтів за покупками",
        "Total revenue this year",
        "Борги клієнтів",
    ]

    client = ollama.Client(host="http://localhost:11434")

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"USER QUERY: {query}")
        print("-" * 60)

        # Get similar examples
        examples = retriever.find_similar(query, top_k=3)

        if examples:
            print(f"\nFound {len(examples)} similar examples:")
            for ex in examples:
                print(f"  - {ex['question_en']} (score: {ex['similarity_score']})")

            # Build prompt with examples
            examples_text = retriever.format_examples_for_prompt(examples)

            prompt = f"""You are a T-SQL expert. Generate SQL for Microsoft SQL Server.

{examples_text}

RULES:
1. Use TOP N instead of LIMIT
2. Use dbo.[Order] with brackets (reserved word)
3. Add WHERE Deleted = 0 for active records
4. Return ONLY the SQL query

QUESTION: {query}

SQL:"""

            print(f"\n--- Calling Ollama (codellama:34b-instruct) ---")

            try:
                response = client.generate(
                    model="codellama:34b-instruct",
                    prompt=prompt,
                )

                sql = response["response"].strip()
                print(f"\nGENERATED SQL:\n{sql}")

            except Exception as e:
                print(f"ERROR calling Ollama: {e}")
        else:
            print("No similar examples found!")

if __name__ == "__main__":
    test_rag_prompt()
