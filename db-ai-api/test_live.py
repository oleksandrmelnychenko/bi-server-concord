"""Live test of query examples + Ollama."""
import os
os.environ.setdefault("DB_NAME", "ConcordDb_v5")
os.environ.setdefault("DB_USER", "test")
os.environ.setdefault("DB_PASSWORD", "test")

from training_data import QueryExampleRetriever
import ollama

def test_ukrainian_queries():
    """Test Ukrainian queries with multilingual embeddings."""

    retriever = QueryExampleRetriever()

    if not retriever.is_available():
        print("ERROR: Query examples not available!")
        return

    print("=" * 70)
    print("LIVE TEST: Ukrainian Queries with Ollama + Query Examples RAG")
    print("=" * 70)

    stats = retriever.get_stats()
    print(f"\nLoaded: {stats['total_examples']} query examples")
    print(f"Model: codellama:34b-instruct")

    # Ukrainian test queries
    test_queries = [
        "Покажи топ 10 товарів за продажами",
        "Скільки клієнтів зареєстровано",
        "Виручка за цей рік",
        "Борги клієнтів",
        "Продажі по місяцях",
        "Товари з низьким запасом",
    ]

    client = ollama.Client(host="http://localhost:11434")

    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"QUERY: {query}")
        print("-" * 70)

        # Get similar examples
        examples = retriever.find_similar(query, top_k=3)

        if examples:
            print(f"\nMatched {len(examples)} examples:")
            for ex in examples:
                print(f"  [{ex['similarity_score']:.2f}] {ex['question_en']}")
                print(f"          {ex['question_uk']}")

            # Build prompt
            examples_text = retriever.format_examples_for_prompt(examples, include_ukrainian=True)

            prompt = f"""You are a T-SQL expert. Generate SQL for Microsoft SQL Server.

{examples_text}

RULES:
1. Use TOP N instead of LIMIT
2. Use dbo.[Order] with brackets
3. Add WHERE Deleted = 0
4. Return ONLY the SQL query, no explanation

QUESTION: {query}

SQL:"""

            print("\n--- Generating SQL with Ollama ---")

            try:
                response = client.generate(
                    model="codellama:34b-instruct",
                    prompt=prompt,
                )

                sql = response["response"].strip()
                # Clean up SQL
                if "```" in sql:
                    sql = sql.split("```")[1].replace("sql", "").strip()

                print(f"\nGENERATED SQL:\n{sql}")

            except Exception as e:
                print(f"ERROR: {e}")
        else:
            print("No similar examples found!")

if __name__ == "__main__":
    test_ukrainian_queries()
