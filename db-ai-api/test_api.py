"""Test script for DB AI API."""
import requests
import json
from typing import Dict, Any


API_BASE_URL = "http://localhost:8000"


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def print_result(result: Dict[str, Any]):
    """Print a formatted result."""
    print(json.dumps(result, indent=2, default=str))


def test_health():
    """Test health check endpoint."""
    print_section("Testing Health Check")

    response = requests.get(f"{API_BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print_result(response.json())


def test_schema():
    """Test schema endpoint."""
    print_section("Testing Schema Endpoint")

    response = requests.get(f"{API_BASE_URL}/schema")
    result = response.json()

    print(f"Status Code: {response.status_code}")
    print(f"Database: {result['database']}")
    print(f"Total Tables: {result['total_tables']}")
    print(f"Total Views: {result['total_views']}")

    if result['tables']:
        print("\nFirst 5 tables:")
        for table in result['tables'][:5]:
            print(f"  - {table['name']} ({table['row_count']:,} rows, {table['column_count']} columns)")


def test_table_search(query: str):
    """Test semantic table search."""
    print_section(f"Testing Table Search: '{query}'")

    response = requests.get(
        f"{API_BASE_URL}/tables/search",
        params={"query": query, "top_k": 5}
    )

    result = response.json()
    print(f"Status Code: {response.status_code}")
    print(f"Query: {result['query']}")
    print(f"Results Found: {result['count']}\n")

    for table in result['results']:
        print(f"{table['rank']}. {table['table_name']}")
        print(f"   Relevance: {table['relevance_score']:.3f}")
        print(f"   Type: {table['type']}")
        print(f"   Rows: {table['row_count']:,}")
        print()


def test_explain_query(question: str):
    """Test query explanation without execution."""
    print_section(f"Testing Query Explanation: '{question}'")

    response = requests.post(
        f"{API_BASE_URL}/explain",
        json={"question": question}
    )

    result = response.json()
    print(f"Status Code: {response.status_code}")
    print(f"\nQuestion: {result['question']}")
    print(f"\nGenerated SQL:")
    print("-" * 70)
    print(result['sql'])
    print("-" * 70)

    if result.get('explanation'):
        print(f"\nExplanation:")
        print(result['explanation'])

    if result.get('relevant_tables'):
        print(f"\nRelevant Tables Used:")
        for table in result['relevant_tables'][:3]:
            print(f"  - {table['table_name']} (relevance: {table['relevance_score']:.3f})")


def test_query(question: str, execute: bool = False):
    """Test full query endpoint."""
    mode = "with execution" if execute else "without execution"
    print_section(f"Testing Query {mode}: '{question}'")

    response = requests.post(
        f"{API_BASE_URL}/query",
        json={
            "question": question,
            "execute": execute,
            "max_rows": 10,
            "include_explanation": True
        }
    )

    result = response.json()
    print(f"Status Code: {response.status_code}")
    print(f"\nQuestion: {result['question']}")
    print(f"Generation Attempts: {result['generation_attempts']}")

    print(f"\nGenerated SQL:")
    print("-" * 70)
    print(result['sql'])
    print("-" * 70)

    if result.get('explanation'):
        print(f"\nExplanation:")
        print(result['explanation'])

    if execute and result.get('execution'):
        exec_result = result['execution']

        if exec_result['success']:
            print(f"\nExecution Result:")
            print(f"  - Rows: {exec_result['row_count']}")
            print(f"  - Execution Time: {exec_result['execution_time_seconds']:.3f}s")
            print(f"  - Columns: {', '.join(exec_result['columns'])}")

            if exec_result['rows']:
                print(f"\nFirst 3 rows:")
                for i, row in enumerate(exec_result['rows'][:3], 1):
                    print(f"  Row {i}: {row}")
        else:
            print(f"\nExecution Failed:")
            print(f"  Error: {exec_result.get('error')}")


def test_table_details(table_name: str):
    """Test table details endpoint."""
    print_section(f"Testing Table Details: '{table_name}'")

    response = requests.get(f"{API_BASE_URL}/schema/table/{table_name}")

    if response.status_code == 200:
        result = response.json()
        print(f"Status Code: {response.status_code}")
        print(f"\n{result['summary']}")
    else:
        print(f"Status Code: {response.status_code}")
        print(f"Error: {response.json()}")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("  DB AI API - Test Suite")
    print("=" * 70)

    try:
        # 1. Health check
        test_health()

        # 2. Schema
        test_schema()

        # 3. Table search
        test_table_search("customer orders")
        test_table_search("product inventory")

        # 4. Explain queries (no execution)
        test_explain_query("Show me the top 10 customers by total order value")
        test_explain_query("What products have stock below 10 units?")

        # 5. Generate SQL without execution
        test_query("How many orders were placed in the last month?", execute=False)

        # 6. Full query with execution (if you want to test)
        # Uncomment to actually execute queries:
        # test_query("Show me the first 5 products", execute=True)

        # 7. Get details of a specific table
        # Replace with an actual table name from your database
        # test_table_details("Products")

        print_section("All Tests Completed")

    except requests.exceptions.ConnectionError:
        print("\nERROR: Could not connect to API. Make sure the server is running:")
        print("  python main.py")
        print("\nOr:")
        print("  docker-compose up")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
