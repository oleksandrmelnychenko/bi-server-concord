"""
Example client for DB AI API.

This demonstrates how to integrate the Text-to-SQL API into your applications.
"""
import requests
from typing import Dict, List, Any, Optional
import json


class DBClient:
    """Simple client for DB AI API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize client.

        Args:
            base_url: API base URL
        """
        self.base_url = base_url.rstrip("/")

    def ask(
        self,
        question: str,
        execute: bool = True,
        max_rows: int = 100,
        include_explanation: bool = False,
    ) -> Dict[str, Any]:
        """Ask a natural language question.

        Args:
            question: Your question in plain English
            execute: Whether to execute the SQL
            max_rows: Maximum rows to return
            include_explanation: Include SQL explanation

        Returns:
            Response dictionary with SQL and optional results

        Example:
            >>> client = DBClient()
            >>> result = client.ask("Show me top 10 customers by revenue")
            >>> print(result['sql'])
            >>> print(result['execution']['rows'])
        """
        response = requests.post(
            f"{self.base_url}/query",
            json={
                "question": question,
                "execute": execute,
                "max_rows": max_rows,
                "include_explanation": include_explanation,
            },
        )
        response.raise_for_status()
        return response.json()

    def execute_sql(self, sql: str, max_rows: int = 100) -> Dict[str, Any]:
        """Execute SQL directly.

        Args:
            sql: SQL query to execute
            max_rows: Maximum rows to return

        Returns:
            Execution results

        Example:
            >>> client = DBClient()
            >>> result = client.execute_sql("SELECT TOP 10 * FROM Products")
            >>> print(result['rows'])
        """
        response = requests.post(
            f"{self.base_url}/execute",
            json={"sql": sql, "max_rows": max_rows},
        )
        response.raise_for_status()
        return response.json()

    def explain(self, question: str) -> Dict[str, Any]:
        """Get SQL explanation without executing.

        Args:
            question: Your question

        Returns:
            SQL query with explanation

        Example:
            >>> client = DBClient()
            >>> result = client.explain("How many customers do we have?")
            >>> print(result['sql'])
            >>> print(result['explanation'])
        """
        response = requests.post(
            f"{self.base_url}/explain",
            json={"question": question},
        )
        response.raise_for_status()
        return response.json()

    def get_schema(self) -> Dict[str, Any]:
        """Get database schema.

        Returns:
            Schema information

        Example:
            >>> client = DBClient()
            >>> schema = client.get_schema()
            >>> print(f"Tables: {schema['total_tables']}")
        """
        response = requests.get(f"{self.base_url}/schema")
        response.raise_for_status()
        return response.json()

    def search_tables(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant tables.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of relevant tables

        Example:
            >>> client = DBClient()
            >>> tables = client.search_tables("customer orders")
            >>> for table in tables:
            ...     print(f"{table['table_name']}: {table['relevance_score']}")
        """
        response = requests.get(
            f"{self.base_url}/tables/search",
            params={"query": query, "top_k": top_k},
        )
        response.raise_for_status()
        return response.json()["results"]

    def health_check(self) -> bool:
        """Check if API is healthy.

        Returns:
            True if healthy

        Example:
            >>> client = DBClient()
            >>> if client.health_check():
            ...     print("API is ready!")
        """
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception:
            return False


# Example usage functions
def example_basic_query():
    """Example: Basic natural language query."""
    print("\n" + "=" * 70)
    print("Example 1: Basic Query")
    print("=" * 70)

    client = DBClient()

    question = "How many products do we have?"
    result = client.ask(question, execute=True)

    print(f"Question: {question}")
    print(f"Generated SQL:\n{result['sql']}")

    if result["execution"]["success"]:
        rows = result["execution"]["rows"]
        print(f"\nResult: {rows}")
    else:
        print(f"Error: {result['execution']['error']}")


def example_with_explanation():
    """Example: Get SQL with explanation."""
    print("\n" + "=" * 70)
    print("Example 2: Query with Explanation")
    print("=" * 70)

    client = DBClient()

    question = "Show me top 5 customers by revenue"
    result = client.explain(question)

    print(f"Question: {question}")
    print(f"\nGenerated SQL:\n{result['sql']}")

    if result.get("explanation"):
        print(f"\nExplanation:\n{result['explanation']}")


def example_table_search():
    """Example: Search for relevant tables."""
    print("\n" + "=" * 70)
    print("Example 3: Semantic Table Search")
    print("=" * 70)

    client = DBClient()

    search_query = "customer orders and revenue"
    tables = client.search_tables(search_query, top_k=5)

    print(f"Search: {search_query}")
    print(f"\nFound {len(tables)} relevant tables:\n")

    for table in tables:
        print(f"{table['rank']}. {table['table_name']}")
        print(f"   Relevance: {table['relevance_score']:.3f}")
        print(f"   Rows: {table['row_count']:,}")
        print()


def example_batch_queries():
    """Example: Multiple queries in sequence."""
    print("\n" + "=" * 70)
    print("Example 4: Batch Queries")
    print("=" * 70)

    client = DBClient()

    questions = [
        "Count total customers",
        "Count total products",
        "Count total orders",
    ]

    results = []
    for question in questions:
        try:
            result = client.ask(question, execute=True, max_rows=1)
            results.append({
                "question": question,
                "sql": result["sql"],
                "result": result["execution"]["rows"][0] if result["execution"]["rows"] else None,
            })
        except Exception as e:
            results.append({"question": question, "error": str(e)})

    print("Batch Results:\n")
    for r in results:
        print(f"Q: {r['question']}")
        if "error" in r:
            print(f"   Error: {r['error']}")
        else:
            print(f"   SQL: {r['sql']}")
            print(f"   Result: {r['result']}")
        print()


def example_error_handling():
    """Example: Proper error handling."""
    print("\n" + "=" * 70)
    print("Example 5: Error Handling")
    print("=" * 70)

    client = DBClient()

    # Try an ambiguous question
    question = "Show me the data"

    try:
        result = client.ask(question, execute=False)
        print(f"Question: {question}")
        print(f"Generated SQL:\n{result['sql']}")
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        print(f"Response: {e.response.json()}")
    except Exception as e:
        print(f"Error: {e}")


def example_integration_use_case():
    """Example: Real-world integration scenario."""
    print("\n" + "=" * 70)
    print("Example 6: Dashboard Integration")
    print("=" * 70)

    client = DBClient()

    # Simulate dashboard widgets
    widgets = {
        "total_customers": "How many customers do we have?",
        "total_revenue": "What is the total revenue this year?",
        "top_products": "Show me top 5 products by sales",
        "recent_orders": "Show me the 10 most recent orders",
    }

    print("Fetching dashboard data...\n")

    dashboard_data = {}

    for widget_name, question in widgets.items():
        try:
            result = client.ask(question, execute=True, max_rows=10)

            if result["execution"]["success"]:
                dashboard_data[widget_name] = {
                    "sql": result["sql"],
                    "data": result["execution"]["rows"],
                    "count": result["execution"]["row_count"],
                }
                print(f"✓ {widget_name}: {result['execution']['row_count']} rows")
            else:
                print(f"✗ {widget_name}: Failed")

        except Exception as e:
            print(f"✗ {widget_name}: Error - {e}")

    print(f"\nDashboard data ready!")
    print(f"Data structure: {json.dumps(list(dashboard_data.keys()), indent=2)}")


def main():
    """Run all examples."""
    client = DBClient()

    # Check health first
    print("Checking API health...")
    if not client.health_check():
        print("❌ API is not available. Please start the server:")
        print("   python main.py")
        return

    print("✅ API is healthy!\n")

    # Run examples
    try:
        example_basic_query()
        example_with_explanation()
        example_table_search()
        example_batch_queries()
        example_error_handling()
        example_integration_use_case()

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("Examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
