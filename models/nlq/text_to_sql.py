"""
Natural Language to SQL Converter

Uses LLM (Llama 3.1) to convert natural language queries to SQL.
"""
import os
import logging
from typing import Dict, List, Any, Optional
import json
import requests
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SchemaInfo:
    """Database schema information"""
    table_name: str
    columns: List[Dict[str, str]]
    sample_data: Optional[List[Dict[str, Any]]] = None
    description: Optional[str] = None


class TextToSQLConverter:
    """Converts natural language queries to SQL using LLM"""

    def __init__(
        self,
        llm_endpoint: str = "http://localhost:11434",
        model_name: str = "llama3.1:8b",
        temperature: float = 0.1
    ):
        self.llm_endpoint = llm_endpoint
        self.model_name = model_name
        self.temperature = temperature
        self.schema_context = []

    def load_schema(self, schema_info: List[SchemaInfo]):
        """
        Load database schema information

        Args:
            schema_info: List of schema information objects
        """
        self.schema_context = schema_info
        logger.info(f"Loaded schema for {len(schema_info)} tables")

    def _build_schema_prompt(self) -> str:
        """Build schema context for LLM prompt"""
        schema_text = "# Database Schema\n\n"

        for table in self.schema_context:
            schema_text += f"## Table: {table.table_name}\n"

            if table.description:
                schema_text += f"Description: {table.description}\n"

            schema_text += "Columns:\n"
            for col in table.columns:
                schema_text += f"  - {col['name']} ({col['type']})"
                if col.get('description'):
                    schema_text += f": {col['description']}"
                schema_text += "\n"

            if table.sample_data:
                schema_text += f"\nSample data:\n{json.dumps(table.sample_data[:3], indent=2)}\n"

            schema_text += "\n"

        return schema_text

    def _build_prompt(self, question: str) -> str:
        """
        Build complete prompt for LLM

        Args:
            question: Natural language question

        Returns:
            Complete prompt
        """
        schema_context = self._build_schema_prompt()

        prompt = f"""You are an expert SQL query generator. Convert the natural language question into a valid PostgreSQL query.

{schema_context}

# Instructions
1. Generate ONLY the SQL query, no explanations
2. Use PostgreSQL syntax
3. Include appropriate WHERE clauses, JOINs, GROUP BY, ORDER BY as needed
4. Return results in a reasonable format
5. Limit results to prevent huge responses (use LIMIT clause)
6. Use table and column names exactly as shown in the schema

# Examples
Question: Show me the top 10 customers by total revenue
SQL: SELECT customer_id, customer_name, SUM(total_amount) as total_revenue
     FROM orders
     JOIN customers ON orders.customer_id = customers.customer_id
     GROUP BY customer_id, customer_name
     ORDER BY total_revenue DESC
     LIMIT 10;

Question: How many orders were placed last month?
SQL: SELECT COUNT(*) as order_count
     FROM orders
     WHERE order_date >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')
     AND order_date < DATE_TRUNC('month', CURRENT_DATE);

# Your Task
Question: {question}
SQL:"""

        return prompt

    def convert_to_sql(self, question: str) -> str:
        """
        Convert natural language question to SQL

        Args:
            question: Natural language question

        Returns:
            Generated SQL query
        """
        logger.info(f"Converting question to SQL: {question[:100]}")

        # Build prompt
        prompt = self._build_prompt(question)

        # Call LLM
        try:
            response = requests.post(
                f"{self.llm_endpoint}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": self.temperature,
                    "stream": False,
                    "options": {
                        "num_predict": 500
                    }
                },
                timeout=30
            )

            response.raise_for_status()
            result = response.json()

            sql_query = result.get('response', '').strip()

            # Clean up the response
            sql_query = self._clean_sql_query(sql_query)

            logger.info(f"Generated SQL: {sql_query}")

            return sql_query

        except requests.RequestException as e:
            logger.error(f"Error calling LLM: {e}")
            raise RuntimeError(f"Failed to generate SQL: {e}")

    def _clean_sql_query(self, sql: str) -> str:
        """
        Clean up generated SQL query

        Args:
            sql: Raw SQL from LLM

        Returns:
            Cleaned SQL
        """
        # Remove markdown code blocks
        sql = sql.replace('```sql', '').replace('```', '')

        # Remove extra whitespace
        sql = ' '.join(sql.split())

        # Ensure it ends with semicolon
        if not sql.endswith(';'):
            sql += ';'

        return sql

    def validate_sql(self, sql: str) -> bool:
        """
        Basic SQL validation

        Args:
            sql: SQL query to validate

        Returns:
            True if valid, False otherwise
        """
        # Basic checks
        sql_lower = sql.lower()

        # Must be a SELECT query (safety check)
        if not sql_lower.strip().startswith('select'):
            logger.warning("Query must be a SELECT statement")
            return False

        # Should not contain dangerous keywords
        dangerous_keywords = ['drop', 'delete', 'truncate', 'alter', 'create', 'insert', 'update']
        for keyword in dangerous_keywords:
            if keyword in sql_lower:
                logger.warning(f"Query contains dangerous keyword: {keyword}")
                return False

        return True


def load_schema_from_datahub() -> List[SchemaInfo]:
    """
    Load schema information from DataHub

    Returns:
        List of schema information
    """
    # TODO: Implement actual DataHub integration
    # For now, return mock schema

    mock_schema = [
        SchemaInfo(
            table_name="customers",
            columns=[
                {"name": "customer_id", "type": "VARCHAR(50)", "description": "Unique customer identifier"},
                {"name": "customer_name", "type": "VARCHAR(255)", "description": "Customer name"},
                {"name": "email", "type": "VARCHAR(255)", "description": "Customer email"},
                {"name": "created_at", "type": "TIMESTAMP", "description": "Account creation date"}
            ],
            description="Customer master data"
        ),
        SchemaInfo(
            table_name="orders",
            columns=[
                {"name": "order_id", "type": "VARCHAR(50)", "description": "Unique order identifier"},
                {"name": "customer_id", "type": "VARCHAR(50)", "description": "Foreign key to customers"},
                {"name": "order_date", "type": "DATE", "description": "Order date"},
                {"name": "total_amount", "type": "DECIMAL(10,2)", "description": "Total order amount"},
                {"name": "status", "type": "VARCHAR(50)", "description": "Order status"}
            ],
            description="Sales orders"
        ),
        SchemaInfo(
            table_name="products",
            columns=[
                {"name": "product_id", "type": "VARCHAR(50)", "description": "Unique product identifier"},
                {"name": "product_name", "type": "VARCHAR(255)", "description": "Product name"},
                {"name": "category", "type": "VARCHAR(100)", "description": "Product category"},
                {"name": "price", "type": "DECIMAL(10,2)", "description": "Product price"}
            ],
            description="Product catalog"
        ),
        SchemaInfo(
            table_name="order_items",
            columns=[
                {"name": "order_item_id", "type": "VARCHAR(50)", "description": "Unique order item ID"},
                {"name": "order_id", "type": "VARCHAR(50)", "description": "Foreign key to orders"},
                {"name": "product_id", "type": "VARCHAR(50)", "description": "Foreign key to products"},
                {"name": "quantity", "type": "INTEGER", "description": "Quantity ordered"},
                {"name": "unit_price", "type": "DECIMAL(10,2)", "description": "Unit price at time of order"}
            ],
            description="Order line items"
        )
    ]

    return mock_schema


def main():
    """Test the text-to-SQL converter"""

    # Initialize converter
    converter = TextToSQLConverter(
        llm_endpoint=os.getenv("LLM_ENDPOINT", "http://localhost:11434"),
        model_name=os.getenv("LLM_MODEL", "llama3.1:8b")
    )

    # Load schema
    schema = load_schema_from_datahub()
    converter.load_schema(schema)

    # Test questions
    test_questions = [
        "Show me the top 10 customers by total revenue",
        "How many orders were placed in the last 30 days?",
        "What are the top 5 selling products?",
        "Show me customers who haven't ordered in the last 90 days",
        "What is the average order value by customer segment?"
    ]

    print("\n=== Text-to-SQL Converter Test ===\n")

    for question in test_questions:
        print(f"Question: {question}")
        try:
            sql = converter.convert_to_sql(question)

            if converter.validate_sql(sql):
                print(f"SQL: {sql}")
                print("✓ Valid\n")
            else:
                print("✗ Invalid SQL generated\n")

        except Exception as e:
            print(f"✗ Error: {e}\n")


if __name__ == "__main__":
    main()
