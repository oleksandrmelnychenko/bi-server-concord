"""SQL generation agent using local LLM via Ollama."""
import re
from typing import Dict, List, Any, Optional
from datetime import datetime

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from loguru import logger
import ollama

from config import settings
from schema_extractor import SchemaExtractor
from table_selector import TableSelector


class SQLAgent:
    """Generate and execute SQL queries using local LLM."""

    def __init__(
        self,
        engine: Optional[Engine] = None,
        schema_extractor: Optional[SchemaExtractor] = None,
        table_selector: Optional[TableSelector] = None,
    ):
        """Initialize SQL agent.

        Args:
            engine: SQLAlchemy engine
            schema_extractor: SchemaExtractor instance
            table_selector: TableSelector instance
        """
        self.engine = engine or create_engine(
            settings.database_url,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
        self.schema_extractor = schema_extractor or SchemaExtractor(self.engine)
        self.table_selector = table_selector or TableSelector(self.schema_extractor)

        # Ensure schema is indexed
        self.table_selector.index_schema()

        self.ollama_client = ollama.Client(host=settings.ollama_base_url)

    def generate_sql(
        self,
        question: str,
        max_retries: int = 2,
        include_explanation: bool = False,
    ) -> Dict[str, Any]:
        """Generate SQL query from natural language question.

        Args:
            question: Natural language question
            max_retries: Maximum retry attempts if SQL is invalid
            include_explanation: Whether to include explanation

        Returns:
            Dictionary with SQL query and metadata
        """
        logger.info(f"Generating SQL for: {question}")

        # Get relevant table context using RAG
        context = self.table_selector.get_context_for_query(question)

        # Generate SQL using LLM
        sql_query = None
        explanation = None
        attempts = []

        for attempt in range(max_retries + 1):
            try:
                # Build prompt
                prompt = self._build_prompt(question, context, attempts)

                # Call Ollama
                response = self.ollama_client.generate(
                    model=settings.ollama_model, prompt=prompt
                )

                # Extract SQL from response
                sql_query = self._extract_sql(response["response"])

                # Validate SQL syntax
                self._validate_sql(sql_query)

                # If we want explanation, extract it
                if include_explanation:
                    explanation = self._extract_explanation(response["response"])

                logger.info(f"Generated SQL successfully on attempt {attempt + 1}")
                break

            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Attempt {attempt + 1} failed: {error_msg}")
                attempts.append({"sql": sql_query, "error": error_msg})

                if attempt == max_retries:
                    raise ValueError(
                        f"Failed to generate valid SQL after {max_retries + 1} attempts. "
                        f"Last error: {error_msg}"
                    )

        return {
            "question": question,
            "sql": sql_query,
            "explanation": explanation,
            "attempts": len(attempts) + 1,
            "generated_at": datetime.now().isoformat(),
        }

    def execute_sql(
        self, sql_query: str, max_rows: Optional[int] = None
    ) -> Dict[str, Any]:
        """Execute SQL query and return results.

        Args:
            sql_query: SQL query to execute
            max_rows: Maximum rows to return

        Returns:
            Dictionary with query results and metadata
        """
        if max_rows is None:
            max_rows = settings.max_rows_returned

        logger.info(f"Executing SQL: {sql_query[:100]}...")

        # Security check for read-only mode
        if settings.read_only_mode:
            self._check_read_only(sql_query)

        start_time = datetime.now()

        try:
            with self.engine.connect() as conn:
                # Set query timeout
                if settings.query_timeout:
                    conn.execute(
                        text(f"SET LOCK_TIMEOUT {settings.query_timeout * 1000}")
                    )

                # Execute query
                result = conn.execute(text(sql_query))

                # Fetch results
                rows = []
                columns = list(result.keys()) if result.returns_rows else []

                if result.returns_rows:
                    for i, row in enumerate(result):
                        if i >= max_rows:
                            logger.warning(
                                f"Limiting results to {max_rows} rows"
                            )
                            break

                        # Convert row to dict
                        row_dict = {}
                        for key, value in row._mapping.items():
                            # Handle non-JSON serializable types
                            if isinstance(value, (datetime,)):
                                row_dict[key] = value.isoformat()
                            elif isinstance(value, bytes):
                                row_dict[key] = value.hex()
                            else:
                                try:
                                    # Test JSON serialization
                                    import json

                                    json.dumps(value)
                                    row_dict[key] = value
                                except (TypeError, ValueError):
                                    row_dict[key] = str(value)

                        rows.append(row_dict)

                execution_time = (datetime.now() - start_time).total_seconds()

                return {
                    "success": True,
                    "columns": columns,
                    "rows": rows,
                    "row_count": len(rows),
                    "execution_time_seconds": execution_time,
                    "limited": len(rows) >= max_rows,
                }

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"SQL execution failed: {e}")

            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "execution_time_seconds": execution_time,
            }

    def query(
        self,
        question: str,
        execute: bool = True,
        max_rows: Optional[int] = None,
        include_explanation: bool = False,
    ) -> Dict[str, Any]:
        """Full pipeline: generate SQL and optionally execute it.

        Args:
            question: Natural language question
            execute: Whether to execute the generated SQL
            max_rows: Maximum rows to return
            include_explanation: Include explanation of the SQL

        Returns:
            Dictionary with SQL, results, and metadata
        """
        # Generate SQL
        sql_result = self.generate_sql(question, include_explanation=include_explanation)

        response = {
            "question": question,
            "sql": sql_result["sql"],
            "explanation": sql_result.get("explanation"),
            "generation_attempts": sql_result["attempts"],
        }

        # Execute if requested
        if execute:
            execution_result = self.execute_sql(sql_result["sql"], max_rows)
            response["execution"] = execution_result
        else:
            response["execution"] = {"message": "SQL generated but not executed"}

        return response

    def _build_prompt(
        self, question: str, context: str, previous_attempts: List[Dict]
    ) -> str:
        """Build prompt for LLM.

        Args:
            question: User question
            context: Table context from RAG
            previous_attempts: Previous failed attempts

        Returns:
            Formatted prompt
        """
        prompt_parts = [
            "You are an expert SQL query generator for Microsoft SQL Server.",
            "",
            "TASK: Generate a SQL query to answer the following question.",
            f"QUESTION: {question}",
            "",
            "RELEVANT DATABASE SCHEMA:",
            context,
            "",
            "INSTRUCTIONS:",
            "1. Generate ONLY valid Microsoft SQL Server (T-SQL) syntax",
            "2. Use proper JOIN syntax when querying multiple tables",
            "3. Always use table aliases for clarity",
            "4. Include appropriate WHERE clauses for filtering",
            "5. Use TOP N for limiting results, not LIMIT",
            "6. Format the SQL query clearly with proper indentation",
            "7. Return ONLY the SQL query, wrapped in ```sql ``` code blocks",
        ]

        if settings.read_only_mode:
            prompt_parts.append(
                "8. IMPORTANT: Generate ONLY SELECT statements (no INSERT, UPDATE, DELETE, DROP, etc.)"
            )

        # Add previous attempts if any
        if previous_attempts:
            prompt_parts.append("\nPREVIOUS FAILED ATTEMPTS:")
            for i, attempt in enumerate(previous_attempts, 1):
                prompt_parts.append(f"Attempt {i}:")
                prompt_parts.append(f"  SQL: {attempt['sql']}")
                prompt_parts.append(f"  Error: {attempt['error']}")
            prompt_parts.append(
                "\nPlease fix the errors and generate a corrected query."
            )

        prompt_parts.append("\nSQL QUERY:")

        return "\n".join(prompt_parts)

    def _extract_sql(self, llm_response: str) -> str:
        """Extract SQL query from LLM response.

        Args:
            llm_response: Raw LLM response

        Returns:
            Extracted SQL query

        Raises:
            ValueError: If no SQL found
        """
        # Try to find SQL in code blocks
        sql_block_pattern = r"```sql\s*(.*?)\s*```"
        matches = re.findall(sql_block_pattern, llm_response, re.DOTALL | re.IGNORECASE)

        if matches:
            sql = matches[0].strip()
        else:
            # Try to find SQL without code blocks
            # Look for SELECT statements
            select_pattern = r"(SELECT\s+.*?)(?:\n\n|$)"
            matches = re.findall(
                select_pattern, llm_response, re.DOTALL | re.IGNORECASE
            )

            if matches:
                sql = matches[0].strip()
            else:
                # Last resort: use entire response
                sql = llm_response.strip()

        if not sql:
            raise ValueError("No SQL query found in LLM response")

        # Clean up the SQL
        sql = sql.strip().rstrip(";")

        return sql

    def _extract_explanation(self, llm_response: str) -> Optional[str]:
        """Extract explanation from LLM response.

        Args:
            llm_response: Raw LLM response

        Returns:
            Extracted explanation or None
        """
        # Remove SQL code blocks
        text = re.sub(r"```sql.*?```", "", llm_response, flags=re.DOTALL)
        explanation = text.strip()

        return explanation if explanation else None

    def _validate_sql(self, sql_query: str) -> None:
        """Validate SQL syntax without executing.

        Args:
            sql_query: SQL query to validate

        Raises:
            ValueError: If SQL is invalid
        """
        # Basic validation
        if not sql_query or not sql_query.strip():
            raise ValueError("Empty SQL query")

        # Check for SQL injection patterns (basic)
        dangerous_patterns = [
            r";\s*DROP",
            r";\s*DELETE",
            r";\s*TRUNCATE",
            r"--.*DROP",
            r"/\*.*DROP.*\*/",
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, sql_query, re.IGNORECASE):
                raise ValueError(f"Potentially dangerous SQL pattern detected: {pattern}")

        # Try to parse with SQLAlchemy (doesn't execute)
        try:
            text(sql_query)
        except Exception as e:
            raise ValueError(f"SQL syntax error: {e}")

    def _check_read_only(self, sql_query: str) -> None:
        """Check if query is read-only.

        Args:
            sql_query: SQL query to check

        Raises:
            ValueError: If query is not read-only
        """
        write_keywords = [
            "INSERT",
            "UPDATE",
            "DELETE",
            "DROP",
            "CREATE",
            "ALTER",
            "TRUNCATE",
            "MERGE",
        ]

        sql_upper = sql_query.upper()

        for keyword in write_keywords:
            if re.search(rf"\b{keyword}\b", sql_upper):
                raise ValueError(
                    f"Write operation '{keyword}' not allowed in read-only mode"
                )


if __name__ == "__main__":
    # Test the SQL agent
    agent = SQLAgent()

    test_questions = [
        "Show me the top 10 customers by total order value",
        "What products have low inventory?",
        "List all employees hired in the last year",
    ]

    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print("="*60)

        result = agent.query(question, execute=False, include_explanation=True)

        print(f"\nGenerated SQL:\n{result['sql']}")

        if result.get("explanation"):
            print(f"\nExplanation:\n{result['explanation']}")
