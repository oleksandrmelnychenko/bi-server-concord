"""
Hybrid SQL + RAG Agent
Intelligently routes queries to either SQL generation or RAG search
"""
import json
import re
from typing import Dict, Any, Optional, Literal
import ollama
from utils.language_utils import detect_language, has_ukrainian, extract_ukrainian_keywords
from rag_engine import RAGQueryEngine
from db_pool import get_connection
import pymssql


class HybridAgent:
    """Hybrid agent that routes between SQL and RAG."""

    def __init__(self,
                 sql_model: str = "qwen2:7b",
                 rag_model: str = "qwen2:7b",
                 sql_prompt_path: str = "prompts/sql_prompt_uk.txt",
                 chroma_dir: str = "chroma_db"):
        """
        Initialize Hybrid Agent.

        Args:
            sql_model: Model for SQL generation
            rag_model: Model for RAG answers
            sql_prompt_path: Path to SQL prompt
            chroma_dir: ChromaDB directory
        """
        print("\n" + "="*60)
        print("🚀 INITIALIZING HYBRID AGENT")
        print("="*60 + "\n")

        self.sql_model = sql_model
        self.rag_model = rag_model

        # Load SQL prompt
        with open(sql_prompt_path, "r", encoding="utf-8") as f:
            self.sql_prompt_template = f.read()

        # Load schema
        with open("schema_cache.json", "r", encoding="utf-8") as f:
            self.schema = json.load(f)

        # Load Ukrainian dictionary
        with open("dictionaries/uk_column_mapping.json", "r", encoding="utf-8") as f:
            self.uk_dictionary = json.load(f)

        # Initialize RAG engine
        print("📦 Initializing RAG engine...")
        self.rag_engine = RAGQueryEngine(
            llm_model=rag_model,
            chroma_dir=chroma_dir
        )

        print("\n✓ Hybrid Agent initialized\n")

    def classify_query(self, question: str) -> Dict[str, Any]:
        """
        Classify query to determine routing.

        Args:
            question: User question

        Returns:
            Classification result
        """
        question_lower = question.lower()

        # SQL indicators
        sql_keywords = [
            "скільки", "кількість", "count", "порахуй",  # Counting
            "покажи", "список", "show", "list",  # Listing
            "топ", "top", "найбільш", "найменш",  # Top/ranking
            "сума", "sum", "загалом", "total",  # Aggregation
            "середнє", "average", "avg",  # Average
            "максимум", "мінімум", "max", "min",  # Min/max
            "за період", "за місяць", "за рік",  # Time-based
            "сьогодні", "вчора", "цього місяця"  # Time filters
        ]

        # RAG indicators
        rag_keywords = [
            "що таке", "як працює", "поясни", "опиши",  # Explanations
            "розкажи про", "інформація про",  # Information requests
            "чому", "навіщо", "why", "how",  # Why/how questions
            "порівняй", "відмінність", "compare",  # Comparisons
            "рекомендації", "порада", "recommend"  # Recommendations
        ]

        # Check for SQL indicators
        sql_score = sum(1 for kw in sql_keywords if kw in question_lower)

        # Check for RAG indicators
        rag_score = sum(1 for kw in rag_keywords if kw in question_lower)

        # Specific patterns
        has_aggregation = any(kw in question_lower for kw in
                             ["скільки", "count", "сума", "sum", "середнє", "avg"])
        has_top = "топ" in question_lower or "top" in question_lower
        has_explanation = any(kw in question_lower for kw in
                             ["що таке", "поясни", "як працює", "чому"])

        # Decision logic
        if has_explanation:
            mode = "rag"
            confidence = 0.9
        elif has_aggregation or has_top:
            mode = "sql"
            confidence = 0.8
        elif sql_score > rag_score:
            mode = "sql"
            confidence = 0.6 + (sql_score * 0.1)
        elif rag_score > sql_score:
            mode = "rag"
            confidence = 0.6 + (rag_score * 0.1)
        else:
            # Default to RAG for ambiguous queries
            mode = "rag"
            confidence = 0.5

        return {
            "mode": mode,
            "confidence": confidence,
            "sql_score": sql_score,
            "rag_score": rag_score,
            "explanation": f"Selected {mode.upper()} mode with {confidence:.0%} confidence"
        }

    def generate_sql(self, question: str) -> Dict[str, Any]:
        """
        Generate SQL query from question.

        Args:
            question: User question

        Returns:
            SQL query and metadata
        """
        # Build schema string (simplified)
        schema_str = self._format_schema_for_prompt()

        # Build prompt
        prompt = self.sql_prompt_template.format(
            schema=schema_str,
            ukrainian_dictionary=json.dumps(self.uk_dictionary, ensure_ascii=False, indent=2),
            question=question
        )

        # Generate SQL
        try:
            response = ollama.generate(
                model=self.sql_model,
                prompt=prompt,
                options={
                    "temperature": 0.1,
                    "top_p": 0.9
                }
            )

            sql = response["response"].strip()

            # Clean up SQL
            sql = self._clean_sql(sql)

            return {
                "success": True,
                "sql": sql,
                "model": self.sql_model,
                "error": None
            }

        except Exception as e:
            return {
                "success": False,
                "sql": None,
                "model": self.sql_model,
                "error": str(e)
            }

    def execute_sql(self, sql: str) -> Dict[str, Any]:
        """
        Execute SQL query safely.

        Args:
            sql: SQL query

        Returns:
            Query results
        """
        # Safety check
        if not self._is_sql_safe(sql):
            return {
                "success": False,
                "results": None,
                "error": "Unsafe SQL query detected"
            }

        conn = get_connection()
        cursor = conn.cursor(as_dict=True)

        try:
            cursor.execute(sql)
            results = cursor.fetchall()

            # Convert to JSON-serializable format
            json_results = []
            for row in results:
                json_row = {}
                for key, value in row.items():
                    json_row[key] = str(value) if value is not None else None
                json_results.append(json_row)

            return {
                "success": True,
                "results": json_results,
                "row_count": len(json_results),
                "error": None
            }

        except Exception as e:
            return {
                "success": False,
                "results": None,
                "error": str(e)
            }

        finally:
            cursor.close()
            conn.close()

    def query(self, question: str, mode: Optional[str] = None) -> Dict[str, Any]:
        """
        Process query using hybrid approach.

        Args:
            question: User question
            mode: Force mode ('sql', 'rag', or None for auto)

        Returns:
            Complete response
        """
        # Detect language
        language = detect_language(question)

        # Classify if mode not specified
        if mode is None:
            classification = self.classify_query(question)
            mode = classification["mode"]
        else:
            classification = {
                "mode": mode,
                "confidence": 1.0,
                "explanation": f"Forced {mode.upper()} mode"
            }

        # Route to appropriate handler
        if mode == "sql":
            return self._handle_sql_query(question, language, classification)
        else:
            return self._handle_rag_query(question, language, classification)

    def _handle_sql_query(self, question: str, language: str,
                         classification: Dict[str, Any]) -> Dict[str, Any]:
        """Handle SQL-based query."""
        # Generate SQL
        sql_result = self.generate_sql(question)

        if not sql_result["success"]:
            return {
                "question": question,
                "language": language,
                "mode": "sql",
                "classification": classification,
                "success": False,
                "answer": "Вибачте, не вдалося згенерувати SQL запит.",
                "error": sql_result["error"]
            }

        # Execute SQL
        exec_result = self.execute_sql(sql_result["sql"])

        if not exec_result["success"]:
            return {
                "question": question,
                "language": language,
                "mode": "sql",
                "classification": classification,
                "sql": sql_result["sql"],
                "success": False,
                "answer": "Вибачте, помилка виконання SQL запиту.",
                "error": exec_result["error"]
            }

        # Format answer
        answer = self._format_sql_answer(question, exec_result["results"], language)

        return {
            "question": question,
            "language": language,
            "mode": "sql",
            "classification": classification,
            "sql": sql_result["sql"],
            "results": exec_result["results"],
            "row_count": exec_result["row_count"],
            "success": True,
            "answer": answer
        }

    def _handle_rag_query(self, question: str, language: str,
                         classification: Dict[str, Any]) -> Dict[str, Any]:
        """Handle RAG-based query."""
        # Query RAG engine
        rag_result = self.rag_engine.query(question, n_results=5)

        return {
            "question": question,
            "language": language,
            "mode": "rag",
            "classification": classification,
            "success": rag_result["success"],
            "answer": rag_result["answer"],
            "n_results": rag_result["n_results"],
            "error": rag_result.get("error")
        }

    def _format_sql_answer(self, question: str, results: list,
                          language: str) -> str:
        """Format SQL results as natural language answer."""
        if not results:
            return "На жаль, за вашим запитом нічого не знайдено."

        # Single value result (count, sum, etc.)
        if len(results) == 1 and len(results[0]) == 1:
            value = list(results[0].values())[0]
            return f"Результат: {value}"

        # Multiple rows
        if len(results) <= 10:
            # Show all results
            answer = f"Знайдено {len(results)} записів:\n\n"
            for i, row in enumerate(results, 1):
                answer += f"{i}. "
                answer += ", ".join(f"{k}: {v}" for k, v in row.items() if v)
                answer += "\n"
            return answer
        else:
            # Show summary
            return f"Знайдено {len(results)} записів. Перші 10:\n\n" + \
                   "\n".join(f"{i}. {list(row.values())[0]}" for i, row in enumerate(results[:10], 1))

    def _format_schema_for_prompt(self, max_tables: int = 50) -> str:
        """Format schema for SQL prompt."""
        schema_parts = []

        for i, (table_name, table_info) in enumerate(self.schema.items()):
            if i >= max_tables:
                break

            columns = table_info.get("columns", [])
            col_list = ", ".join([col["column_name"] for col in columns[:10]])

            schema_parts.append(f"[dbo].[{table_name}] ({col_list})")

        return "\n".join(schema_parts)

    def _clean_sql(self, sql: str) -> str:
        """Clean up generated SQL."""
        # Remove markdown code blocks
        sql = re.sub(r"```sql\n?", "", sql)
        sql = re.sub(r"```\n?", "", sql)

        # Remove comments
        sql = re.sub(r"--.*$", "", sql, flags=re.MULTILINE)

        # Clean whitespace
        sql = sql.strip()

        return sql

    def _is_sql_safe(self, sql: str) -> bool:
        """Check if SQL is safe to execute."""
        sql_upper = sql.upper()

        # Block dangerous keywords
        dangerous = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "TRUNCATE", "EXEC", "EXECUTE"]

        for keyword in dangerous:
            if keyword in sql_upper:
                return False

        return True


def main():
    """Test hybrid agent."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Hybrid Agent")
    parser.add_argument("--query", "-q", required=True, help="Query text")
    parser.add_argument("--mode", choices=["sql", "rag"], help="Force mode")

    args = parser.parse_args()

    # Initialize agent
    agent = HybridAgent()

    # Query
    print("\n" + "="*60)
    print("🔍 PROCESSING QUERY")
    print("="*60)
    print(f"Query: {args.query}\n")

    result = agent.query(question=args.query, mode=args.mode)

    # Print results
    print("="*60)
    print(f"📝 RESULT ({result['mode'].upper()} mode)")
    print("="*60)
    print(f"Language: {result['language']}")
    print(f"Success: {result['success']}")
    print(f"Classification: {result['classification']['explanation']}")
    print(f"\nAnswer:\n{result['answer']}\n")

    if result.get("sql"):
        print(f"SQL Query:\n{result['sql']}\n")

    if result.get("error"):
        print(f"⚠️  Error: {result['error']}")


if __name__ == "__main__":
    main()
