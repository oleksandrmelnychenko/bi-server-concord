"""SQL query validator for training data examples."""
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import settings


class SQLValidator:
    """Validate SQL queries against the database."""

    def __init__(self, engine: Optional[Engine] = None):
        """Initialize validator with database connection.

        Args:
            engine: SQLAlchemy engine (creates one if not provided)
        """
        self.engine = engine or create_engine(
            settings.database_url,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
        self.templates_dir = Path(__file__).parent / "templates"
        self.results_file = Path(__file__).parent / "data" / "validation_results.json"

    def validate_syntax(self, sql: str) -> Dict[str, Any]:
        """Validate SQL syntax without executing.

        Args:
            sql: SQL query to validate

        Returns:
            Validation result dictionary
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
        }

        # Check for empty query
        if not sql or not sql.strip():
            result["valid"] = False
            result["errors"].append("Empty SQL query")
            return result

        # Check for dangerous operations
        dangerous_patterns = [
            (r"\bDROP\b", "DROP statement detected"),
            (r"\bDELETE\b(?!\s+0)", "DELETE statement detected (not WHERE Deleted = 0)"),
            (r"\bTRUNCATE\b", "TRUNCATE statement detected"),
            (r"\bINSERT\b", "INSERT statement detected"),
            (r"\bUPDATE\b(?!\s+0)", "UPDATE statement detected"),
            (r";\s*--", "SQL injection pattern detected"),
        ]

        for pattern, msg in dangerous_patterns:
            if re.search(pattern, sql, re.IGNORECASE):
                result["warnings"].append(msg)

        # Check for common issues
        if "LIMIT" in sql.upper() and "TOP" not in sql.upper():
            result["warnings"].append("Use TOP instead of LIMIT for T-SQL")

        # Check for required Deleted = 0 filter
        tables_with_soft_delete = ["Product", "Client", "Order", "OrderItem", "Sale", "Debt", "Payment", "Stock"]
        for table in tables_with_soft_delete:
            if re.search(rf"\b{table}\b", sql, re.IGNORECASE):
                if "Deleted = 0" not in sql and "Deleted=0" not in sql:
                    result["warnings"].append(f"Missing 'Deleted = 0' filter for table {table}")

        return result

    def execute_and_verify(
        self, sql: str, timeout: int = 30
    ) -> Dict[str, Any]:
        """Execute SQL query and verify it returns results.

        Args:
            sql: SQL query to execute
            timeout: Query timeout in seconds

        Returns:
            Execution result dictionary
        """
        result = {
            "success": False,
            "row_count": 0,
            "columns": [],
            "execution_time_ms": 0,
            "error": None,
            "sample_data": None,
        }

        start_time = datetime.now()

        try:
            with self.engine.connect() as conn:
                # Set timeout
                conn.execute(text(f"SET LOCK_TIMEOUT {timeout * 1000}"))

                # Execute query
                query_result = conn.execute(text(sql))

                # Get results
                if query_result.returns_rows:
                    result["columns"] = list(query_result.keys())
                    rows = query_result.fetchall()
                    result["row_count"] = len(rows)
                    result["success"] = True

                    # Store sample data (first 3 rows)
                    if rows:
                        sample = []
                        for row in rows[:3]:
                            row_dict = {}
                            for i, col in enumerate(result["columns"]):
                                value = row[i]
                                if isinstance(value, datetime):
                                    value = value.isoformat()
                                elif isinstance(value, bytes):
                                    value = value.hex()[:50]
                                elif value is not None:
                                    value = str(value)[:100]
                                row_dict[col] = value
                            sample.append(row_dict)
                        result["sample_data"] = sample
                else:
                    result["success"] = True
                    result["row_count"] = 0

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"SQL execution failed: {e}")

        result["execution_time_ms"] = int(
            (datetime.now() - start_time).total_seconds() * 1000
        )

        return result

    def validate_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single query example.

        Args:
            example: Query example dictionary

        Returns:
            Validation result
        """
        sql = example.get("sql", "")
        example_id = example.get("id", "unknown")

        logger.info(f"Validating example: {example_id}")

        # Syntax validation
        syntax_result = self.validate_syntax(sql)

        # Execution validation
        exec_result = self.execute_and_verify(sql)

        return {
            "id": example_id,
            "question_en": example.get("question_en", ""),
            "question_uk": example.get("question_uk", ""),
            "sql": sql,
            "syntax_valid": syntax_result["valid"],
            "syntax_errors": syntax_result["errors"],
            "syntax_warnings": syntax_result["warnings"],
            "execution_success": exec_result["success"],
            "row_count": exec_result["row_count"],
            "columns": exec_result["columns"],
            "execution_time_ms": exec_result["execution_time_ms"],
            "execution_error": exec_result["error"],
            "sample_data": exec_result["sample_data"],
            "validated_at": datetime.now().isoformat(),
        }

    def validate_template_file(self, filepath: Path) -> Dict[str, Any]:
        """Validate all examples in a template file.

        Args:
            filepath: Path to template JSON file

        Returns:
            Validation summary
        """
        logger.info(f"Validating template file: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            template = json.load(f)

        domain = template.get("domain", "unknown")
        examples = template.get("examples", [])

        results = []
        success_count = 0
        error_count = 0

        for example in examples:
            result = self.validate_example(example)
            results.append(result)

            if result["execution_success"]:
                success_count += 1
            else:
                error_count += 1

        return {
            "domain": domain,
            "filepath": str(filepath),
            "total_examples": len(examples),
            "success_count": success_count,
            "error_count": error_count,
            "success_rate": success_count / len(examples) if examples else 0,
            "results": results,
            "validated_at": datetime.now().isoformat(),
        }

    def validate_all_templates(self) -> Dict[str, Any]:
        """Validate all template files in the templates directory.

        Returns:
            Complete validation summary
        """
        logger.info("Validating all template files...")

        # Ensure data directory exists
        data_dir = Path(__file__).parent / "data"
        data_dir.mkdir(exist_ok=True)

        template_files = list(self.templates_dir.glob("*.json"))
        all_results = []
        total_examples = 0
        total_success = 0
        total_errors = 0

        for filepath in template_files:
            try:
                result = self.validate_template_file(filepath)
                all_results.append(result)
                total_examples += result["total_examples"]
                total_success += result["success_count"]
                total_errors += result["error_count"]
            except Exception as e:
                logger.error(f"Failed to validate {filepath}: {e}")
                all_results.append({
                    "filepath": str(filepath),
                    "error": str(e),
                })

        summary = {
            "total_files": len(template_files),
            "total_examples": total_examples,
            "total_success": total_success,
            "total_errors": total_errors,
            "overall_success_rate": total_success / total_examples if total_examples else 0,
            "file_results": all_results,
            "validated_at": datetime.now().isoformat(),
        }

        # Save results
        with open(self.results_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Validation complete: {total_success}/{total_examples} examples passed "
            f"({summary['overall_success_rate']:.1%})"
        )

        return summary

    def get_failed_examples(self) -> List[Dict[str, Any]]:
        """Get list of failed examples from last validation.

        Returns:
            List of failed example results
        """
        if not self.results_file.exists():
            return []

        with open(self.results_file, "r", encoding="utf-8") as f:
            results = json.load(f)

        failed = []
        for file_result in results.get("file_results", []):
            for example in file_result.get("results", []):
                if not example.get("execution_success", False):
                    failed.append(example)

        return failed


if __name__ == "__main__":
    # Run validation
    validator = SQLValidator()
    summary = validator.validate_all_templates()

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total files: {summary['total_files']}")
    print(f"Total examples: {summary['total_examples']}")
    print(f"Successful: {summary['total_success']}")
    print(f"Failed: {summary['total_errors']}")
    print(f"Success rate: {summary['overall_success_rate']:.1%}")

    if summary['total_errors'] > 0:
        print("\nFailed examples:")
        for failed in validator.get_failed_examples():
            print(f"  - {failed['id']}: {failed.get('execution_error', 'Unknown error')}")
