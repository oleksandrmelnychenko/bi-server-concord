"""
Pydantic schemas for natural language query endpoints
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class NLQueryRequest(BaseModel):
    """Request for natural language query"""
    query: str = Field(..., min_length=5, description="Natural language query")
    return_sql: bool = Field(True, description="Return generated SQL")
    max_rows: int = Field(100, ge=1, le=1000, description="Maximum rows to return")


class NLQueryResponse(BaseModel):
    """Response with query results"""
    query: str
    sql: Optional[str] = None
    results: List[Dict[str, Any]]
    row_count: int
    columns: List[str]
    execution_time_ms: float
    generated_at: datetime

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Show me top 10 customers by total purchase amount",
                "sql": "SELECT customer_id, customer_name, SUM(total_amount) as total FROM orders GROUP BY customer_id ORDER BY total DESC LIMIT 10",
                "results": [
                    {
                        "customer_id": "CUST-001",
                        "customer_name": "Acme Corp",
                        "total": 125430.50
                    }
                ],
                "row_count": 10,
                "columns": ["customer_id", "customer_name", "total"],
                "execution_time_ms": 45.2,
                "generated_at": "2025-01-15T10:30:00Z"
            }
        }
