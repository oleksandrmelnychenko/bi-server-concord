"""Configuration management for the DB AI API."""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database Configuration
    db_host: str = "localhost"
    db_port: int = 1433
    db_name: str
    db_user: str
    db_password: str
    db_driver: str = "ODBC Driver 18 for SQL Server"

    # Ollama Configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "codellama:34b"

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True

    # Security
    read_only_mode: bool = True
    query_timeout: int = 60  # Increased for 34B model
    max_rows_returned: int = 1000

    # RAG Configuration
    vector_db_path: str = "./vector_db"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k_tables: int = 10

    # Query Examples RAG Configuration
    query_examples_db: str = "./chroma_db_examples_v2"
    query_examples_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    query_examples_top_k: int = 3

    # Join/template prompt budgeting
    precompute_max_pairs: int = 50000  # total join pairs to precompute (priority-ordered; high to approach full coverage)
    precomputed_per_query_limit: int = 120  # join templates to include per query
    join_rulebook_max_chars: int = 20000  # FK rulebook size budget in chars
    prompt_max_chars: int = 32000  # overall prompt budget guardrail

    # Logging
    log_level: str = "INFO"
    log_sql_queries: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = False

    @property
    def database_url(self) -> str:
        """Generate SQLAlchemy database URL."""
        # For pymssql (MSSQL)
        return (
            f"mssql+pymssql://{self.db_user}:{self.db_password}@"
            f"{self.db_host}:{self.db_port}/{self.db_name}"
        )

    @property
    def connection_string(self) -> str:
        """Generate raw connection string for pyodbc."""
        return (
            f"DRIVER={{{self.db_driver}}};"
            f"SERVER={self.db_host},{self.db_port};"
            f"DATABASE={self.db_name};"
            f"UID={self.db_user};"
            f"PWD={self.db_password};"
            f"TrustServerCertificate=yes;"
        )


# Global settings instance
settings = Settings()
