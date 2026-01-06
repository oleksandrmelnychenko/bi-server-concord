"""Configuration management for the DB AI API."""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database Configuration
    db_host: str = "localhost"
    db_port: int = 1433
    db_name: str = "ConcordDb_v5"
    db_user: str = ""
    db_password: str = ""
    db_driver: str = "ODBC Driver 17 for SQL Server"
    db_trusted_connection: str = "yes"

    # Ollama Configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3-coder:30b"
    llm_temperature: float = 0.1
    llm_num_predict: int = 2048
    llm_top_k: int = 10
    llm_top_p: float = 0.9

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True

    # Security
    read_only_mode: bool = True
    query_timeout: int = 60  # Increased for 34B model
    max_rows_returned: int = 1000

    # Vector/RAG Configuration
    vector_db_path: str = "./vector_db"
    vector_store_backend: str = "faiss"  # faiss | chroma
    faiss_index_path: str = "./faiss_index"
    faiss_index_type: str = "flat"  # flat | ivf
    faiss_nlist: int = 100  # only used for ivf
    embedding_model: str = "intfloat/multilingual-e5-large"
    top_k_tables: int = 10

    # Full RAG Collection Configuration
    rag_chroma_dir: str = "./chroma_db_full"
    rag_collection_name: str = "concorddb_full"
    rag_embedding_model: str = "intfloat/multilingual-e5-large"

    # Query Examples RAG Configuration
    query_examples_db: str = "./faiss_examples"
    query_examples_collection: str = "query_examples"
    query_examples_model: str = "intfloat/multilingual-e5-large"
    query_examples_top_k: int = 3

    # Join/template prompt budgeting
    precompute_max_pairs: int = 5000  # total join pairs to precompute (reduced for faster startup)
    precomputed_per_query_limit: int = 120  # join templates to include per query
    join_rulebook_max_chars: int = 20000  # FK rulebook size budget in chars
    prompt_max_chars: int = 32000  # overall prompt budget guardrail

    # Prompt Templates
    sql_prompt_path: str = "./prompts/sql_prompt.txt"
    system_prompt_path: str = "./prompts/system_prompt.txt"

    # Logging
    log_level: str = "INFO"
    log_sql_queries: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = False

    @property
    def database_url(self) -> str:
        """Generate SQLAlchemy database URL using pyodbc."""
        # Use pyodbc with proper escaping for special characters
        from urllib.parse import quote_plus
        if self.db_trusted_connection.lower() in ('yes', 'true', '1'):
            conn_str = (
                f"DRIVER={self.db_driver};"
                f"SERVER={self.db_host};"
                f"DATABASE={self.db_name};"
                f"Trusted_Connection=yes;"
                f"TrustServerCertificate=yes;"
            )
        else:
            conn_str = (
                f"DRIVER={self.db_driver};"
                f"SERVER={self.db_host},{self.db_port};"
                f"DATABASE={self.db_name};"
                f"UID={self.db_user};"
                f"PWD={self.db_password};"
                f"TrustServerCertificate=yes;"
            )
        return f"mssql+pyodbc:///?odbc_connect={quote_plus(conn_str)}"

    @property
    def connection_string(self) -> str:
        """Generate raw connection string for pyodbc."""
        if self.db_trusted_connection.lower() in ('yes', 'true', '1'):
            return (
                f"DRIVER={{{self.db_driver}}};"
                f"SERVER={self.db_host},{self.db_port};"
                f"DATABASE={self.db_name};"
                f"Trusted_Connection=yes;"
                f"TrustServerCertificate=yes;"
            )
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
