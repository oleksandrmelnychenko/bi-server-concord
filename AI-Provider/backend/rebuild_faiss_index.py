"""Rebuild FAISS index for schema RAG using multilingual-e5-large embeddings."""
import argparse
from pathlib import Path

from loguru import logger

from config import settings
from schema_extractor import SchemaExtractor
from table_selector import TableSelector


def set_setting(field: str, value):
    """Mutate pydantic settings object safely."""
    object.__setattr__(settings, field, value)


def main():
    parser = argparse.ArgumentParser(description="Rebuild FAISS index for table selection")
    parser.add_argument(
        "--index-type",
        choices=["flat", "ivf"],
        default=settings.faiss_index_type,
        help="FAISS index type (flat = exact, ivf = approximate/faster)",
    )
    parser.add_argument(
        "--nlist",
        type=int,
        default=settings.faiss_nlist,
        help="Number of clusters for IVF (ignored for flat)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if an index exists",
    )

    args = parser.parse_args()

    # Force FAISS backend + overrides
    set_setting("vector_store_backend", "faiss")
    set_setting("faiss_index_type", args.index_type)
    set_setting("faiss_nlist", args.nlist)

    logger.info(f"Building FAISS index at {settings.faiss_index_path} ({args.index_type})")
    extractor = SchemaExtractor()
    selector = TableSelector(schema_extractor=extractor)
    selector.index_schema(force_refresh=args.force)

    index_path = Path(settings.faiss_index_path) / "index.faiss"
    logger.info(f"FAISS index ready: {index_path}")


if __name__ == "__main__":
    main()
