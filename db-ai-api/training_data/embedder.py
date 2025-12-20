"""Embed query examples into ChromaDB for RAG retrieval."""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions
from loguru import logger

# Multilingual model for Ukrainian + English support
MULTILINGUAL_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Import variation generator (optional, graceful fallback)
VARIATIONS_AVAILABLE = False
VariationGenerator = None
VariationConfig = None

try:
    from .variations import VariationGenerator, VariationConfig
    VARIATIONS_AVAILABLE = True
except ImportError:
    pass

if not VARIATIONS_AVAILABLE:
    try:
        # Fallback for running as main script
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        from variations import VariationGenerator, VariationConfig
        VARIATIONS_AVAILABLE = True
    except ImportError:
        pass


class QueryExampleEmbedder:
    """Embed query examples into ChromaDB for semantic search."""

    COLLECTION_NAME = "query_examples"

    def __init__(self, db_path: Optional[str] = None):
        """Initialize embedder with ChromaDB.

        Args:
            db_path: Path to ChromaDB storage (default: chroma_db_examples)
        """
        self.db_path = Path(db_path or Path(__file__).parent.parent / "chroma_db_examples_v2")
        self.db_path.mkdir(parents=True, exist_ok=True)

        self.templates_dir = Path(__file__).parent / "templates"

        # Initialize multilingual embedding function
        logger.info(f"Loading multilingual model: {MULTILINGUAL_MODEL}")
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=MULTILINGUAL_MODEL
        )

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        """Get or create ChromaDB collection."""
        try:
            collection = self.client.get_collection(
                name=self.COLLECTION_NAME,
                embedding_function=self.embedding_fn
            )
            logger.info(f"Loaded existing collection: {self.COLLECTION_NAME} ({collection.count()} items)")
            return collection
        except Exception:
            logger.info(f"Creating new collection: {self.COLLECTION_NAME}")
            return self.client.create_collection(
                name=self.COLLECTION_NAME,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )

    def _init_variation_generator(self):
        """Initialize variation generator if available."""
        if VARIATIONS_AVAILABLE and VariationGenerator:
            config = VariationConfig(representative_sample_size=15)
            self.variation_generator = VariationGenerator(config)
            logger.info("Variation generator initialized")
        else:
            self.variation_generator = None
            logger.warning("Variation generator not available")

    def _create_document(self, example: Dict[str, Any], domain: str) -> str:
        """Create searchable document from example.

        Combines all question variations for better semantic matching.

        Args:
            example: Query example dictionary
            domain: Domain name (products, sales, etc.)

        Returns:
            Combined document text for embedding
        """
        parts = []

        # Add main questions
        if example.get("question_en"):
            parts.append(example["question_en"])
        if example.get("question_uk"):
            parts.append(example["question_uk"])

        # Add English variations
        for variation in example.get("variations_en", []):
            parts.append(variation)

        # Add Ukrainian variations
        for variation in example.get("variations_uk", []):
            parts.append(variation)

        # Add domain context
        parts.append(f"Domain: {domain}")

        # Add table context
        tables = example.get("tables_used", [])
        if tables:
            parts.append(f"Tables: {', '.join(tables)}")

        return " | ".join(parts)

    def _create_metadata(self, example: Dict[str, Any], domain: str, filepath: str) -> Dict[str, Any]:
        """Create metadata for ChromaDB storage.

        Args:
            example: Query example dictionary
            domain: Domain name
            filepath: Source file path

        Returns:
            Metadata dictionary
        """
        return {
            "id": example.get("id", ""),
            "domain": domain,
            "category": example.get("category", ""),
            "complexity": example.get("complexity", ""),
            "question_en": example.get("question_en", ""),
            "question_uk": example.get("question_uk", ""),
            "sql": example.get("sql", ""),
            "tables_used": ",".join(example.get("tables_used", [])),
            "source_file": filepath,
            "embedded_at": datetime.now().isoformat(),
        }

    def embed_template_file(self, filepath: Path, include_variations: bool = False) -> Dict[str, Any]:
        """Embed all examples from a template file.

        Args:
            filepath: Path to template JSON file
            include_variations: If True, also embed variations

        Returns:
            Embedding summary
        """
        logger.info(f"Embedding template file: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            template = json.load(f)

        domain = template.get("domain", "unknown")
        examples = template.get("examples", [])

        documents = []
        metadatas = []
        ids = []
        variation_count = 0

        for example in examples:
            example_id = example.get("id", "")
            if not example_id:
                continue

            # Original document
            doc = self._create_document(example, domain)
            metadata = self._create_metadata(example, domain, str(filepath))
            doc_id = f"{domain}_{example_id}"

            documents.append(doc)
            metadatas.append(metadata)
            ids.append(doc_id)

            # Add variations if enabled and generator available
            if include_variations and self.variation_generator:
                variations = self.variation_generator.get_representative_variations(example)
                for var in variations:
                    # Skip original (already added)
                    if not var.transformation_types:
                        continue

                    var_doc = var.text
                    var_metadata = {
                        **metadata,
                        "is_variation": True,
                        "variation_type": ",".join(t.value for t in var.transformation_types),
                        "variation_language": var.language,
                        "original_id": example_id,
                    }
                    var_id = var.variation_id

                    documents.append(var_doc)
                    metadatas.append(var_metadata)
                    ids.append(var_id)
                    variation_count += 1

        # Add to collection
        if documents:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )
            logger.info(f"Embedded {len(documents)} items from {domain} ({variation_count} variations)")

        return {
            "domain": domain,
            "filepath": str(filepath),
            "examples_embedded": len(examples),
            "variations_embedded": variation_count,
            "total_documents": len(documents),
        }

    def embed_all_templates(
        self,
        force_refresh: bool = False,
        include_variations: bool = False
    ) -> Dict[str, Any]:
        """Embed all template files.

        Args:
            force_refresh: If True, recreate collection from scratch
            include_variations: If True, also embed variations

        Returns:
            Complete embedding summary
        """
        logger.info("Embedding all template files...")

        # Initialize variation generator if needed
        if include_variations:
            self._init_variation_generator()

        # Clear collection if force refresh
        if force_refresh:
            logger.info("Force refresh: recreating collection")
            try:
                self.client.delete_collection(name=self.COLLECTION_NAME)
            except Exception:
                pass
            self.collection = self.client.create_collection(
                name=self.COLLECTION_NAME,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )

        # Skip if already populated
        if self.collection.count() > 0 and not force_refresh:
            logger.info(f"Collection already has {self.collection.count()} items, skipping")
            return {
                "status": "skipped",
                "existing_count": self.collection.count(),
            }

        template_files = list(self.templates_dir.glob("*.json"))
        results = []
        total_examples = 0
        total_variations = 0

        for filepath in template_files:
            try:
                result = self.embed_template_file(filepath, include_variations=include_variations)
                results.append(result)
                total_examples += result["examples_embedded"]
                total_variations += result.get("variations_embedded", 0)
            except Exception as e:
                logger.error(f"Failed to embed {filepath}: {e}")
                results.append({
                    "filepath": str(filepath),
                    "error": str(e),
                })

        summary = {
            "status": "completed",
            "total_files": len(template_files),
            "total_examples": total_examples,
            "total_variations": total_variations,
            "total_documents": total_examples + total_variations,
            "collection_count": self.collection.count(),
            "file_results": results,
            "include_variations": include_variations,
            "embedded_at": datetime.now().isoformat(),
        }

        logger.info(
            f"Embedding complete: {total_examples} examples + {total_variations} variations "
            f"= {self.collection.count()} documents"
        )

        return summary

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection.

        Returns:
            Collection statistics
        """
        count = self.collection.count()

        # Get domain distribution
        if count > 0:
            results = self.collection.get(include=["metadatas"])
            domains = {}
            categories = {}
            complexities = {}

            for metadata in results["metadatas"]:
                domain = metadata.get("domain", "unknown")
                domains[domain] = domains.get(domain, 0) + 1

                category = metadata.get("category", "unknown")
                categories[category] = categories.get(category, 0) + 1

                complexity = metadata.get("complexity", "unknown")
                complexities[complexity] = complexities.get(complexity, 0) + 1

            return {
                "total_examples": count,
                "domains": domains,
                "categories": categories,
                "complexities": complexities,
            }

        return {"total_examples": 0}

    def delete_collection(self) -> None:
        """Delete the collection entirely."""
        try:
            self.client.delete_collection(name=self.COLLECTION_NAME)
            logger.info(f"Deleted collection: {self.COLLECTION_NAME}")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Embed query examples into ChromaDB")
    parser.add_argument("--variations", action="store_true", help="Include variations")
    parser.add_argument("--force", action="store_true", help="Force refresh collection")
    args = parser.parse_args()

    # Embed all templates
    embedder = QueryExampleEmbedder()

    print("\n" + "=" * 60)
    print("QUERY EXAMPLE EMBEDDER")
    print("=" * 60)
    print(f"Include variations: {args.variations}")
    print(f"Force refresh: {args.force}")

    # Force refresh to rebuild
    summary = embedder.embed_all_templates(
        force_refresh=args.force,
        include_variations=args.variations
    )

    print("\n" + "=" * 60)
    print("EMBEDDING SUMMARY")
    print("=" * 60)
    print(f"Status: {summary['status']}")
    print(f"Total files: {summary.get('total_files', 0)}")
    print(f"Total examples: {summary.get('total_examples', 0)}")
    print(f"Total variations: {summary.get('total_variations', 0)}")
    print(f"Total documents: {summary.get('total_documents', 0)}")
    print(f"Collection size: {summary.get('collection_count', 0)}")

    # Show stats
    stats = embedder.get_collection_stats()
    print("\nCollection Statistics:")
    print(f"  Total documents: {stats['total_examples']}")
    if stats.get('domains'):
        print("  By domain:")
        for domain, count in stats['domains'].items():
            print(f"    - {domain}: {count}")
