"""Rebuild FAISS index for training examples using multilingual-e5-large."""
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from config import settings


class TrainingExamplesFAISS:
    """FAISS-based storage for SQL training examples."""

    def __init__(self, index_dir: Path = None):
        self.index_dir = Path(index_dir or settings.query_examples_db)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.index_dir / "examples.faiss"
        self.meta_file = self.index_dir / "examples_meta.json"

    def rebuild(
        self,
        model: SentenceTransformer,
        examples: List[Dict[str, Any]],
    ) -> int:
        """Rebuild FAISS index from training examples.

        Args:
            model: SentenceTransformer model for embeddings
            examples: List of training examples

        Returns:
            Number of indexed examples
        """
        import faiss

        if not examples:
            raise ValueError("No examples provided")

        # Create search texts: combine English + Ukrainian questions + variations
        texts = []
        for ex in examples:
            # Primary: Ukrainian question (most common user input)
            parts = [ex.get("question_uk", "")]
            # Secondary: English question
            if ex.get("question_en"):
                parts.append(ex["question_en"])
            # Add key variations
            for var in ex.get("variations_uk", [])[:2]:
                parts.append(var)
            for var in ex.get("variations_en", [])[:1]:
                parts.append(var)

            # E5 format: "passage: <text>" for documents (training examples)
            text = "passage: " + " | ".join(filter(None, parts))
            texts.append(text)

        logger.info(f"Embedding {len(texts)} training examples...")

        # Generate embeddings
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32,
        )
        embeddings = np.array(embeddings, dtype=np.float32)

        # Create FAISS index (inner product for normalized vectors = cosine similarity)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        # Save index
        faiss.write_index(index, str(self.index_file))

        # Save metadata
        metadata = []
        for ex in examples:
            metadata.append({
                "id": ex.get("id", ""),
                "domain": ex.get("domain", ""),
                "category": ex.get("category", ""),
                "complexity": ex.get("complexity", ""),
                "question_en": ex.get("question_en", ""),
                "question_uk": ex.get("question_uk", ""),
                "sql": ex.get("sql", ""),
                "tables_used": ",".join(ex.get("tables_used", [])),
                "notes": ex.get("notes", ""),
            })

        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"FAISS training examples index saved: {len(examples)} examples, {dim}d")
        return len(examples)


def load_all_templates(templates_dir: Path) -> List[Dict[str, Any]]:
    """Load all training examples from template files.

    Args:
        templates_dir: Directory containing JSON template files

    Returns:
        List of all examples with domain attached
    """
    all_examples = []

    for json_file in sorted(templates_dir.glob("*.json")):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            domain = data.get("domain", json_file.stem)
            examples = data.get("examples", [])

            # Attach domain to each example
            for ex in examples:
                ex["domain"] = domain

            all_examples.extend(examples)
            logger.info(f"Loaded {len(examples)} examples from {json_file.name}")

        except Exception as e:
            logger.error(f"Failed to load {json_file}: {e}")

    return all_examples


def main():
    """Rebuild training examples FAISS index."""
    import argparse

    parser = argparse.ArgumentParser(description="Rebuild FAISS index for training examples")
    parser.add_argument(
        "--model",
        default=settings.embedding_model,
        help=f"Embedding model (default: {settings.embedding_model})",
    )
    parser.add_argument(
        "--output",
        default=settings.query_examples_db,
        help=f"Output directory (default: {settings.query_examples_db})",
    )
    args = parser.parse_args()

    # Load embedding model
    logger.info(f"Loading embedding model: {args.model}")
    model = SentenceTransformer(args.model)
    logger.info(f"Model dimension: {model.get_sentence_embedding_dimension()}d")

    # Load all templates
    templates_dir = Path(__file__).parent / "training_data" / "templates"
    logger.info(f"Loading templates from: {templates_dir}")

    examples = load_all_templates(templates_dir)
    logger.info(f"Total examples loaded: {len(examples)}")

    if not examples:
        logger.error("No examples found!")
        return

    # Rebuild FAISS index
    store = TrainingExamplesFAISS(index_dir=Path(args.output))
    count = store.rebuild(model, examples)

    logger.info(f"Successfully indexed {count} training examples")
    logger.info(f"Index location: {store.index_file}")
    logger.info(f"Metadata location: {store.meta_file}")


if __name__ == "__main__":
    main()
