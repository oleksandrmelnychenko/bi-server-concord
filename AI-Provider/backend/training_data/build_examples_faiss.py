"""Build a FAISS index of query examples from JSON templates.

This ingests the templates in training_data/templates, expands variations,
embeds them with the configured SentenceTransformer model, and writes
examples.faiss + examples_meta.json (consumed by QueryExampleRetriever).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import faiss  # type: ignore
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from config import settings


def _format_for_embedding(text: str, model_name: str) -> str:
    """Apply E5-style prefixing when needed."""
    if "e5" in model_name.lower():
        return f"passage: {text}"
    return text


def load_examples(template_dir: Path, include_variations: bool = True) -> List[Dict[str, Any]]:
    """Load examples + variations from JSON templates."""
    examples: List[Dict[str, Any]] = []
    for path in sorted(template_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        domain = data.get("domain", "general")
        for ex in data.get("examples", []):
            base = {
                "id": ex.get("id") or path.stem,
                "domain": domain,
                "category": ex.get("category", ""),
                "complexity": ex.get("complexity", ""),
                "question_en": ex.get("question_en", ""),
                "question_uk": ex.get("question_uk", ""),
                "sql": ex.get("sql", ""),
                "tables_used": ",".join(ex.get("tables_used", [])),
                "notes": ex.get("notes", ""),
                "source_file": path.name,
            }

            surfaces: List[str] = []
            for field in ("question_en", "question_uk"):
                val = ex.get(field)
                if val:
                    surfaces.append(val)
            if include_variations:
                for key in ("variations_en", "variations_uk"):
                    for val in ex.get(key, []) or []:
                        if val:
                            surfaces.append(val)

            for surface in surfaces:
                meta = dict(base)
                meta["surface"] = surface
                examples.append(meta)
    return examples


def build_faiss_index(examples: List[Dict[str, Any]], output_dir: Path) -> Dict[str, Any]:
    """Embed examples and write FAISS + metadata."""
    if not examples:
        raise ValueError("No examples loaded; nothing to index.")

    output_dir.mkdir(parents=True, exist_ok=True)
    index_file = output_dir / "examples.faiss"
    meta_file = output_dir / "examples_meta.json"

    model_name = settings.embedding_model
    logger.info(f"Embedding {len(examples)} examples with model: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [_format_for_embedding(ex["surface"], model_name) for ex in examples]
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(index_file))

    with meta_file.open("w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)

    logger.info(f"Wrote FAISS index to {index_file} and metadata to {meta_file}")
    return {
        "examples": len(examples),
        "dimension": dim,
        "index_path": str(index_file),
        "meta_path": str(meta_file),
    }


def main():
    parser = argparse.ArgumentParser(description="Build FAISS query examples index")
    parser.add_argument(
        "--templates",
        type=Path,
        default=Path(__file__).parent / "templates",
        help="Directory containing example templates (default: training_data/templates)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(settings.query_examples_db),
        help="Output directory for FAISS index + metadata (default: settings.QUERY_EXAMPLES_DB)",
    )
    parser.add_argument(
        "--no-variations",
        action="store_true",
        help="Exclude variations_en/variations_uk from the index",
    )
    args = parser.parse_args()

    examples = load_examples(args.templates, include_variations=not args.no_variations)
    stats = build_faiss_index(examples, args.output)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
