"""
RAG Query Engine
Handles semantic search and answer generation with Ukrainian support
"""
import json
import os
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import ollama
from utils.language_utils import detect_language, has_ukrainian


class RAGQueryEngine:
    """RAG query engine with Ukrainian support."""

    def __init__(self,
                 embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 llm_model: str = "qwen2.5:14b",
                 chroma_dir: str = "chroma_db_full",
                 collection_name: str = "concorddb_full",
                 system_prompt_path: str = "prompts/system_prompt_uk.txt"):
        """
        Initialize RAG Query Engine.

        Args:
            embedding_model: Embedding model name
            llm_model: LLM model name (Ollama)
            chroma_dir: ChromaDB directory
            collection_name: Collection name
            system_prompt_path: Path to system prompt
        """
        print("\n" + "="*60)
        print("INITIALIZING RAG QUERY ENGINE")
        print("="*60 + "\n")

        embedding_model = os.getenv("RAG_EMBEDDING_MODEL", embedding_model)
        chroma_dir = os.getenv("RAG_CHROMA_DIR", chroma_dir)
        collection_name = os.getenv("RAG_COLLECTION_NAME", collection_name)

        self.embedding_model_name = embedding_model
        self.collection_name = collection_name
        self.chroma_dir = chroma_dir

        # Load embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        print(f"[OK] Embedding model loaded")

        # Connect to ChromaDB
        print(f"\nConnecting to ChromaDB: {chroma_dir}")
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(anonymized_telemetry=False)
        )

        # Get collection
        try:
            self.collection = self.chroma_client.get_collection(name=collection_name)
            doc_count = self.collection.count()
            print(f"[OK] Loaded collection: {collection_name}")
            print(f"  Documents: {doc_count}")
        except Exception as e:
            print(f"[ERROR] Error loading collection: {e}")
            raise

        self._validate_embedding_dimension()

        # Set LLM model
        self.llm_model = llm_model
        print(f"\nLLM Model: {llm_model}")

        # Load system prompt
        self.system_prompt_template = self._load_system_prompt(system_prompt_path)
        print(f"[OK] Loaded system prompt from: {system_prompt_path}")

        # Load schema description
        self.schema_description = self._load_schema_description()
        print(f"[OK] Loaded schema description\n")

    def _load_system_prompt(self, path: str) -> str:
        """Load system prompt template."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            # Fallback prompt
            return """Ти — інтелектуальний асистент бази даних.
Відповідай українською на основі наданого контексту.

КОНТЕКСТ:
{context}

ЗАПИТАННЯ:
{question}

ВІДПОВІДЬ:"""

    def _load_schema_description(self) -> str:
        """Load schema description from cache."""
        try:
            with open("schema_cache.json", "r", encoding="utf-8") as f:
                schema = json.load(f)

            # Create brief description
            table_list = list(schema.keys())[:50]  # First 50 tables
            description = f"База даних містить {len(schema)} таблиць, включаючи: " + \
                         ", ".join(table_list[:20])

            return description
        except:
            return "База даних ConcordDb"

    def _get_collection_embedding_dimension(self) -> Optional[int]:
        try:
            sample = self.collection.get(limit=1, include=["embeddings"])
        except Exception as exc:
            print(f"[WARN] Failed to read collection embeddings: {exc}")
            return None

        embeddings = sample.get("embeddings")
        if embeddings is None or len(embeddings) == 0 or embeddings[0] is None:
            return None

        try:
            return len(embeddings[0])
        except Exception:
            return None

    def _validate_embedding_dimension(self) -> None:
        model_dim = self.embedding_model.get_sentence_embedding_dimension()
        collection_dim = self._get_collection_embedding_dimension()
        if collection_dim is None:
            return
        if collection_dim != model_dim:
            raise ValueError(
                f"Embedding dimension mismatch: collection '{self.collection_name}' "
                f"({self.chroma_dir}) has {collection_dim}, model "
                f"'{self.embedding_model_name}' has {model_dim}. "
                "Reindex the collection or set RAG_EMBEDDING_MODEL/RAG_COLLECTION_NAME to match."
            )


    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Semantic search in vector database.

        Args:
            query: Search query
            n_results: Number of results

        Returns:
            Search results with documents and metadata
        """
        # Create query embedding
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True
        )[0]

        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        return {
            "query": query,
            "n_results": len(results["ids"][0]) if results["ids"] else 0,
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else []
        }

    def format_context(self, search_results: Dict[str, Any],
                      max_context_length: int = 4000) -> str:
        """
        Format search results into context for LLM.

        Args:
            search_results: Results from search()
            max_context_length: Maximum context length in characters

        Returns:
            Formatted context string
        """
        context_parts = []
        current_length = 0

        for i, (doc, metadata, distance) in enumerate(zip(
            search_results["documents"],
            search_results["metadatas"],
            search_results["distances"]
        ), 1):
            # Format document
            doc_text = f"=== ДЖЕРЕЛО {i} (релевантність: {1-distance:.2f}) ===\n"
            doc_text += f"Таблиця: {metadata.get('table', 'Unknown')}\n"
            doc_text += f"{doc}\n\n"

            # Check length
            if current_length + len(doc_text) > max_context_length:
                break

            context_parts.append(doc_text)
            current_length += len(doc_text)

        return "\n".join(context_parts)

    def generate_answer(self, query: str, context: str) -> Dict[str, Any]:
        """
        Generate answer using LLM.

        Args:
            query: User question
            context: Context from search

        Returns:
            Generated answer and metadata
        """
        # Build prompt
        prompt = self.system_prompt_template.format(
            schema_description=self.schema_description,
            context=context,
            question=query
        )

        # Generate with Ollama
        try:
            response = ollama.generate(
                model=self.llm_model,
                prompt=prompt,
                options={
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "num_predict": 1000
                }
            )

            answer = response["response"].strip()

            return {
                "success": True,
                "answer": answer,
                "model": self.llm_model,
                "error": None
            }

        except Exception as e:
            return {
                "success": False,
                "answer": None,
                "model": self.llm_model,
                "error": str(e)
            }

    def query(self, question: str,
             n_results: int = 5,
             return_context: bool = False) -> Dict[str, Any]:
        """
        Full RAG query: search + generate answer.

        Args:
            question: User question
            n_results: Number of search results
            return_context: Include context in response

        Returns:
            Complete query response
        """
        # Detect language
        language = detect_language(question)

        # Search
        search_results = self.search(question, n_results=n_results)

        # Format context
        context = self.format_context(search_results)

        # Generate answer
        answer_result = self.generate_answer(question, context)

        # Build response
        response = {
            "question": question,
            "language": language,
            "answer": answer_result["answer"],
            "success": answer_result["success"],
            "error": answer_result["error"],
            "n_results": search_results["n_results"],
            "model": self.llm_model
        }

        if return_context:
            response["context"] = context
            response["search_results"] = search_results

        return response

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "collection_documents": self.collection.count(),
            "embedding_model": self.embedding_model_name,
            "llm_model": self.llm_model,
            "embedding_dimension": self.embedding_model.get_sentence_embedding_dimension()
        }


def main():
    """Test RAG engine."""
    import argparse

    parser = argparse.ArgumentParser(description="Test RAG Query Engine")
    parser.add_argument("--query", "-q", required=True, help="Query text")
    parser.add_argument("--n-results", type=int, default=5, help="Number of search results")
    parser.add_argument("--model", default="qwen2:7b", help="LLM model")
    parser.add_argument("--show-context", action="store_true", help="Show context")

    args = parser.parse_args()

    # Initialize engine
    engine = RAGQueryEngine(llm_model=args.model)

    # Query
    print("\n" + "="*60)
    print("QUERYING RAG ENGINE")
    print("="*60)
    print(f"Query: {args.query}\n")

    result = engine.query(
        question=args.query,
        n_results=args.n_results,
        return_context=args.show_context
    )

    # Print results
    print("="*60)
    print("ANSWER")
    print("="*60)
    print(f"Language: {result['language']}")
    print(f"Success: {result['success']}")
    print(f"Results used: {result['n_results']}")
    print(f"\n{result['answer']}\n")

    if args.show_context and "context" in result:
        print("="*60)
        print("📚 CONTEXT")
        print("="*60)
        print(result["context"])

    if result["error"]:
        print(f"\n⚠️  Error: {result['error']}")


if __name__ == "__main__":
    main()
