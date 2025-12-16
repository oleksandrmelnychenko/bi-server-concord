#!/usr/bin/env python3
"""
RAG CLI Management Tool
Command-line interface for managing the RAG system
"""
import argparse
import sys
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress
from rich import print as rprint

console = Console()


def cmd_extract(args):
    """Extract data from database."""
    from data_extractor import DataExtractor

    console.print("\n[bold cyan]📦 Data Extraction[/bold cyan]\n")

    extractor = DataExtractor(output_dir=args.output_dir)

    max_tables = 5 if args.test else args.max_tables

    if args.test:
        console.print("[yellow]⚠️  Test mode: extracting only 5 tables[/yellow]\n")

    stats = extractor.extract_all_data(max_tables=max_tables)

    # Show summary
    console.print("\n[bold green]✅ Extraction Complete![/bold green]\n")
    console.print(f"Tables processed: {stats['extracted_tables']}/{stats['total_tables']}")
    console.print(f"Documents created: {stats['total_documents']}")
    console.print(f"Output: {args.output_dir}/extracted_documents.json\n")


def cmd_embed(args):
    """Embed documents into vector database."""
    from embedder import RAGEmbedder

    console.print("\n[bold cyan]🎯 Document Embedding[/bold cyan]\n")

    # Load documents
    console.print(f"Loading documents from: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        documents = json.load(f)

    console.print(f"Loaded {len(documents)} documents")

    # Test mode
    if args.test:
        documents = documents[:100]
        console.print(f"[yellow]⚠️  Test mode: embedding only 100 documents[/yellow]")

    # Initialize embedder
    embedder = RAGEmbedder(
        chroma_dir=args.chroma_dir,
        collection_name=args.collection
    )

    # Reset if requested
    if args.reset:
        embedder.reset_collection()

    # Embed
    stats = embedder.embed_documents(
        documents=documents,
        batch_size=args.batch_size
    )

    # Show summary
    console.print("\n[bold green]✅ Embedding Complete![/bold green]\n")
    collection_stats = embedder.get_collection_stats()
    console.print(f"Collection: {collection_stats['collection_name']}")
    console.print(f"Total documents: {collection_stats['document_count']}")
    console.print(f"Embedding dimension: {collection_stats['embedding_dimension']}\n")


def cmd_query(args):
    """Query the RAG system."""
    from hybrid_agent import HybridAgent

    console.print("\n[bold cyan]🔍 Querying RAG System[/bold cyan]\n")

    # Initialize agent
    agent = HybridAgent()

    # Query
    console.print(f"[bold]Question:[/bold] {args.query}\n")

    result = agent.query(question=args.query, mode=args.mode)

    # Display result
    console.print(Panel(
        result['answer'],
        title=f"Answer ({result['mode'].upper()} mode)",
        border_style="green" if result['success'] else "red"
    ))

    # Show details
    console.print(f"\n[dim]Language: {result['language']}[/dim]")
    console.print(f"[dim]Classification: {result['classification']['explanation']}[/dim]")

    if result.get('sql'):
        console.print(f"\n[bold]SQL Query:[/bold]\n{result['sql']}")

    if result.get('error'):
        console.print(f"\n[red]⚠️  Error: {result['error']}[/red]")


def cmd_search(args):
    """Semantic search in RAG database."""
    from rag_engine import RAGQueryEngine

    console.print("\n[bold cyan]🔎 Semantic Search[/bold cyan]\n")

    # Initialize engine
    engine = RAGQueryEngine()

    # Search
    console.print(f"[bold]Query:[/bold] {args.query}\n")

    result = engine.search(query=args.query, n_results=args.n_results)

    # Display results
    console.print(f"Found {result['n_results']} results:\n")

    for i, (doc, metadata, distance) in enumerate(zip(
        result['documents'],
        result['metadatas'],
        result['distances']
    ), 1):
        relevance = 1 - distance
        console.print(f"[bold cyan]{i}.[/bold cyan] [green]Relevance: {relevance:.2%}[/green]")
        console.print(f"   Table: {metadata['table']}")
        console.print(f"   ID: {metadata['primary_key_value']}")
        console.print(f"   {doc[:200]}...\n")


def cmd_stats(args):
    """Show system statistics."""
    console.print("\n[bold cyan]📊 System Statistics[/bold cyan]\n")

    # Schema stats
    if Path("schema_cache.json").exists():
        with open("schema_cache.json", "r") as f:
            schema = json.load(f)

        table = Table(title="Database Schema")
        table.add_column("Item", style="cyan")
        table.add_column("Count", style="green")

        table.add_row("Tables", str(len(schema)))

        console.print(table)
        console.print()

    # Extraction stats
    if Path("data/extraction_stats.json").exists():
        with open("data/extraction_stats.json", "r") as f:
            stats = json.load(f)

        table = Table(title="Data Extraction")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Tables extracted", str(stats.get('extracted_tables', 0)))
        table.add_row("Documents created", str(stats.get('total_documents', 0)))
        table.add_row("Errors", str(len(stats.get('errors', []))))

        console.print(table)
        console.print()

    # Vector DB stats
    if Path("chroma_db").exists():
        try:
            from embedder import RAGEmbedder

            embedder = RAGEmbedder()
            coll_stats = embedder.get_collection_stats()

            table = Table(title="Vector Database")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Collection", coll_stats['collection_name'])
            table.add_row("Documents", str(coll_stats['document_count']))
            table.add_row("Embedding dim", str(coll_stats['embedding_dimension']))

            console.print(table)
        except:
            console.print("[yellow]⚠️  ChromaDB not initialized[/yellow]")

    console.print()


def cmd_pipeline(args):
    """Run full pipeline: extract -> embed."""
    console.print("\n[bold cyan]🚀 Running Full Pipeline[/bold cyan]\n")

    # Step 1: Extract
    console.print("[bold]Step 1/2: Extracting data...[/bold]\n")
    from data_extractor import DataExtractor

    extractor = DataExtractor(output_dir="data")
    max_tables = 5 if args.test else None
    extract_stats = extractor.extract_all_data(max_tables=max_tables)

    console.print(f"\n✓ Extracted {extract_stats['total_documents']} documents\n")

    # Step 2: Embed
    console.print("[bold]Step 2/2: Embedding documents...[/bold]\n")
    from embedder import RAGEmbedder

    # Load documents
    with open("data/extracted_documents.json", "r", encoding="utf-8") as f:
        documents = json.load(f)

    if args.test:
        documents = documents[:100]

    embedder = RAGEmbedder()
    if args.reset:
        embedder.reset_collection()

    embed_stats = embedder.embed_documents(documents, batch_size=32)

    console.print(f"\n✓ Embedded {embed_stats['embedded_documents']} documents\n")

    # Summary
    console.print("\n[bold green]✅ Pipeline Complete![/bold green]\n")
    console.print(f"Documents extracted: {extract_stats['total_documents']}")
    console.print(f"Documents embedded: {embed_stats['embedded_documents']}\n")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RAG System Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline (test mode)
  python rag_cli.py pipeline --test

  # Extract data from database
  python rag_cli.py extract --test

  # Embed documents
  python rag_cli.py embed --test --reset

  # Query the system
  python rag_cli.py query "Скільки клієнтів з Києва?"

  # Semantic search
  python rag_cli.py search "клієнти Київ"

  # Show statistics
  python rag_cli.py stats
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract data from database")
    extract_parser.add_argument("--output-dir", default="data", help="Output directory")
    extract_parser.add_argument("--max-tables", type=int, help="Max tables to process")
    extract_parser.add_argument("--test", action="store_true", help="Test mode (5 tables)")

    # Embed command
    embed_parser = subparsers.add_parser("embed", help="Embed documents")
    embed_parser.add_argument("--input", default="data/extracted_documents.json",
                             help="Input documents JSON")
    embed_parser.add_argument("--chroma-dir", default="chroma_db",
                             help="ChromaDB directory")
    embed_parser.add_argument("--collection", default="concorddb_ukrainian",
                             help="Collection name")
    embed_parser.add_argument("--batch-size", type=int, default=32,
                             help="Batch size")
    embed_parser.add_argument("--reset", action="store_true",
                             help="Reset collection")
    embed_parser.add_argument("--test", action="store_true",
                             help="Test mode (100 docs)")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query RAG system")
    query_parser.add_argument("query", help="Query text")
    query_parser.add_argument("--mode", choices=["sql", "rag", "auto"], default="auto",
                             help="Query mode")

    # Search command
    search_parser = subparsers.add_parser("search", help="Semantic search")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--n-results", type=int, default=5,
                              help="Number of results")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show system statistics")

    # Pipeline command
    pipeline_parser = subparsers.add_parser("pipeline", help="Run full pipeline")
    pipeline_parser.add_argument("--test", action="store_true",
                                help="Test mode (5 tables, 100 docs)")
    pipeline_parser.add_argument("--reset", action="store_true",
                                help="Reset vector DB")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Route to command handler
    try:
        if args.command == "extract":
            cmd_extract(args)
        elif args.command == "embed":
            cmd_embed(args)
        elif args.command == "query":
            cmd_query(args)
        elif args.command == "search":
            cmd_search(args)
        elif args.command == "stats":
            cmd_stats(args)
        elif args.command == "pipeline":
            cmd_pipeline(args)

    except KeyboardInterrupt:
        console.print("\n\n[yellow]⚠️  Interrupted by user[/yellow]\n")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]❌ Error: {e}[/red]\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
