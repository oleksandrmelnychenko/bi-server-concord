"""Index database schema into ChromaDB for semantic search."""
import json
import chromadb
from chromadb.config import Settings as ChromaSettings
from pathlib import Path


def index_schema():
    """Index schema from schema_cache.json into ChromaDB."""
    print("="*70)
    print("Indexing Schema into ChromaDB")
    print("="*70)
    print()

    # Load schema
    print("Step 1: Loading schema from schema_cache.json...")
    with open('schema_cache.json', 'r') as f:
        schema = json.load(f)

    total_tables = len(schema['tables']) + len(schema['views'])
    print(f"Loaded {total_tables} tables/views")
    print()

    # Initialize ChromaDB
    print("Step 2: Initializing ChromaDB...")
    db_path = Path("./vector_db")
    db_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(db_path),
        settings=ChromaSettings(anonymized_telemetry=False),
    )

    collection_name = "tables_ConcordDb_v5"

    # Delete existing collection if exists
    try:
        client.delete_collection(name=collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except Exception:
        pass

    # Create new collection
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"Created collection: {collection_name}")
    print()

    # Prepare documents for indexing
    print("Step 3: Preparing documents...")
    documents = []
    metadatas = []
    ids = []

    # Index tables
    for table_name, table_info in schema['tables'].items():
        doc, metadata, doc_id = create_table_document(table_name, table_info)
        documents.append(doc)
        metadatas.append(metadata)
        ids.append(doc_id)

    # Index views
    for view_name, view_info in schema['views'].items():
        doc, metadata, doc_id = create_table_document(view_name, view_info)
        documents.append(doc)
        metadatas.append(metadata)
        ids.append(doc_id)

    print(f"Prepared {len(documents)} documents")
    print()

    # Index in batches
    print("Step 4: Indexing into ChromaDB...")
    batch_size = 50
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i : i + batch_size]
        batch_metas = metadatas[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]

        collection.add(
            documents=batch_docs,
            metadatas=batch_metas,
            ids=batch_ids
        )

        progress = min(i + batch_size, len(documents))
        print(f"  Indexed {progress}/{len(documents)} documents ({progress*100//len(documents)}%)")

    print()
    print("="*70)
    print("Indexing Complete!")
    print("="*70)
    print(f"Total documents indexed: {collection.count()}")
    print(f"Vector database location: {db_path}")
    print()

    # Test semantic search
    print("Testing semantic search...")
    print()

    test_queries = [
        "product prices and pricing",
        "customer orders and sales",
        "inventory and stock",
        "client information"
    ]

    for query in test_queries:
        results = collection.query(
            query_texts=[query],
            n_results=3
        )

        print(f"Query: '{query}'")
        if results and results['documents'] and results['documents'][0]:
            for i, (table_name, distance) in enumerate(zip(
                results['metadatas'][0],
                results['distances'][0]
            ), 1):
                relevance = 1 - distance
                print(f"  {i}. {table_name['table_name']} (relevance: {relevance:.3f})")
        print()

    print("="*70)


def create_table_document(table_name, table_info):
    """Create searchable document from table information."""
    # Create rich text description
    doc_parts = [
        f"Table: {table_name}",
        f"Type: {table_info['type']}",
        f"Row Count: {table_info.get('row_count', 0):,}",
    ]

    # Add column information
    column_descriptions = []
    for col in table_info.get('columns', []):
        col_desc = f"{col['name']} ({col['type']})"
        if not col.get('nullable', True):
            col_desc += " NOT NULL"
        column_descriptions.append(col_desc)

    if column_descriptions:
        doc_parts.append(f"Columns: {', '.join(column_descriptions[:20])}")  # First 20 cols

    # Add primary keys
    if table_info.get('primary_keys'):
        doc_parts.append(f"Primary Keys: {', '.join(table_info['primary_keys'])}")

    # Add foreign key relationships
    if table_info.get('foreign_keys'):
        fk_descriptions = []
        for fk in table_info['foreign_keys'][:5]:  # First 5 FKs
            fk_desc = f"{fk['column']} -> {fk['references_table']}"
            fk_descriptions.append(fk_desc)
        if fk_descriptions:
            doc_parts.append(f"Foreign Keys: {'; '.join(fk_descriptions)}")

    # Add sample data context
    if table_info.get('sample_data') and len(table_info['sample_data']) > 0:
        sample = table_info['sample_data'][0]
        sample_str = ", ".join(
            f"{k}={v}" for k, v in list(sample.items())[:5]
        )  # First 5 columns
        doc_parts.append(f"Sample: {sample_str}")

    document = "\n".join(doc_parts)

    # Metadata for filtering
    metadata = {
        "table_name": table_name,
        "type": table_info['type'],
        "row_count": table_info.get('row_count', 0),
        "column_count": len(table_info.get('columns', [])),
    }

    doc_id = f"{table_info['type']}_{table_name.replace('.', '_')}"

    return document, metadata, doc_id


if __name__ == "__main__":
    try:
        index_schema()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
