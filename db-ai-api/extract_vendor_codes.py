# -*- coding: utf-8 -*-
"""Extract unique vendor code prefixes from database for training data."""
import json
from collections import Counter
from sqlalchemy import create_engine, text
from config import settings
from loguru import logger


def extract_vendor_codes(prefix_length: int = 3, min_count: int = 10):
    """Extract unique vendor code prefixes from Product table.

    Args:
        prefix_length: Length of prefix to extract (default 3)
        min_count: Minimum count to include prefix (default 10)

    Returns:
        List of tuples (prefix, count, sample_products)
    """
    engine = create_engine(settings.database_url, pool_pre_ping=True)

    # Query to get vendor code prefixes with counts
    sql = f"""
    SELECT
        LEFT(VendorCode, {prefix_length}) as CodePrefix,
        COUNT(*) as ProductCount,
        STRING_AGG(CAST(Name as NVARCHAR(MAX)), ', ') WITHIN GROUP (ORDER BY Name) as SampleNames
    FROM (
        SELECT TOP 5 VendorCode, Name
        FROM dbo.Product
        WHERE Deleted = 0
          AND VendorCode IS NOT NULL
          AND LEN(VendorCode) >= {prefix_length}
          AND LEFT(VendorCode, {prefix_length}) = p.CodePrefix
    ) samples
    FROM dbo.Product p
    WHERE Deleted = 0
      AND VendorCode IS NOT NULL
      AND LEN(VendorCode) >= {prefix_length}
    GROUP BY LEFT(VendorCode, {prefix_length})
    HAVING COUNT(*) >= {min_count}
    ORDER BY COUNT(*) DESC
    """

    # Simpler query that works
    simple_sql = f"""
    SELECT TOP 100
        LEFT(VendorCode, {prefix_length}) as CodePrefix,
        COUNT(*) as ProductCount
    FROM dbo.Product
    WHERE Deleted = 0
      AND VendorCode IS NOT NULL
      AND LEN(VendorCode) >= {prefix_length}
    GROUP BY LEFT(VendorCode, {prefix_length})
    HAVING COUNT(*) >= {min_count}
    ORDER BY COUNT(*) DESC
    """

    logger.info(f"Extracting vendor code prefixes (length={prefix_length}, min_count={min_count})")

    try:
        with engine.connect() as conn:
            result = conn.execute(text(simple_sql))
            prefixes = []
            for row in result:
                prefixes.append({
                    "prefix": row[0],
                    "count": row[1]
                })

            logger.info(f"Found {len(prefixes)} unique prefixes")
            return prefixes
    except Exception as e:
        logger.error(f"Failed to extract vendor codes: {e}")
        return []


def get_sample_products_by_prefix(prefix: str, limit: int = 5):
    """Get sample products for a given prefix."""
    engine = create_engine(settings.database_url, pool_pre_ping=True)

    sql = f"""
    SELECT TOP {limit} ID, Name, VendorCode
    FROM dbo.Product
    WHERE Deleted = 0
      AND VendorCode LIKE :prefix
    ORDER BY VendorCode
    """

    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql), {"prefix": f"{prefix}%"})
            products = []
            for row in result:
                products.append({
                    "id": row[0],
                    "name": row[1],
                    "vendor_code": row[2]
                })
            return products
    except Exception as e:
        logger.error(f"Failed to get samples for {prefix}: {e}")
        return []


def save_vendor_codes(output_file: str = "training_data/vendor_codes.json"):
    """Extract and save vendor codes to JSON file."""
    # Get prefixes of different lengths
    prefixes_3 = extract_vendor_codes(prefix_length=3, min_count=50)
    prefixes_2 = extract_vendor_codes(prefix_length=2, min_count=100)

    # Get samples for top prefixes
    enriched_prefixes = []
    for p in prefixes_3[:30]:  # Top 30 prefixes
        samples = get_sample_products_by_prefix(p["prefix"], limit=3)
        enriched_prefixes.append({
            **p,
            "samples": samples
        })

    data = {
        "description": "Unique vendor code prefixes extracted from ConcordDb_v5.dbo.Product",
        "extracted_at": str(__import__("datetime").datetime.now()),
        "prefixes_3char": prefixes_3,
        "prefixes_2char": prefixes_2,
        "top_prefixes_with_samples": enriched_prefixes
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved vendor codes to {output_file}")

    # Print summary
    print("\n" + "="*60)
    print("TOP 20 VENDOR CODE PREFIXES (3 char):")
    print("="*60)
    for p in prefixes_3[:20]:
        print(f"  {p['prefix']:6} - {p['count']:,} products")

    print("\n" + "="*60)
    print("TOP 20 VENDOR CODE PREFIXES (2 char):")
    print("="*60)
    for p in prefixes_2[:20]:
        print(f"  {p['prefix']:4} - {p['count']:,} products")

    return data


if __name__ == "__main__":
    save_vendor_codes()
