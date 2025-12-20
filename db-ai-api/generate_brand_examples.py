# -*- coding: utf-8 -*-
"""Generate training examples for all vendor code brands."""
import json
from pathlib import Path


def generate_brand_examples(vendor_codes_file: str = "training_data/vendor_codes.json",
                            output_file: str = "training_data/templates/brands.json",
                            min_count: int = 500):
    """Generate training examples for all brands with min_count products.

    Args:
        vendor_codes_file: Path to extracted vendor codes
        output_file: Path to output JSON file
        min_count: Minimum product count to include brand
    """
    # Load vendor codes
    with open(vendor_codes_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Filter prefixes with enough products (3-char prefixes are more specific)
    brands = [p for p in data['prefixes_3char'] if p['count'] >= min_count]

    print(f"Generating examples for {len(brands)} brands (>= {min_count} products)")

    examples = []

    for i, brand in enumerate(brands):
        prefix = brand['prefix']
        count = brand['count']

        # Skip numeric-only prefixes (less useful for training)
        if prefix.isdigit():
            continue

        example_id = f"brand_{prefix.lower()}_{i+1:03d}"

        # Search example
        examples.append({
            "id": f"{example_id}_search",
            "category": "search",
            "complexity": "simple",
            "question_en": f"Find products with {prefix} code",
            "question_uk": f"Знайди товари з кодом {prefix}",
            "variations_en": [
                f"{prefix} products",
                f"show {prefix} items",
                f"products with {prefix}",
                f"list {prefix} products",
                f"show all {prefix} products",
                f"all {prefix} items"
            ],
            "variations_uk": [
                f"товари {prefix}",
                f"покажи {prefix}",
                f"продукти з {prefix}",
                f"покажи всі товари {prefix}",
                f"всі товари {prefix}",
                f"список товарів {prefix}",
                f"знайти {prefix}"
            ],
            "sql": f"SELECT TOP 100 ID, Name, VendorCode FROM dbo.Product WHERE Deleted = 0 AND VendorCode LIKE N'{prefix}%' ORDER BY VendorCode",
            "tables_used": ["Product"],
            "metadata": {"brand": prefix, "product_count": count}
        })

        # Sales ranking example (only for brands with >1000 products)
        if count >= 1000:
            examples.append({
                "id": f"{example_id}_top_sales",
                "category": "ranking",
                "complexity": "medium",
                "question_en": f"Top selling {prefix} products",
                "question_uk": f"Топ продажів {prefix}",
                "variations_en": [
                    f"best selling {prefix}",
                    f"{prefix} sales ranking",
                    f"most sold {prefix} products",
                    f"top {prefix} sellers"
                ],
                "variations_uk": [
                    f"найкращі продажі {prefix}",
                    f"лідери продажів {prefix}",
                    f"топ {prefix}",
                    f"найпопулярніші {prefix}"
                ],
                "sql": f"SELECT TOP 10 p.Name, p.VendorCode, SUM(oi.Qty) as TotalSold, COUNT(DISTINCT oi.OrderID) as OrderCount FROM dbo.Product p JOIN dbo.OrderItem oi ON oi.ProductID = p.ID WHERE p.Deleted = 0 AND oi.Deleted = 0 AND p.VendorCode LIKE N'{prefix}%' GROUP BY p.ID, p.Name, p.VendorCode ORDER BY TotalSold DESC",
                "tables_used": ["Product", "OrderItem"],
                "metadata": {"brand": prefix, "product_count": count}
            })

    # Create output structure
    output = {
        "domain": "brands",
        "description": f"Auto-generated brand-specific query examples for {len(brands)} brands",
        "generated_from": "vendor_codes.json",
        "min_product_count": min_count,
        "examples": examples
    }

    # Save to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nGenerated {len(examples)} examples for {len(brands)} brands")
    print(f"Saved to: {output_file}")

    # Print summary
    print("\n" + "="*60)
    print("BRANDS INCLUDED:")
    print("="*60)
    for brand in brands[:30]:  # First 30
        if not brand['prefix'].isdigit():
            print(f"  {brand['prefix']:6} - {brand['count']:,} products")

    return output


if __name__ == "__main__":
    generate_brand_examples(min_count=500)
