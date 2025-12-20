# -*- coding: utf-8 -*-
"""Add training examples with correct JOIN patterns based on actual DB structure."""
import json

# New examples with CORRECT JOIN patterns
# KEY RELATIONSHIPS:
# - Client.ID <- ClientAgreement.ClientID
# - ClientAgreement.ID <- Order.ClientAgreementID
# - Order.ID <- OrderItem.OrderID
# - Product.ID <- OrderItem.ProductID
# - Order.ID <- Sale.OrderID
# - ClientAgreement.ID <- Sale.ClientAgreementID

new_examples = {
    "customers": [
        {
            "id": "client_region_khmelnytskyi",
            "category": "filter",
            "complexity": "simple",
            "question_en": "Clients from Khmelnytskyi region",
            "question_uk": "Клієнти з Хмельницького",
            "variations_en": ["customers from Khmelnytskyi", "Khmelnytskyi clients"],
            "variations_uk": ["клієнт із хмельницького", "хмельницькі клієнти", "покупці з хмельниччини"],
            "sql": "SELECT TOP 20 c.ID, c.Name, c.LegalAddress, c.MobileNumber FROM dbo.Client c WHERE c.Deleted = 0 AND (c.LegalAddress LIKE N'%Хмельниц%' OR c.ActualAddress LIKE N'%Хмельниц%') ORDER BY c.Name",
            "tables_used": ["Client"]
        },
        {
            "id": "client_orders_correct_join",
            "category": "ranking",
            "complexity": "complex",
            "question_en": "Top clients by number of orders",
            "question_uk": "Топ клієнтів за кількістю замовлень",
            "variations_en": ["best customers by orders", "most orders"],
            "variations_uk": ["найактивніші клієнти", "клієнти з найбільше замовлень"],
            "sql": "SELECT TOP 10 c.Name, COUNT(DISTINCT o.ID) as OrderCount FROM dbo.Client c JOIN dbo.ClientAgreement ca ON ca.ClientID = c.ID JOIN dbo.[Order] o ON o.ClientAgreementID = ca.ID WHERE c.Deleted = 0 AND o.Deleted = 0 GROUP BY c.ID, c.Name ORDER BY OrderCount DESC",
            "tables_used": ["Client", "ClientAgreement", "Order"]
        },
        {
            "id": "client_products_purchased",
            "category": "listing",
            "complexity": "complex",
            "question_en": "Products purchased by client",
            "question_uk": "Товари що купляв клієнт",
            "variations_en": ["what did client buy", "client purchase history"],
            "variations_uk": ["що купляв клієнт", "історія покупок клієнта", "товари клієнта"],
            "sql": "SELECT DISTINCT p.Name, p.VendorCode, SUM(oi.Qty) as TotalQty FROM dbo.Client c JOIN dbo.ClientAgreement ca ON ca.ClientID = c.ID JOIN dbo.[Order] o ON o.ClientAgreementID = ca.ID JOIN dbo.OrderItem oi ON oi.OrderID = o.ID JOIN dbo.Product p ON p.ID = oi.ProductID WHERE c.Deleted = 0 AND c.Name LIKE N'%keyword%' GROUP BY p.ID, p.Name, p.VendorCode ORDER BY TotalQty DESC",
            "tables_used": ["Client", "ClientAgreement", "Order", "OrderItem", "Product"]
        },
        {
            "id": "client_total_purchases",
            "category": "aggregation",
            "complexity": "complex",
            "question_en": "Client total purchase amount",
            "question_uk": "Загальна сума покупок клієнта",
            "variations_en": ["how much did client spend", "client spending"],
            "variations_uk": ["скільки витратив клієнт", "обсяг покупок клієнта"],
            "sql": "SELECT c.Name, SUM(oi.Qty * oi.PricePerItem) as TotalAmount FROM dbo.Client c JOIN dbo.ClientAgreement ca ON ca.ClientID = c.ID JOIN dbo.[Order] o ON o.ClientAgreementID = ca.ID JOIN dbo.OrderItem oi ON oi.OrderID = o.ID WHERE c.Deleted = 0 AND o.Deleted = 0 AND oi.Deleted = 0 AND c.Name LIKE N'%keyword%' GROUP BY c.ID, c.Name",
            "tables_used": ["Client", "ClientAgreement", "Order", "OrderItem"]
        },
        {
            "id": "client_region_generic",
            "category": "filter",
            "complexity": "simple",
            "question_en": "Clients from specific region",
            "question_uk": "Клієнти з області",
            "variations_en": ["customers by region", "regional clients"],
            "variations_uk": ["клієнти з регіону", "регіональні клієнти"],
            "sql": "SELECT TOP 30 c.ID, c.Name, c.LegalAddress FROM dbo.Client c WHERE c.Deleted = 0 AND (c.LegalAddress LIKE N'%keyword%' OR c.ActualAddress LIKE N'%keyword%' OR c.DeliveryAddress LIKE N'%keyword%') ORDER BY c.Name",
            "tables_used": ["Client"]
        }
    ],
    "sales": [
        {
            "id": "sales_product_top_sem",
            "category": "ranking",
            "complexity": "medium",
            "question_en": "Top SEM products by sales",
            "question_uk": "Топ продажів SEM",
            "variations_en": ["best selling SEM", "SEM sales ranking"],
            "variations_uk": ["покажи топ продажів SEM", "найкращі продажі SEM", "лідери продажів SEM"],
            "sql": "SELECT TOP 10 p.Name, p.VendorCode, SUM(oi.Qty) as TotalSold, COUNT(DISTINCT oi.OrderID) as OrderCount FROM dbo.Product p JOIN dbo.OrderItem oi ON oi.ProductID = p.ID WHERE p.Deleted = 0 AND oi.Deleted = 0 AND p.VendorCode LIKE N'SEM%' GROUP BY p.ID, p.Name, p.VendorCode ORDER BY TotalSold DESC",
            "tables_used": ["Product", "OrderItem"]
        },
        {
            "id": "sales_product_top_vendorcode",
            "category": "ranking",
            "complexity": "medium",
            "question_en": "Top products by vendor code prefix",
            "question_uk": "Топ товарів за кодом",
            "variations_en": ["sales by product code", "product sales by code"],
            "variations_uk": ["продажі за кодом товару", "топ товарів за артикулом"],
            "sql": "SELECT TOP 10 p.Name, p.VendorCode, SUM(oi.Qty) as TotalSold FROM dbo.Product p JOIN dbo.OrderItem oi ON oi.ProductID = p.ID WHERE p.Deleted = 0 AND oi.Deleted = 0 AND p.VendorCode LIKE N'%keyword%' GROUP BY p.ID, p.Name, p.VendorCode ORDER BY TotalSold DESC",
            "tables_used": ["Product", "OrderItem"]
        },
        {
            "id": "sales_by_client_correct",
            "category": "aggregation",
            "complexity": "complex",
            "question_en": "Sales by client",
            "question_uk": "Продажі по клієнтах",
            "variations_en": ["client sales", "revenue by customer"],
            "variations_uk": ["виручка по клієнтах", "продажі клієнтів"],
            "sql": "SELECT TOP 20 c.Name, COUNT(DISTINCT s.ID) as SalesCount, SUM(oi.Qty * oi.PricePerItem) as TotalAmount FROM dbo.Client c JOIN dbo.ClientAgreement ca ON ca.ClientID = c.ID JOIN dbo.Sale s ON s.ClientAgreementID = ca.ID JOIN dbo.[Order] o ON o.ID = s.OrderID JOIN dbo.OrderItem oi ON oi.OrderID = o.ID WHERE c.Deleted = 0 AND s.Deleted = 0 GROUP BY c.ID, c.Name ORDER BY TotalAmount DESC",
            "tables_used": ["Client", "ClientAgreement", "Sale", "Order", "OrderItem"]
        },
        {
            "id": "sales_order_items_detail",
            "category": "listing",
            "complexity": "medium",
            "question_en": "Order items with prices",
            "question_uk": "Позиції замовлення з цінами",
            "variations_en": ["order details", "items in order"],
            "variations_uk": ["деталі замовлення", "товари в замовленні"],
            "sql": "SELECT p.Name, p.VendorCode, oi.Qty, oi.PricePerItem, (oi.Qty * oi.PricePerItem) as Total FROM dbo.OrderItem oi JOIN dbo.Product p ON p.ID = oi.ProductID WHERE oi.Deleted = 0 AND oi.OrderID = @OrderID",
            "tables_used": ["OrderItem", "Product"]
        },
        {
            "id": "sales_top_products_overall",
            "category": "ranking",
            "complexity": "medium",
            "question_en": "Top selling products overall",
            "question_uk": "Топ продажів товарів",
            "variations_en": ["best sellers", "most sold products"],
            "variations_uk": ["найкращі продажі", "топ продаж", "лідери продажів"],
            "sql": "SELECT TOP 10 p.Name, p.VendorCode, SUM(oi.Qty) as TotalSold, COUNT(DISTINCT oi.OrderID) as OrderCount FROM dbo.Product p JOIN dbo.OrderItem oi ON oi.ProductID = p.ID WHERE p.Deleted = 0 AND oi.Deleted = 0 GROUP BY p.ID, p.Name, p.VendorCode ORDER BY TotalSold DESC",
            "tables_used": ["Product", "OrderItem"]
        }
    ],
    "products": [
        {
            "id": "product_sales_ranking_correct",
            "category": "ranking",
            "complexity": "medium",
            "question_en": "Top 10 products by sales quantity",
            "question_uk": "Топ 10 товарів за кількістю продажів",
            "variations_en": ["best sellers", "most sold products", "popular products"],
            "variations_uk": ["найпопулярніші товари", "лідери продажів", "топ продаж"],
            "sql": "SELECT TOP 10 p.Name, p.VendorCode, SUM(oi.Qty) as TotalSold FROM dbo.Product p JOIN dbo.OrderItem oi ON oi.ProductID = p.ID WHERE p.Deleted = 0 AND oi.Deleted = 0 GROUP BY p.ID, p.Name, p.VendorCode ORDER BY TotalSold DESC",
            "tables_used": ["Product", "OrderItem"]
        },
        {
            "id": "product_by_code_search",
            "category": "search",
            "complexity": "simple",
            "question_en": "Find products by vendor code",
            "question_uk": "Знайти товар за артикулом",
            "variations_en": ["search by code", "product lookup by code"],
            "variations_uk": ["пошук за кодом", "товар з кодом", "знайти за артикулом"],
            "sql": "SELECT TOP 20 p.ID, p.Name, p.VendorCode, p.Description FROM dbo.Product p WHERE p.Deleted = 0 AND p.VendorCode LIKE N'%keyword%' ORDER BY p.VendorCode",
            "tables_used": ["Product"]
        },
        {
            "id": "product_never_sold",
            "category": "filter",
            "complexity": "medium",
            "question_en": "Products never sold",
            "question_uk": "Товари без продажів",
            "variations_en": ["unsold products", "products with no sales"],
            "variations_uk": ["непродані товари", "товари без замовлень"],
            "sql": "SELECT p.ID, p.Name, p.VendorCode FROM dbo.Product p WHERE p.Deleted = 0 AND NOT EXISTS (SELECT 1 FROM dbo.OrderItem oi WHERE oi.ProductID = p.ID AND oi.Deleted = 0)",
            "tables_used": ["Product", "OrderItem"]
        }
    ]
}

# Add to existing templates
for domain, examples in new_examples.items():
    filepath = f'training_data/templates/{domain}.json'
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except:
        data = {"domain": domain, "examples": []}

    existing_ids = {e['id'] for e in data.get('examples', [])}
    added = 0
    for ex in examples:
        if ex['id'] not in existing_ids:
            data['examples'].append(ex)
            added += 1
            print(f"Added: {ex['id']}")

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"{domain}: added {added} new examples")

print("\nDone adding examples!")
