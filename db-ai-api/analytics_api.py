"""
Analytics API - SQL queries for business reports
Connects directly to MSSQL database for aggregate calculations
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
import pyodbc
import uvicorn
import os
from datetime import datetime

# Database configuration (supports Docker environment variables)
DB_HOST = os.getenv('DB_HOST', '78.152.175.67')
DB_PORT = int(os.getenv('DB_PORT', '1433'))
DB_USER = os.getenv('DB_USER', 'ef_migrator')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'Grimm_jow92')
DB_NAME = os.getenv('DB_NAME', 'ConcordDb_v5')
DB_DRIVER = os.getenv('DB_DRIVER', 'ODBC Driver 18 for SQL Server')

app = FastAPI(
    title="Analytics API - Business Reports",
    description="SQL-based analytics for ConcordDb",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




def cursor_to_dicts(cursor):
    """Convert cursor rows to list of dictionaries."""
    columns = [column[0] for column in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]

def get_connection():
    """Create database connection using pyodbc with TLS support."""
    conn_str = (
        f'DRIVER={{{DB_DRIVER}}};'
        f'SERVER={DB_HOST},{DB_PORT};'
        f'DATABASE={DB_NAME};'
        f'UID={DB_USER};'
        f'PWD={DB_PASSWORD};'
        f'TrustServerCertificate=yes;'
        f'Connection Timeout=60;'
    )
    return pyodbc.connect(conn_str)


@app.get("/")
async def root():
    """API info."""
    return {
        "service": "Analytics API",
        "database": DB_NAME,
        "endpoints": [
            "/sales/yearly - Продажі по роках",
            "/sales/monthly?year=2025 - Продажі по місяцях",
            "/sales/by-region - Продажі по регіонах",
            "/products/top - Топ товарів",
            "/clients/top - Топ клієнтів",
            "/debts/summary - Сума боргів"
        ]
    }


@app.get("/health")
async def health():
    """Health check endpoint for Docker."""
    try:
        conn = get_connection()
        conn.close()
        return {"status": "healthy", "database": DB_NAME}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database connection failed: {str(e)}")


@app.get("/sales/yearly")
async def sales_yearly():
    """Продажі по роках (скільки товару продали за рік)."""
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Get yearly sales summary
        cursor.execute("""
            SELECT
                YEAR(s.Created) as year,
                COUNT(DISTINCT s.ID) as total_sales,
                COUNT(DISTINCT s.OrderID) as total_orders,
                COUNT(DISTINCT oi.ID) as total_items
            FROM dbo.Sale s
            LEFT JOIN dbo.[Order] o ON s.OrderID = o.ID
            LEFT JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
            WHERE s.Deleted = 0
            GROUP BY YEAR(s.Created)
            ORDER BY year DESC
        """)

        results = cursor_to_dicts(cursor)
        conn.close()

        return {
            "title": "Продажі по роках",
            "data": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sales/monthly")
async def sales_monthly(year: int = Query(default=2025, description="Рік")):
    """Продажі по місяцях за вказаний рік."""
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                MONTH(s.Created) as month,
                COUNT(DISTINCT s.ID) as total_sales,
                COUNT(DISTINCT s.OrderID) as total_orders
            FROM dbo.Sale s
            WHERE s.Deleted = 0 AND YEAR(s.Created) = %s
            GROUP BY MONTH(s.Created)
            ORDER BY month
        """, (year,))

        results = cursor_to_dicts(cursor)
        conn.close()

        month_names = {
            1: "Січень", 2: "Лютий", 3: "Березень", 4: "Квітень",
            5: "Травень", 6: "Червень", 7: "Липень", 8: "Серпень",
            9: "Вересень", 10: "Жовтень", 11: "Листопад", 12: "Грудень"
        }

        for r in results:
            r["month_name"] = month_names.get(r["month"], "")

        return {
            "title": f"Продажі по місяцях за {year} рік",
            "year": year,
            "data": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sales/by-region")
async def sales_by_region(year: Optional[int] = Query(default=None, description="Рік (опціонально)")):
    """Продажі по регіонах."""
    try:
        conn = get_connection()
        cursor = conn.cursor()

        year_filter = ""
        params = ()
        if year:
            year_filter = "AND YEAR(s.Created) = %s"
            params = (year,)

        query = f"""
            SELECT
                c.RegionID as region_id,
                COUNT(DISTINCT s.ID) as total_sales,
                COUNT(DISTINCT c.ID) as unique_clients
            FROM dbo.Sale s
            JOIN dbo.[Order] o ON s.OrderID = o.ID
            JOIN dbo.ClientAgreement ca ON o.ClientAgreementID = ca.ID
            JOIN dbo.Client c ON ca.ClientID = c.ID
            WHERE s.Deleted = 0 {year_filter}
            GROUP BY c.RegionID
            ORDER BY total_sales DESC
        """

        cursor.execute(query, params)
        results = cursor_to_dicts(cursor)
        conn.close()

        return {
            "title": f"Продажі по регіонах" + (f" за {year} рік" if year else ""),
            "data": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/products/top")
async def top_products(
    limit: int = Query(default=20, ge=1, le=100, description="Кількість"),
    year: Optional[int] = Query(default=None, description="Рік")
):
    """Топ товарів за кількістю продажів."""
    try:
        conn = get_connection()
        cursor = conn.cursor()

        year_filter = ""
        params = (limit,)
        if year:
            year_filter = "AND YEAR(oi.Created) = %s"
            params = (year, limit)

        query = f"""
            SELECT TOP (%s)
                p.ID as product_id,
                p.Name as product_name,
                SUM(oi.Qty) as total_qty,
                COUNT(DISTINCT oi.OrderID) as order_count
            FROM dbo.OrderItem oi
            JOIN dbo.Product p ON oi.ProductID = p.ID
            WHERE oi.Deleted = 0 {year_filter}
            GROUP BY p.ID, p.Name
            ORDER BY total_qty DESC
        """

        if year:
            cursor.execute(query, (year, limit))
        else:
            cursor.execute(query, (limit,))

        results = cursor_to_dicts(cursor)
        conn.close()

        return {
            "title": f"Топ {limit} товарів" + (f" за {year} рік" if year else ""),
            "data": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/clients/top")
async def top_clients(
    limit: int = Query(default=20, ge=1, le=100, description="Кількість"),
    year: Optional[int] = Query(default=None, description="Рік")
):
    """Топ клієнтів за кількістю замовлень."""
    try:
        conn = get_connection()
        cursor = conn.cursor()

        year_filter = ""
        if year:
            year_filter = f"AND YEAR(s.Created) = {year}"

        cursor.execute(f"""
            SELECT TOP ({limit})
                c.ID as client_id,
                c.Name as client_name,
                c.RegionID as region_id,
                COUNT(DISTINCT s.ID) as total_sales,
                COUNT(DISTINCT o.ID) as total_orders
            FROM dbo.Client c
            JOIN dbo.ClientAgreement ca ON c.ID = ca.ClientID
            JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
            JOIN dbo.Sale s ON o.ID = s.OrderID
            WHERE c.Deleted = 0 AND s.Deleted = 0 {year_filter}
            GROUP BY c.ID, c.Name, c.RegionID
            ORDER BY total_sales DESC
        """)

        results = cursor_to_dicts(cursor)
        conn.close()

        return {
            "title": f"Топ {limit} клієнтів" + (f" за {year} рік" if year else ""),
            "data": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debts/summary")
async def debts_summary():
    """Загальна сума боргів."""
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                COUNT(*) as total_debts,
                SUM(Total) as total_amount,
                AVG(Total) as avg_amount,
                MIN(Total) as min_amount,
                MAX(Total) as max_amount
            FROM dbo.Debt
            WHERE Deleted = 0 AND Total > 0
        """)

        columns = [column[0] for column in cursor.description]
        summary = dict(zip(columns, cursor.fetchone())) if cursor.description else None

        # Get by year
        cursor.execute("""
            SELECT
                YEAR(Created) as year,
                COUNT(*) as debt_count,
                SUM(Total) as total_amount
            FROM dbo.Debt
            WHERE Deleted = 0 AND Total > 0
            GROUP BY YEAR(Created)
            ORDER BY year DESC
        """)

        by_year = cursor_to_dicts(cursor)
        conn.close()

        return {
            "title": "Статистика боргів",
            "summary": summary,
            "by_year": by_year
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/orderitems/yearly")
async def orderitems_yearly():
    """Кількість проданого товару по роках."""
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                YEAR(oi.Created) as year,
                COUNT(*) as total_items,
                SUM(oi.Qty) as total_quantity,
                COUNT(DISTINCT oi.ProductID) as unique_products,
                COUNT(DISTINCT oi.OrderID) as total_orders
            FROM dbo.OrderItem oi
            WHERE oi.Deleted = 0
            GROUP BY YEAR(oi.Created)
            ORDER BY year DESC
        """)

        results = cursor_to_dicts(cursor)
        conn.close()

        return {
            "title": "Кількість проданого товару по роках",
            "data": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/products/vendor-codes")
async def vendor_codes_with_sales(
    limit: int = Query(default=50, ge=1, le=500, description="Кількість"),
    prefix: Optional[str] = Query(default=None, description="Фільтр по префіксу (напр. SEM)")
):
    """Всі вендор коди з кількістю проданих товарів."""
    try:
        conn = get_connection()
        cursor = conn.cursor()

        prefix_filter = ""
        params = (limit,)
        if prefix:
            prefix_filter = "AND p.VendorCode LIKE %s"
            params = (f'{prefix}%', limit)

        query = f"""
            SELECT TOP (%s)
                p.VendorCode as vendor_code,
                p.Name as product_name,
                COALESCE(SUM(oi.Qty), 0) as total_sold,
                COUNT(DISTINCT oi.OrderID) as order_count
            FROM dbo.Product p
            LEFT JOIN dbo.OrderItem oi ON p.ID = oi.ProductID AND oi.Deleted = 0
            WHERE p.Deleted = 0 AND p.VendorCode IS NOT NULL {prefix_filter}
            GROUP BY p.VendorCode, p.Name
            ORDER BY total_sold DESC
        """

        if prefix:
            cursor.execute(query, (prefix + '%', limit))
        else:
            cursor.execute(query, (limit,))

        results = cursor_to_dicts(cursor)
        conn.close()

        return {
            "title": f"Вендор коди за продажами" + (f" (фільтр: {prefix})" if prefix else ""),
            "count": len(results),
            "data": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/products/by-code/{vendor_code}")
async def product_by_code(vendor_code: str):
    """Статистика конкретного товару по артикулу (наприклад SEM9401)."""
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Get product info with sales
        cursor.execute("""
            SELECT
                p.ID as product_id,
                p.Name as product_name,
                p.VendorCode as vendor_code,
                COALESCE(SUM(oi.Qty), 0) as total_sold,
                COUNT(DISTINCT oi.OrderID) as order_count,
                COUNT(DISTINCT YEAR(oi.Created)) as years_sold
            FROM dbo.Product p
            LEFT JOIN dbo.OrderItem oi ON p.ID = oi.ProductID AND oi.Deleted = 0
            WHERE p.Deleted = 0 AND p.VendorCode = %s
            GROUP BY p.ID, p.Name, p.VendorCode
        """, (vendor_code,))

        columns = [column[0] for column in cursor.description]
        row = cursor.fetchone()
        product = dict(zip(columns, row)) if row else None

        if not product:
            conn.close()
            raise HTTPException(status_code=404, detail=f"Товар {vendor_code} не знайдено")

        # Get sales by year
        cursor.execute("""
            SELECT
                YEAR(oi.Created) as year,
                SUM(oi.Qty) as qty_sold,
                COUNT(DISTINCT oi.OrderID) as orders
            FROM dbo.OrderItem oi
            JOIN dbo.Product p ON oi.ProductID = p.ID
            WHERE p.VendorCode = %s AND oi.Deleted = 0
            GROUP BY YEAR(oi.Created)
            ORDER BY year DESC
        """, (vendor_code,))

        by_year = cursor_to_dicts(cursor)
        conn.close()

        return {
            "title": f"Статистика товару {vendor_code}",
            "product": product,
            "by_year": by_year
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/products/zero-sales")
async def zero_sales_products(
    limit: int = Query(default=50, ge=1, le=500, description="Кількість"),
    prefix: Optional[str] = Query(default=None, description="Фільтр по префіксу")
):
    """Товари з нульовими продажами."""
    try:
        conn = get_connection()
        cursor = conn.cursor()

        prefix_filter = ""
        if prefix:
            prefix_filter = f"AND p.VendorCode LIKE '{prefix}%'"

        cursor.execute(f"""
            SELECT TOP ({limit})
                p.ID as product_id,
                p.Name as product_name,
                p.VendorCode as vendor_code,
                p.Created as created_date
            FROM dbo.Product p
            LEFT JOIN dbo.OrderItem oi ON p.ID = oi.ProductID AND oi.Deleted = 0
            WHERE p.Deleted = 0
                AND p.VendorCode IS NOT NULL
                AND p.IsForSale = 1
                {prefix_filter}
            GROUP BY p.ID, p.Name, p.VendorCode, p.Created
            HAVING COALESCE(SUM(oi.Qty), 0) = 0
            ORDER BY p.Created DESC
        """)

        results = cursor_to_dicts(cursor)
        conn.close()

        return {
            "title": "Товари без продажів" + (f" ({prefix}*)" if prefix else ""),
            "count": len(results),
            "data": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/products/compare-brands")
async def compare_brands(
    brand1: str = Query(..., description="Перший бренд (напр. SEM)"),
    brand2: str = Query(..., description="Другий бренд (напр. MG)"),
    year: Optional[int] = Query(default=None, description="Рік")
):
    """Порівняння продажів двох брендів."""
    try:
        conn = get_connection()
        cursor = conn.cursor()

        year_filter = ""
        if year:
            year_filter = f"AND YEAR(oi.Created) = {year}"

        results = {}
        for brand in [brand1, brand2]:
            cursor.execute(f"""
                SELECT
                    COUNT(DISTINCT p.ID) as product_count,
                    COALESCE(SUM(oi.Qty), 0) as total_sold,
                    COUNT(DISTINCT oi.OrderID) as order_count,
                    COUNT(DISTINCT oi.ID) as item_count
                FROM dbo.Product p
                LEFT JOIN dbo.OrderItem oi ON p.ID = oi.ProductID AND oi.Deleted = 0 {year_filter}
                WHERE p.Deleted = 0 AND p.VendorCode LIKE %s
            """, (f'{brand}%',))

            columns = [column[0] for column in cursor.description]
            stats = dict(zip(columns, cursor.fetchone())) if cursor.description else None

            # Top 5 products for this brand
            cursor.execute(f"""
                SELECT TOP (5)
                    p.VendorCode as vendor_code,
                    p.Name as product_name,
                    COALESCE(SUM(oi.Qty), 0) as total_sold
                FROM dbo.Product p
                LEFT JOIN dbo.OrderItem oi ON p.ID = oi.ProductID AND oi.Deleted = 0 {year_filter}
                WHERE p.Deleted = 0 AND p.VendorCode LIKE %s
                GROUP BY p.VendorCode, p.Name
                ORDER BY total_sold DESC
            """, (f'{brand}%',))

            top_products = cursor_to_dicts(cursor)

            results[brand] = {
                "stats": stats,
                "top_products": top_products
            }

        conn.close()

        return {
            "title": f"Порівняння {brand1} vs {brand2}" + (f" за {year} рік" if year else ""),
            "brand1": brand1,
            "brand2": brand2,
            "comparison": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/products/search")
async def search_products(
    q: str = Query(..., description="Пошуковий запит"),
    limit: int = Query(default=20, ge=1, le=100, description="Кількість"),
    sort_by_sales: bool = Query(default=False, description="Сортувати по продажах")
):
    """Пошук товарів по назві з опцією сортування по продажах."""
    try:
        conn = get_connection()
        cursor = conn.cursor()

        if sort_by_sales:
            # Search with sales ranking
            cursor.execute(f"""
                SELECT TOP ({limit})
                    p.ID as product_id,
                    p.Name as product_name,
                    p.VendorCode as vendor_code,
                    COALESCE(SUM(oi.Qty), 0) as total_sold,
                    COUNT(DISTINCT oi.OrderID) as order_count
                FROM dbo.Product p
                LEFT JOIN dbo.OrderItem oi ON p.ID = oi.ProductID AND oi.Deleted = 0
                WHERE p.Deleted = 0 AND (p.Name LIKE %s OR p.VendorCode LIKE %s)
                GROUP BY p.ID, p.Name, p.VendorCode
                ORDER BY total_sold DESC
            """, (f'%{q}%', f'%{q}%'))
        else:
            cursor.execute(f"""
                SELECT TOP ({limit})
                    p.ID as product_id,
                    p.Name as product_name,
                    p.VendorCode as vendor_code,
                    p.Created as created
                FROM dbo.Product p
                WHERE p.Deleted = 0 AND (p.Name LIKE %s OR p.VendorCode LIKE %s)
                ORDER BY p.Name
            """, (f'%{q}%', f'%{q}%'))

        results = cursor_to_dicts(cursor)
        conn.close()

        return {
            "query": q,
            "count": len(results),
            "sorted_by_sales": sort_by_sales,
            "products": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("\nStarting Analytics API...")
    print("API: http://localhost:8001")
    print("Docs: http://localhost:8001/docs")
    uvicorn.run(app, host="0.0.0.0", port=8001)
