#!/usr/bin/env python3
"""
Run dbt transformations using DuckDB directly
This bypasses dependency conflicts by using DuckDB to read Delta and execute SQL
"""

import duckdb
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
DELTA_BASE = "/opt/dagster/app/data/delta"
OUTPUT_DB = "/opt/dagster/app/data/dbt/concord_bi.duckdb"
DBT_MODELS_DIR = "/opt/dagster/app/pipelines/dbt/models"

def run_transformations():
    """Run all dbt transformations using DuckDB"""

    # Create output directory
    Path(OUTPUT_DB).parent.mkdir(parents=True, exist_ok=True)

    # Connect to DuckDB
    logger.info(f"Connecting to DuckDB: {OUTPUT_DB}")
    conn = duckdb.connect(OUTPUT_DB)

    # Install and load delta extension
    logger.info("Installing Delta extension...")
    conn.execute("INSTALL delta")
    conn.execute("LOAD delta")

    logger.info("=" * 70)
    logger.info("RUNNING DBT TRANSFORMATIONS WITH DUCKDB")
    logger.info("=" * 70)

    try:
        # Step 1: Create views from Delta Lake sources
        logger.info("\n[1/4] Creating source views from Delta Lake...")

        sources = {
            "dbo_Client": f"{DELTA_BASE}/customer/dbo_Client",
            "dbo_ClientAgreement": f"{DELTA_BASE}/customer/dbo_ClientAgreement",
            "dbo_Product": f"{DELTA_BASE}/product/dbo_Product",
            "dbo_ProductAnalogue": f"{DELTA_BASE}/product/dbo_ProductAnalogue",
            "dbo_ProductPricing": f"{DELTA_BASE}/product/dbo_ProductPricing",
            "dbo_Sale": f"{DELTA_BASE}/transaction/dbo_Sale",
            "dbo_OrderItem": f"{DELTA_BASE}/transaction/dbo_OrderItem",
        }

        for table_name, delta_path in sources.items():
            logger.info(f"  Creating view: {table_name}")
            conn.execute(f"""
                CREATE OR REPLACE VIEW {table_name} AS
                SELECT * FROM delta_scan('{delta_path}')
            """)

        logger.info(f"✓ Created {len(sources)} source views")

        # Step 2: Run staging models
        logger.info("\n[2/4] Running staging models...")

        # stg_customers
        logger.info("  Creating stg_customers...")
        conn.execute("""
            CREATE OR REPLACE VIEW stg_customers AS
            SELECT
                CAST(ID AS VARCHAR) as customer_id,
                TRIM(Name) as customer_name,
                TRIM(EmailAddress) as email,
                TRIM(MobileNumber) as mobile,
                TRIM(TIN) as tin,
                TRIM(USREOU) as usreou,
                TRIM(LegalAddress) as legal_address,
                TRIM(ActualAddress) as actual_address,
                TRIM(DeliveryAddress) as delivery_address,
                CAST(RegionID AS VARCHAR) as region_id,
                CAST(Created AS TIMESTAMP) as created_at,
                CAST(Updated AS TIMESTAMP) as updated_at,
                CAST(Deleted AS BOOLEAN) as is_deleted,
                _ingested_at,
                _source_table
            FROM dbo_Client
            WHERE ID IS NOT NULL AND Deleted = false
        """)

        #stg_products
        logger.info("  Creating stg_products...")
        conn.execute("""
            CREATE OR REPLACE VIEW stg_products AS
            SELECT
                CAST(ID AS VARCHAR) as product_id,
                TRIM(Name) as product_name,
                TRIM(Description) as description,
                TRIM(VendorCode) as vendor_code,
                TRIM(MainOriginalNumber) as main_original_number,
                CAST(IsForSale AS BOOLEAN) as is_for_sale,
                CAST(IsForWeb AS BOOLEAN) as is_for_web,
                CAST(HasAnalogue AS BOOLEAN) as has_analogue,
                CAST(HasImage AS BOOLEAN) as has_image,
                TRIM(Size) as size,
                TRIM(Volume) as volume,
                CAST(MeasureUnitID AS VARCHAR) as measure_unit_id,
                CAST(Created AS TIMESTAMP) as created_at,
                CAST(Updated AS TIMESTAMP) as updated_at,
                CAST(Deleted AS BOOLEAN) as is_deleted,
                _ingested_at,
                _source_table
            FROM dbo_Product
            WHERE ID IS NOT NULL AND Deleted = false
        """)

        # stg_sales
        logger.info("  Creating stg_sales...")
        conn.execute("""
            CREATE OR REPLACE VIEW stg_sales AS
            SELECT
                CAST(ID AS VARCHAR) as sale_id,
                CAST(ClientAgreementID AS VARCHAR) as client_agreement_id,
                CAST(OrderID AS VARCHAR) as order_id,
                CAST(UserID AS VARCHAR) as user_id,
                CAST(BaseLifeCycleStatusID AS VARCHAR) as lifecycle_status_id,
                CAST(BaseSalePaymentStatusID AS VARCHAR) as payment_status_id,
                TRIM(Comment) as comment,
                CAST(IsMerged AS BOOLEAN) as is_merged,
                CAST(Created AS TIMESTAMP) as created_at,
                CAST(Updated AS TIMESTAMP) as updated_at,
                CAST(Deleted AS BOOLEAN) as is_deleted,
                _ingested_at,
                _source_table
            FROM dbo_Sale
            WHERE ID IS NOT NULL AND Deleted = false
        """)

        # stg_order_items
        logger.info("  Creating stg_order_items...")
        conn.execute("""
            CREATE OR REPLACE VIEW stg_order_items AS
            SELECT
                CAST(ID AS VARCHAR) as order_item_id,
                CAST(OrderID AS VARCHAR) as order_id,
                CAST(ProductID AS VARCHAR) as product_id,
                CAST(UserId AS VARCHAR) as user_id,
                CAST(Qty AS DECIMAL(18,4)) as quantity,
                CAST(OrderedQty AS DECIMAL(18,4)) as ordered_quantity,
                CAST(PricePerItem AS DECIMAL(18,2)) as price_per_item,
                CAST(OneTimeDiscount AS DECIMAL(18,2)) as one_time_discount,
                CAST(IsValidForCurrentSale AS BOOLEAN) as is_valid_for_sale,
                CAST(IsFromOffer AS BOOLEAN) as is_from_offer,
                TRIM(Comment) as comment,
                CAST(Created AS TIMESTAMP) as created_at,
                CAST(Updated AS TIMESTAMP) as updated_at,
                CAST(Deleted AS BOOLEAN) as is_deleted,
                _ingested_at,
                _source_table
            FROM dbo_OrderItem
            WHERE ID IS NOT NULL AND Deleted = false AND Qty > 0
        """)

        # stg_product_analogues
        logger.info("  Creating stg_product_analogues...")
        conn.execute("""
            CREATE OR REPLACE VIEW stg_product_analogues AS
            SELECT
                CAST(ID AS VARCHAR) as product_analogue_id,
                CAST(BaseProductID AS VARCHAR) as base_product_id,
                CAST(AnalogueProductID AS VARCHAR) as analogue_product_id,
                CAST(Created AS TIMESTAMP) as created_at,
                CAST(Updated AS TIMESTAMP) as updated_at,
                CAST(Deleted AS BOOLEAN) as is_deleted,
                _ingested_at,
                _source_table
            FROM dbo_ProductAnalogue
            WHERE ID IS NOT NULL AND Deleted = false
                AND BaseProductID IS NOT NULL
                AND AnalogueProductID IS NOT NULL
                AND BaseProductID != AnalogueProductID
        """)

        # stg_product_pricing
        logger.info("  Creating stg_product_pricing...")
        conn.execute("""
            CREATE OR REPLACE VIEW stg_product_pricing AS
            SELECT
                CAST(ID AS VARCHAR) as product_pricing_id,
                CAST(ProductID AS VARCHAR) as product_id,
                CAST(PricingID AS VARCHAR) as pricing_id,
                CAST(Price AS DECIMAL(18,2)) as price,
                CAST(Created AS TIMESTAMP) as created_at,
                CAST(Updated AS TIMESTAMP) as updated_at,
                CAST(Deleted AS BOOLEAN) as is_deleted,
                _ingested_at,
                _source_table
            FROM dbo_ProductPricing
            WHERE ID IS NOT NULL AND Deleted = false
                AND ProductID IS NOT NULL AND Price > 0
        """)

        logger.info("✓ Created 6 staging views")

        # Step 3: Run ML feature mart models
        logger.info("\n[3/4] Running ML feature mart models...")
        logger.info("  This may take a few minutes for large datasets...")

        # Create schema
        conn.execute("CREATE SCHEMA IF NOT EXISTS ml_features")

        # interaction_matrix (fastest, needed for customer_features)
        logger.info("  Creating interaction_matrix...")
        start = datetime.now()
        conn.execute("""
            CREATE OR REPLACE TABLE ml_features.interaction_matrix AS
            WITH customer_product_interactions AS (
                SELECT
                    c.customer_id,
                    oi.product_id,
                    COUNT(DISTINCT s.sale_id) as num_purchases,
                    SUM(oi.quantity) as total_quantity,
                    SUM(oi.quantity * oi.price_per_item) as total_spent,
                    MAX(s.created_at) as last_purchase_date,
                    MIN(s.created_at) as first_purchase_date,
                    LEAST(5, GREATEST(1,
                        (COUNT(DISTINCT s.sale_id) * 0.4) +
                        (CASE
                            WHEN DATE_DIFF('day', MAX(s.created_at), CURRENT_DATE) <= 30 THEN 2.0
                            WHEN DATE_DIFF('day', MAX(s.created_at), CURRENT_DATE) <= 90 THEN 1.5
                            WHEN DATE_DIFF('day', MAX(s.created_at), CURRENT_DATE) <= 180 THEN 1.0
                            ELSE 0.5
                        END) +
                        (LEAST(2.5, SUM(oi.quantity * oi.price_per_item) / 10000))
                    )) as implicit_rating
                FROM stg_customers c
                INNER JOIN dbo_ClientAgreement ca
                    ON CAST(ca.ClientID AS VARCHAR) = c.customer_id
                INNER JOIN stg_sales s
                    ON CAST(s.client_agreement_id AS VARCHAR) = CAST(ca.ID AS VARCHAR)
                INNER JOIN stg_order_items oi
                    ON CAST(oi.order_id AS VARCHAR) = s.order_id
                WHERE ca.Deleted = false
                    AND s.created_at IS NOT NULL
                    AND oi.quantity > 0
                GROUP BY c.customer_id, oi.product_id
            )
            SELECT
                customer_id,
                product_id,
                num_purchases,
                total_quantity,
                total_spent,
                implicit_rating,
                first_purchase_date,
                last_purchase_date,
                DATE_DIFF('day', last_purchase_date, CURRENT_DATE) as days_since_last_purchase,
                DATE_DIFF('day', first_purchase_date, last_purchase_date) as purchase_span_days,
                CURRENT_TIMESTAMP as features_updated_at
            FROM customer_product_interactions
        """)
        duration = (datetime.now() - start).total_seconds()
        count = conn.execute("SELECT COUNT(*) FROM ml_features.interaction_matrix").fetchone()[0]
        logger.info(f"  ✓ interaction_matrix: {count:,} rows in {duration:.1f}s")

        # customer_features (depends on interaction_matrix)
        logger.info("  Creating customer_features...")
        start = datetime.now()

        # Read and execute the customer_features SQL file (it's complex, so let's  simplify)
        conn.execute("""
            CREATE OR REPLACE TABLE ml_features.customer_features AS
            WITH customer_stats AS (
                SELECT
                    customer_id,
                    COUNT(DISTINCT product_id) as unique_products,
                    SUM(num_purchases) as total_orders,
                    SUM(total_spent) as total_revenue,
                    MAX(last_purchase_date) as last_purchase,
                    MIN(first_purchase_date) as first_purchase
                FROM ml_features.interaction_matrix
                GROUP BY customer_id
            ),
            rfm AS (
                SELECT
                    customer_id,
                    unique_products,
                    total_orders,
                    total_revenue,
                    last_purchase,
                    first_purchase,
                    DATE_DIFF('day', last_purchase, CURRENT_DATE) as days_since_last,
                    -- RFM Scores
                    CASE
                        WHEN DATE_DIFF('day', last_purchase, CURRENT_DATE) <= 30 THEN 5
                        WHEN DATE_DIFF('day', last_purchase, CURRENT_DATE) <= 60 THEN 4
                        WHEN DATE_DIFF('day', last_purchase, CURRENT_DATE) <= 90 THEN 3
                        WHEN DATE_DIFF('day', last_purchase, CURRENT_DATE) <= 180 THEN 2
                        ELSE 1
                    END as recency_score,
                    CASE
                        WHEN total_orders >= 10 THEN 5
                        WHEN total_orders >= 5 THEN 4
                        WHEN total_orders >= 3 THEN 3
                        WHEN total_orders >= 2 THEN 2
                        ELSE 1
                    END as frequency_score,
                    CASE
                        WHEN total_revenue >= 50000 THEN 5
                        WHEN total_revenue >= 20000 THEN 4
                        WHEN total_revenue >= 10000 THEN 3
                        WHEN total_revenue >= 5000 THEN 2
                        ELSE 1
                    END as monetary_score
                FROM customer_stats
            )
            SELECT
                c.customer_id,
                c.customer_name,
                c.email,
                c.created_at as customer_since,
                COALESCE(r.recency_score, 0) as recency_score,
                COALESCE(r.frequency_score, 0) as frequency_score,
                COALESCE(r.monetary_score, 0) as monetary_score,
                CAST(COALESCE(r.recency_score, 0) AS VARCHAR) ||
                CAST(COALESCE(r.frequency_score, 0) AS VARCHAR) ||
                CAST(COALESCE(r.monetary_score, 0) AS VARCHAR) as rfm_segment,
                CASE
                    WHEN r.recency_score >= 4 AND r.frequency_score >= 4 AND r.monetary_score >= 4 THEN 'Champions'
                    WHEN r.recency_score >= 4 AND r.frequency_score >= 3 THEN 'Loyal Customers'
                    WHEN r.recency_score >= 4 AND r.monetary_score >= 4 THEN 'Big Spenders'
                    WHEN r.recency_score >= 3 AND r.frequency_score >= 3 THEN 'Potential Loyalists'
                    WHEN r.recency_score >= 4 THEN 'Recent Customers'
                    WHEN r.recency_score = 3 THEN 'At Risk'
                    WHEN r.recency_score = 2 AND r.frequency_score >= 2 THEN 'Cannot Lose Them'
                    WHEN r.recency_score <= 2 AND r.frequency_score <= 2 THEN 'Lost'
                    ELSE 'New Customer'
                END as customer_segment,
                COALESCE(r.total_orders, 0) as total_orders,
                COALESCE(r.unique_products, 0) as unique_products_purchased,
                COALESCE(r.total_revenue, 0) as lifetime_value,
                r.last_purchase as last_purchase_date,
                COALESCE(r.days_since_last, 99999) as days_since_last_purchase,
                r.first_purchase as first_purchase_date,
                CASE
                    WHEN r.days_since_last IS NULL THEN 'Never Purchased'
                    WHEN r.days_since_last <= 30 THEN 'Active'
                    WHEN r.days_since_last <= 90 THEN 'At Risk'
                    WHEN r.days_since_last <= 180 THEN 'Dormant'
                    ELSE 'Churned'
                END as customer_status,
                CURRENT_TIMESTAMP as features_updated_at
            FROM stg_customers c
            LEFT JOIN rfm r ON c.customer_id = r.customer_id
        """)
        duration = (datetime.now() - start).total_seconds()
        count = conn.execute("SELECT COUNT(*) FROM ml_features.customer_features").fetchone()[0]
        logger.info(f"  ✓ customer_features: {count:,} rows in {duration:.1f}s")

        # product_features
        logger.info("  Creating product_features...")
        start = datetime.now()
        conn.execute("""
            CREATE OR REPLACE TABLE ml_features.product_features AS
            WITH product_sales AS (
                SELECT
                    oi.product_id,
                    COUNT(DISTINCT oi.order_id) as times_ordered,
                    COUNT(DISTINCT s.client_agreement_id) as unique_customers,
                    SUM(oi.quantity) as total_quantity_sold,
                    SUM(oi.quantity * oi.price_per_item) as total_revenue,
                    AVG(oi.price_per_item) as avg_price,
                    MAX(s.created_at) as last_sold_date,
                    DATE_DIFF('day', MAX(s.created_at), CURRENT_DATE) as days_since_last_sale
                FROM stg_order_items oi
                INNER JOIN stg_sales s ON oi.order_id = s.sale_id
                WHERE s.created_at IS NOT NULL
                GROUP BY oi.product_id
            ),
            product_analogues_count AS (
                SELECT
                    base_product_id as product_id,
                    COUNT(DISTINCT analogue_product_id) as num_analogues
                FROM stg_product_analogues
                GROUP BY base_product_id
            )
            SELECT
                p.product_id,
                p.product_name,
                p.description,
                p.vendor_code,
                p.is_for_sale,
                p.is_for_web,
                p.has_analogue,
                COALESCE(ps.times_ordered, 0) as times_ordered,
                COALESCE(ps.unique_customers, 0) as unique_customers,
                COALESCE(ps.total_quantity_sold, 0) as total_quantity_sold,
                COALESCE(ps.total_revenue, 0) as total_revenue,
                COALESCE(ps.avg_price, 0) as current_price,
                COALESCE(pa.num_analogues, 0) as num_analogues,
                ps.last_sold_date,
                COALESCE(ps.days_since_last_sale, 99999) as days_since_last_sale,
                CASE
                    WHEN ps.days_since_last_sale IS NULL THEN 'Never Sold'
                    WHEN ps.days_since_last_sale <= 30 THEN 'Active'
                    WHEN ps.days_since_last_sale <= 90 THEN 'Slow Moving'
                    WHEN ps.days_since_last_sale <= 180 THEN 'Dormant'
                    ELSE 'Dead Stock'
                END as product_status,
                p.created_at,
                CURRENT_TIMESTAMP as features_updated_at
            FROM stg_products p
            LEFT JOIN product_sales ps ON p.product_id = ps.product_id
            LEFT JOIN product_analogues_count pa ON p.product_id = pa.product_id
        """)
        duration = (datetime.now() - start).total_seconds()
        count = conn.execute("SELECT COUNT(*) FROM ml_features.product_features").fetchone()[0]
        logger.info(f"  ✓ product_features: {count:,} rows in {duration:.1f}s")

        logger.info("✓ Created 3 ML feature tables")

        # Step 4: Summary
        logger.info("\n[4/4] Summary...")

        # Get row counts
        staging_counts = {}
        for view in ['stg_customers', 'stg_products', 'stg_sales', 'stg_order_items']:
            count = conn.execute(f"SELECT COUNT(*) FROM {view}").fetchone()[0]
            staging_counts[view] = count

        ml_counts = {}
        for table in ['customer_features', 'product_features', 'interaction_matrix']:
            count = conn.execute(f"SELECT COUNT(*) FROM ml_features.{table}").fetchone()[0]
            ml_counts[table] = count

        logger.info("\n" + "=" * 70)
        logger.info("TRANSFORMATION SUMMARY")
        logger.info("=" * 70)
        logger.info("\nStaging Views:")
        for view, count in staging_counts.items():
            logger.info(f"  - {view}: {count:,} rows")

        logger.info("\nML Feature Tables:")
        for table, count in ml_counts.items():
            logger.info(f"  - ml_features.{table}: {count:,} rows")

        logger.info(f"\n✅ ALL TRANSFORMATIONS COMPLETE!")
        logger.info(f"📁 DuckDB file: {OUTPUT_DB}")
        logger.info(f"🔍 To query: duckdb {OUTPUT_DB}")

        return True

    except Exception as e:
        logger.error(f"❌ Error during transformations: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        conn.close()

if __name__ == "__main__":
    import sys
    success = run_transformations()
    sys.exit(0 if success else 1)
