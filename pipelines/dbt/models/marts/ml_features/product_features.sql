-- Product features for ML models
-- Aggregates product metrics, popularity, and relationships for recommendations

{{
    config(
        materialized='table',
        schema='ml_features'
    )
}}

with product_sales as (
    -- Aggregate sales metrics for each product
    select
        oi.product_id,
        count(distinct oi.order_id) as times_ordered,
        count(distinct s.client_agreement_id) as unique_customers,
        sum(oi.quantity) as total_quantity_sold,
        sum(oi.quantity * oi.price_per_item) as total_revenue,
        avg(oi.price_per_item) as avg_price,
        max(s.created_at) as last_sold_date,
        datediff('day', max(s.created_at), current_date) as days_since_last_sale

    from {{ ref('stg_order_items') }} oi
    inner join {{ ref('stg_sales') }} s
        on oi.order_id = s.sale_id
    where s.created_at is not null
    group by oi.product_id
),

product_analogues_count as (
    -- Count number of alternatives for each product
    select
        base_product_id as product_id,
        count(distinct analogue_product_id) as num_analogues
    from {{ ref('stg_product_analogues') }}
    group by base_product_id
),

product_pricing_latest as (
    -- Get latest pricing for each product
    select
        product_id,
        price as current_price,
        created_at as price_updated_at,
        row_number() over (partition by product_id order by created_at desc) as rn
    from {{ ref('stg_product_pricing') }}
),

product_popularity as (
    -- Calculate popularity scores
    select
        product_id,
        times_ordered,
        unique_customers,
        total_quantity_sold,
        total_revenue,

        -- Popularity score (normalized 0-100)
        case
            when total_revenue > 0 then
                least(100,
                    (times_ordered * 0.3) +
                    (unique_customers * 0.5) +
                    (total_quantity_sold * 0.2)
                )
            else 0
        end as popularity_score,

        -- Trending score (recent activity)
        case
            when days_since_last_sale <= 7 then 10
            when days_since_last_sale <= 30 then 8
            when days_since_last_sale <= 90 then 5
            when days_since_last_sale <= 180 then 3
            else 1
        end as trending_score,

        avg_price,
        last_sold_date,
        days_since_last_sale

    from product_sales
),

final as (
    select
        p.product_id,
        p.product_name,
        p.description,
        p.vendor_code,
        p.main_original_number,

        -- Flags
        p.is_for_sale,
        p.is_for_web,
        p.has_analogue,
        p.has_image,

        -- Sales metrics
        coalesce(ps.times_ordered, 0) as times_ordered,
        coalesce(ps.unique_customers, 0) as unique_customers,
        coalesce(ps.total_quantity_sold, 0) as total_quantity_sold,
        coalesce(ps.total_revenue, 0) as total_revenue,

        -- Popularity metrics
        coalesce(pp.popularity_score, 0) as popularity_score,
        coalesce(pp.trending_score, 1) as trending_score,

        -- Pricing
        coalesce(pl.current_price, ps.avg_price, 0) as current_price,
        coalesce(ps.avg_price, 0) as avg_historical_price,
        pl.price_updated_at,

        -- Alternatives
        coalesce(pac.num_analogues, 0) as num_analogues,

        -- Temporal
        ps.last_sold_date,
        coalesce(ps.days_since_last_sale, 99999) as days_since_last_sale,

        -- Product status
        case
            when ps.days_since_last_sale is null then 'Never Sold'
            when ps.days_since_last_sale <= 30 then 'Active'
            when ps.days_since_last_sale <= 90 then 'Slow Moving'
            when ps.days_since_last_sale <= 180 then 'Dormant'
            else 'Dead Stock'
        end as product_status,

        -- Metadata
        p.created_at,
        current_timestamp as features_updated_at

    from {{ ref('stg_products') }} p
    left join product_sales ps on p.product_id = ps.product_id
    left join product_popularity pp on p.product_id = pp.product_id
    left join product_analogues_count pac on p.product_id = pac.product_id
    left join product_pricing_latest pl on p.product_id = pl.product_id and pl.rn = 1
)

select * from final
