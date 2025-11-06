-- Customer features for ML models (Enhanced for 5-Tier Ensemble)
-- Generates RFM analysis, temporal patterns, and behavioral metrics
-- Supports LightFM, LSTM, Survival Analysis, Bandits, and GNN models

{{
    config(
        materialized='table',
        schema='ml_features',
        indexes=[
            {'columns': ['customer_id']},
            {'columns': ['customer_segment']},
            {'columns': ['customer_status']}
        ]
    )
}}

with customer_purchases as (
    -- Get all customer purchases via ClientAgreement -> Sale -> OrderItem
    select
        c.customer_id,
        s.sale_id,
        s.created_at as purchase_date,
        oi.product_id,
        oi.quantity,
        oi.price_per_item,
        oi.quantity * oi.price_per_item as line_total
    from {{ ref('stg_customers') }} c
    inner join {{ source('delta_lake', 'dbo_ClientAgreement') }} ca
        on cast(ca.ClientID as string) = c.customer_id
    inner join {{ ref('stg_sales') }} s
        on cast(s.client_agreement_id as string) = cast(ca.ID as string)
    inner join {{ ref('stg_order_items') }} oi
        on cast(oi.order_id as string) = s.order_id
    where ca.Deleted = false
        and s.created_at is not null
),

customer_aggregates as (
    select
        customer_id,

        -- Order metrics
        count(distinct sale_id) as total_orders,
        count(distinct product_id) as unique_products_purchased,

        -- Revenue metrics (Monetary)
        sum(line_total) as total_revenue,
        avg(line_total) as avg_line_value,
        sum(line_total) / nullif(count(distinct sale_id), 0) as avg_order_value,
        stddev(line_total) as stddev_line_value,

        -- Recency
        max(purchase_date) as last_purchase_date,
        datediff('day', max(purchase_date), current_date) as days_since_last_purchase,

        -- Frequency
        min(purchase_date) as first_purchase_date,
        datediff('day', min(purchase_date), max(purchase_date)) as customer_lifespan_days,

        -- Purchase frequency (orders per month)
        count(distinct sale_id)::float / nullif(
            greatest(1, datediff('month', min(purchase_date), max(purchase_date))),
            0
        ) as orders_per_month,

        -- Inter-order intervals (for Survival Analysis)
        avg(
            datediff('day',
                lag(purchase_date) over (partition by customer_id order by purchase_date),
                purchase_date
            )
        ) as avg_days_between_orders,

        stddev(
            datediff('day',
                lag(purchase_date) over (partition by customer_id order by purchase_date),
                purchase_date
            )
        ) as stddev_days_between_orders,

        -- Quantity
        sum(quantity) as total_quantity_purchased,
        avg(quantity) as avg_quantity_per_order,

        -- Seasonality metrics (for LSTM)
        count(distinct extract(month from purchase_date)) as num_active_months,
        count(distinct extract(quarter from purchase_date)) as num_active_quarters,
        count(distinct extract(year from purchase_date)) as num_active_years,

        -- Purchase behavior patterns
        max(sale_id) as most_recent_sale_id,
        min(sale_id) as first_sale_id,

        -- Cohort analysis
        extract(year from min(purchase_date)) as cohort_year,
        extract(month from min(purchase_date)) as cohort_month

    from customer_purchases
    group by customer_id
),

rfm_scores as (
    select
        customer_id,

        -- RFM Scores (1-5 scale, 5 = best)
        -- Recency: Lower days = Better score
        case
            when days_since_last_purchase <= 30 then 5
            when days_since_last_purchase <= 60 then 4
            when days_since_last_purchase <= 90 then 3
            when days_since_last_purchase <= 180 then 2
            else 1
        end as recency_score,

        -- Frequency: More orders = Better score
        case
            when total_orders >= 10 then 5
            when total_orders >= 5 then 4
            when total_orders >= 3 then 3
            when total_orders >= 2 then 2
            else 1
        end as frequency_score,

        -- Monetary: Higher revenue = Better score
        case
            when total_revenue >= 50000 then 5
            when total_revenue >= 20000 then 4
            when total_revenue >= 10000 then 3
            when total_revenue >= 5000 then 2
            else 1
        end as monetary_score,

        -- All aggregated metrics
        total_orders,
        unique_products_purchased,
        total_revenue,
        avg_line_value,
        avg_order_value,
        last_purchase_date,
        days_since_last_purchase,
        first_purchase_date,
        customer_lifespan_days,
        total_quantity_purchased

    from customer_aggregates
),

customer_segments as (
    select
        customer_id,
        recency_score,
        frequency_score,
        monetary_score,

        -- RFM Segment (concatenated for easy filtering)
        cast(recency_score as string) || cast(frequency_score as string) || cast(monetary_score as string) as rfm_segment,

        -- Customer Segment Labels
        case
            when recency_score >= 4 and frequency_score >= 4 and monetary_score >= 4 then 'Champions'
            when recency_score >= 4 and frequency_score >= 3 then 'Loyal Customers'
            when recency_score >= 4 and monetary_score >= 4 then 'Big Spenders'
            when recency_score >= 3 and frequency_score >= 3 then 'Potential Loyalists'
            when recency_score >= 4 then 'Recent Customers'
            when recency_score = 3 then 'At Risk'
            when recency_score = 2 and frequency_score >= 2 then 'Cannot Lose Them'
            when recency_score <= 2 and frequency_score <= 2 then 'Lost'
            else 'Needs Attention'
        end as customer_segment,

        -- All metrics
        total_orders,
        unique_products_purchased,
        total_revenue,
        avg_line_value,
        avg_order_value,
        last_purchase_date,
        days_since_last_purchase,
        first_purchase_date,
        customer_lifespan_days,
        total_quantity_purchased

    from rfm_scores
),

final as (
    select
        c.customer_id,
        c.customer_name,
        c.email,
        c.created_at as customer_since,

        -- RFM Analysis
        coalesce(cs.recency_score, 0) as recency_score,
        coalesce(cs.frequency_score, 0) as frequency_score,
        coalesce(cs.monetary_score, 0) as monetary_score,
        coalesce(cs.rfm_segment, '000') as rfm_segment,
        coalesce(cs.customer_segment, 'New Customer') as customer_segment,

        -- Purchase metrics
        coalesce(cs.total_orders, 0) as total_orders,
        coalesce(cs.unique_products_purchased, 0) as unique_products_purchased,
        coalesce(cs.total_revenue, 0) as lifetime_value,
        coalesce(cs.avg_line_value, 0) as avg_line_value,
        coalesce(cs.avg_order_value, 0) as avg_order_value,
        coalesce(cs.stddev_line_value, 0) as stddev_line_value,
        coalesce(cs.total_quantity_purchased, 0) as total_quantity_purchased,
        coalesce(cs.avg_quantity_per_order, 0) as avg_quantity_per_order,

        -- Temporal metrics
        cs.last_purchase_date,
        coalesce(cs.days_since_last_purchase, 99999) as days_since_last_purchase,
        cs.first_purchase_date,
        coalesce(cs.customer_lifespan_days, 0) as customer_lifespan_days,

        -- Purchase frequency metrics (for Survival Analysis)
        coalesce(cs.orders_per_month, 0) as orders_per_month,
        coalesce(cs.avg_days_between_orders, 999) as avg_days_between_orders,
        coalesce(cs.stddev_days_between_orders, 0) as stddev_days_between_orders,

        -- Expected next order date (Survival Analysis prediction)
        case
            when cs.avg_days_between_orders is not null then
                dateadd('day', cast(cs.avg_days_between_orders as int), cs.last_purchase_date)
            else null
        end as expected_next_order_date,

        -- Reorder probability (0-1 scale)
        case
            when cs.avg_days_between_orders is not null and cs.avg_days_between_orders > 0 then
                least(1.0, greatest(0.0,
                    1.0 - (cs.days_since_last_purchase::float / cs.avg_days_between_orders)
                ))
            else 0.0
        end as reorder_probability,

        -- Seasonality metrics (for LSTM)
        coalesce(cs.num_active_months, 0) as num_active_months,
        coalesce(cs.num_active_quarters, 0) as num_active_quarters,
        coalesce(cs.num_active_years, 0) as num_active_years,

        -- Purchase behavior flags
        case when cs.total_orders >= 10 then true else false end as is_high_frequency,
        case when cs.total_revenue >= 50000 then true else false end as is_high_value,
        case
            when cs.avg_days_between_orders is not null
                and cs.days_since_last_purchase > (cs.avg_days_between_orders * 1.5)
            then true else false
        end as is_overdue_for_reorder,

        -- Customer cohort
        coalesce(cs.cohort_year, extract(year from c.created_at)) as cohort_year,
        coalesce(cs.cohort_month, extract(month from c.created_at)) as cohort_month,

        -- Customer value tier (for Contextual Bandits)
        case
            when cs.total_revenue >= 100000 then 'Platinum'
            when cs.total_revenue >= 50000 then 'Gold'
            when cs.total_revenue >= 20000 then 'Silver'
            when cs.total_revenue >= 5000 then 'Bronze'
            else 'New'
        end as customer_tier,

        -- Customer status
        case
            when cs.days_since_last_purchase is null then 'Never Purchased'
            when cs.days_since_last_purchase <= 30 then 'Active'
            when cs.days_since_last_purchase <= 90 then 'At Risk'
            when cs.days_since_last_purchase <= 180 then 'Dormant'
            else 'Churned'
        end as customer_status,

        -- Engagement score (composite metric for model weighting)
        least(100, greatest(0,
            (cs.recency_score * 15) +
            (cs.frequency_score * 10) +
            (cs.monetary_score * 15) +
            (coalesce(cs.num_active_quarters, 0) * 5)
        )) as engagement_score,

        -- Metadata
        current_timestamp as features_updated_at

    from {{ ref('stg_customers') }} c
    left join customer_segments cs on c.customer_id = cs.customer_id
)

select * from final
