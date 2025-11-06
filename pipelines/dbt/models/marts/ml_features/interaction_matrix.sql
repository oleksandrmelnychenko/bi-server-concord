-- Customer-Product Interaction Matrix (Enhanced for 5-Tier Ensemble)
-- Creates interaction data for LightFM, LSTM, Survival Analysis, GNN, and Bandits
-- Now handles 183K+ interactions from actual production data

{{
    config(
        materialized='table',
        schema='ml_features',
        indexes=[
            {'columns': ['customer_id']},
            {'columns': ['product_id']},
            {'columns': ['last_purchase_date']}
        ]
    )
}}

with customer_product_interactions as (
    select
        c.customer_id,
        oi.product_id,

        -- Basic interaction metrics
        count(distinct s.sale_id) as num_purchases,
        sum(oi.quantity) as total_quantity,
        sum(oi.quantity * oi.price_per_item) as total_spent,
        avg(oi.quantity) as avg_quantity_per_order,
        avg(oi.price_per_item) as avg_price_per_item,

        -- Temporal metrics for Survival Analysis & LSTM
        max(s.created_at) as last_purchase_date,
        min(s.created_at) as first_purchase_date,

        -- Inter-purchase intervals (for repurchase prediction)
        avg(
            datediff('day',
                lag(s.created_at) over (partition by c.customer_id, oi.product_id order by s.created_at),
                s.created_at
            )
        ) as avg_days_between_purchases,

        stddev(
            datediff('day',
                lag(s.created_at) over (partition by c.customer_id, oi.product_id order by s.created_at),
                s.created_at
            )
        ) as stddev_days_between_purchases,

        -- Seasonality features (for LSTM)
        count(distinct extract(month from s.created_at)) as num_distinct_months,
        count(distinct extract(quarter from s.created_at)) as num_distinct_quarters,

        -- Purchase trend (increasing/decreasing)
        case
            when count(distinct s.sale_id) >= 3 then
                corr(
                    extract(epoch from s.created_at),
                    row_number() over (partition by c.customer_id, oi.product_id order by s.created_at)
                )
            else 0
        end as purchase_trend,

        -- Calculate implicit rating (1-5 scale)
        -- Enhanced with temporal decay for better recommendations
        least(5, greatest(1,
            (count(distinct s.sale_id) * 0.3) +  -- Frequency weight
            (case  -- Recency weight (exponential decay)
                when datediff('day', max(s.created_at), current_date) <= 30 then 2.5
                when datediff('day', max(s.created_at), current_date) <= 60 then 2.0
                when datediff('day', max(s.created_at), current_date) <= 90 then 1.5
                when datediff('day', max(s.created_at), current_date) <= 180 then 1.0
                when datediff('day', max(s.created_at), current_date) <= 365 then 0.5
                else 0.2
            end) +
            (least(2.0, log(1 + sum(oi.quantity * oi.price_per_item)) / 5))  -- Monetary weight (log scale)
        )) as implicit_rating

    from {{ ref('stg_customers') }} c
    inner join {{ source('delta_lake', 'dbo_ClientAgreement') }} ca
        on cast(ca.ClientID as string) = c.customer_id
    inner join {{ ref('stg_sales') }} s
        on cast(s.client_agreement_id as string) = cast(ca.ID as string)
    inner join {{ ref('stg_order_items') }} oi
        on cast(oi.order_id as string) = s.order_id

    where ca.Deleted = false
        and s.created_at is not null
        and oi.quantity > 0

    group by c.customer_id, oi.product_id
),

final as (
    select
        customer_id,
        product_id,

        -- Basic interaction strength
        num_purchases,
        total_quantity,
        total_spent,
        avg_quantity_per_order,
        avg_price_per_item,

        -- Implicit rating for collaborative filtering (LightFM, Neural CF)
        implicit_rating,

        -- Temporal context
        first_purchase_date,
        last_purchase_date,
        datediff('day', last_purchase_date, current_date) as days_since_last_purchase,
        datediff('day', first_purchase_date, last_purchase_date) as purchase_span_days,

        -- Repurchase prediction features (Survival Analysis)
        coalesce(avg_days_between_purchases, 999) as avg_days_between_purchases,
        coalesce(stddev_days_between_purchases, 0) as stddev_days_between_purchases,

        -- Expected next purchase date (Weibull prediction)
        case
            when avg_days_between_purchases is not null then
                dateadd('day', cast(avg_days_between_purchases as int), last_purchase_date)
            else null
        end as expected_next_purchase_date,

        -- Repurchase likelihood score (0-1)
        case
            when avg_days_between_purchases is not null and avg_days_between_purchases > 0 then
                least(1.0, greatest(0.0,
                    1.0 - (datediff('day', last_purchase_date, current_date)::float / avg_days_between_purchases)
                ))
            else 0.0
        end as repurchase_likelihood,

        -- Seasonality indicators (LSTM features)
        num_distinct_months,
        num_distinct_quarters,
        purchase_trend,

        -- Customer behavior flags
        case when num_purchases >= 5 then true else false end as is_repeat_customer,
        case when datediff('day', last_purchase_date, current_date) > 180 then true else false end as is_at_risk,
        case
            when avg_days_between_purchases is not null
                and datediff('day', last_purchase_date, current_date) > (avg_days_between_purchases * 1.5)
            then true else false
        end as is_overdue_for_repurchase,

        -- Revenue potential
        case
            when repurchase_likelihood > 0.7 then 'High'
            when repurchase_likelihood > 0.4 then 'Medium'
            else 'Low'
        end as revenue_potential,

        -- Metadata
        current_timestamp as features_updated_at

    from customer_product_interactions
)

select * from final
