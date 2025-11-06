-- Product Demand Time Series
-- Aggregates product sales by day/week/month for forecasting models

{{
    config(
        materialized='table',
        schema='ml_features'
    )
}}

with daily_sales as (
    select
        oi.product_id,
        cast(s.created_at as date) as sale_date,
        count(distinct s.sale_id) as num_orders,
        sum(oi.quantity) as quantity_sold,
        sum(oi.quantity * oi.price_per_item) as revenue

    from {{ ref('stg_order_items') }} oi
    inner join {{ ref('stg_sales') }} s
        on oi.order_id = s.sale_id
    where s.created_at is not null
        and oi.quantity > 0
    group by oi.product_id, cast(s.created_at as date)
),

weekly_sales as (
    select
        product_id,
        date_trunc('week', sale_date) as week_start_date,
        count(distinct sale_date) as num_sale_days,
        sum(num_orders) as num_orders,
        sum(quantity_sold) as quantity_sold,
        sum(revenue) as revenue,
        avg(quantity_sold) as avg_daily_quantity

    from daily_sales
    group by product_id, date_trunc('week', sale_date)
),

monthly_sales as (
    select
        product_id,
        date_trunc('month', sale_date) as month_start_date,
        count(distinct sale_date) as num_sale_days,
        sum(num_orders) as num_orders,
        sum(quantity_sold) as quantity_sold,
        sum(revenue) as revenue,
        avg(quantity_sold) as avg_daily_quantity

    from daily_sales
    group by product_id, date_trunc('month', sale_date)
),

product_demand_stats as (
    -- Calculate demand volatility and trends
    select
        product_id,
        count(*) as num_months_with_sales,
        min(month_start_date) as first_sale_month,
        max(month_start_date) as last_sale_month,
        avg(quantity_sold) as avg_monthly_quantity,
        stddev(quantity_sold) as stddev_monthly_quantity,
        -- Coefficient of variation (demand volatility)
        case
            when avg(quantity_sold) > 0 then stddev(quantity_sold) / avg(quantity_sold)
            else null
        end as demand_volatility

    from monthly_sales
    group by product_id
),

final as (
    select
        -- Daily granularity (for recent detailed forecasting)
        ds.product_id,
        ds.sale_date as date,
        'daily' as granularity,
        ds.num_orders,
        ds.quantity_sold,
        ds.revenue,
        null as num_sale_days,
        null as avg_daily_quantity,

        -- Demand characteristics
        pds.avg_monthly_quantity,
        pds.stddev_monthly_quantity,
        pds.demand_volatility,
        pds.num_months_with_sales,

        current_timestamp as features_updated_at

    from daily_sales ds
    left join product_demand_stats pds on ds.product_id = pds.product_id

    union all

    -- Weekly granularity (for medium-term forecasting)
    select
        ws.product_id,
        ws.week_start_date as date,
        'weekly' as granularity,
        ws.num_orders,
        ws.quantity_sold,
        ws.revenue,
        ws.num_sale_days,
        ws.avg_daily_quantity,

        pds.avg_monthly_quantity,
        pds.stddev_monthly_quantity,
        pds.demand_volatility,
        pds.num_months_with_sales,

        current_timestamp as features_updated_at

    from weekly_sales ws
    left join product_demand_stats pds on ws.product_id = pds.product_id

    union all

    -- Monthly granularity (for long-term forecasting and seasonality)
    select
        ms.product_id,
        ms.month_start_date as date,
        'monthly' as granularity,
        ms.num_orders,
        ms.quantity_sold,
        ms.revenue,
        ms.num_sale_days,
        ms.avg_daily_quantity,

        pds.avg_monthly_quantity,
        pds.stddev_monthly_quantity,
        pds.demand_volatility,
        pds.num_months_with_sales,

        current_timestamp as features_updated_at

    from monthly_sales ms
    left join product_demand_stats pds on ms.product_id = pds.product_id
)

select * from final
order by product_id, granularity, date
