-- Customer Purchase Sequences for LSTM Model
-- Creates temporal sequences of customer purchases for next-product prediction
-- Captures order history, product sequences, and temporal patterns

{{
    config(
        materialized='table',
        schema='ml_features',
        indexes=[
            {'columns': ['customer_id']},
            {'columns': ['sequence_position']}
        ]
    )
}}

with customer_order_sequences as (
    -- Get all orders for each customer in chronological order
    select
        c.customer_id,
        s.sale_id as order_id,
        s.created_at as order_date,
        row_number() over (partition by c.customer_id order by s.created_at) as sequence_position,

        -- Time features
        extract(year from s.created_at) as order_year,
        extract(month from s.created_at) as order_month,
        extract(day from s.created_at) as order_day,
        extract(dow from s.created_at) as order_day_of_week,  -- 0=Sunday, 6=Saturday
        extract(quarter from s.created_at) as order_quarter,

        -- Days since last order
        datediff('day',
            lag(s.created_at) over (partition by c.customer_id order by s.created_at),
            s.created_at
        ) as days_since_last_order,

        -- Order-level aggregates
        count(oi.product_id) over (partition by s.sale_id) as num_items_in_order,
        sum(oi.quantity) over (partition by s.sale_id) as total_quantity_in_order,
        sum(oi.quantity * oi.price_per_item) over (partition by s.sale_id) as order_value

    from {{ ref('stg_customers') }} c
    inner join {{ source('delta_lake', 'dbo_ClientAgreement') }} ca
        on cast(ca.ClientID as string) = c.customer_id
    inner join {{ ref('stg_sales') }} s
        on cast(s.client_agreement_id as string) = cast(ca.ID as string)
    left join {{ ref('stg_order_items') }} oi
        on cast(oi.order_id as string) = s.order_id

    where ca.Deleted = false
        and s.created_at is not null

    qualify row_number() over (partition by c.customer_id, s.sale_id order by s.created_at) = 1
),

product_sequences as (
    -- Product-level sequences (what products in what order)
    select
        c.customer_id,
        s.sale_id as order_id,
        s.created_at as order_date,
        oi.product_id,
        p.product_name,
        oi.quantity,
        oi.price_per_item,
        oi.quantity * oi.price_per_item as line_total,

        -- Sequence position (overall across all orders)
        row_number() over (
            partition by c.customer_id
            order by s.created_at, oi.created_at
        ) as global_product_sequence,

        -- Position within this specific order
        row_number() over (
            partition by c.customer_id, s.sale_id
            order by oi.created_at
        ) as product_position_in_order,

        -- Order-level sequence number
        dense_rank() over (
            partition by c.customer_id
            order by s.created_at
        ) as order_sequence_number,

        -- Product features for LSTM input
        coalesce(pf.num_analogues, 0) as product_num_analogues,
        coalesce(pf.graph_centrality, 0) as product_graph_centrality,
        coalesce(pf.avg_price, oi.price_per_item) as product_avg_price,
        coalesce(pf.popularity_tier, 'Unknown') as product_popularity_tier

    from {{ ref('stg_customers') }} c
    inner join {{ source('delta_lake', 'dbo_ClientAgreement') }} ca
        on cast(ca.ClientID as string) = c.customer_id
    inner join {{ ref('stg_sales') }} s
        on cast(s.client_agreement_id as string) = cast(ca.ID as string)
    inner join {{ ref('stg_order_items') }} oi
        on cast(oi.order_id as string) = s.order_id
    left join {{ source('delta_lake', 'dbo_Product') }} p
        on cast(p.ID as string) = oi.product_id
    left join {{ ref('product_graph_features') }} pf
        on pf.product_id = oi.product_id

    where ca.Deleted = false
        and s.created_at is not null
        and oi.quantity > 0
),

aggregated_sequences as (
    -- Aggregate products per order for LSTM input
    select
        ps.customer_id,
        ps.order_id,
        ps.order_date,
        ps.order_sequence_number,

        -- List of products in this order (for sequence modeling)
        array_agg(ps.product_id order by ps.product_position_in_order) as product_ids_in_order,
        array_agg(ps.product_name order by ps.product_position_in_order) as product_names_in_order,
        array_agg(ps.quantity order by ps.product_position_in_order) as quantities_in_order,

        -- Order summary stats
        count(distinct ps.product_id) as num_unique_products,
        sum(ps.quantity) as total_quantity,
        sum(ps.line_total) as total_order_value,
        avg(ps.line_total) as avg_item_value,

        -- Product diversity metrics
        count(distinct ps.product_popularity_tier) as num_popularity_tiers,
        avg(ps.product_graph_centrality) as avg_product_centrality

    from product_sequences ps
    group by
        ps.customer_id,
        ps.order_id,
        ps.order_date,
        ps.order_sequence_number
),

customer_sequence_features as (
    -- Customer-level sequence features for LSTM
    select
        o.customer_id,
        o.order_id,
        o.order_date,
        o.sequence_position,

        -- Temporal features
        o.order_year,
        o.order_month,
        o.order_day,
        o.order_day_of_week,
        o.order_quarter,
        coalesce(o.days_since_last_order, 999) as days_since_last_order,

        -- Order-level features
        o.num_items_in_order,
        o.total_quantity_in_order,
        o.order_value,

        -- Product sequence data
        a.product_ids_in_order,
        a.product_names_in_order,
        a.quantities_in_order,
        a.num_unique_products,
        a.avg_product_centrality,

        -- Sequence context (for LSTM)
        -- Previous N orders (lookback window)
        lag(a.product_ids_in_order, 1) over (partition by o.customer_id order by o.sequence_position) as prev_order_products_1,
        lag(a.product_ids_in_order, 2) over (partition by o.customer_id order by o.sequence_position) as prev_order_products_2,
        lag(a.product_ids_in_order, 3) over (partition by o.customer_id order by o.sequence_position) as prev_order_products_3,

        -- Next order (target for prediction)
        lead(a.product_ids_in_order, 1) over (partition by o.customer_id order by o.sequence_position) as next_order_products,
        lead(o.order_date, 1) over (partition by o.customer_id order by o.sequence_position) as next_order_date,

        -- Seasonality indicators
        case when o.order_month in (12, 1, 2) then true else false end as is_winter,
        case when o.order_month in (3, 4, 5) then true else false end as is_spring,
        case when o.order_month in (6, 7, 8) then true else false end as is_summer,
        case when o.order_month in (9, 10, 11) then true else false end as is_fall,
        case when o.order_day_of_week in (0, 6) then true else false end as is_weekend

    from customer_order_sequences o
    left join aggregated_sequences a
        on o.customer_id = a.customer_id
        and o.order_id = a.order_id
),

final as (
    select
        customer_id,
        order_id,
        order_date,
        sequence_position,

        -- Temporal features (LSTM input)
        order_year,
        order_month,
        order_day,
        order_day_of_week,
        order_quarter,
        days_since_last_order,

        -- Seasonality flags
        is_winter,
        is_spring,
        is_summer,
        is_fall,
        is_weekend,

        -- Order features
        num_items_in_order,
        total_quantity_in_order,
        order_value,
        num_unique_products,
        avg_product_centrality,

        -- Product sequences (LSTM core data)
        product_ids_in_order,
        product_names_in_order,
        quantities_in_order,

        -- Historical context (lookback window for LSTM)
        prev_order_products_1,
        prev_order_products_2,
        prev_order_products_3,

        -- Target (what to predict)
        next_order_products,
        next_order_date,

        -- Days until next order (survival target)
        case
            when next_order_date is not null then
                datediff('day', order_date, next_order_date)
            else null
        end as days_until_next_order,

        -- Metadata
        current_timestamp as features_updated_at

    from customer_sequence_features
)

select * from final
order by customer_id, sequence_position
