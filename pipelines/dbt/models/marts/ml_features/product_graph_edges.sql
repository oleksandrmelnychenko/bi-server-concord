-- Product Graph Edge List for GNN
-- Contains all product-to-product relationships (1.7M analogues)
-- Used for Graph Neural Network training and inference

{{
    config(
        materialized='table',
        schema='ml_features',
        indexes=[
            {'columns': ['source_product_id', 'target_product_id']},
            {'columns': ['relationship_type']}
        ]
    )
}}

with product_analogues_edges as (
    -- Direct analogue relationships (directed edges)
    select
        cast(BaseProductID as string) as source_product_id,
        cast(AnalogueProductID as string) as target_product_id,
        1.0 as edge_weight,
        'analogue' as relationship_type,
        'ProductAnalogue' as source_table,
        cast(Created as timestamp) as relationship_created_at
    from {{ source('delta_lake', 'dbo_ProductAnalogue') }}
    where Deleted = false
        and BaseProductID is not null
        and AnalogueProductID is not null
),

product_compatibility_edges as (
    -- Products compatible with same car brands (implicit similarity)
    select distinct
        cast(p1.ProductID as string) as source_product_id,
        cast(p2.ProductID as string) as target_product_id,
        0.5 as edge_weight,
        'car_compatibility' as relationship_type,
        'ProductCarBrand' as source_table,
        current_timestamp as relationship_created_at
    from {{ source('delta_lake', 'dbo_ProductCarBrand') }} p1
    inner join {{ source('delta_lake', 'dbo_ProductCarBrand') }} p2
        on p1.CarBrandID = p2.CarBrandID
        and p1.ProductID < p2.ProductID  -- Avoid duplicates
    where p1.Deleted = false
        and p2.Deleted = false
        and p1.ProductID is not null
        and p2.ProductID is not null
    limit 100000  -- Cap compatibility edges to manageable size
),

product_category_edges as (
    -- Products in same category (weak similarity)
    select distinct
        cast(p1.ProductID as string) as source_product_id,
        cast(p2.ProductID as string) as target_product_id,
        0.3 as edge_weight,
        'same_category' as relationship_type,
        'ProductProductGroup' as source_table,
        current_timestamp as relationship_created_at
    from {{ source('delta_lake', 'dbo_ProductProductGroup') }} p1
    inner join {{ source('delta_lake', 'dbo_ProductProductGroup') }} p2
        on p1.ProductGroupID = p2.ProductGroupID
        and p1.ProductID < p2.ProductID  -- Avoid duplicates
    where p1.Deleted = false
        and p2.Deleted = false
        and p1.ProductID is not null
        and p2.ProductID is not null
    limit 50000  -- Cap category edges
),

frequently_bought_together as (
    -- Products bought together in same orders (strong behavioral signal)
    select
        cast(oi1.ProductID as string) as source_product_id,
        cast(oi2.ProductID as string) as target_product_id,
        count(*) / 10.0 as edge_weight,  -- Normalize by dividing by 10
        'bought_together' as relationship_type,
        'OrderItem' as source_table,
        max(oi1.Created) as relationship_created_at
    from {{ source('delta_lake', 'dbo_OrderItem') }} oi1
    inner join {{ source('delta_lake', 'dbo_OrderItem') }} oi2
        on oi1.OrderID = oi2.OrderID
        and oi1.ProductID < oi2.ProductID  -- Avoid duplicates
    where oi1.Deleted = false
        and oi2.Deleted = false
        and oi1.ProductID is not null
        and oi2.ProductID is not null
    group by oi1.ProductID, oi2.ProductID
    having count(*) >= 2  -- Products bought together at least twice
),

final as (
    -- Combine all edge types
    select * from product_analogues_edges

    union all

    select * from frequently_bought_together

    -- Uncomment to add more edge types (may slow down GNN training)
    -- union all
    -- select * from product_compatibility_edges
    -- union all
    -- select * from product_category_edges
)

select
    source_product_id,
    target_product_id,
    edge_weight,
    relationship_type,
    source_table,
    relationship_created_at,
    current_timestamp as features_updated_at
from final
