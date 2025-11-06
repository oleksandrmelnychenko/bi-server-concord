-- Product Graph Features for Graph Neural Network (GNN)
-- Leverages 1.7M product analogue relationships to build product knowledge graph
-- Creates node features + edge features for GraphSAGE / GAT models

{{
    config(
        materialized='table',
        schema='ml_features',
        indexes=[
            {'columns': ['product_id']},
            {'columns': ['graph_centrality']}
        ]
    )
}}

with product_analogues_bidirectional as (
    -- Make analogue relationships bidirectional for undirected graph
    select
        cast(BaseProductID as string) as source_product_id,
        cast(AnalogueProductID as string) as target_product_id,
        1.0 as edge_weight,
        'analogue' as relationship_type
    from {{ source('delta_lake', 'dbo_ProductAnalogue') }}
    where Deleted = false
        and BaseProductID is not null
        and AnalogueProductID is not null

    union all

    select
        cast(AnalogueProductID as string) as source_product_id,
        cast(BaseProductID as string) as target_product_id,
        1.0 as edge_weight,
        'analogue' as relationship_type
    from {{ source('delta_lake', 'dbo_ProductAnalogue') }}
    where Deleted = false
        and BaseProductID is not null
        and AnalogueProductID is not null
),

product_graph_metrics as (
    -- Calculate graph centrality metrics for each product node
    select
        source_product_id as product_id,

        -- Degree centrality (number of connections)
        count(distinct target_product_id) as num_analogues,

        -- Weighted degree
        sum(edge_weight) as total_edge_weight,

        -- Average weight
        avg(edge_weight) as avg_edge_weight

    from product_analogues_bidirectional
    group by source_product_id
),

product_purchase_metrics as (
    -- Purchase-based features from interaction matrix
    select
        product_id,
        count(distinct customer_id) as num_customers,
        sum(num_purchases) as total_purchases,
        sum(total_quantity) as total_quantity_sold,
        sum(total_spent) as total_revenue,
        avg(implicit_rating) as avg_customer_rating
    from {{ ref('interaction_matrix') }}
    group by product_id
),

product_category_features as (
    -- Product group / category features
    select
        cast(ProductID as string) as product_id,
        count(distinct ProductGroupID) as num_categories
    from {{ source('delta_lake', 'dbo_ProductProductGroup') }}
    where Deleted = false
        and ProductID is not null
    group by ProductID
),

product_specifications as (
    -- Product specifications for content-based features
    select
        cast(ProductID as string) as product_id,
        count(*) as num_specifications
    from {{ source('delta_lake', 'dbo_ProductSpecification') }}
    where Deleted = false
        and ProductID is not null
    group by ProductID
),

product_car_compatibility as (
    -- Car brand compatibility for automotive parts
    select
        cast(ProductID as string) as product_id,
        count(distinct CarBrandID) as num_compatible_brands
    from {{ source('delta_lake', 'dbo_ProductCarBrand') }}
    where Deleted = false
        and ProductID is not null
    group by ProductID
),

product_pricing_features as (
    -- Pricing history features
    select
        cast(ProductID as string) as product_id,
        count(*) as num_price_points,
        avg(Price) as avg_price,
        min(Price) as min_price,
        max(Price) as max_price,
        stddev(Price) as price_volatility
    from {{ source('delta_lake', 'dbo_ProductPricing') }}
    where Deleted = false
        and ProductID is not null
        and Price > 0
    group by ProductID
),

product_base_info as (
    -- Base product information
    select
        cast(ID as string) as product_id,
        trim(Name) as product_name,
        cast(IsForSale as boolean) as is_for_sale,
        cast(IsForWeb as boolean) as is_for_web,
        cast(HasAnalogue as boolean) as has_analogue,
        cast(HasImage as boolean) as has_image,
        cast(Weight as float) as weight
    from {{ source('delta_lake', 'dbo_Product') }}
    where Deleted = false
        and ID is not null
),

final as (
    select
        p.product_id,
        p.product_name,

        -- Product attributes (node features for GNN)
        coalesce(p.is_for_sale, false) as is_for_sale,
        coalesce(p.is_for_web, false) as is_for_web,
        coalesce(p.has_analogue, false) as has_analogue,
        coalesce(p.has_image, false) as has_image,
        coalesce(p.weight, 0) as weight,

        -- Graph structure metrics (GNN topology features)
        coalesce(g.num_analogues, 0) as num_analogues,
        coalesce(g.total_edge_weight, 0) as total_edge_weight,
        coalesce(g.avg_edge_weight, 0) as avg_edge_weight,

        -- Graph centrality score (normalized)
        case
            when g.num_analogues > 0 then
                least(1.0, g.num_analogues::float / 100.0)  -- Normalize to 0-1
            else 0.0
        end as graph_centrality,

        -- Purchase behavior (node features)
        coalesce(pm.num_customers, 0) as num_customers,
        coalesce(pm.total_purchases, 0) as total_purchases,
        coalesce(pm.total_quantity_sold, 0) as total_quantity_sold,
        coalesce(pm.total_revenue, 0) as total_revenue,
        coalesce(pm.avg_customer_rating, 0) as avg_customer_rating,

        -- Product content features
        coalesce(cat.num_categories, 0) as num_categories,
        coalesce(spec.num_specifications, 0) as num_specifications,
        coalesce(car.num_compatible_brands, 0) as num_compatible_brands,

        -- Pricing features
        coalesce(price.num_price_points, 0) as num_price_points,
        coalesce(price.avg_price, 0) as avg_price,
        coalesce(price.min_price, 0) as min_price,
        coalesce(price.max_price, 0) as max_price,
        coalesce(price.price_volatility, 0) as price_volatility,

        -- Product popularity tier (for model weighting)
        case
            when pm.total_purchases >= 100 then 'Very Popular'
            when pm.total_purchases >= 50 then 'Popular'
            when pm.total_purchases >= 10 then 'Moderate'
            when pm.total_purchases >= 1 then 'Low'
            else 'Never Sold'
        end as popularity_tier,

        -- Graph hub score (products with many connections and high sales)
        case
            when g.num_analogues >= 100 and pm.total_purchases >= 50 then 'Hub Product'
            when g.num_analogues >= 50 and pm.total_purchases >= 10 then 'Popular Node'
            when g.num_analogues >= 10 then 'Connected Node'
            else 'Isolated Node'
        end as graph_role,

        -- Composite feature for GNN message passing
        (coalesce(g.num_analogues, 0) * 0.3) +
        (coalesce(pm.num_customers, 0) * 0.3) +
        (coalesce(cat.num_categories, 0) * 0.2) +
        (coalesce(car.num_compatible_brands, 0) * 0.2) as node_importance_score,

        -- Metadata
        current_timestamp as features_updated_at

    from product_base_info p
    left join product_graph_metrics g on p.product_id = g.product_id
    left join product_purchase_metrics pm on p.product_id = pm.product_id
    left join product_category_features cat on p.product_id = cat.product_id
    left join product_specifications spec on p.product_id = spec.product_id
    left join product_car_compatibility car on p.product_id = car.product_id
    left join product_pricing_features price on p.product_id = price.product_id
)

select * from final
