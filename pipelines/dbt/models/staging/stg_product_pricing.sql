-- Staging model for product pricing
-- Cleans and standardizes product pricing history

{{
    config(
        materialized='view'
    )
}}

select
    -- Primary key
    cast(ID as string) as product_pricing_id,

    -- Foreign keys
    cast(ProductID as string) as product_id,
    cast(PricingID as string) as pricing_id,

    -- Pricing
    cast(Price as decimal(18,2)) as price,

    -- Metadata
    cast(Created as timestamp) as created_at,
    cast(Updated as timestamp) as updated_at,
    cast(Deleted as boolean) as is_deleted,

    -- Ingestion metadata
    _ingested_at,
    _source_table

from {{ source('delta_lake', 'dbo_ProductPricing') }}
where ID is not null
    and Deleted = false  -- Only active pricing
    and ProductID is not null
    and Price > 0  -- Only positive prices
