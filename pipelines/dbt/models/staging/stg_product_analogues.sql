-- Staging model for product analogues
-- Cleans and standardizes product alternative/analogue relationships

{{
    config(
        materialized='view'
    )
}}

select
    -- Primary key
    cast(ID as string) as product_analogue_id,

    -- Foreign keys
    cast(BaseProductID as string) as base_product_id,
    cast(AnalogueProductID as string) as analogue_product_id,

    -- Metadata
    cast(Created as timestamp) as created_at,
    cast(Updated as timestamp) as updated_at,
    cast(Deleted as boolean) as is_deleted,

    -- Ingestion metadata
    _ingested_at,
    _source_table

from {{ source('delta_lake', 'dbo_ProductAnalogue') }}
where ID is not null
    and Deleted = false  -- Only active relationships
    and BaseProductID is not null
    and AnalogueProductID is not null
    and BaseProductID != AnalogueProductID  -- Avoid self-references
