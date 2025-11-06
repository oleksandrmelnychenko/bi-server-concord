-- Staging model for order items
-- Cleans and standardizes order line items

{{
    config(
        materialized='view'
    )
}}

select
    -- Primary key
    cast(ID as string) as order_item_id,

    -- Foreign keys
    cast(OrderID as string) as order_id,
    cast(ProductID as string) as product_id,
    cast(UserId as string) as user_id,

    -- Quantities
    cast(Qty as decimal(18,4)) as quantity,
    cast(OrderedQty as decimal(18,4)) as ordered_quantity,

    -- Pricing
    cast(PricePerItem as decimal(18,2)) as price_per_item,
    cast(OneTimeDiscount as decimal(18,2)) as one_time_discount,

    -- Flags
    cast(IsValidForCurrentSale as boolean) as is_valid_for_sale,
    cast(IsFromOffer as boolean) as is_from_offer,

    -- Metadata
    trim(Comment) as comment,
    cast(Created as timestamp) as created_at,
    cast(Updated as timestamp) as updated_at,
    cast(Deleted as boolean) as is_deleted,

    -- Ingestion metadata
    _ingested_at,
    _source_table

from {{ source('delta_lake', 'dbo_OrderItem') }}
where ID is not null
    and Deleted = false  -- Only active items
    and Qty > 0  -- Only items with positive quantity
