-- Staging model for products
-- Cleans and standardizes product catalog data

{{
    config(
        materialized='view'
    )
}}

select
    -- Primary key
    cast(ID as string) as product_id,

    -- Product information
    trim(Name) as product_name,
    trim(Description) as description,
    trim(VendorCode) as vendor_code,
    trim(MainOriginalNumber) as main_original_number,

    -- Flags
    cast(IsForSale as boolean) as is_for_sale,
    cast(IsForWeb as boolean) as is_for_web,
    cast(HasAnalogue as boolean) as has_analogue,
    cast(HasImage as boolean) as has_image,

    -- Specifications
    trim(Size) as size,
    trim(Volume) as volume,
    cast(MeasureUnitID as string) as measure_unit_id,

    -- Metadata
    cast(Created as timestamp) as created_at,
    cast(Updated as timestamp) as updated_at,
    cast(Deleted as boolean) as is_deleted,

    -- Ingestion metadata
    _ingested_at,
    _source_table

from {{ source('delta_lake', 'dbo_Product') }}
where ID is not null
    and Deleted = false  -- Only active products
