-- Staging model for customers
-- Cleans and standardizes raw customer data from Delta Lake

{{
    config(
        materialized='view'
    )
}}

select
    -- Primary key
    cast(ID as string) as customer_id,

    -- Customer information
    trim(Name) as customer_name,
    trim(EmailAddress) as email,
    trim(MobileNumber) as mobile,
    trim(TIN) as tin,
    trim(USREOU) as usreou,

    -- Address information
    trim(LegalAddress) as legal_address,
    trim(ActualAddress) as actual_address,
    trim(DeliveryAddress) as delivery_address,
    cast(RegionID as string) as region_id,

    -- Metadata
    cast(Created as timestamp) as created_at,
    cast(Updated as timestamp) as updated_at,
    cast(Deleted as boolean) as is_deleted,

    -- Ingestion metadata
    _ingested_at,
    _source_table

from {{ source('delta_lake', 'dbo_Client') }}
where ID is not null
    and Deleted = false  -- Only active customers
