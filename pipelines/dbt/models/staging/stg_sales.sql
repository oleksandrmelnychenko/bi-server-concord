-- Staging model for sales
-- Cleans and standardizes sales/order data

{{
    config(
        materialized='view'
    )
}}

select
    -- Primary key
    cast(ID as string) as sale_id,

    -- Foreign keys
    cast(ClientAgreementID as string) as client_agreement_id,
    cast(OrderID as string) as order_id,
    cast(UserID as string) as user_id,

    -- Status
    cast(BaseLifeCycleStatusID as string) as lifecycle_status_id,
    cast(BaseSalePaymentStatusID as string) as payment_status_id,

    -- Additional information
    trim(Comment) as comment,
    cast(IsMerged as boolean) as is_merged,

    -- Metadata
    cast(Created as timestamp) as created_at,
    cast(Updated as timestamp) as updated_at,
    cast(Deleted as boolean) as is_deleted,

    -- Ingestion metadata
    _ingested_at,
    _source_table

from {{ source('delta_lake', 'dbo_Sale') }}
where ID is not null
    and Deleted = false  -- Only active sales
