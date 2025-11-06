"""
Dagster Repository - Defines all pipelines and schedules
"""
from dagster import repository, job, op, schedule, ScheduleDefinition
import os
import logging

logger = logging.getLogger(__name__)


@op
def ingest_mssql_to_delta():
    """Ingest data from MSSQL to Delta Lake"""
    from pipelines.ingestion.mssql_to_delta import MSSQLToDeltaIngestion

    config = {
        "mssql_host": os.getenv("MSSQL_HOST"),
        "mssql_database": os.getenv("MSSQL_DATABASE"),
        "mssql_user": os.getenv("MSSQL_USER"),
        "mssql_password": os.getenv("MSSQL_PASSWORD"),
        "delta_base_path": "/opt/dagster/app/data/delta"
    }

    ingestion = MSSQLToDeltaIngestion(**config)
    results = ingestion.ingest_all_tables(mode="append")

    successful = sum(1 for r in results if r['status'] == 'success')
    total_rows = sum(r.get('rows', 0) for r in results)

    logger.info(f"Ingestion complete: {successful} tables, {total_rows} rows")

    return {"successful_tables": successful, "total_rows": total_rows}


@job
def mssql_ingestion_job():
    """Job for ingesting MSSQL data to Delta Lake"""
    ingest_mssql_to_delta()


@schedule(
    job=mssql_ingestion_job,
    cron_schedule="0 2 * * *"  # Daily at 2 AM
)
def daily_mssql_ingestion_schedule():
    """Daily schedule for MSSQL ingestion"""
    return {}


@repository
def concord_bi_repository():
    """Repository containing all jobs and schedules"""
    return [
        mssql_ingestion_job,
        daily_mssql_ingestion_schedule
    ]
