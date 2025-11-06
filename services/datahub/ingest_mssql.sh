#!/bin/bash
# Script to run DataHub ingestion for MSSQL

set -e

echo "Starting DataHub MSSQL ingestion..."

# Load environment variables
source /opt/datahub/.env 2>/dev/null || true

# Run ingestion
datahub ingest -c /opt/datahub/mssql-recipe.yml

echo "DataHub ingestion complete!"
