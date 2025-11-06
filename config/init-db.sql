-- Initialize databases for different services

-- MLflow database
CREATE DATABASE mlflow;

-- Dagster database
CREATE DATABASE dagster;

-- DataHub database
CREATE DATABASE datahub;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE mlflow TO concord;
GRANT ALL PRIVILEGES ON DATABASE dagster TO concord;
GRANT ALL PRIVILEGES ON DATABASE datahub TO concord;
