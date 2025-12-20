# Dockerfile for DB AI API
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for MSSQL
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    unixodbc-dev \
    unixodbc \
    apt-transport-https \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Microsoft ODBC Driver for SQL Server (modern approach for Debian 12+)
RUN curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg \
    && curl -fsSL https://packages.microsoft.com/config/debian/12/prod.list | tee /etc/apt/sources.list.d/mssql-release.list \
    && sed -i 's|signed-by=/usr/share/keyrings/microsoft-prod.gpg||g' /etc/apt/sources.list.d/mssql-release.list || true \
    && echo "deb [arch=amd64,arm64,armhf signed-by=/usr/share/keyrings/microsoft-prod.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql18 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create logs directory
RUN mkdir -p logs

# Create vector database directory
RUN mkdir -p vector_db

# Create query examples database directory
RUN mkdir -p chroma_db_examples_v2

# Expose API ports (8000=RAG, 8001=Analytics, 8003=Text-to-SQL)
EXPOSE 8000 8001 8003

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "main.py"]
