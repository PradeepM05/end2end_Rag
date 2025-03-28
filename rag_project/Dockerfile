FROM python:3.10-slim

WORKDIR /app

# Install dependencies for Google Cloud SDK and curl for health check
RUN apt-get update && \
    apt-get install -y curl gnupg apt-transport-https ca-certificates && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update && \
    apt-get install -y google-cloud-sdk && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/vector_store /app/logs

# Set environment variables
ENV PYTHONPATH="${PYTHONPATH}:/app"
ENV PORT=8000

# Run the FastAPI application
CMD exec uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT}