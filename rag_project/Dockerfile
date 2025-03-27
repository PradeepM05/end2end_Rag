FROM python:3.10-slim

WORKDIR /app

# Install wget for health checks
RUN apt-get update && apt-get install -y wget && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/logs

# Set environment variables
ENV PYTHONPATH="${PYTHONPATH}:/app"
ENV PORT=8000

# Run the FastAPI application
CMD exec uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT}