# SentimentSense: Vultr-Optimized Production Environment
# This Dockerfile is designed for high-performance deployment on Vultr Cloud Compute
# or Vultr Cloud GPUs for privacy-first, self-hosted AI inference.

FROM python:3.9-slim-buster

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DEBIAN_FRONTEND noninteractive

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Exposure port for FastAPI
EXPOSE 8000

# Start command with Uvicorn optimized for production
# Performance Note: Increasing workers scales with Vultr vCPU count
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
