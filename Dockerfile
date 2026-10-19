# Multi-stage Dockerfile for FactForge Grounded Fact Layer
FROM python:3.11-slim as base

WORKDIR /app

# Install system dependencies for PyMuPDF and C++ build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose FastAPI backend (8000) and Streamlit UI (8501)
EXPOSE 8000
EXPOSE 8501

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
