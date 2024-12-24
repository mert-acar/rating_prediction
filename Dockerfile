FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and models
COPY src/ src/
COPY models/ models/

# Set environment variables
ENV PYTHONPATH=/app

# Expose the port
EXPOSE 9001

# Run the API server
CMD ["python", "src/api.py"]
