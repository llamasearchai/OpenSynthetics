FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml README.md ./
RUN pip install --upgrade pip && \
    pip install . && \
    pip install uvicorn gunicorn

# Copy application code
COPY opensynthetics ./opensynthetics

# Create directories
RUN mkdir -p /data/opensynthetics

# Environment variables
ENV OPENSYNTHETICS_DATA_DIR=/data/opensynthetics
ENV OPENSYNTHETICS_ENV=prod

# Expose ports
EXPOSE 8000
EXPOSE 8001

# Run API server by default
CMD ["gunicorn", "opensynthetics.api.main:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]