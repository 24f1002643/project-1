# Start from official Python image
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (Git + pip prerequisites)
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

# Copy app code
COPY app/ /app

# Create writable logs directory
RUN mkdir -p /app/logs && chmod -R 777 /app/logs

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (Hugging Face Spaces typically uses 7860)
EXPOSE 7860

# Environment setup
ENV PYTHONUNBUFFERED=1

# (Optional) Run as non-root user for safety (especially on HF Spaces)
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Start the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers"]
