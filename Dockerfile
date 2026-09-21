# Use official Python lightweight image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy dependency requirements
COPY requirements.txt .

# Install dependencies (ignoring jupyter and visualization libs to save space if desired, but we just use requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and necessary source modules
COPY app/ ./app/
COPY src/ ./src/
COPY artifacts/ ./artifacts/
COPY results/calibration/ ./results/calibration/

# Expose API port
EXPOSE 8000

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Run the FastAPI server via Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
