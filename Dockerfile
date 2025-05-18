FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    libpoppler-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt')"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/uploads static/graphs logs

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV FLASK_DEBUG=0

# Expose port
EXPOSE 10000

# Run the application with Gunicorn
CMD gunicorn wsgi:app --bind 0.0.0.0:$PORT --workers=4 --threads=2 --timeout=120 --log-level=info
