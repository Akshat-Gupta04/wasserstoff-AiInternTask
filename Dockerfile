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

# Upgrade pip and install dependencies with conflict resolution
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir pip-tools && \
    (pip-compile requirements.txt --output-file requirements.lock || echo "pip-compile failed, using alternative method") && \
    (pip install --no-cache-dir -r requirements.lock || \
     (echo "Installing packages individually..." && \
      # Core packages first
      pip install flask==2.2.5 werkzeug==2.2.3 gunicorn==21.2.0 python-dotenv==1.0.0 && \
      # OpenAI and LangChain (older versions for compatibility)
      pip install openai==0.28.1 tiktoken==0.5.1 && \
      pip install langchain==0.0.267 && \
      pip install langchain-openai==0.0.2 && \
      # Document processing
      pip install pdfplumber==0.10.2 pymupdf==1.23.5 pytesseract==0.3.10 pillow==10.0.1 && \
      # Data processing and visualization (compatible versions)
      pip install numpy==1.24.3 pandas==2.0.3 && \
      pip install scikit-learn==1.3.0 networkx==3.1 plotly==5.15.0 && \
      # Vector database and search
      pip install chromadb==0.4.13 rank-bm25==0.2.2 && \
      # NLP
      pip install nltk==3.8.1))

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
