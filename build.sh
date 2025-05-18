#!/usr/bin/env bash
set -o errexit

echo "Starting build process..."

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install packages in groups with compatible versions
echo "Installing Python dependencies..."

# Core packages first
echo "Installing core packages..."
pip install flask==2.2.5 werkzeug==2.2.3 gunicorn==21.2.0 python-dotenv==1.0.0

# OpenAI and LangChain (newer versions for compatibility)
echo "Installing OpenAI and LangChain..."
pip install openai==1.6.1 tiktoken==0.5.2
pip install langchain==0.0.335
pip install langchain-openai==0.0.2

# Document processing
echo "Installing document processing packages..."
pip install pdfplumber==0.10.2 pymupdf==1.23.5 pytesseract==0.3.10 pillow==10.0.1

# Data processing and visualization (compatible versions)
echo "Installing data processing packages..."
pip install numpy==1.24.3 pandas==2.0.3
pip install scikit-learn==1.3.0 networkx==3.1 plotly==5.15.0

# Vector database and search
echo "Installing vector database..."
pip install chromadb==0.4.13 rank-bm25==0.2.2

# NLP
echo "Installing NLP packages..."
pip install nltk==3.8.1

# Download NLTK data
echo "Downloading NLTK data..."
python -m nltk.downloader punkt

echo "Build completed successfully!"
