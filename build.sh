#!/usr/bin/env bash
set -o errexit

echo "Starting build process..."

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install packages one by one to identify problematic packages
echo "Installing Python dependencies individually..."

# Core packages first
pip install flask==2.3.3
pip install python-dotenv==1.0.0
pip install gunicorn==21.2.0

# OpenAI and LangChain
pip install openai==1.3.0
pip install langchain==0.0.335
pip install langchain-openai==0.0.2
pip install tiktoken==0.5.1

# Document processing
pip install pdfplumber==0.10.2
pip install pymupdf==1.23.5
pip install pytesseract==0.3.10
pip install pillow==10.0.1

# Data processing and visualization
pip install numpy==1.26.1
pip install pandas==2.1.1
pip install scikit-learn==1.3.2
pip install networkx==3.2.1
pip install plotly==5.18.0

# Vector database and search
pip install chromadb==0.4.18
pip install rank-bm25==0.2.2

# NLP
pip install nltk==3.8.1
pip install werkzeug==2.3.7

# Download NLTK data
python -m nltk.downloader punkt

echo "Build completed successfully!"
