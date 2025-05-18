#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

echo "Starting build process..."

# Install system dependencies
echo "Installing system dependencies..."
apt-get update
apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    libpoppler-dev

# Clean up apt cache
apt-get clean
rm -rf /var/lib/apt/lists/*

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies with verbose output
echo "Installing Python dependencies..."
pip install --no-cache-dir -v -r requirements.txt

echo "Build completed successfully!"
