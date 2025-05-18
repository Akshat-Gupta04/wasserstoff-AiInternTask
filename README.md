# Document Research & Theme Identification Chatbot

[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/Akshat-Gupta04/wasserstoff-AiInternTask)
![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/flask-2.0%2B-green)
![License](https://img.shields.io/badge/license-MIT-orange)

## Developer

This application was developed by **Akshat Gupta**.

GitHub: [Akshat-Gupta04](https://github.com/Akshat-Gupta04)

Repository: [wasserstoff-AiInternTask](https://github.com/Akshat-Gupta04/wasserstoff-AiInternTask)

A comprehensive Flask-based web application for document analysis, theme identification, and knowledge graph visualization. This application allows users to upload multiple documents, ask questions about their content, and explore the relationships between themes and information through interactive visualizations.

## 📋 Table of Contents

- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Methodology](#-methodology)
- [Installation](#-installation)
- [Usage](#-usage)
- [Deployment](#-deployment)
- [Technologies Used](#-technologies-used)
- [Project Structure](#-project-structure)
- [License](#-license)

## ✨ Features

- **Multi-Document Processing**: Upload and analyze multiple PDF and image files simultaneously
- **Advanced Text Extraction**: Combines PyMuPDF, pdfplumber, and OCR for robust text extraction
- **Semantic Search**: Vector embeddings with OpenAI's text-embedding-3-small model
- **Theme Identification**: Automatic discovery of key themes using NMF and TF-IDF
- **Knowledge Graph Visualization**: Interactive graph showing relationships between documents and themes
- **Continuous Chat Interface**: Ask follow-up questions while maintaining context
- **Multiple Chat Sessions**: Create and manage separate research sessions
- **Detailed Citations**: Answers include specific document, page, and paragraph citations
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Processing Animations**: Visual feedback during document analysis and query processing

## 🏗 System Architecture

The application follows a modular architecture with the following components:

1. **Frontend Layer**:
   - HTML/CSS/JavaScript interface with Bootstrap 5
   - Interactive visualizations using Plotly
   - Dynamic animations for processing feedback
   - Responsive design for all device sizes

2. **Application Layer**:
   - Flask web framework for routing and request handling
   - Session management for multiple chat contexts
   - Document processing pipeline
   - Query processing and response generation

3. **Data Processing Layer**:
   - Text extraction from multiple document formats
   - Vector embedding generation
   - Theme identification using NMF
   - Knowledge graph construction

4. **Storage Layer**:
   - SQLite database for document text and metadata
   - ChromaDB for vector embeddings and semantic search
   - Session-specific data isolation

## 🔍 Methodology

### Document Processing

1. **Text Extraction**:
   - Primary: PyMuPDF for direct text extraction from PDFs
   - Secondary: pdfplumber as an alternative PDF text extractor
   - Tertiary: OCR using pytesseract for image-based or scanned documents

2. **Text Processing**:
   - Splits text into paragraphs for granular analysis
   - Filters out short or empty paragraphs
   - Normalizes whitespace and formatting

3. **Vector Embedding**:
   - Generates embeddings using OpenAI's text-embedding-3-small
   - Stores embeddings in ChromaDB for semantic search
   - Includes metadata for citation tracking

### Query Processing

1. **Semantic Search**:
   - Generates vector embedding for the query
   - Performs similarity search against document embeddings
   - Retrieves the most semantically relevant paragraphs

2. **Hybrid Ranking**:
   - Combines results from semantic and keyword search
   - Weights and normalizes scores for balanced retrieval
   - Ensures both semantic meaning and keyword relevance are considered

3. **Response Generation**:
   - Uses OpenAI's GPT model to generate comprehensive answers
   - Provides explicit instructions to cite sources properly
   - Formats response with citations for traceability

### Theme Identification

1. **Vectorization**:
   - Applies TF-IDF vectorization to identify important terms
   - Uses Non-negative Matrix Factorization (NMF) for topic modeling
   - Optimizes number of topics based on coherence scores

2. **Theme Refinement**:
   - Uses LLM to refine raw themes into coherent concepts
   - Considers query context to ensure themes are relevant
   - Produces concise, human-readable theme descriptions

3. **Knowledge Graph Construction**:
   - Creates a directed graph using NetworkX
   - Represents documents and themes as nodes
   - Establishes weighted edges based on document-theme relationships
   - Marks query-relevant documents with special attributes

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- OpenAI API key
- Tesseract OCR (for image processing)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Akshat-Gupta04/wasserstoff-AiInternTask.git
   cd wasserstoff-AiInternTask
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install Tesseract OCR:
   - On macOS: `brew install tesseract`
   - On Ubuntu: `sudo apt-get install tesseract-ocr`
   - On Windows: Download and install from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

5. Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

6. Run the application:
   ```bash
   python app.py
   ```

7. Open your browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

> **Note**: The application now processes files in memory without saving them to disk, making it more suitable for deployment on platforms with limited storage like Render.

## 📖 Usage

### Uploading Documents

1. On the home page, click the "Browse" button to select documents
2. Hold Ctrl/Cmd to select multiple files (PDF, PNG, JPG, JPEG supported)
3. Enter your initial question in the text field
4. Click "Analyze Documents" to process the files and start a new chat session

### Asking Questions

1. Type your question in the input field at the bottom of the chat interface
2. Click the send button or press Enter to submit your question
3. View the response, which includes:
   - A comprehensive answer to your question
   - Key themes identified in the relevant documents
   - A knowledge graph showing relationships between documents and themes
   - A table of extracted answers with citations

### Using the Knowledge Graph

1. Click the "Graph" tab to view the knowledge graph
2. Use the dropdown menu to filter by relevance or specific themes
3. Hover over nodes to see detailed information
4. Click the fullscreen button for a larger view
5. Zoom and pan to explore complex graphs

### Managing Sessions

1. Create a new chat session by clicking "New Chat" in the sidebar
2. Switch between existing sessions by clicking on them in the sidebar
3. Delete a session by clicking the "×" button next to its name

## 🚢 Deployment

This application is configured for easy deployment on Render.com and other cloud platforms.

### Deploying to Render

1. **Create a Render account**:
   - Sign up at [render.com](https://render.com)

2. **Connect your GitHub repository**:
   - In the Render dashboard, click "New" and select "Web Service"
   - Connect to your GitHub repository

3. **Configure the service**:
   - Name: `document-research-chatbot` (or your preferred name)
   - Environment: `Python 3`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
   - Select the appropriate instance type (at least 1GB RAM recommended)

4. **Add environment variables**:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `LLM_MODEL`: `gpt-3.5-turbo`
   - `EMBEDDING_MODEL`: `text-embedding-3-small`
   - `SECRET_KEY`: Generate a secure random string

5. **Create a disk**:
   - In the Render dashboard, go to "Disks"
   - Create a new disk with at least 10GB
   - Mount path: `/data`
   - Attach it to your web service

6. **Deploy**:
   - Click "Create Web Service"
   - Render will automatically build and deploy your application

### Using render.yaml (Recommended)

This repository includes a `render.yaml` file for Blueprint deployments:

1. Fork this repository to your GitHub account
2. In Render, click "New" and select "Blueprint"
3. Connect to your forked repository
4. Render will automatically configure all services based on the render.yaml file
5. Add your `OPENAI_API_KEY` when prompted
6. Click "Apply" to deploy

### Memory-Efficient Processing

The application has been optimized for deployment on platforms with limited storage:

- **In-Memory File Processing**: Files are processed directly in memory without being saved to disk
- **Session Isolation**: Each user session has its own isolated vector database
- **Efficient Resource Usage**: Minimizes disk I/O operations for better performance
- **Reduced Storage Requirements**: No need for large persistent storage for uploaded files

### Data Persistence

The application stores only essential data in a persistent disk on Render:

- **Disk Configuration**: A 10GB disk is mounted at `/data` to store:
  - SQLite database (`documents.db`)
  - ChromaDB vector database
  - Generated knowledge graphs

- **Environment Variables**: The application uses these paths:
  - `DATA_FOLDER=/data`: Main data directory
  - `UPLOAD_FOLDER=/data/uploads`: For temporary processing (not used for storage)
  - `STATIC_FOLDER=static`: Static assets (CSS, JS, etc.)

### Verifying Deployment

- The application includes a `/health` endpoint that checks:
  - Database connection
  - Vector database (ChromaDB) status
  - OpenAI API configuration
- Use this endpoint to monitor the health of your deployment

### Scaling Considerations

- **Worker Configuration**: The application uses Gunicorn with:
  - 4 worker processes for handling concurrent requests
  - 2 threads per worker for improved throughput
  - 120-second timeout for long-running operations

- **Memory Usage**:
  - Each worker requires approximately 250-500MB of RAM
  - Choose an instance with at least 2GB RAM for production use
  - The application implements efficient memory management for processing large documents

## 🛠 Technologies Used

- **Backend**:
  - Flask: Web framework
  - Gunicorn: WSGI HTTP Server for production
  - SQLite: Relational database
  - ChromaDB: Vector database
  - OpenAI API: Embeddings and LLM
  - PyMuPDF & pdfplumber: PDF processing
  - pytesseract: OCR for images
  - scikit-learn: NMF and TF-IDF
  - NetworkX: Graph construction

- **Frontend**:
  - HTML5/CSS3/JavaScript
  - Bootstrap 5: UI framework
  - Plotly.js: Interactive visualizations
  - Font Awesome: Icons
  - Animate.css: Animations

## 📁 Project Structure

```
document-research-chatbot/
├── app.py                  # Main Flask application
├── wsgi.py                 # WSGI entry point for Gunicorn
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker configuration for containerized deployment
├── build.sh                # Build script for deployment
├── render.yaml             # Render deployment configuration
├── .env.example            # Example environment variables
├── .gitignore              # Git ignore file
├── README.md               # Project documentation
├── PRODUCTION.md           # Production deployment guide
├── static/
│   ├── css/
│   │   └── style.css       # Custom styles
│   ├── js/
│   │   ├── main.js         # Main JavaScript
│   │   ├── new-animation.js # Processing animations
│   │   └── analysis-animation.js # Analysis animations
│   └── graphs/             # Generated knowledge graphs (not in repo)
└── templates/
    ├── index.html          # Home page template
    └── chat.html           # Chat interface template
```

> **Note**: The application now processes files in memory, so there's no need for permanent `uploads/` or `data/` directories in the repository. These directories are created dynamically when needed.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
