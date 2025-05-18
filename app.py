"""
Document Research & Theme Identification Chatbot
------------------------------------------------

This application implements a comprehensive document research system that allows users to:
1. Upload and process multiple documents (PDF, images, etc.)
2. Extract text using OCR and text extraction techniques
3. Create vector embeddings for semantic search
4. Query documents using natural language
5. Identify themes across documents
6. Generate knowledge graphs of document relationships
7. Provide detailed, cited responses to user queries

Methodology:
-----------
1. Document Processing:
   - Text extraction using PyMuPDF, pdfplumber, and OCR (pytesseract)
   - Document chunking into paragraphs for granular analysis
   - Vector embeddings generation using OpenAI's text-embedding-3-small model

2. Vector Database:
   - ChromaDB for efficient semantic search
   - Session-specific collections to isolate user data
   - Fallback mechanisms for reliability

3. Theme Identification:
   - TF-IDF vectorization to identify important terms
   - Non-negative Matrix Factorization (NMF) for topic modeling
   - Entity extraction and relationship mapping

4. Knowledge Graph:
   - NetworkX for graph construction
   - Plotly for interactive visualization
   - Force-directed layout for intuitive representation

5. Query Processing:
   - Hybrid search combining vector similarity and BM25 ranking
   - Context retrieval with citation tracking
   - LLM-based response generation with OpenAI's GPT models

6. Session Management:
   - Isolated vector databases per chat session
   - SQLite for persistent storage
   - Thread-safe database operations

Technical Implementation:
-----------------------
- Flask web framework for the backend
- SQLite for relational data storage
- ChromaDB for vector embeddings
- OpenAI API for embeddings and LLM responses
- Plotly for interactive visualizations
"""

import os
import sqlite3
import chromadb
import pdfplumber
import pytesseract
import shutil
import json
import uuid
import datetime
import time
import traceback
import logging
import logging.handlers
import threading
import functools
import re
import numpy as np
import pandas as pd
import tiktoken
import fitz  # PyMuPDF for PDF processing
import io  # For BytesIO
from PIL import Image
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from rank_bm25 import BM25Okapi
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import plotly.graph_objects as go
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from dotenv import load_dotenv
import nltk
nltk.download('punkt')
from nltk import bigrams, word_tokenize

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Helper function to convert NumPy types to Python native types
def convert_numpy_types(obj):
    """Convert NumPy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

# Set up logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Configure logging
logger = logging.getLogger('document_research_app')
logger.setLevel(logging.INFO)

# Create handlers
console_handler = logging.StreamHandler()
file_handler = logging.handlers.RotatingFileHandler(
    os.path.join(log_dir, 'app.log'),
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)

# Create formatters
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Flask app setup
app = Flask(__name__)

# Load environment variables
load_dotenv()

# Configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-change-in-production')
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'uploads')
app.config['DATA_FOLDER'] = os.getenv('DATA_FOLDER', 'data')
app.config['STATIC_FOLDER'] = os.getenv('STATIC_FOLDER', 'static')
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'png', 'jpg', 'jpeg'}
# Set max content length with fallback
max_content_env = os.getenv('MAX_CONTENT_LENGTH')
if max_content_env:
    try:
        app.config['MAX_CONTENT_LENGTH'] = int(max_content_env.split('#')[0].strip())  # Remove any comments
    except (ValueError, TypeError):
        logger.warning(f"Invalid MAX_CONTENT_LENGTH value: {max_content_env}, using default")
        app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB default
else:
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB default
app.config['DEBUG'] = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DATA_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)

# Get OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not found in .env file")
    raise ValueError("OPENAI_API_KEY not found in .env file")

# Initialize OpenAI and LangChain
# Using OpenAI client v1.x
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Get LLM model from environment or use default
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
llm = ChatOpenAI(model_name=LLM_MODEL, openai_api_key=OPENAI_API_KEY)

# Database connection pool with thread safety
class DBConnectionPool:
    def __init__(self, db_path, max_connections=5):
        self.db_path = db_path
        self.max_connections = max_connections
        self.connections = []
        self.lock = threading.Lock()
        # Create a single shared connection with check_same_thread=False
        self.shared_connection = sqlite3.connect(
            self.db_path,
            check_same_thread=False,  # Allow usage across threads
            isolation_level=None      # Enable autocommit mode
        )
        # Enable foreign keys
        self.shared_connection.execute("PRAGMA foreign_keys = ON")
        logger.info(f"Created shared SQLite connection to {self.db_path}")

    def get_connection(self):
        # Always return the shared connection
        return self.shared_connection

    def return_connection(self, conn):
        # No need to return the connection since we're using a shared one
        pass

    def close_all(self):
        with self.lock:
            try:
                if self.shared_connection:
                    self.shared_connection.close()
                    self.shared_connection = None
                    logger.info("Closed shared SQLite connection")
            except Exception as e:
                logger.error(f"Error closing shared connection: {str(e)}")

            # Close any remaining connections in the pool (should be empty)
            for conn in self.connections:
                try:
                    conn.close()
                except Exception as e:
                    logger.error(f"Error closing connection: {str(e)}")
            self.connections = []

# Database connection context manager with thread safety
class DBConnection:
    def __init__(self, pool):
        self.pool = pool
        self.conn = None
        self.cursor = None
        self.transaction_lock = threading.Lock()

    def __enter__(self):
        # Get the shared connection
        self.conn = self.pool.get_connection()

        # Acquire lock for this transaction
        self.transaction_lock.acquire()

        # Start a transaction explicitly
        self.conn.execute("BEGIN")

        # Create a cursor for this transaction
        self.cursor = self.conn.cursor()

        # Return the connection for backward compatibility
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            try:
                if exc_type is not None:
                    # If there was an exception, rollback
                    self.conn.execute("ROLLBACK")
                    logger.error(f"Transaction rolled back due to: {str(exc_type)}")
                else:
                    # Otherwise commit
                    self.conn.execute("COMMIT")
            except Exception as e:
                logger.error(f"Error in transaction: {str(e)}")
                try:
                    self.conn.execute("ROLLBACK")
                except Exception:
                    pass
            finally:
                # Close the cursor
                if self.cursor:
                    self.cursor.close()

                # Release the lock
                self.transaction_lock.release()

# Token usage tracking
class TokenUsageTracker:
    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_embedding_tokens = 0
        self.session_usage = {}
        self.lock = threading.Lock()

    def add_llm_usage(self, session_id, prompt_tokens, completion_tokens):
        with self.lock:
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens

            if session_id not in self.session_usage:
                self.session_usage[session_id] = {
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'embedding_tokens': 0
                }

            self.session_usage[session_id]['prompt_tokens'] += prompt_tokens
            self.session_usage[session_id]['completion_tokens'] += completion_tokens

            logger.info(f"Session {session_id} used {prompt_tokens} prompt tokens and {completion_tokens} completion tokens")

    def add_embedding_usage(self, session_id, embedding_tokens):
        with self.lock:
            self.total_embedding_tokens += embedding_tokens

            if session_id not in self.session_usage:
                self.session_usage[session_id] = {
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'embedding_tokens': 0
                }

            self.session_usage[session_id]['embedding_tokens'] += embedding_tokens

            logger.info(f"Session {session_id} used {embedding_tokens} embedding tokens")

    def get_session_usage(self, session_id):
        with self.lock:
            if session_id in self.session_usage:
                return self.session_usage[session_id]
            return {
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'embedding_tokens': 0
            }

    def get_total_usage(self):
        with self.lock:
            return {
                'prompt_tokens': self.total_prompt_tokens,
                'completion_tokens': self.total_completion_tokens,
                'embedding_tokens': self.total_embedding_tokens,
                'total_tokens': self.total_prompt_tokens + self.total_completion_tokens + self.total_embedding_tokens
            }

    def estimate_cost(self):
        # Approximate costs based on OpenAI pricing (may need updates)
        with self.lock:
            # GPT-3.5-turbo: $0.0015 / 1K prompt tokens, $0.002 / 1K completion tokens
            # text-embedding-3-small: $0.00002 / 1K tokens
            prompt_cost = (self.total_prompt_tokens / 1000) * 0.0015
            completion_cost = (self.total_completion_tokens / 1000) * 0.002
            embedding_cost = (self.total_embedding_tokens / 1000) * 0.00002

            return {
                'prompt_cost': prompt_cost,
                'completion_cost': completion_cost,
                'embedding_cost': embedding_cost,
                'total_cost': prompt_cost + completion_cost + embedding_cost
            }
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

# Initialize databases with thread safety
def init_database(reset=False):
    db_path = os.path.join(app.config['DATA_FOLDER'], "documents.db")

    # If reset is True, delete the existing database
    if reset and os.path.exists(db_path):
        try:
            os.remove(db_path)
            logger.info(f"Deleted existing database at {db_path}")
        except Exception as e:
            logger.error(f"Error deleting database: {str(e)}")
            # Try to rename it instead
            backup_path = f"{db_path}.bak.{int(time.time())}"
            try:
                shutil.move(db_path, backup_path)
                logger.info(f"Moved existing database to {backup_path}")
            except Exception as move_err:
                logger.error(f"Error moving database: {str(move_err)}")
                raise

    # Create a thread-safe connection
    conn = sqlite3.connect(
        db_path,
        check_same_thread=False,
        isolation_level=None  # Enable autocommit mode
    )

    # Enable foreign keys
    conn.execute("PRAGMA foreign_keys = ON")

    # Start a transaction
    conn.execute("BEGIN")

    try:
        cursor = conn.cursor()

        # Documents table
        cursor.execute('''CREATE TABLE IF NOT EXISTS documents
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          filename TEXT,
                          page INTEGER,
                          paragraph INTEGER,
                          text TEXT,
                          session_id TEXT DEFAULT 'default')''')

        # Chat sessions table
        cursor.execute('''CREATE TABLE IF NOT EXISTS chat_sessions
                         (id TEXT PRIMARY KEY,
                          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                          last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                          title TEXT DEFAULT 'New Chat')''')

        # Chat messages table
        cursor.execute('''CREATE TABLE IF NOT EXISTS chat_messages
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          session_id TEXT,
                          timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                          role TEXT,
                          content TEXT,
                          metadata TEXT,
                          FOREIGN KEY (session_id) REFERENCES chat_sessions(id))''')

        # Add indexes for better performance
        cursor.execute('''CREATE INDEX IF NOT EXISTS idx_documents_session_id ON documents(session_id)''')
        cursor.execute('''CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages(session_id)''')

        # Commit the transaction
        conn.execute("COMMIT")
        logger.info("Database schema initialized successfully")
    except Exception as e:
        # Rollback on error
        conn.execute("ROLLBACK")
        logger.error(f"Error initializing database: {str(e)}")
        raise
    finally:
        # Close the connection
        conn.close()

    return db_path

# Initialize ChromaDB
def init_chromadb(session_id=None, reset=False):
    """Initialize ChromaDB with proper error handling

    Args:
        session_id: If provided, creates a session-specific collection
        reset: If True, resets the collection
    """
    try:
        # Use persistent client with a specific path to ensure we can fully delete it
        chroma_path = os.path.join(app.config['DATA_FOLDER'], "chroma_db")

        # If reset is True, delete the entire ChromaDB directory
        if reset and os.path.exists(chroma_path):
            try:
                # Delete the entire directory to ensure complete reset
                shutil.rmtree(chroma_path)
                print(f"Successfully deleted ChromaDB at {chroma_path}")
            except Exception as e:
                print(f"Error deleting ChromaDB: {str(e)}")
                # If we can't delete it, try to rename it as a backup
                backup_path = f"{chroma_path}_backup_{int(time.time())}"
                try:
                    shutil.move(chroma_path, backup_path)
                    print(f"Moved existing ChromaDB to {backup_path}")
                except Exception as move_error:
                    print(f"Error moving ChromaDB: {str(move_error)}")

        # Ensure the directory exists
        os.makedirs(chroma_path, exist_ok=True)

        # Create a new client with the persistent path
        try:
            chroma_client = chromadb.PersistentClient(path=chroma_path)
        except Exception as client_error:
            print(f"Error creating ChromaDB client: {str(client_error)}")
            # Fallback to in-memory client if persistent fails
            print("Falling back to in-memory ChromaDB client")
            chroma_client = chromadb.Client()

        # Create or get the collection
        try:
            # Use session-specific collection name if provided
            collection_name = f"documents_{session_id}" if session_id else "documents"

            # First try to get the collection
            try:
                collection = chroma_client.get_collection(collection_name)
                if reset:  # If reset is True, delete and recreate
                    chroma_client.delete_collection(collection_name)
                    collection = chroma_client.create_collection(collection_name)
            except Exception as get_error:
                # If getting fails, try to create it
                print(f"Error getting collection {collection_name}: {str(get_error)}")
                collection = chroma_client.create_collection(collection_name)

            # Test the collection with a simple operation to ensure it works
            test_embedding = [0.0] * 1536  # Default OpenAI embedding size
            collection.add(
                embeddings=[test_embedding],
                documents=["Test document"],
                metadatas=[{"test": True}],
                ids=["test_id"]
            )
            # If we get here, the collection is working
            print("ChromaDB collection initialized successfully")

            # Clean up test document if it's not a reset
            if not reset:
                try:
                    collection.delete(ids=["test_id"])
                except:
                    pass

            return collection

        except Exception as collection_error:
            print(f"Error with ChromaDB collection: {str(collection_error)}")
            # Last resort: create a dummy collection wrapper that won't crash the app
            return DummyCollection()

    except Exception as e:
        print(f"Critical error initializing ChromaDB: {str(e)}")
        # Return a dummy collection that won't crash the app
        return DummyCollection()

# Dummy Collection class for fallback when ChromaDB fails
class DummyCollection:
    """A fallback collection that mimics ChromaDB but doesn't crash"""

    def __init__(self):
        self.documents = {}
        self.is_dummy = True
        print("WARNING: Using dummy vector database. Search functionality will be limited.")

    def add(self, embeddings=None, documents=None, metadatas=None, ids=None):
        """Store documents without embeddings"""
        if documents and ids:
            for i, doc_id in enumerate(ids):
                metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
                self.documents[doc_id] = {
                    "document": documents[i] if i < len(documents) else "",
                    "metadata": metadata
                }
        return {"success": True}

    def query(self, query_embeddings=None, n_results=10, where=None):
        """Return simple results without semantic search"""
        filtered_docs = []
        filtered_metadatas = []
        filtered_ids = []

        # Apply where filter if provided
        for doc_id, data in self.documents.items():
            if where:
                # Simple filtering based on metadata
                metadata_match = True
                for key, value in where.items():
                    if key not in data["metadata"] or data["metadata"][key] != value:
                        metadata_match = False
                        break

                if not metadata_match:
                    continue

            filtered_docs.append(data["document"])
            filtered_metadatas.append(data["metadata"])
            filtered_ids.append(doc_id)

            if len(filtered_docs) >= n_results:
                break

        return {
            "documents": [filtered_docs],
            "metadatas": [filtered_metadatas],
            "ids": [filtered_ids],
            "distances": [[1.0] * len(filtered_docs)]  # Dummy distances
        }

    def delete(self, ids=None):
        """Delete documents by ID"""
        if ids:
            for doc_id in ids:
                if doc_id in self.documents:
                    del self.documents[doc_id]
        return {"success": True}

# Initialize the SQL database
db_path = init_database()

# Create database connection pool
db_pool = DBConnectionPool(db_path, max_connections=10)

# Dictionary to store session-specific collections
session_collections = {}

# Initialize token usage tracker
token_tracker = TokenUsageTracker()

# Create embedding cache
embedding_cache = {}

# Create LLM response cache
llm_cache = {}
LLM_CACHE_SIZE = 100  # Maximum number of cached responses

# Function to get a cursor with transaction management
def get_db_cursor():
    """Get a database cursor with transaction management"""
    conn = db_pool.get_connection()
    try:
        conn.execute("BEGIN")
    except sqlite3.OperationalError as e:
        if "transaction is active" not in str(e):
            raise
        # If a transaction is already active, that's fine
        logger.debug("Transaction already active when calling get_db_cursor")
    return conn.cursor()

# Function to commit a transaction
def commit_db_transaction():
    """Commit the current transaction"""
    conn = db_pool.get_connection()
    try:
        conn.execute("COMMIT")
    except sqlite3.OperationalError as e:
        if "no transaction is active" not in str(e):
            raise
        # If no transaction is active, that's fine
        logger.debug("No transaction active when calling commit_db_transaction")

# Function to rollback a transaction
def rollback_db_transaction():
    """Rollback the current transaction"""
    conn = db_pool.get_connection()
    try:
        conn.execute("ROLLBACK")
    except sqlite3.OperationalError as e:
        if "no transaction is active" not in str(e):
            raise
        # If no transaction is active, that's fine
        logger.debug("No transaction active when calling rollback_db_transaction")

# Error handling decorator
def handle_errors(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Log the error
            logger.error(f"Error in {func.__name__}: {str(e)}")
            logger.error(traceback.format_exc())

            # Return a safe fallback or re-raise depending on the function
            if func.__name__ in ['index', 'chat', 'reset', 'delete_chat', 'update_chat_title']:
                # For route handlers, flash an error message and redirect
                flash(f"An error occurred: {str(e)}", "danger")
                return redirect(url_for('index'))
            else:
                # For utility functions, re-raise the exception
                raise
    return wrapper

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_session_collection(session_id):
    """Get or create a session-specific vector database collection"""
    global session_collections

    # If we already have a collection for this session, return it
    if session_id in session_collections:
        return session_collections[session_id]

    # Otherwise, create a new collection for this session
    collection = init_chromadb(session_id=session_id)
    session_collections[session_id] = collection
    return collection

# Session management functions
def create_session():
    """Create a new chat session and return its ID using thread-safe approach"""
    session_id = str(uuid.uuid4())
    try:
        # Get a cursor
        cursor = get_db_cursor()

        # Execute query
        cursor.execute("INSERT INTO chat_sessions (id) VALUES (?)", (session_id,))

        # Commit transaction
        commit_db_transaction()

        logger.info(f"Created new session: {session_id}")
        return session_id
    except Exception as e:
        # Rollback on error
        rollback_db_transaction()
        logger.error(f"Error creating session: {str(e)}")
        raise

def get_session(session_id):
    """Get session details by ID using thread-safe approach"""
    try:
        # Get a cursor
        cursor = get_db_cursor()

        # Execute query
        cursor.execute("SELECT * FROM chat_sessions WHERE id = ?", (session_id,))
        session = cursor.fetchone()

        # Log result
        if session:
            logger.debug(f"Retrieved session: {session_id}")
        else:
            logger.debug(f"Session not found: {session_id}")

        # Commit transaction
        commit_db_transaction()

        return session
    except Exception as e:
        # Rollback on error
        rollback_db_transaction()
        logger.error(f"Error getting session {session_id}: {str(e)}")
        raise

def delete_session(session_id):
    """Delete a chat session and its associated data using thread-safe approach"""
    try:
        # Get a cursor
        cursor = get_db_cursor()

        # First, delete associated documents
        cursor.execute("DELETE FROM documents WHERE session_id = ?", (session_id,))
        doc_count = cursor.rowcount

        # Delete chat messages
        cursor.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
        msg_count = cursor.rowcount

        # Delete the session itself
        cursor.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))

        # Commit transaction
        commit_db_transaction()

        logger.info(f"Deleted session {session_id} with {doc_count} documents and {msg_count} messages")

        # Delete the session-specific vector database collection
        try:
            # Get the ChromaDB client
            chroma_path = os.path.join(app.config['DATA_FOLDER'], "chroma_db")
            chroma_client = chromadb.PersistentClient(path=chroma_path)

            # Delete the collection for this session
            collection_name = f"documents_{session_id}"
            try:
                logger.info(f"Deleting vector database collection for session {session_id}")
                chroma_client.delete_collection(collection_name)
                logger.info(f"Successfully deleted vector database collection for session {session_id}")
            except Exception as coll_err:
                logger.error(f"Error deleting collection {collection_name}: {str(coll_err)}")

            # Remove from session collections dictionary
            global session_collections
            if session_id in session_collections:
                del session_collections[session_id]
                logger.info(f"Removed session {session_id} from session_collections dictionary")
        except Exception as e:
            logger.error(f"Error deleting session-specific vector database: {str(e)}")
            logger.error(traceback.format_exc())
            # Continue even if ChromaDB deletion fails

        return True
    except Exception as e:
        # Rollback on error
        rollback_db_transaction()
        logger.error(f"Error deleting session: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def get_or_create_session(session_id=None):
    """Get an existing session or create a new one"""
    if session_id:
        session = get_session(session_id)
        if session:
            return session_id
    return create_session()

def get_chat_history(session_id):
    """Get all messages for a session using thread-safe approach"""
    try:
        # Get a cursor
        cursor = get_db_cursor()

        # Execute query
        cursor.execute("SELECT role, content, metadata FROM chat_messages WHERE session_id = ? ORDER BY timestamp", (session_id,))
        messages = cursor.fetchall()

        # Log result
        logger.debug(f"Retrieved {len(messages)} messages for session {session_id}")

        # Commit transaction
        commit_db_transaction()

        # Format messages
        formatted_messages = []
        for role, content, metadata in messages:
            try:
                # Parse JSON metadata with error handling
                if metadata:
                    metadata_dict = json.loads(metadata)
                    # Convert any NumPy types that might have been missed during serialization
                    metadata_dict = convert_numpy_types(metadata_dict)
                else:
                    metadata_dict = {}
            except Exception as e:
                logger.error(f"Error parsing message metadata: {str(e)}")
                metadata_dict = {}

            formatted_messages.append({
                "role": role,
                "content": content,
                "metadata": metadata_dict
            })
        return formatted_messages
    except Exception as e:
        # Rollback on error
        rollback_db_transaction()
        logger.error(f"Error getting chat history for session {session_id}: {str(e)}")
        raise

def add_message(session_id, role, content, metadata=None):
    """Add a message to the chat history using thread-safe approach"""
    try:
        # Convert NumPy types to Python native types before serialization
        if metadata:
            metadata = convert_numpy_types(metadata)

        # Use the custom encoder for JSON serialization
        metadata_json = json.dumps(metadata, cls=NumpyEncoder) if metadata else None

        # Get a cursor
        cursor = get_db_cursor()

        # Execute queries
        cursor.execute(
            "INSERT INTO chat_messages (session_id, role, content, metadata) VALUES (?, ?, ?, ?)",
            (session_id, role, content, metadata_json)
        )
        cursor.execute("UPDATE chat_sessions SET last_updated = CURRENT_TIMESTAMP WHERE id = ?", (session_id,))

        # Commit transaction
        commit_db_transaction()

        logger.debug(f"Added {role} message to session {session_id}")
    except Exception as e:
        # Rollback on error
        rollback_db_transaction()
        logger.error(f"Error adding message to session {session_id}: {str(e)}")
        raise

def get_all_sessions():
    """Get all chat sessions using thread-safe approach"""
    try:
        # Get a cursor
        cursor = get_db_cursor()

        # Execute query
        cursor.execute("SELECT id, title, created_at, last_updated FROM chat_sessions ORDER BY last_updated DESC")
        sessions = cursor.fetchall()

        # Log result
        logger.debug(f"Retrieved {len(sessions)} chat sessions")

        # Commit transaction
        commit_db_transaction()

        # Format sessions
        formatted_sessions = []
        for id, title, created_at, last_updated in sessions:
            formatted_sessions.append({
                "id": id,
                "title": title,
                "created_at": created_at,
                "last_updated": last_updated
            })
        return formatted_sessions
    except Exception as e:
        # Rollback on error
        rollback_db_transaction()
        logger.error(f"Error getting all sessions: {str(e)}")
        # Return empty list on error to avoid crashing the app
        return []

def update_session_title(session_id, title):
    """Update the title of a chat session using thread-safe approach"""
    try:
        # Get a cursor
        cursor = get_db_cursor()

        # Execute query
        cursor.execute("UPDATE chat_sessions SET title = ? WHERE id = ?", (title, session_id))

        # Commit transaction
        commit_db_transaction()

        logger.debug(f"Updated title for session {session_id}: {title}")
    except Exception as e:
        # Rollback on error
        rollback_db_transaction()
        logger.error(f"Error updating title for session {session_id}: {str(e)}")
        raise

# Function to count tokens
def count_tokens(text, model="gpt-3.5-turbo"):
    """Count the number of tokens in a text string"""
    try:
        # Use tiktoken for accurate token counting
        encoding = tiktoken.encoding_for_model(model)
        tokens = len(encoding.encode(text))
        return tokens
    except Exception as e:
        logger.warning(f"Error counting tokens: {str(e)}")
        # Fallback to approximate counting (4 chars ~= 1 token)
        return len(text) // 4

# Function to get cached LLM response or generate a new one
def get_llm_response(prompt_text, session_id='default'):
    """Get LLM response with caching to reduce API calls"""
    # Create a hash of the prompt for cache key
    prompt_hash = str(hash(prompt_text))

    # Check if we have this response cached
    if prompt_hash in llm_cache:
        logger.debug(f"Using cached LLM response for prompt: {prompt_text[:50]}...")
        return llm_cache[prompt_hash]

    # Count tokens for the prompt
    prompt_tokens = count_tokens(prompt_text)

    # Call the LLM
    response = llm.invoke(prompt_text)

    # Count tokens for the response
    completion_tokens = count_tokens(response.content)

    # Track token usage
    token_tracker.add_llm_usage(session_id, prompt_tokens, completion_tokens)

    # Cache the result
    llm_cache[prompt_hash] = response

    # Limit cache size
    if len(llm_cache) > LLM_CACHE_SIZE:
        # Remove oldest items (first 10)
        for key in list(llm_cache.keys())[:10]:
            del llm_cache[key]

    # Log token usage
    logger.debug(f"LLM call used {prompt_tokens} prompt tokens and {completion_tokens} completion tokens")

    return response

def get_openai_embedding(text, session_id='default'):
    """Get embedding with caching to reduce API calls"""
    # Skip very short texts
    if len(text.strip()) < 10:
        logger.debug("Text too short for embedding, returning zeros")
        return [0.0] * 1536

    # Create a hash of the text for cache key
    text_hash = str(hash(text))

    # Check if we have this embedding cached
    if text_hash in embedding_cache:
        logger.debug(f"Using cached embedding for text: {text[:30]}...")
        return embedding_cache[text_hash]

    try:
        # Count tokens for tracking
        token_count = count_tokens(text, "text-embedding-3-small")

        # Call OpenAI API with OpenAI client v1.x
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        # Get the embedding
        embedding = response.data[0].embedding

        # Cache the result
        embedding_cache[text_hash] = embedding

        # Track token usage
        token_tracker.add_embedding_usage(session_id, token_count)

        # Limit cache size (keep most recent 1000 embeddings)
        if len(embedding_cache) > 1000:
            # Remove oldest items (first 100)
            for key in list(embedding_cache.keys())[:100]:
                del embedding_cache[key]

        return embedding
    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}")
        raise Exception(f"Embedding generation failed: {str(e)}")

def process_document(file_path, filename, session_id='default'):
    """Process a document with multiple extraction methods for maximum reliability

    Methodology:
    -----------
    1. Document Text Extraction:
       - Uses a multi-layered approach with fallback mechanisms
       - Primary: PyMuPDF for direct text extraction from PDFs
       - Secondary: pdfplumber as an alternative PDF text extractor
       - Tertiary: OCR using pytesseract for image-based or scanned documents

    2. Text Processing:
       - Splits text into paragraphs for granular analysis
       - Filters out short or empty paragraphs
       - Normalizes whitespace and formatting

    3. Vector Embedding:
       - Generates embeddings for each paragraph using OpenAI's text-embedding-3-small
       - Stores embeddings in ChromaDB for semantic search
       - Includes metadata for citation tracking (filename, page, paragraph)

    4. Database Storage:
       - Stores raw text in SQLite for retrieval and display
       - Creates BM25 index for keyword-based search as backup
       - Maintains session isolation for multi-user support

    5. Error Handling:
       - Implements comprehensive error recovery at each stage
       - Provides fallback mechanisms when primary methods fail
       - Ensures at least minimal functionality even with problematic documents

    Args:
        file_path (str): Path to the document file
        filename (str): Name of the document file
        session_id (str): ID of the current chat session

    Returns:
        tuple: (paragraphs, bm25_index, document_info)
            - paragraphs: List of extracted text paragraphs
            - bm25_index: BM25 search index for keyword search
            - document_info: Metadata about extracted paragraphs for UI display
    """
    texts = []

    try:
        # Log the document processing attempt
        logger.info(f"Processing document: {filename} (path: {file_path}) for session {session_id}")

        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"File does not exist: {file_path}")
            raise Exception(f"File not found: {file_path}")

        # Check file size
        file_size = os.path.getsize(file_path)
        logger.info(f"File size: {file_size / (1024*1024):.2f} MB")
        if file_size == 0:
            logger.error(f"File is empty: {file_path}")
            raise Exception(f"File is empty: {filename}")

        # SIMPLIFIED APPROACH: Try multiple methods in sequence until one works

        # Step 1: Extract text from document
        if file_path.lower().endswith(".pdf"):
            logger.info(f"Processing PDF file: {filename}")

            # Method 1: Try PyMuPDF first (most reliable)
            if not texts:
                try:
                    logger.info(f"Trying PyMuPDF for {filename}")
                    doc = fitz.open(file_path)
                    logger.info(f"PyMuPDF opened document with {len(doc)} pages")

                    for page_num in range(len(doc)):
                        try:
                            page_text = doc[page_num].get_text()
                            # Add even small amounts of text - we'll filter later if needed
                            if page_text and len(page_text.strip()) > 0:
                                logger.info(f"PyMuPDF extracted {len(page_text)} characters from page {page_num+1}")
                                texts.append((page_num + 1, page_text))
                            else:
                                logger.warning(f"PyMuPDF found no text on page {page_num+1}")
                        except Exception as page_err:
                            logger.error(f"PyMuPDF error on page {page_num+1}: {str(page_err)}")
                except Exception as fitz_err:
                    logger.error(f"PyMuPDF failed: {str(fitz_err)}")

            # Method 2: Try pdfplumber if PyMuPDF didn't work
            if not texts:
                try:
                    logger.info(f"Trying pdfplumber for {filename}")
                    with pdfplumber.open(file_path) as pdf:
                        logger.info(f"pdfplumber opened document with {len(pdf.pages)} pages")

                        for page_num, page in enumerate(pdf.pages):
                            try:
                                text = page.extract_text()
                                if text and len(text.strip()) > 0:
                                    logger.info(f"pdfplumber extracted {len(text)} characters from page {page_num+1}")
                                    texts.append((page_num + 1, text))
                                else:
                                    logger.warning(f"pdfplumber found no text on page {page_num+1}")
                            except Exception as page_err:
                                logger.error(f"pdfplumber error on page {page_num+1}: {str(page_err)}")
                except Exception as plumber_err:
                    logger.error(f"pdfplumber failed: {str(plumber_err)}")

            # Method 3: Last resort - try to extract images and run OCR
            if not texts:
                try:
                    logger.info(f"Trying OCR on PDF images for {filename}")
                    doc = fitz.open(file_path)

                    for page_num in range(len(doc)):
                        try:
                            # Get page as image
                            pix = doc[page_num].get_pixmap()
                            img_data = pix.tobytes("png")
                            img = Image.open(io.BytesIO(img_data))

                            # Run OCR
                            text = pytesseract.image_to_string(img)
                            if text and len(text.strip()) > 0:
                                logger.info(f"OCR extracted {len(text)} characters from page {page_num+1}")
                                texts.append((page_num + 1, text))
                            else:
                                logger.warning(f"OCR found no text on page {page_num+1}")
                        except Exception as page_err:
                            logger.error(f"OCR error on page {page_num+1}: {str(page_err)}")
                except Exception as ocr_err:
                    logger.error(f"PDF OCR failed: {str(ocr_err)}")

        elif file_path.lower().endswith((".png", ".jpg", ".jpeg")):
            logger.info(f"Processing image file: {filename}")

            # For images, just try OCR directly
            try:
                # Open the image
                image = Image.open(file_path)
                logger.info(f"Image opened: size={image.size}, mode={image.mode}")

                # Convert to RGB if needed
                if image.mode not in ('RGB', 'L'):
                    logger.info(f"Converting image from {image.mode} to RGB")
                    image = image.convert('RGB')

                # Use pytesseract for OCR
                text = pytesseract.image_to_string(image)
                if text and len(text.strip()) > 0:
                    logger.info(f"OCR extracted {len(text)} characters from image")
                    texts.append((1, text))
                else:
                    logger.warning(f"OCR found no text in image")
            except Exception as img_err:
                logger.error(f"Image processing failed: {str(img_err)}")

        else:
            # For unsupported file types, try a simple text extraction
            try:
                logger.info(f"Trying to read {filename} as text file")
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                    if text and len(text.strip()) > 0:
                        logger.info(f"Read {len(text)} characters from text file")
                        texts.append((1, text))
                    else:
                        logger.warning(f"No text found in file")
            except Exception as txt_err:
                logger.error(f"Text file reading failed: {str(txt_err)}")
                logger.error(f"Unsupported file type: {file_path}")
                raise Exception(f"Unsupported file type: {filename}")

        # FALLBACK: If we still have no text, create a dummy paragraph
        if not texts:
            logger.warning(f"No text extracted from {filename} using any method, creating dummy text")
            dummy_text = f"[This document '{filename}' could not be processed for text extraction. " \
                        f"It may be an image-only PDF, a scanned document without OCR, or contain no extractable text.]"
            texts.append((1, dummy_text))

            # Log this as a special case
            logger.info(f"Using dummy text for {filename}")

        # Log success
        logger.info(f"Successfully extracted {len(texts)} text sections from {filename}")

        # Step 2: Process extracted text
        all_paragraphs = []
        doc_info = []

        try:
            # Get a database cursor
            cursor = get_db_cursor()

            # Process each text section
            for page_num, text in texts:
                # Split text into paragraphs
                paragraphs = text.split("\n\n")

                # Filter out very short paragraphs and normalize text
                filtered_paragraphs = []
                for para_num, para in enumerate(paragraphs):
                    para = para.strip()
                    # Skip empty or very short paragraphs (less than 20 chars)
                    if para and len(para) >= 20:
                        # Normalize whitespace
                        para = ' '.join(para.split())
                        filtered_paragraphs.append((para_num, para))

                # Process filtered paragraphs
                for para_num, para in filtered_paragraphs:
                    doc_id = f"{filename}_{page_num}_{para_num}"

                    # Get the collection for this session
                    collection = get_session_collection(session_id)

                    # Check if it's a dummy collection
                    is_dummy_collection = hasattr(collection, 'is_dummy') and collection.is_dummy

                    # Get embedding if not using dummy collection
                    embedding = None
                    try:
                        # Pass session_id for token tracking
                        embedding = get_openai_embedding(para, session_id)
                    except Exception as embed_err:
                        logger.error(f"Error generating embedding: {str(embed_err)}")
                        # Create a dummy embedding if OpenAI fails
                        embedding = [0.0] * 1536

                    # Add to session-specific ChromaDB collection
                    try:
                        metadata = {
                            "doc_id": filename,
                            "page": page_num,
                            "paragraph": para_num,
                            "session_id": session_id
                        }

                        if is_dummy_collection:
                            collection.add(
                                documents=[para],
                                metadatas=[metadata],
                                ids=[doc_id]
                            )
                        else:
                            collection.add(
                                embeddings=[embedding],
                                documents=[para],
                                metadatas=[metadata],
                                ids=[doc_id]
                            )
                        logger.debug(f"Added document to vector database: {doc_id}")
                    except Exception as chroma_err:
                        logger.error(f"Error adding to vector database: {str(chroma_err)}")

                    # Add to SQLite database
                    try:
                        cursor.execute(
                            "INSERT INTO documents (filename, page, paragraph, text, session_id) VALUES (?, ?, ?, ?, ?)",
                            (filename, page_num, para_num, para, session_id)
                        )
                        commit_db_transaction()
                        logger.debug(f"Added document to SQL database: {doc_id}")
                    except Exception as sql_err:
                        logger.error(f"Error adding to SQL database: {str(sql_err)}")
                        rollback_db_transaction()

                    # Add to return values
                    all_paragraphs.append(para)
                    doc_info.append({
                        "id": doc_id,
                        "page": page_num,
                        "paragraph": para_num,
                        "text_preview": para[:100] + "..." if len(para) > 100 else para
                    })

            # Create BM25 index for search
            if all_paragraphs:
                tokenized_paragraphs = [para.lower().split() for para in all_paragraphs]
                bm25 = BM25Okapi(tokenized_paragraphs)
                logger.info(f"Created BM25 index with {len(all_paragraphs)} paragraphs")
            else:
                # Create a dummy BM25 index if no paragraphs were processed
                bm25 = BM25Okapi([[""]])
                logger.warning("Created empty BM25 index")

            # Convert any NumPy types before returning
            converted_doc_info = convert_numpy_types(doc_info)

            # Return document info for UI display
            logger.info(f"Returning {len(all_paragraphs)} paragraphs and {len(doc_info)} document info items")
            return all_paragraphs, bm25, converted_doc_info

        except Exception as process_err:
            logger.error(f"Text processing failed: {str(process_err)}")
            traceback.print_exc()

            # Create a minimal result with the extracted text
            minimal_paragraphs = []
            minimal_doc_info = []

            # Use the raw text sections as paragraphs
            for page_num, text in texts:
                para = text.strip()
                if para:
                    doc_id = f"{filename}_{page_num}_0"
                    minimal_paragraphs.append(para)
                    minimal_doc_info.append({
                        "id": doc_id,
                        "page": page_num,
                        "paragraph": 0,
                        "text_preview": para[:100] + "..." if len(para) > 100 else para
                    })

            # Create a BM25 index with whatever we have
            if minimal_paragraphs:
                tokenized_paragraphs = [para.lower().split() for para in minimal_paragraphs]
                minimal_bm25 = BM25Okapi(tokenized_paragraphs)
            else:
                minimal_bm25 = BM25Okapi([[""]])

            logger.info(f"Returning minimal result with {len(minimal_paragraphs)} paragraphs after processing error")
            return minimal_paragraphs, minimal_bm25, minimal_doc_info

    except Exception as e:
        # Log the full error with traceback
        logger.error(f"Document processing failed: {str(e)}")
        traceback.print_exc()

        # Create a dummy paragraph with error information
        dummy_text = f"[Error processing document '{filename}': {str(e)}]"
        dummy_paragraphs = [dummy_text]
        dummy_doc_info = [{
            "id": f"{filename}_0_0",
            "page": 0,
            "paragraph": 0,
            "text_preview": dummy_text
        }]

        # Return minimal data to avoid crashing the app
        logger.info("Returning dummy data after document processing failure")
        return dummy_paragraphs, BM25Okapi([dummy_text.lower().split()]), dummy_doc_info

def process_query(query, paragraphs, bm25, session_id='default', chat_history=None):
    """Process a user query against document collection with semantic search

    Methodology:
    -----------
    1. Semantic Search:
       - Generates vector embedding for the query using OpenAI's text-embedding-3-small
       - Performs similarity search against document embeddings in ChromaDB
       - Retrieves the most semantically relevant paragraphs

    2. Keyword Search:
       - Uses BM25 algorithm for traditional keyword-based search
       - Provides complementary results to semantic search
       - Helps capture exact term matches that semantic search might miss

    3. Hybrid Ranking:
       - Combines results from semantic and keyword search
       - Weights and normalizes scores for balanced retrieval
       - Ensures both semantic meaning and keyword relevance are considered

    4. Context Assembly:
       - Selects top-ranked paragraphs as context for the LLM
       - Includes citation information (document name, page, paragraph)
       - Optimizes context length to balance completeness with token limits

    5. Response Generation:
       - Uses OpenAI's GPT model to generate a comprehensive answer
       - Provides explicit instructions to cite sources properly
       - Formats response with citations for traceability

    6. Theme Identification:
       - Analyzes patterns across retrieved paragraphs
       - Identifies common topics and concepts
       - Generates a knowledge graph of related themes

    Args:
        query (str): User's natural language query
        paragraphs (list): List of document paragraphs
        bm25 (BM25Okapi): BM25 search index for keyword search
        session_id (str): ID of the current chat session
        chat_history (list): Previous messages in the conversation

    Returns:
        dict: Response containing answer, citations, themes, and knowledge graph
    """
    try:
        # Log the query for debugging
        logger.info(f"Processing query for session {session_id}: {query[:50]}...")

        # Step 1: Get query embedding
        try:
            # Use the session_id for token tracking
            query_embedding = get_openai_embedding(query, session_id)
        except Exception as embed_err:
            logger.error(f"Error generating query embedding: {str(embed_err)}")
            # Create a dummy embedding if OpenAI fails
            query_embedding = [0.0] * 1536

        # Step 2: Query the session-specific vector database
        try:
            # Get the collection for this session
            collection = get_session_collection(session_id)

            # Check if collection is a dummy
            is_dummy_collection = hasattr(collection, 'is_dummy') and collection.is_dummy

            if is_dummy_collection:
                print("Using dummy collection for query")
                semantic_results = collection.query(
                    n_results=50
                    # No need to filter by session_id since we're using a session-specific collection
                )
            else:
                print(f"Querying vector database for session: {session_id}")
                semantic_results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=50
                    # No need to filter by session_id since we're using a session-specific collection
                )
        except Exception as query_err:
            print(f"Vector database query failed: {str(query_err)}")
            # Create empty results if ChromaDB query fails
            semantic_results = {
                "documents": [[]],
                "metadatas": [[]],
                "ids": [[]],
                "distances": [[]]
            }

        # Step 3: BM25 search
        try:
            tokenized_query = query.lower().split()
            bm25_scores = bm25.get_scores(tokenized_query)
            bm25_top_indices = np.argsort(bm25_scores)[::-1][:50]
        except Exception as bm25_err:
            print(f"BM25 search failed: {str(bm25_err)}")
            # Create empty results if BM25 fails
            bm25_scores = np.zeros(len(paragraphs) if paragraphs else 1)
            bm25_top_indices = []

        # Step 4: Combine vector and BM25 results
        combined_results = {}

        # Add vector results
        try:
            if semantic_results["documents"][0]:
                for i, (doc, meta) in enumerate(zip(semantic_results["documents"][0], semantic_results["metadatas"][0])):
                    try:
                        doc_id = meta["doc_id"] + f"_{meta['page']}_{meta['paragraph']}"
                        score = 1 - semantic_results["distances"][0][i] if "distances" in semantic_results and i < len(semantic_results["distances"][0]) else 0.9
                        combined_results[doc_id] = {"doc": doc, "meta": meta, "score": score}
                    except Exception as item_err:
                        print(f"Error processing vector result item: {str(item_err)}")
        except Exception as vector_err:
            print(f"Error processing vector results: {str(vector_err)}")

        # Add BM25 results
        try:
            # Get last metadata or use default
            if semantic_results["metadatas"][0]:
                last_meta = semantic_results["metadatas"][0][-1]
            else:
                last_meta = {"doc_id": "unknown", "page": 1, "paragraph": 0}

            for idx in bm25_top_indices:
                if idx < len(paragraphs):
                    doc_id = f"{last_meta['doc_id']}_1_{idx}"
                    if doc_id not in combined_results:
                        combined_results[doc_id] = {
                            "doc": paragraphs[idx],
                            "meta": {"doc_id": last_meta['doc_id'], "page": 1, "paragraph": idx},
                            "score": bm25_scores[idx] / max(bm25_scores) if max(bm25_scores) > 0 else 0
                        }
        except Exception as bm25_combine_err:
            print(f"Error combining BM25 results: {str(bm25_combine_err)}")

        # If we have no results at all, create a dummy result
        if not combined_results:
            print("No results found, creating dummy result")
            combined_results["dummy_0_0"] = {
                "doc": "No relevant information found in the document.",
                "meta": {"doc_id": "dummy", "page": 0, "paragraph": 0},
                "score": 0.5
            }

        # Sort results by score
        sorted_results = sorted(combined_results.items(), key=lambda x: x[1]["score"], reverse=True)

        # Step 5: Prepare context batches for LLM
        try:
            max_tokens = 15000
            context_batches = []
            current_batch = []
            current_tokens = len(tokenizer.encode(query + "\nContexts:\n"))

            for res in sorted_results:
                doc = res[1]["doc"]
                context_tokens = len(tokenizer.encode(f"[Context {len(current_batch)+1}] {doc}"))
                if current_tokens + context_tokens < max_tokens:
                    current_batch.append(res)
                    current_tokens += context_tokens
                else:
                    if current_batch:
                        context_batches.append(current_batch)
                    current_batch = [res]
                    current_tokens = len(tokenizer.encode(query + "\nContexts:\n")) + context_tokens

            if current_batch:
                context_batches.append(current_batch)
        except Exception as batch_err:
            print(f"Error preparing context batches: {str(batch_err)}")
            # Create a single batch with all results if batching fails
            context_batches = [sorted_results[:20]]  # Limit to 20 to avoid token issues

        # Step 6: Include chat history in the prompt if available
        chat_context = ""
        if chat_history and len(chat_history) > 0:
            try:
                chat_context = "Previous conversation:\n"
                for msg in chat_history[-5:]:  # Include last 5 messages
                    role = "User" if msg["role"] == "user" else "Assistant"
                    chat_context += f"{role}: {msg['content']}\n"
                chat_context += "\n"
            except Exception as history_err:
                print(f"Error processing chat history: {str(history_err)}")
                chat_context = ""

        # Step 7: Process with LLM
        all_extracted_answers = []
        context_counter = 0

        # Create a more efficient prompt template with chat history
        prompt = PromptTemplate(
            input_variables=["chat_history", "query", "contexts"],
            template="{chat_history}Query: {query}\nContexts:\n{contexts}\nExtract ONLY information that directly answers the query. For EACH relevant context, provide a concise answer (1-2 sentences) prefixed with '[Context X]'. Extract from MULTIPLE contexts when possible. Skip irrelevant contexts."
        )

        for batch in context_batches:
            try:
                # Get contexts from the batch
                contexts = [res[1]["doc"] for res in batch]

                # Format contexts more efficiently
                formatted_contexts = "\n".join([f"[Context {context_counter + i + 1}] {ctx}" for i, ctx in enumerate(contexts)])

                # Create input text
                input_text = prompt.format(chat_history=chat_context, query=query, contexts=formatted_contexts)

                # Use the caching function to get the response
                llm_response = get_llm_response(input_text, session_id)

                lines = llm_response.content.split("\n")
                for line in lines:
                    match = re.match(r'\[Context (\d+)\]\s*(.+)', line)
                    if match:
                        try:
                            context_num = int(match.group(1)) - 1
                            answer = match.group(2).strip()
                            # More lenient check to allow more answers
                            if context_num < context_counter + len(contexts) and answer and len(answer) > 5:
                                try:
                                    # Use a simpler relevance check to reduce token usage
                                    # First try a keyword-based relevance check
                                    query_keywords = set(word.lower() for word in query.split() if len(word) > 3)
                                    answer_keywords = set(word.lower() for word in answer.split() if len(word) > 3)

                                    # If there's keyword overlap, consider it relevant
                                    if query_keywords.intersection(answer_keywords):
                                        is_relevant = True
                                    else:
                                        # Fall back to a more efficient LLM check
                                        relevance_prompt = PromptTemplate(
                                            input_variables=["query", "answer"],
                                            template="Q: {query}\nA: {answer}\nIs A relevant to Q? Answer only Yes/No."
                                        )
                                        prompt_text = relevance_prompt.format(query=query, answer=answer)

                                        # Use the caching function to get the response
                                        relevance_check = get_llm_response(prompt_text, session_id)

                                        is_relevant = "yes" in relevance_check.content.strip().lower()

                                    if is_relevant:
                                        meta = batch[context_num - context_counter][1]["meta"]
                                        filename = meta.get("doc_id", "unknown")
                                        page_num = meta.get("page", 0)
                                        para_num = meta.get("paragraph", 0)
                                        all_extracted_answers.append({
                                            "DocumentID": f"DOC{context_num + 1:03d}",
                                            "Filename": filename,
                                            "Extracted_Answer": answer,
                                            "Citation": f"{filename} (Page {page_num}, Para {para_num})",
                                            "score": batch[context_num - context_counter][1]["score"],
                                            "doc_id": filename,
                                            "page": page_num,
                                            "paragraph": para_num
                                        })
                                except Exception as relevance_err:
                                    print(f"Error checking relevance: {str(relevance_err)}")
                                    # Add the answer anyway if relevance check fails
                                    meta = batch[context_num - context_counter][1]["meta"]
                                    filename = meta.get("doc_id", "unknown")
                                    page_num = meta.get("page", 0)
                                    para_num = meta.get("paragraph", 0)
                                    all_extracted_answers.append({
                                        "DocumentID": f"DOC{context_num + 1:03d}",
                                        "Filename": filename,
                                        "Extracted_Answer": answer,
                                        "Citation": f"{filename} (Page {page_num}, Para {para_num})",
                                        "score": batch[context_num - context_counter][1]["score"],
                                        "doc_id": filename,
                                        "page": page_num,
                                        "paragraph": para_num
                                    })
                        except Exception as line_err:
                            print(f"Error processing LLM response line: {str(line_err)}")
                context_counter += len(contexts)
            except Exception as batch_process_err:
                print(f"Error processing batch: {str(batch_process_err)}")

        # Step 8: Generate a detailed response
        try:
            if not all_extracted_answers:
                # If no answers were extracted, create a default answer
                all_extracted_answers.append({
                    "DocumentID": "DOC001",
                    "Filename": "N/A",
                    "Extracted_Answer": "No specific information found in the document that directly answers your query.",
                    "Citation": "No relevant information found",
                    "score": 0.5,
                    "doc_id": "unknown",
                    "page": 0,
                    "paragraph": 0
                })

            if chat_history:
                # For follow-up questions, use a different prompt that considers conversation context
                response_prompt = PromptTemplate(
                    input_variables=["chat_history", "query", "answers"],
                    template="{chat_history}Current Query: {query}\nExtracted Answers:\n{answers}\nProvide a comprehensive response that directly addresses the current query while maintaining context from the conversation history."
                )

                answers_text = "\n".join([f"- {ans['Extracted Answer']} (Page {ans['page']}, Para {ans['paragraph']})" for ans in all_extracted_answers])
                response_input = response_prompt.format(
                    chat_history=chat_context,
                    query=query,
                    answers=answers_text
                )

                response_result = llm.invoke(response_input)
                detailed_response = response_result.content
            else:
                # For initial questions, use a simpler summary
                detailed_response = "Summary of findings: " + " ".join([ans["Extracted Answer"] for ans in all_extracted_answers])
        except Exception as response_err:
            print(f"Error generating detailed response: {str(response_err)}")
            # Fallback response if generation fails
            detailed_response = "I found some information in the document, but I'm having trouble summarizing it. Here are the key points: " + " ".join([ans.get("Extracted Answer", "No answer found") for ans in all_extracted_answers])

        # Convert any NumPy types in the extracted answers
        converted_answers = convert_numpy_types(all_extracted_answers)

        return detailed_response, converted_answers

    except Exception as e:
        # Log the full error with traceback
        print(f"Query processing failed: {str(e)}")
        traceback.print_exc()

        # Return minimal data to avoid crashing the app
        fallback_answer = {
            "DocumentID": "ERROR",
            "Filename": "Error",
            "Extracted_Answer": "An error occurred while processing your query. Please try again or reset the application.",
            "Citation": "Error processing query",
            "score": 0,
            "doc_id": "error",
            "page": 0,
            "paragraph": 0
        }

        fallback_response = "I encountered an error while processing your query. Please try again with a different question or reset the application if the problem persists."

        return fallback_response, [fallback_answer]

def compute_topic_coherence(topics, vectorizer, X):
    feature_names = vectorizer.get_feature_names_out()
    coherence_scores = []
    for topic in topics:
        top_indices = topic.argsort()[-10:]
        top_words = [feature_names[i] for i in top_indices]
        if len(top_words) < 2:
            continue
        word_vectors = X[:, top_indices].toarray()
        sims = cosine_similarity(word_vectors.T)
        coherence = sims[np.triu_indices(len(top_words), k=1)].mean()
        coherence_scores.append(coherence)
    return np.mean(coherence_scores) if coherence_scores else 0

def synthesize_response(query, response, extracted_answers, _, session_id='default'):
    """Synthesize a response with themes and knowledge graph visualization

    Methodology:
    -----------
    1. Theme Identification:
       - Retrieves document texts from the database
       - Applies TF-IDF vectorization to identify important terms
       - Uses Non-negative Matrix Factorization (NMF) for topic modeling
       - Optimizes number of topics based on coherence scores
       - Extracts keywords for each identified theme

    2. Theme Refinement:
       - Uses LLM to refine raw themes into coherent concepts
       - Considers query context to ensure themes are relevant
       - Weights documents by relevance to prioritize important content
       - Produces concise, human-readable theme descriptions

    3. Knowledge Graph Construction:
       - Creates a directed graph using NetworkX
       - Represents documents and themes as nodes
       - Establishes weighted edges based on document-theme relationships
       - Marks query-relevant documents with special attributes
       - Stores citation information for traceability

    4. Response Synthesis:
       - Combines identified themes with extracted answers
       - Generates a comprehensive response addressing the query
       - Incorporates multiple document sources when available
       - Includes proper citations to source documents

    5. Visualization:
       - Generates an interactive knowledge graph using Plotly
       - Provides filtering options for exploring themes
       - Highlights query-relevant documents
       - Creates a session-specific visualization file

    Args:
        query (str): User's natural language query
        response (str): Initial response from query processing
        extracted_answers (list): List of extracted answers with citations
        _ (list): Unused parameter (kept for backward compatibility)
        session_id (str): ID of the current chat session

    Returns:
        dict: Contains synthesized response, identified themes, and knowledge graph
    """
    conn = None
    try:
        # Step 1: Get document texts from database
        try:
            conn = sqlite3.connect(os.path.join(app.config['DATA_FOLDER'], "documents.db"))
            cursor = conn.cursor()
            cursor.execute("SELECT text FROM documents")
            texts = [row[0] for row in cursor.fetchall()]
            conn.close()
            conn = None

            if not texts:
                # If no texts found, use extracted answers as texts
                texts = [ans.get("Extracted Answer", "No text available") for ans in extracted_answers]
                if not texts:
                    # If still no texts, create a dummy text
                    texts = ["No document text available"]
        except Exception as db_err:
            print(f"Error retrieving texts from database: {str(db_err)}")
            # Use extracted answers as fallback
            texts = [ans.get("Extracted Answer", "No text available") for ans in extracted_answers]
            if not texts:
                texts = ["No document text available"]

        # Step 2: Prepare text for vectorization
        try:
            custom_stop_words = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of']

            # Process texts with error handling for each step
            tokenized_texts = []
            for text in texts:
                try:
                    tokenized_texts.append(word_tokenize(text.lower()))
                except Exception as token_err:
                    print(f"Error tokenizing text: {str(token_err)}")
                    tokenized_texts.append(text.lower().split())

            # Generate bigrams
            bigram_texts = []
            for tokens in tokenized_texts:
                try:
                    bigram_list = [
                        '_'.join(bg) for bg in bigrams(tokens)
                        if all(w not in custom_stop_words for w in bg)
                    ]
                    bigram_texts.append(' '.join(bigram_list))
                except Exception as bigram_err:
                    print(f"Error generating bigrams: {str(bigram_err)}")
                    bigram_texts.append('')

            # Combine original texts with bigrams
            combined_texts = [f"{texts[i]} {bigram_texts[i]}" for i in range(len(texts))]
        except Exception as text_prep_err:
            print(f"Error in text preparation: {str(text_prep_err)}")
            # Use original texts as fallback
            combined_texts = texts

        # Step 3: Vectorize texts
        try:
            vectorizer = TfidfVectorizer(
                max_df=1.0,  # Allow terms that appear in all documents
                min_df=1,    # Include terms that appear in at least 1 document
                stop_words=custom_stop_words,
                ngram_range=(1, 2)
            )
            X = vectorizer.fit_transform(combined_texts)
        except Exception as vectorize_err:
            print(f"Error in text vectorization: {str(vectorize_err)}")
            # Create a simple vectorizer with minimal parameters as fallback
            try:
                vectorizer = TfidfVectorizer(stop_words=None)
                X = vectorizer.fit_transform(combined_texts)
            except Exception as simple_vec_err:
                print(f"Error in simple vectorization: {str(simple_vec_err)}")
                # Return basic response without themes if vectorization fails
                return {
                    "response": response,
                    "themes": ["Theme extraction failed"],
                    "graph": nx.DiGraph()
                }

        # Step 4: Generate themes
        try:
            # Check if we have enough documents for NMF
            if len(texts) < 2:
                # Fallback for very few documents: use simple keyword extraction
                print("Too few documents for NMF. Using keyword extraction fallback.")
                try:
                    feature_names = vectorizer.get_feature_names_out()
                    feature_scores = np.asarray(X.sum(axis=0)).ravel()
                    top_indices = feature_scores.argsort()[-20:][::-1]
                    top_keywords = [feature_names[i] for i in top_indices]

                    # Group keywords into themes
                    initial_themes = []
                    theme_count = min(3, len(top_keywords) // 5 + 1)
                    for i in range(theme_count):
                        start_idx = i * (len(top_keywords) // theme_count)
                        end_idx = min((i + 1) * (len(top_keywords) // theme_count), len(top_keywords))
                        theme_keywords = top_keywords[start_idx:end_idx]
                        if theme_keywords:
                            initial_themes.append(f"Theme {i + 1}: {', '.join(theme_keywords)}")

                    if not initial_themes:
                        initial_themes = ["Theme 1: Document analysis"]

                    # Create dummy W matrix for graph visualization
                    best_n_components = len(initial_themes)
                    W = np.zeros((len(texts), best_n_components))
                    for i in range(len(texts)):
                        for j in range(best_n_components):
                            W[i, j] = 1.0 / best_n_components  # Equal distribution
                except Exception as keyword_err:
                    print(f"Error in keyword extraction: {str(keyword_err)}")
                    initial_themes = ["Theme 1: Document analysis"]
                    best_n_components = 1
                    W = np.ones((len(texts), 1))
            else:
                # Normal NMF approach for multiple documents
                try:
                    max_components = min(10, len(texts), X.shape[1])
                    best_n_components = min(3, len(texts))  # Ensure we don't try more components than documents

                    # Only try to find optimal components if we have enough documents
                    if len(texts) > 3:
                        try:
                            best_coherence = 0
                            for n in range(2, max_components + 1):
                                try:
                                    nmf = NMF(n_components=n, random_state=42)
                                    W = nmf.fit_transform(X)
                                    coherence = compute_topic_coherence(nmf.components_, vectorizer, X)
                                    if coherence > best_coherence:
                                        best_coherence = coherence
                                        best_n_components = n
                                except Exception as nmf_iter_err:
                                    print(f"Error in NMF iteration {n}: {str(nmf_iter_err)}")
                        except Exception as coherence_err:
                            print(f"Error finding optimal components: {str(coherence_err)}")
                            # Use default value
                            best_n_components = min(3, len(texts))

                    # Final NMF with best components
                    nmf = NMF(n_components=best_n_components, random_state=42)
                    W = nmf.fit_transform(X)

                    # Extract themes
                    feature_names = vectorizer.get_feature_names_out()
                    initial_themes = []
                    for topic_idx, topic in enumerate(nmf.components_):
                        top_indices = topic.argsort()[-10:]
                        top_words = [feature_names[i].replace('_', ' ') for i in top_indices]
                        initial_themes.append(f"Theme {topic_idx + 1}: {', '.join(top_words)}")
                except Exception as nmf_err:
                    print(f"Error in NMF: {str(nmf_err)}")
                    # Fallback to simple themes
                    initial_themes = [f"Theme {i+1}: Document analysis" for i in range(min(3, len(texts)))]
                    best_n_components = len(initial_themes)
                    W = np.ones((len(texts), best_n_components)) / best_n_components
        except Exception as theme_err:
            print(f"Error generating themes: {str(theme_err)}")
            # Fallback to basic themes
            initial_themes = ["Theme 1: Document content"]
            best_n_components = 1
            W = np.ones((len(texts), 1))

        # Step 5: Weight documents by relevance
        try:
            doc_weights = np.ones(len(texts))
            for ans in extracted_answers:
                try:
                    doc_id_str = ans.get("DocumentID", "DOC001")
                    if doc_id_str.startswith("DOC"):
                        doc_id = int(doc_id_str[3:]) - 1
                        if 0 <= doc_id < len(texts):
                            doc_weights[doc_id] += ans.get("score", 0.5)
                except Exception as weight_err:
                    print(f"Error weighting document: {str(weight_err)}")

            W_weighted = W * doc_weights[:, np.newaxis]
        except Exception as weighting_err:
            print(f"Error in document weighting: {str(weighting_err)}")
            # Use unweighted matrix as fallback
            W_weighted = W

        # Step 6: Refine themes with LLM
        try:
            prompt = PromptTemplate(
                input_variables=["query", "themes", "texts", "answers"],
                template="Query: {query}\nInitial Themes:\n{themes}\nRelevant Texts:\n{texts}\nExtracted Answers:\n{answers}\nRefine these themes to be specific, coherent, and aligned with the query and document content. Output each theme as a concise phrase (e.g., 'Forbidden romantic encounters'). Return exactly {n_themes} themes."
            )

            # Get relevant texts safely
            relevant_texts = []
            for ans in extracted_answers:
                try:
                    doc_id_str = ans.get("DocumentID", "")
                    if doc_id_str.startswith("DOC"):
                        doc_id = int(doc_id_str[3:]) - 1
                        if 0 <= doc_id < len(texts):
                            relevant_texts.append(texts[doc_id])
                except Exception as rel_text_err:
                    print(f"Error getting relevant text: {str(rel_text_err)}")

            # If no relevant texts, use the first few texts
            if not relevant_texts and texts:
                relevant_texts = texts[:min(3, len(texts))]

            # Format answers for prompt
            answers_text = []
            for a in extracted_answers:
                try:
                    answers_text.append(f"{a.get('DocumentID', 'DOC')}:" +
                                       f" {a.get('Extracted Answer', 'No answer')} " +
                                       f"({a.get('Citation', 'No citation')})")
                except Exception as ans_format_err:
                    print(f"Error formatting answer: {str(ans_format_err)}")

            # Call LLM to refine themes
            refined_themes_response = llm.invoke(prompt.format(
                query=query,
                themes="\n".join(initial_themes),
                texts="\n".join(relevant_texts[:10]),
                answers="\n".join(answers_text),
                n_themes=best_n_components
            ))

            # Process LLM response
            refined_themes = [line.strip() for line in refined_themes_response.content.split("\n")
                             if line.strip()][:best_n_components]

            final_themes = refined_themes if refined_themes else initial_themes
        except Exception as refine_err:
            print(f"Error refining themes: {str(refine_err)}")
            # Use initial themes as fallback
            final_themes = initial_themes

        # Step 7: Create knowledge graph
        try:
            G = nx.DiGraph()

            # Add document nodes
            for i, text in enumerate(texts):
                try:
                    doc_id = f"doc_{i}"
                    text_preview = text[:100] + "..." if len(text) > 100 else text
                    G.add_node(doc_id, type="document", text=text_preview)

                    # Connect to themes
                    if i < W_weighted.shape[0]:
                        for theme_idx, weight in enumerate(W_weighted[i]):
                            if theme_idx < len(final_themes) and weight > 0.1:
                                theme_id = f"theme_{theme_idx}"
                                if theme_id not in G:
                                    G.add_node(theme_id, type="theme",
                                              description=final_themes[theme_idx])
                                G.add_edge(doc_id, theme_id, weight=float(weight))
                except Exception as node_err:
                    print(f"Error adding node to graph: {str(node_err)}")

            # Mark relevant documents
            for ans in extracted_answers:
                try:
                    doc_id_str = ans.get("DocumentID", "")
                    if doc_id_str.startswith("DOC"):
                        doc_id = f"doc_{int(doc_id_str[3:]) - 1}"
                        if doc_id in G.nodes:
                            G.nodes[doc_id]["relevant"] = True
                            G.nodes[doc_id]["score"] = ans.get("score", 0)
                            G.nodes[doc_id]["citation"] = ans.get("Citation", "")
                except Exception as mark_err:
                    print(f"Error marking relevant document: {str(mark_err)}")

            # Save graph to file with session-specific name
            try:
                if session_id:
                    # Create a session-specific filename
                    import hashlib
                    session_hash = hashlib.md5(session_id.encode()).hexdigest()[:10]
                    graph_filename = f"knowledge_graph_{session_hash}.graphml"
                else:
                    graph_filename = "knowledge_graph.graphml"

                nx.write_graphml(G, os.path.join(app.config['DATA_FOLDER'], graph_filename))
                print(f"Saved graph file for session {session_id} to {graph_filename}")
            except Exception as save_err:
                print(f"Error saving graph file: {str(save_err)}")
        except Exception as graph_err:
            print(f"Error creating knowledge graph: {str(graph_err)}")
            # Create minimal graph as fallback
            G = nx.DiGraph()
            G.add_node("doc_0", type="document", text="Document")
            G.add_node("theme_0", type="theme", description=final_themes[0] if final_themes else "Theme")
            G.add_edge("doc_0", "theme_0", weight=1.0)

        # Step 8: Generate final synthesized response with token optimization
        try:
            # Use a more efficient prompt template
            prompt = PromptTemplate(
                input_variables=["query", "themes", "answers"],
                template="Query: {query}\nThemes: {themes}\nAnswers: {answers}\nSynthesize a concise response that addresses the query, highlights key themes, and incorporates information from multiple answers when available. Include specific citations."
            )
            # Format the prompt
            prompt_text = prompt.format(
                query=query,
                themes="\n".join(final_themes),
                answers="\n".join([f"{a.get('DocumentID', 'DOC')}: {a.get('Extracted Answer', 'No answer')} ({a.get('Citation', 'No citation')})" for a in extracted_answers])
            )

            # Use the caching function to get the response
            synthesized = get_llm_response(prompt_text, session_id)

            # Log usage
            logger.info(f"Synthesized response for session {session_id}")

            final_response = synthesized.content
        except Exception as synth_err:
            print(f"Error synthesizing final response: {str(synth_err)}")
            # Use original response as fallback
            final_response = response

        # Return the synthesized response
        return {
            "response": final_response,
            "themes": final_themes,
            "graph": G
        }
    except Exception as e:
        # Log the full error with traceback
        print(f"Theme synthesis failed: {str(e)}")
        traceback.print_exc()

        # Return minimal response to avoid crashing the app
        G = nx.DiGraph()
        G.add_node("doc_0", type="document", text="Document")
        G.add_node("theme_0", type="theme", description="Document theme")
        G.add_edge("doc_0", "theme_0", weight=1.0)

        return {
            "response": response,
            "themes": ["Document analysis"],
            "graph": G
        }
    finally:
        # Ensure database connection is closed
        if conn:
            try:
                conn.close()
            except:
                pass

def visualize_graph(G, session_id=None):
    """Create an interactive knowledge graph visualization using Plotly

    Methodology:
    -----------
    1. Graph Preparation:
       - Converts NetworkX graph to Plotly-compatible format
       - Handles NumPy data types for JSON serialization
       - Uses Kamada-Kawai layout algorithm for optimal node positioning

    2. Visual Encoding:
       - Documents represented as circles (blue for regular, red for query-relevant)
       - Themes represented as green squares
       - Node size encodes relevance and connectivity
       - Edge width represents relationship strength

    3. Interactive Features:
       - Hover information for detailed node data
       - Filtering options for exploring specific themes
       - Relevance-based filtering to focus on important content
       - Zoom and pan capabilities for large graphs

    4. Session Isolation:
       - Creates session-specific filenames to prevent conflicts
       - Uses MD5 hashing of session ID for unique identification
       - Stores visualizations in the static folder for web access

    5. Responsive Design:
       - Optimized for both desktop and mobile viewing
       - Maintains readability at different screen sizes
       - Provides clear visual hierarchy of information

    Args:
        G (nx.DiGraph): NetworkX graph containing documents and themes
        session_id (str): ID of the current chat session for file naming

    Returns:
        str: Filename of the generated HTML visualization
    """
    try:
        # Convert any NumPy types in the graph attributes
        for node, attrs in G.nodes(data=True):
            for key, value in list(attrs.items()):
                attrs[key] = convert_numpy_types(value)

        for _, _, attrs in G.edges(data=True):
            for key, value in list(attrs.items()):
                attrs[key] = convert_numpy_types(value)

        pos = nx.kamada_kawai_layout(G, scale=2)

        edge_traces = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = edge[2].get('weight', 1)
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(width=1 + 2 * weight, color='#888'),
                hoverinfo='none',
                mode='lines'
            )
            edge_traces.append(edge_trace)

        node_x_doc, node_y_doc, node_text_doc, node_size_doc, node_color_doc, node_symbol_doc = [], [], [], [], [], []
        node_x_theme, node_y_theme, node_text_theme, node_size_theme, node_color_theme, node_symbol_theme = [], [], [], [], [], []

        max_score = max((G.nodes[n].get('score', 0) for n in G.nodes if G.nodes[n]['type'] == 'document'), default=1)
        for node in G.nodes():
            x, y = pos[node]
            node_info = G.nodes[node]
            node_type = node_info['type']
            text = f"{node_type.capitalize()}: "
            if node_type == 'document':
                doc_id = node_info.get('text', 'Unknown')[:50] + "..." if len(node_info.get('text', '')) > 50 else node_info.get('text', 'Unknown')
                text += f"DOC{node[4:]} - {doc_id}"
                if node_info.get('citation'):
                    text += f"\nCitation: {node_info['citation']}"
                if node_info.get('score'):
                    text += f"\nRelevance Score: {node_info['score']:.2f}"
                size = 15
                color = '#1f78b4'
                symbol = 'circle'
                if node_info.get('relevant', False):
                    color = '#d62728'
                    size = 20 + 15 * (node_info.get('score', 0) / max_score if max_score > 0 else 1)
                node_x_doc.append(x)
                node_y_doc.append(y)
                node_text_doc.append(text)
                node_size_doc.append(size)
                node_color_doc.append(color)
                node_symbol_doc.append(symbol)
            else:
                text += node_info.get('description', 'Unknown')
                size = 25 + 10 * G.degree(node)
                color = '#2ca02c'
                symbol = 'square'
                node_x_theme.append(x)
                node_y_theme.append(y)
                node_text_theme.append(text)
                node_size_theme.append(size)
                node_color_theme.append(color)
                node_symbol_theme.append(symbol)

        node_trace_doc = go.Scatter(
            x=node_x_doc,
            y=node_y_doc,
            mode='markers+text',
            text=[f"DOC{node[4:]}" for node in G.nodes if G.nodes[node]['type'] == 'document'],
            textposition='middle center',
            textfont=dict(size=10, color='#ffffff'),
            hoverinfo='text',
            hovertext=node_text_doc,
            marker=dict(
                showscale=False,
                color=node_color_doc,
                size=node_size_doc,
                symbol=node_symbol_doc,
                line=dict(width=2, color='#ffffff')
            ),
            name='Documents'
        )

        node_trace_theme = go.Scatter(
            x=node_x_theme,
            y=node_y_theme,
            mode='markers+text',
            text=[t.split(': ')[1][:20] for t in node_text_theme],
            textposition='middle center',
            textfont=dict(size=12, color='#ffffff'),
            hoverinfo='text',
            hovertext=node_text_theme,
            marker=dict(
                showscale=False,
                color=node_color_theme,
                size=node_size_theme,
                symbol=node_symbol_theme,
                line=dict(width=2, color='#ffffff')
            ),
            name='Themes'
        )

        # Create a title that includes the session ID for debugging
        graph_title = 'Advanced Interactive Knowledge Graph'
        if session_id:
            # Only show the first 8 characters of the session ID in the title
            short_id = session_id[:8] + '...' if len(session_id) > 8 else session_id
            graph_title += f' (Session: {short_id})'

        fig = go.Figure(
            data=edge_traces + [node_trace_doc, node_trace_theme],
            layout=go.Layout(
                title=graph_title,
                titlefont=dict(size=20, color='#333'),
                showlegend=True,
                legend=dict(
                    x=1.1,
                    y=1,
                    bgcolor='rgba(255,255,255,0.5)',
                    bordercolor='#333',
                    borderwidth=1
                ),
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[
                    dict(
                        text="Blue circles: Documents | Red circles: Query-Relevant | Green squares: Themes",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.02
                    ),
                    dict(
                        text="Use dropdown to filter, click nodes to highlight, zoom/pan to explore",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.05
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='rgba(245,245,245,1)',
                paper_bgcolor='rgba(255,255,255,1)',
                font=dict(family="Arial", size=12, color='#333'),
                height=800
            )
        )

        theme_options = [
            dict(
                label=f"Theme: {G.nodes[t]['description'][:20]}...",
                method="restyle",
                args=[{
                    "visible": [
                        [True if G.has_edge(n, t) or n == t else False for n in G.nodes] * len(edge_traces),
                        [G.nodes[n]['type'] == 'document' and G.has_edge(n, t) for n in G.nodes],
                        [G.nodes[n]['type'] == 'theme' and n == t for n in G.nodes]
                    ]
                }]
            ) for t in G.nodes if G.nodes[t]['type'] == 'theme'
        ]

        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=[
                        dict(
                            label="All Nodes",
                            method="restyle",
                            args=[{"visible": [True] * (len(edge_traces) + 2)}]
                        ),
                        dict(
                            label="High Relevance (>0.5)",
                            method="restyle",
                            args=[{
                                "visible": [
                                    [True] * len(edge_traces),
                                    [G.nodes[n].get('score', 0) > 0.5 or not G.nodes[n].get('relevant', False) for n in G.nodes if G.nodes[n]['type'] == 'document'],
                                    [True for _ in range(len(node_x_theme))]
                                ]
                            }]
                        ),
                        dict(
                            label="Query-Relevant Only",
                            method="restyle",
                            args=[{
                                "visible": [
                                    [True if G.nodes[e[0]].get('relevant', False) or G.nodes[e[1]].get('relevant', False) else False for e in G.edges] * len(edge_traces),
                                    [G.nodes[n].get('relevant', False) for n in G.nodes if G.nodes[n]['type'] == 'document'],
                                    [True for _ in range(len(node_x_theme))]
                                ]
                            }]
                        )
                    ] + theme_options,
                    direction="down",
                    showactive=True,
                    x=0.1,
                    xanchor="left",
                    y=1.1,
                    yanchor="top"
                ),
                dict(
                    buttons=[
                        dict(
                            label="Reset View",
                            method="relayout",
                            args=[{
                                "xaxis.range": None,
                                "yaxis.range": None
                            }]
                        )
                    ],
                    direction="down",
                    showactive=False,
                    x=0.3,
                    xanchor="left",
                    y=1.1,
                    yanchor="top"
                )
            ]
        )

        # Create a session-specific filename to prevent conflicts
        if session_id:
            # Use a hash of the session ID to create a unique filename
            import hashlib
            session_hash = hashlib.md5(session_id.encode()).hexdigest()[:10]
            graph_filename = f"knowledge_graph_{session_hash}.html"
        else:
            graph_filename = "knowledge_graph.html"

        graph_path = os.path.join(app.config['STATIC_FOLDER'], graph_filename)
        fig.write_html(graph_path, include_plotlyjs='cdn')
        print(f"Knowledge graph saved to {graph_path} for session {session_id}")
        return graph_filename
    except Exception as e:
        print(f"Graph visualization error: {str(e)}")
        raise Exception(f"Graph visualization failed: {str(e)}")

# Flask routes
@app.route('/', methods=['GET', 'POST'])
@handle_errors
def index():
    # Check if database has the required schema
    try:
        conn = sqlite3.connect(os.path.join(app.config['DATA_FOLDER'], "documents.db"))
        cursor = conn.cursor()

        # Check if the chat_sessions table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chat_sessions'")
        if not cursor.fetchone():
            # Database schema is outdated, show a warning
            flash("Database schema is outdated. Please reset the application to update it.", "warning")
            conn.close()
            # Create a temporary session ID
            session_id = "temp_" + str(uuid.uuid4())
            chat_history = []
            all_sessions = []
            return render_template('index.html',
                                session_id=session_id,
                                all_sessions=all_sessions)

        conn.close()
    except Exception as e:
        print(f"Database check error: {str(e)}")
        # If there's an error, we'll continue and let the normal flow handle it

    # Get or create a session
    session_id = request.args.get('session_id')
    if not session_id:
        try:
            session_id = get_or_create_session()
        except Exception as e:
            flash(f"Error creating session: {str(e)}. Please reset the application.", "danger")
            session_id = "temp_" + str(uuid.uuid4())
            chat_history = []
            all_sessions = []
            return render_template('index.html',
                                session_id=session_id,
                                all_sessions=all_sessions)

    # Get chat history for this session
    try:
        chat_history = get_chat_history(session_id)
    except Exception as e:
        flash(f"Error loading chat history: {str(e)}. Please reset the application.", "danger")
        chat_history = []

    # Get all available sessions for the sidebar
    try:
        all_sessions = get_all_sessions()
    except Exception as e:
        flash(f"Error loading sessions: {str(e)}. Please reset the application.", "danger")
        all_sessions = []

    if request.method == 'POST':
        # Handle file upload with query
        if 'file' in request.files and request.form.get('query'):
            files = request.files.getlist('file')
            query = request.form['query']

            # Check if any files were selected
            if not files or all(file.filename == '' for file in files):
                flash('No files selected.', 'danger')
                return redirect(url_for('index', session_id=session_id))

            # Check if all files have valid extensions
            invalid_files = [file.filename for file in files if not allowed_file(file.filename)]
            if invalid_files:
                flash(f'Invalid file type(s): {", ".join(invalid_files)}. Please upload only PDF, PNG, JPG, or JPEG files.', 'danger')
                return redirect(url_for('index', session_id=session_id))

            try:
                # Process all documents
                all_paragraphs = []
                all_doc_info = []
                processed_files = []
                failed_files = []
                error_messages = []

                # Create upload directory if it doesn't exist
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

                # Save and process each file
                for file in files:
                    if file and file.filename != '':
                        try:
                            # Secure the filename
                            filename = secure_filename(file.filename)
                            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

                            # Log file details
                            logger.info(f"Saving file: {filename} (size: {len(file.read())} bytes)")
                            file.seek(0)  # Reset file pointer after reading

                            # Save the file
                            file.save(file_path)

                            # Check if file was saved correctly
                            if not os.path.exists(file_path):
                                raise Exception(f"File was not saved correctly: {filename}")

                            # Check file size
                            file_size = os.path.getsize(file_path)
                            if file_size == 0:
                                raise Exception(f"File is empty: {filename}")

                            logger.info(f"Successfully saved file: {filename} (size: {file_size} bytes)")
                            processed_files.append(filename)

                            # Process document with session ID
                            paragraphs, _, doc_info = process_document(file_path, filename, session_id)

                            # Check if any text was extracted
                            if paragraphs:
                                all_paragraphs.extend(paragraphs)
                                all_doc_info.extend(doc_info)
                                logger.info(f"Successfully processed {filename}: extracted {len(paragraphs)} paragraphs")
                            else:
                                failed_files.append(filename)
                                error_messages.append(f"No text could be extracted from {filename}")
                                logger.warning(f"No text extracted from {filename}")
                        except Exception as doc_err:
                            failed_files.append(filename)
                            error_message = str(doc_err)
                            error_messages.append(f"Error processing {filename}: {error_message}")
                            logger.error(f"Error processing document {filename}: {error_message}")
                            traceback.print_exc()
                            # Continue with other files even if one fails

                # If no documents were processed successfully
                if not all_paragraphs:
                    error_details = "\n".join(error_messages[:5])  # Show first 5 errors
                    if len(error_messages) > 5:
                        error_details += f"\n...and {len(error_messages) - 5} more errors"

                    flash(f"None of the uploaded files could be processed. Please try different files. Error details: {error_details}", 'danger')
                    logger.error(f"All document processing failed. Errors: {error_messages}")
                    return redirect(url_for('index', session_id=session_id))

                # If some documents failed but others succeeded
                if failed_files and all_paragraphs:
                    flash(f"Some files could not be processed: {', '.join(failed_files)}. Continuing with the successfully processed files.", 'warning')
                    logger.warning(f"Partial document processing success. Failed files: {failed_files}")

                # Create a combined BM25 index for all paragraphs
                tokenized_paragraphs = [para.lower().split() for para in all_paragraphs]
                combined_bm25 = BM25Okapi(tokenized_paragraphs)

                # Add user query to chat history with file information
                file_info = f"Uploaded {len(processed_files)} file(s): {', '.join(processed_files)}"
                add_message(session_id, "user", f"{query}\n\n[{file_info}]")

                # Get updated chat history for processing
                chat_history = get_chat_history(session_id)

                # Process query with session context
                response, extracted_answers = process_query(query, all_paragraphs, combined_bm25, session_id, chat_history)

                # Generate themes and knowledge graph
                synthesized = synthesize_response(query, response, extracted_answers, all_paragraphs, session_id)
                graph_filename = visualize_graph(synthesized["graph"], session_id)

                # Add assistant response to chat history
                metadata = {
                    "themes": synthesized["themes"],
                    "graph_path": graph_filename,
                    "extracted_answers": extracted_answers,
                    "processed_files": processed_files
                }
                add_message(session_id, "assistant", synthesized["response"], metadata)

                # Update session title based on first query
                if len(chat_history) == 0:
                    update_session_title(session_id, f"Chat about {query[:30]}...")

                # Format table for display
                # Convert NumPy types in extracted_answers before creating DataFrame
                converted_answers = convert_numpy_types(extracted_answers)
                df = pd.DataFrame(converted_answers)
                if not df.empty:
                    # Rename columns for better display
                    df_display = df.copy()
                    df_display.columns = [col.replace('_', ' ') for col in df_display.columns]
                    table_html = df_display[["Filename", "Extracted Answer", "Citation"]].to_html(classes='table table-striped table-hover', index=False)
                else:
                    table_html = "<p>No relevant answers found.</p>"

                # Get updated chat history
                chat_history = get_chat_history(session_id)

                # Render the continuous chat interface without current_query and current_response
                # to prevent duplicate messages
                return render_template('chat.html',
                                     session_id=session_id,
                                     chat_history=chat_history,
                                     all_sessions=all_sessions,
                                     current_themes=synthesized["themes"],
                                     current_table=table_html,
                                     current_graph_path=graph_filename,
                                     documents=all_doc_info)
            except Exception as e:
                error_message = str(e)
                user_friendly_message = "There was an error processing your documents."

                if "no such table" in error_message.lower():
                    user_friendly_message = "Database error: The application database needs to be initialized. Please try resetting the application."
                elif "permission" in error_message.lower():
                    user_friendly_message = "Permission error: The application doesn't have permission to access some files. Please check your file permissions."
                elif "memory" in error_message.lower():
                    user_friendly_message = "Memory error: The document is too large to process. Please try with a smaller document or split it into parts."
                elif "timeout" in error_message.lower() or "timed out" in error_message.lower():
                    user_friendly_message = "Timeout error: The operation took too long. Please try with a smaller document or try again later."
                elif "file format" in error_message.lower() or "not supported" in error_message.lower():
                    user_friendly_message = "File format error: The document format is not supported or is corrupted. Please try a different file."

                flash(f"{user_friendly_message} (Technical details: {error_message})", 'danger')
                print(f"Processing error: {str(e)}")
                traceback.print_exc()
                return redirect(url_for('index', session_id=session_id))

        # Handle just a query (follow-up question) or query with new files
        elif request.form.get('query'):
            query = request.form['query']
            files = request.files.getlist('file') if 'file' in request.files else []
            has_new_files = files and any(file.filename != '' for file in files)

            try:
                # Get existing documents for this session
                conn = sqlite3.connect(os.path.join(app.config['DATA_FOLDER'], "documents.db"))
                cursor = conn.cursor()
                cursor.execute("SELECT text FROM documents WHERE session_id = ?", (session_id,))
                existing_texts = [row[0] for row in cursor.fetchall()]

                # Process new files if any
                new_paragraphs = []
                new_doc_info = []
                processed_files = []

                if has_new_files:
                    # Check if all files have valid extensions
                    invalid_files = [file.filename for file in files if file.filename != '' and not allowed_file(file.filename)]
                    if invalid_files:
                        flash(f'Invalid file type(s): {", ".join(invalid_files)}. Please upload only PDF, PNG, JPG, or JPEG files.', 'danger')
                        return redirect(url_for('index', session_id=session_id))

                    # Create upload directory if it doesn't exist
                    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

                    # Track failed files and error messages
                    failed_files = []
                    error_messages = []

                    # Save and process each new file
                    for file in files:
                        if file and file.filename != '':
                            try:
                                # Secure the filename
                                filename = secure_filename(file.filename)
                                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

                                # Log file details
                                logger.info(f"Saving file: {filename} (size: {len(file.read())} bytes)")
                                file.seek(0)  # Reset file pointer after reading

                                # Save the file
                                file.save(file_path)

                                # Check if file was saved correctly
                                if not os.path.exists(file_path):
                                    raise Exception(f"File was not saved correctly: {filename}")

                                # Check file size
                                file_size = os.path.getsize(file_path)
                                if file_size == 0:
                                    raise Exception(f"File is empty: {filename}")

                                logger.info(f"Successfully saved file: {filename} (size: {file_size} bytes)")
                                processed_files.append(filename)

                                # Process document with session ID
                                paragraphs, _, doc_info = process_document(file_path, filename, session_id)

                                # Check if any text was extracted
                                if paragraphs:
                                    new_paragraphs.extend(paragraphs)
                                    new_doc_info.extend(doc_info)
                                    logger.info(f"Successfully processed {filename}: extracted {len(paragraphs)} paragraphs")
                                else:
                                    failed_files.append(filename)
                                    error_messages.append(f"No text could be extracted from {filename}")
                                    logger.warning(f"No text extracted from {filename}")
                            except Exception as doc_err:
                                failed_files.append(filename)
                                error_message = str(doc_err)
                                error_messages.append(f"Error processing {filename}: {error_message}")
                                logger.error(f"Error processing document {filename}: {error_message}")
                                traceback.print_exc()
                                # Continue with other files even if one fails

                    # If some files failed but others succeeded
                    if failed_files and new_paragraphs:
                        flash(f"Some files could not be processed: {', '.join(failed_files)}. Continuing with the successfully processed files.", 'warning')
                        logger.warning(f"Partial document processing success. Failed files: {failed_files}")

                    # If all files failed
                    if has_new_files and not new_paragraphs and failed_files:
                        error_details = "\n".join(error_messages[:5])  # Show first 5 errors
                        if len(error_messages) > 5:
                            error_details += f"\n...and {len(error_messages) - 5} more errors"

                        flash(f"None of the new files could be processed. Error details: {error_details}", 'danger')
                        logger.error(f"All new document processing failed. Errors: {error_messages}")

                # Combine existing and new texts
                all_paragraphs = existing_texts + new_paragraphs

                if not all_paragraphs:
                    flash('Please upload at least one document first.', 'warning')
                    return redirect(url_for('index', session_id=session_id))

                # Create BM25 index for all paragraphs
                tokenized_paragraphs = [para.lower().split() for para in all_paragraphs]
                bm25 = BM25Okapi(tokenized_paragraphs)

                # Add user query to chat history with file information if new files were uploaded
                if has_new_files:
                    file_info = f"Uploaded {len(processed_files)} new file(s): {', '.join(processed_files)}"
                    add_message(session_id, "user", f"{query}\n\n[{file_info}]")
                else:
                    add_message(session_id, "user", query)

                # Get updated chat history
                chat_history = get_chat_history(session_id)

                # Process follow-up query
                response, extracted_answers = process_query(query, all_paragraphs, bm25, session_id, chat_history)

                # Generate themes and knowledge graph
                synthesized = synthesize_response(query, response, extracted_answers, all_paragraphs, session_id)
                graph_filename = visualize_graph(synthesized["graph"], session_id)

                # Add assistant response to chat history
                metadata = {
                    "themes": synthesized["themes"],
                    "graph_path": graph_filename,
                    "extracted_answers": extracted_answers
                }
                if has_new_files:
                    metadata["processed_files"] = processed_files

                add_message(session_id, "assistant", synthesized["response"], metadata)

                # Format table for display
                # Convert NumPy types in extracted_answers before creating DataFrame
                converted_answers = convert_numpy_types(extracted_answers)
                df = pd.DataFrame(converted_answers)
                if not df.empty:
                    # Rename columns for better display
                    df_display = df.copy()
                    df_display.columns = [col.replace('_', ' ') for col in df_display.columns]
                    table_html = df_display[["Filename", "Extracted Answer", "Citation"]].to_html(classes='table table-striped table-hover', index=False)
                else:
                    table_html = "<p>No relevant answers found.</p>"

                # Get updated chat history
                chat_history = get_chat_history(session_id)

                # Get document info for display
                try:
                    conn = sqlite3.connect(os.path.join(app.config['DATA_FOLDER'], "documents.db"))
                    cursor = conn.cursor()
                    cursor.execute("SELECT filename, page, paragraph, text FROM documents WHERE session_id = ? LIMIT 20", (session_id,))
                    doc_rows = cursor.fetchall()
                    doc_info = [{"id": f"{row[0]}_{row[1]}_{row[2]}", "page": row[1], "paragraph": row[2], "text_preview": row[3][:100] + "..." if len(row[3]) > 100 else row[3]} for row in doc_rows]
                    conn.close()
                except Exception as db_err:
                    print(f"Error getting document info: {str(db_err)}")
                    # Use new_doc_info as fallback
                    doc_info = new_doc_info if new_doc_info else []

                # Render the continuous chat interface without current_query and current_response
                # to prevent duplicate messages
                return render_template('chat.html',
                                     session_id=session_id,
                                     chat_history=chat_history,
                                     all_sessions=all_sessions,
                                     current_themes=synthesized["themes"],
                                     current_table=table_html,
                                     current_graph_path=graph_filename,
                                     documents=doc_info)
            except Exception as e:
                error_message = str(e)
                user_friendly_message = "There was an error processing your query."

                if "no such table" in error_message.lower():
                    user_friendly_message = "Database error: The application database needs to be initialized. Please try resetting the application."
                elif "no documents" in error_message.lower() or "no texts" in error_message.lower():
                    user_friendly_message = "No documents found: Please upload at least one document before asking questions."
                elif "openai" in error_message.lower() or "api" in error_message.lower():
                    user_friendly_message = "AI service error: There was an issue connecting to the AI service. Please try again later."
                elif "memory" in error_message.lower():
                    user_friendly_message = "Memory error: The operation requires more memory than available. Please try with simpler queries."
                elif "timeout" in error_message.lower() or "timed out" in error_message.lower():
                    user_friendly_message = "Timeout error: The operation took too long. Please try again with a simpler query."

                flash(f"{user_friendly_message} (Technical details: {error_message})", 'danger')
                print(f"Query processing error: {str(e)}")
                traceback.print_exc()
                return redirect(url_for('index', session_id=session_id))

    # GET request - show chat interface or start new chat
    if chat_history:
        # Get document info for display
        conn = sqlite3.connect(os.path.join(app.config['DATA_FOLDER'], "documents.db"))
        cursor = conn.cursor()
        cursor.execute("SELECT filename, page, paragraph, text FROM documents WHERE session_id = ? LIMIT 10", (session_id,))
        doc_rows = cursor.fetchall()
        doc_info = [{"id": f"{row[0]}_{row[1]}_{row[2]}", "page": row[1], "paragraph": row[2], "text_preview": row[3][:100] + "..." if len(row[3]) > 100 else row[3]} for row in doc_rows]
        conn.close()

        # Continue existing chat
        return render_template('chat.html',
                             session_id=session_id,
                             chat_history=chat_history,
                             all_sessions=all_sessions,
                             documents=doc_info)
    else:
        # Start new chat
        return render_template('index.html',
                             session_id=session_id,
                             all_sessions=all_sessions)

@app.route('/chat/<session_id>', methods=['GET', 'POST'])
@handle_errors
def chat(session_id):
    """View a specific chat session or process follow-up questions"""
    # Check if session exists
    session = get_session(session_id)
    if not session:
        flash("Chat session not found. Starting a new session.", "warning")
        return redirect(url_for('index'))

    # Get all available sessions for the sidebar
    try:
        all_sessions = get_all_sessions()
    except Exception as e:
        flash(f"Error loading sessions: {str(e)}. Please reset the application.", "danger")
        all_sessions = []

    # Handle POST request (follow-up question)
    if request.method == 'POST':
        if request.form.get('query'):
            query = request.form['query']
            files = request.files.getlist('file') if 'file' in request.files else []
            has_new_files = files and any(file.filename != '' for file in files)

            try:
                # Get chat history for processing
                chat_history = get_chat_history(session_id)

                # Add user query to chat history
                add_message(session_id, "user", query)

                # Process new files if any
                processed_files = []

                if has_new_files:
                    # Process uploaded files similar to index route
                    # This is simplified - you may need to add more code from the index route
                    for file in files:
                        if file and allowed_file(file.filename):
                            filename = secure_filename(file.filename)
                            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                            file.save(file_path)
                            processed_files.append(filename)

                    # Process documents and update vector store
                    # This would need the document processing code from the index route

                # Get paragraphs and BM25 index from database
                conn = sqlite3.connect(os.path.join(app.config['DATA_FOLDER'], "documents.db"))
                cursor = conn.cursor()
                cursor.execute("SELECT text FROM documents WHERE session_id = ?", (session_id,))
                all_paragraphs = [row[0] for row in cursor.fetchall()]
                conn.close()

                # Create BM25 index
                tokenized_paragraphs = [para.lower().split() for para in all_paragraphs]
                bm25 = BM25Okapi(tokenized_paragraphs)

                # Get updated chat history after adding user message
                chat_history = get_chat_history(session_id)

                # Get session-specific collection
                # No need to reference global collection anymore

                # Process follow-up query with strict session isolation
                print(f"Processing follow-up query for session: {session_id}")
                response, extracted_answers = process_query(query, all_paragraphs, bm25, session_id, chat_history)

                # Generate themes and knowledge graph
                synthesized = synthesize_response(query, response, extracted_answers, all_paragraphs, session_id)
                graph_filename = visualize_graph(synthesized["graph"])

                # Format table for display
                # Convert NumPy types in extracted_answers before creating DataFrame
                converted_answers = convert_numpy_types(extracted_answers)
                df = pd.DataFrame(converted_answers)
                if not df.empty:
                    # Rename columns for better display
                    df_display = df.copy()
                    df_display.columns = [col.replace('_', ' ') for col in df_display.columns]
                    table_html = df_display[["Filename", "Extracted Answer", "Citation"]].to_html(classes='table table-striped table-hover', index=False)
                else:
                    table_html = "<p>No relevant answers found.</p>"

                # Add assistant response to chat history
                metadata = {
                    "themes": synthesized["themes"],
                    "graph_path": graph_filename,
                    "extracted_answers": extracted_answers
                }
                if has_new_files:
                    metadata["processed_files"] = processed_files

                add_message(session_id, "assistant", synthesized["response"], metadata)

                # Get updated chat history
                chat_history = get_chat_history(session_id)

                # Get document info for display
                conn = sqlite3.connect(os.path.join(app.config['DATA_FOLDER'], "documents.db"))
                cursor = conn.cursor()
                cursor.execute("SELECT filename, page, paragraph, text FROM documents WHERE session_id = ? LIMIT 20", (session_id,))
                doc_rows = cursor.fetchall()
                doc_info = [{"id": f"{row[0]}_{row[1]}_{row[2]}", "page": row[1], "paragraph": row[2], "text_preview": row[3][:100] + "..." if len(row[3]) > 100 else row[3]} for row in doc_rows]
                conn.close()

                # Render the chat interface with current graph path
                # This ensures the knowledge graph is updated
                return render_template('chat.html',
                                     session_id=session_id,
                                     chat_history=chat_history,
                                     all_sessions=all_sessions,
                                     documents=doc_info,
                                     current_table=table_html,
                                     current_graph_path=graph_filename,
                                     current_themes=synthesized["themes"])

            except Exception as e:
                error_message = str(e)
                user_friendly_message = "There was an error processing your query."

                if "vector database" in error_message.lower():
                    user_friendly_message = "Vector database error: Please try resetting the application."
                elif "document" in error_message.lower() and "process" in error_message.lower():
                    user_friendly_message = "Document processing error: There was an issue with one of your documents."

                flash(f"{user_friendly_message} (Technical details: {error_message})", 'danger')
                print(f"Processing error: {str(e)}")
                traceback.print_exc()

                # Get chat history and document info for display
                chat_history = get_chat_history(session_id)

                conn = sqlite3.connect(os.path.join(app.config['DATA_FOLDER'], "documents.db"))
                cursor = conn.cursor()
                cursor.execute("SELECT filename, page, paragraph, text FROM documents WHERE session_id = ? LIMIT 20", (session_id,))
                doc_rows = cursor.fetchall()
                doc_info = [{"id": f"{row[0]}_{row[1]}_{row[2]}", "page": row[1], "paragraph": row[2], "text_preview": row[3][:100] + "..." if len(row[3]) > 100 else row[3]} for row in doc_rows]
                conn.close()

                # Create a simple fallback graph for error cases
                try:
                    # Create a minimal fallback graph
                    G = nx.DiGraph()
                    G.add_node("error_node", type="document", text="Error processing query")
                    G.add_node("error_theme", type="theme", description="Error")
                    G.add_edge("error_node", "error_theme", weight=1.0)

                    # Generate a simple graph visualization
                    fallback_graph = visualize_graph(G)
                except Exception:
                    # If even that fails, use a static filename
                    fallback_graph = "knowledge_graph.html"

                return render_template('chat.html',
                                     session_id=session_id,
                                     chat_history=chat_history,
                                     all_sessions=all_sessions,
                                     documents=doc_info,
                                     current_table="<p>No results available.</p>",
                                     current_graph_path=fallback_graph,
                                     current_themes=["Error processing query"])

    # GET request - show chat interface

    try:
        chat_history = get_chat_history(session_id)
        all_sessions = get_all_sessions()

        # Get document info for display
        conn = sqlite3.connect(os.path.join(app.config['DATA_FOLDER'], "documents.db"))
        cursor = conn.cursor()

        # Check if the documents table has the session_id column
        cursor.execute("PRAGMA table_info(documents)")
        columns = [column[1] for column in cursor.fetchall()]

        if 'session_id' in columns:
            cursor.execute("SELECT filename, page, paragraph, text FROM documents WHERE session_id = ? LIMIT 10", (session_id,))
        else:
            # Handle the case where the session_id column doesn't exist yet
            cursor.execute("SELECT filename, page, paragraph, text FROM documents LIMIT 10")
            # Show a warning to the user
            flash("Database schema needs to be updated. Please reset the application.", "warning")

        doc_rows = cursor.fetchall()
        doc_info = [{"id": f"{row[0]}_{row[1]}_{row[2]}", "page": row[1], "paragraph": row[2], "text_preview": row[3][:100] + "..." if len(row[3]) > 100 else row[3]} for row in doc_rows]
        conn.close()

        return render_template('chat.html',
                            session_id=session_id,
                            chat_history=chat_history,
                            all_sessions=all_sessions,
                            documents=doc_info)
    except Exception as e:
        flash(f"Error loading chat session: {str(e)}. Please reset the application.", "danger")
        return redirect(url_for('index'))

@app.route('/delete_chat/<session_id>', methods=['POST'])
@handle_errors
def delete_chat(session_id):
    """Delete a specific chat session"""
    try:
        # Check if session exists
        session = get_session(session_id)
        if not session:
            return jsonify({"success": False, "message": "Chat session not found"}), 404

        # Delete the session
        success = delete_session(session_id)

        if success:
            flash("Chat session deleted successfully.", "success")
            return jsonify({"success": True, "message": "Chat session deleted successfully"})
        else:
            flash("Error deleting chat session. Please try again.", "danger")
            return jsonify({"success": False, "message": "Error deleting chat session"}), 500
    except Exception as e:
        error_message = str(e)
        print(f"Error in delete_chat route: {error_message}")
        traceback.print_exc()
        flash(f"Error deleting chat session: {error_message}", "danger")
        return jsonify({"success": False, "message": f"Error: {error_message}"}), 500

@app.route('/reset', methods=['POST'])
@handle_errors
def reset():
    """Reset the application by deleting and recreating both databases"""
    success_messages = []
    error_messages = []

    # Declare global collection to ensure we're updating the global variable
    global collection

    try:
        # Step 1: Reset SQLite database
        try:
            # Close the existing connection pool
            try:
                global db_pool
                if db_pool:
                    db_pool.close_all()
                    logger.info("Closed existing database connection pool")
            except Exception as close_err:
                logger.error(f"Error closing connection pool: {str(close_err)}")

            # Initialize a fresh database with reset=True
            # This will handle deleting/renaming the old database
            db_path = init_database(reset=True)

            # Create a new connection pool
            db_pool = DBConnectionPool(db_path, max_connections=10)
            logger.info("Created new database connection pool")

            success_messages.append("SQL database reset successfully")
        except Exception as e:
            error_message = str(e)
            print(f"Error resetting SQL database: {error_message}")
            traceback.print_exc()
            error_messages.append(f"Error resetting SQL database: {error_message}")

        # Step 2: Reset ChromaDB
        try:
            chroma_path = os.path.join(app.config['DATA_FOLDER'], "chroma_db")
            if os.path.exists(chroma_path):
                try:
                    # Try to delete the directory
                    shutil.rmtree(chroma_path)
                    print(f"Deleted ChromaDB directory: {chroma_path}")
                except PermissionError:
                    # If permission error, try to rename it instead
                    backup_path = f"{chroma_path}_backup_{int(time.time())}"
                    shutil.move(chroma_path, backup_path)
                    print(f"Moved existing ChromaDB to {backup_path}")
                except Exception as rm_err:
                    print(f"Error removing ChromaDB directory: {str(rm_err)}")
                    # Try to delete files individually
                    for root, dirs, files in os.walk(chroma_path, topdown=False):
                        for name in files:
                            try:
                                os.remove(os.path.join(root, name))
                            except:
                                pass
                        for name in dirs:
                            try:
                                os.rmdir(os.path.join(root, name))
                            except:
                                pass

            # Initialize a fresh ChromaDB
            collection = init_chromadb(reset=True)
            success_messages.append("Vector database reset successfully")
        except Exception as e:
            error_message = str(e)
            print(f"Error resetting vector database: {error_message}")
            traceback.print_exc()
            error_messages.append(f"Error resetting vector database: {error_message}")

        # Step 3: Clear uploads folder
        try:
            deleted_files = 0
            for file in os.listdir(app.config['UPLOAD_FOLDER']):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        deleted_files += 1
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                        deleted_files += 1
                except Exception as file_err:
                    print(f"Error removing file {file_path}: {str(file_err)}")
            success_messages.append(f"Cleared {deleted_files} uploaded files")
        except Exception as e:
            error_message = str(e)
            print(f"Error clearing uploads: {error_message}")
            error_messages.append(f"Error clearing uploads: {error_message}")

        # Step 4: Clear static folder (except CSS and JS)
        try:
            deleted_files = 0
            for file in os.listdir(app.config['STATIC_FOLDER']):
                if file not in ['css', 'js']:
                    file_path = os.path.join(app.config['STATIC_FOLDER'], file)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            deleted_files += 1
                        elif os.path.isdir(file_path) and file not in ['css', 'js']:
                            shutil.rmtree(file_path)
                            deleted_files += 1
                    except Exception as file_err:
                        print(f"Error removing file {file_path}: {str(file_err)}")
            success_messages.append(f"Cleared {deleted_files} generated files")
        except Exception as e:
            error_message = str(e)
            print(f"Error clearing static files: {error_message}")
            error_messages.append(f"Error clearing static files: {error_message}")

        # Step 5: Create a default session
        try:
            default_session_id = create_session()
            success_messages.append(f"Created default session: {default_session_id}")
        except Exception as e:
            error_message = str(e)
            print(f"Error creating default session: {error_message}")
            error_messages.append(f"Error creating default session: {error_message}")

        # Report results
        if error_messages:
            flash(f"Reset completed with some issues: {'; '.join(error_messages)}", 'warning')
        else:
            flash(f"Application has been reset successfully! {'; '.join(success_messages)}", 'success')

    except Exception as e:
        error_message = str(e)
        print(f"Error during reset process: {error_message}")
        traceback.print_exc()
        flash(f'Error during reset process: {error_message}', 'danger')

    # Always redirect to the index page without a session ID to start fresh
    return redirect(url_for('index'))

# Cache statistics and management endpoint
@app.route('/cache', methods=['GET', 'POST'])
@handle_errors
def cache_management():
    """Get cache statistics or clear caches"""
    if request.method == 'POST':
        # Check if the request is authorized (you might want to add authentication)
        action = request.form.get('action', '')
        if action == 'clear':
            clear_caches()
            return jsonify({
                "status": "success",
                "message": "Caches cleared successfully",
                "timestamp": datetime.datetime.now().isoformat()
            })
        else:
            return jsonify({
                "status": "error",
                "message": f"Unknown action: {action}",
                "timestamp": datetime.datetime.now().isoformat()
            }), 400
    else:
        # Get cache statistics
        embedding_cache_size = len(embedding_cache)
        llm_cache_size = len(llm_cache)

        # Format the response
        response = {
            "embedding_cache": {
                "size": embedding_cache_size,
                "memory_estimate": embedding_cache_size * 1536 * 4 / (1024 * 1024)  # Rough estimate in MB
            },
            "llm_cache": {
                "size": llm_cache_size
            },
            "timestamp": datetime.datetime.now().isoformat()
        }

        return jsonify(response)

# Token usage statistics endpoint
@app.route('/usage', methods=['GET'])
@handle_errors
def token_usage():
    """Get token usage statistics"""
    # Check if the request is authorized (you might want to add authentication)
    try:
        # Get total usage
        total_usage = token_tracker.get_total_usage()
        cost_estimate = token_tracker.estimate_cost()

        # Get session-specific usage
        session_usage = {}
        for session_id in token_tracker.session_usage:
            session_usage[session_id] = token_tracker.get_session_usage(session_id)

        # Format the response
        response = {
            "total_usage": total_usage,
            "cost_estimate": cost_estimate,
            "session_usage": session_usage,
            "timestamp": datetime.datetime.now().isoformat()
        }

        return jsonify(response)
    except Exception as e:
        logger.error(f"Error getting token usage: {str(e)}")
        return jsonify({
            "error": str(e),
            "timestamp": datetime.datetime.now().isoformat()
        }), 500

# Diagnostic endpoint to check document processing capabilities
@app.route('/diagnostics', methods=['GET'])
@handle_errors
def diagnostics():
    """Run diagnostics on document processing capabilities"""
    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "system": {},
        "libraries": {},
        "tesseract": {},
        "folders": {},
        "test_extraction": {}
    }

    # Check system
    try:
        import platform
        results["system"] = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "processor": platform.processor()
        }
    except Exception as e:
        results["system"] = {"error": str(e)}

    # Check libraries
    libraries = ["pdfplumber", "pytesseract", "PIL", "fitz"]
    for lib in libraries:
        try:
            if lib == "PIL":
                from PIL import __version__ as pil_version
                results["libraries"][lib] = pil_version
            elif lib == "pdfplumber":
                import pdfplumber
                results["libraries"][lib] = pdfplumber.__version__
            elif lib == "pytesseract":
                import pytesseract
                # Convert Version object to string to avoid JSON serialization issues
                results["libraries"][lib] = "Installed"
            elif lib == "fitz":
                import fitz
                # Convert Version object to string to avoid JSON serialization issues
                results["libraries"][lib] = "Installed"
        except Exception as e:
            results["libraries"][lib] = {"error": str(e)}

    # Check Tesseract
    try:
        # Just check if pytesseract is available without getting version
        import pytesseract
        results["tesseract"] = {
            "version": "Installed",
            "available": True
        }

        # Try to run a simple OCR test
        try:
            from PIL import Image, ImageDraw

            # Create a simple test image with text
            img = Image.new('RGB', (200, 50), color=(255, 255, 255))
            d = ImageDraw.Draw(img)
            d.text((10, 10), "Testing OCR", fill=(0, 0, 0))

            # Save the test image
            test_img_path = os.path.join(app.config['UPLOAD_FOLDER'], "test_ocr.png")
            img.save(test_img_path)

            # Run OCR on the test image
            ocr_text = pytesseract.image_to_string(img)
            results["tesseract"]["test_ocr"] = {
                "text": ocr_text,
                "success": "Testing" in ocr_text
            }
        except Exception as ocr_err:
            results["tesseract"]["test_ocr"] = {"error": str(ocr_err)}
    except Exception as e:
        results["tesseract"] = {"error": str(e)}

    # Check folders
    folders = ["UPLOAD_FOLDER", "DATA_FOLDER", "STATIC_FOLDER"]
    for folder in folders:
        folder_path = app.config.get(folder)
        try:
            results["folders"][folder] = {
                "path": folder_path,
                "exists": os.path.exists(folder_path),
                "writable": os.access(folder_path, os.W_OK) if os.path.exists(folder_path) else False
            }
        except Exception as e:
            results["folders"][folder] = {"error": str(e)}

    # Test PDF extraction with both libraries
    try:
        # Create a simple PDF with text
        test_pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], "test.pdf")

        # Try to find an existing PDF to test with
        pdf_files = []
        for root, _, files in os.walk(app.config['UPLOAD_FOLDER']):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))

        if pdf_files:
            test_pdf_path = pdf_files[0]
            results["test_extraction"]["pdf_file"] = test_pdf_path

            # Test with pdfplumber
            try:
                with pdfplumber.open(test_pdf_path) as pdf:
                    if len(pdf.pages) > 0:
                        text = pdf.pages[0].extract_text()
                        results["test_extraction"]["pdfplumber"] = {
                            "success": text is not None and len(text) > 0,
                            "text_length": len(text) if text else 0,
                            "text_sample": text[:100] if text else None
                        }
            except Exception as plumber_err:
                results["test_extraction"]["pdfplumber"] = {"error": str(plumber_err)}

            # Test with PyMuPDF
            try:
                doc = fitz.open(test_pdf_path)
                if len(doc) > 0:
                    text = doc[0].get_text()
                    results["test_extraction"]["pymupdf"] = {
                        "success": text is not None and len(text) > 0,
                        "text_length": len(text) if text else 0,
                        "text_sample": text[:100] if text else None
                    }
            except Exception as fitz_err:
                results["test_extraction"]["pymupdf"] = {"error": str(fitz_err)}
        else:
            results["test_extraction"]["status"] = "No PDF files found for testing"
    except Exception as e:
        results["test_extraction"] = {"error": str(e)}

    # Convert any non-serializable objects to strings
    def make_json_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_json_serializable(item) for item in obj]
        elif isinstance(obj, (int, float, bool, str, type(None))):
            return obj
        else:
            return str(obj)

    # Make sure all results are JSON serializable
    results = make_json_serializable(results)

    return jsonify(results)

# Test file upload endpoint
@app.route('/test-upload', methods=['GET', 'POST'])
@handle_errors
def test_upload():
    """Test file upload and processing in isolation"""
    if request.method == 'POST':
        results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "upload_status": {},
            "processing_status": {},
            "extraction_results": {}
        }

        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({
                "error": "No file part in the request",
                "timestamp": datetime.datetime.now().isoformat()
            }), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({
                "error": "No file selected",
                "timestamp": datetime.datetime.now().isoformat()
            }), 400

        # Process the file
        try:
            # Create upload directory if it doesn't exist
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

            # Save the file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Log file details
            file_size = len(file.read())
            file.seek(0)  # Reset file pointer

            results["upload_status"] = {
                "filename": filename,
                "size": file_size,
                "content_type": file.content_type
            }

            # Save the file
            file.save(file_path)

            # Check if file was saved correctly
            if os.path.exists(file_path):
                saved_size = os.path.getsize(file_path)
                results["upload_status"]["saved"] = True
                results["upload_status"]["saved_size"] = saved_size
                results["upload_status"]["path"] = file_path
            else:
                results["upload_status"]["saved"] = False
                return jsonify(results), 500

            # Try to process the file with each method separately

            # 1. Try PyMuPDF
            try:
                pymupdf_results = {"success": False, "pages": []}
                doc = fitz.open(file_path)
                pymupdf_results["page_count"] = len(doc)

                for page_num in range(len(doc)):
                    try:
                        page_text = doc[page_num].get_text()
                        pymupdf_results["pages"].append({
                            "page": page_num + 1,
                            "text_length": len(page_text),
                            "text_sample": page_text[:100] if page_text else None
                        })
                    except Exception as page_err:
                        pymupdf_results["pages"].append({
                            "page": page_num + 1,
                            "error": str(page_err)
                        })

                pymupdf_results["success"] = any(page.get("text_length", 0) > 0 for page in pymupdf_results["pages"])
                results["extraction_results"]["pymupdf"] = pymupdf_results
            except Exception as fitz_err:
                results["extraction_results"]["pymupdf"] = {"error": str(fitz_err)}

            # 2. Try pdfplumber
            if file_path.lower().endswith(".pdf"):
                try:
                    pdfplumber_results = {"success": False, "pages": []}
                    with pdfplumber.open(file_path) as pdf:
                        pdfplumber_results["page_count"] = len(pdf.pages)

                        for page_num, page in enumerate(pdf.pages):
                            try:
                                text = page.extract_text()
                                pdfplumber_results["pages"].append({
                                    "page": page_num + 1,
                                    "text_length": len(text) if text else 0,
                                    "text_sample": text[:100] if text else None
                                })
                            except Exception as page_err:
                                pdfplumber_results["pages"].append({
                                    "page": page_num + 1,
                                    "error": str(page_err)
                                })

                    pdfplumber_results["success"] = any(page.get("text_length", 0) > 0 for page in pdfplumber_results["pages"])
                    results["extraction_results"]["pdfplumber"] = pdfplumber_results
                except Exception as plumber_err:
                    results["extraction_results"]["pdfplumber"] = {"error": str(plumber_err)}

            # 3. Try OCR if it's an image or PDF
            try:
                ocr_results = {"success": False}

                if file_path.lower().endswith((".png", ".jpg", ".jpeg")):
                    # Direct OCR for images
                    image = Image.open(file_path)
                    if image.mode not in ('RGB', 'L'):
                        image = image.convert('RGB')

                    text = pytesseract.image_to_string(image)
                    ocr_results["text_length"] = len(text) if text else 0
                    ocr_results["text_sample"] = text[:100] if text else None
                    ocr_results["success"] = len(text) > 0 if text else False

                elif file_path.lower().endswith(".pdf"):
                    # Try OCR on first page of PDF
                    ocr_results["pages"] = []
                    doc = fitz.open(file_path)

                    for page_num in range(min(1, len(doc))):  # Just try first page
                        try:
                            pix = doc[page_num].get_pixmap()
                            img_data = pix.tobytes("png")
                            img = Image.open(io.BytesIO(img_data))

                            text = pytesseract.image_to_string(img)
                            ocr_results["pages"].append({
                                "page": page_num + 1,
                                "text_length": len(text) if text else 0,
                                "text_sample": text[:100] if text else None
                            })
                        except Exception as page_err:
                            ocr_results["pages"].append({
                                "page": page_num + 1,
                                "error": str(page_err)
                            })

                    if "pages" in ocr_results:
                        ocr_results["success"] = any(page.get("text_length", 0) > 0 for page in ocr_results["pages"])

                results["extraction_results"]["ocr"] = ocr_results
            except Exception as ocr_err:
                results["extraction_results"]["ocr"] = {"error": str(ocr_err)}

            # 4. Try our full document processing function
            try:
                texts, _, doc_info = process_document(file_path, filename, "test_session")

                results["processing_status"] = {
                    "success": len(texts) > 0,
                    "text_sections": len(texts),
                    "doc_info_items": len(doc_info),
                    "samples": []
                }

                # Add samples of extracted text
                for page_num, text in texts[:3]:  # Show first 3 sections
                    results["processing_status"]["samples"].append({
                        "page": page_num,
                        "text_length": len(text),
                        "text_sample": text[:100] + "..." if len(text) > 100 else text
                    })

            except Exception as proc_err:
                results["processing_status"] = {
                    "success": False,
                    "error": str(proc_err)
                }

            return jsonify(results)

        except Exception as e:
            return jsonify({
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.datetime.now().isoformat()
            }), 500

    # GET request - show upload form
    return '''
    <!doctype html>
    <title>Test File Upload</title>
    <h1>Test File Upload</h1>
    <p>This endpoint tests file upload and processing in isolation.</p>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    <p>After uploading, you'll see detailed diagnostic information about the file processing.</p>
    '''

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    try:
        # Check database connection
        with DBConnection(db_pool) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            db_status = cursor.fetchone() is not None

        # Check ChromaDB connection
        try:
            chroma_path = os.path.join(app.config['DATA_FOLDER'], "chroma_db")
            # Just try to create a client to check if ChromaDB is accessible
            _ = chromadb.PersistentClient(path=chroma_path)
            chroma_status = True
        except Exception:
            chroma_status = False

        # Check OpenAI API
        openai_status = OPENAI_API_KEY is not None

        # Overall status
        status = db_status and chroma_status and openai_status

        response = {
            "status": "healthy" if status else "unhealthy",
            "timestamp": datetime.datetime.now().isoformat(),
            "components": {
                "database": "up" if db_status else "down",
                "vector_db": "up" if chroma_status else "down",
                "openai_api": "configured" if openai_status else "not_configured"
            },
            "version": "1.0.0"
        }

        return jsonify(response), 200 if status else 503
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.datetime.now().isoformat()
        }), 503

# Function to clear caches periodically
def clear_caches():
    """Clear embedding and LLM caches to free up memory"""
    global embedding_cache, llm_cache

    # Get current cache sizes
    embedding_cache_size = len(embedding_cache)
    llm_cache_size = len(llm_cache)

    # Clear caches
    embedding_cache.clear()
    llm_cache.clear()

    logger.info(f"Cleared caches: {embedding_cache_size} embeddings and {llm_cache_size} LLM responses")

# Graceful shutdown
def shutdown_handler():
    """Clean up resources on application shutdown"""
    logger.info("Application shutting down, cleaning up resources...")

    # Close all database connections
    try:
        db_pool.close_all()
        logger.info("Closed all database connections")
    except Exception as e:
        logger.error(f"Error closing database connections: {str(e)}")

    # Clear session collections
    try:
        global session_collections
        session_collections.clear()
        logger.info("Cleared session collections")
    except Exception as e:
        logger.error(f"Error clearing session collections: {str(e)}")

    # Clear caches
    try:
        clear_caches()
    except Exception as e:
        logger.error(f"Error clearing caches: {str(e)}")

# Register shutdown handler
import atexit
atexit.register(shutdown_handler)

# Schedule cache clearing every 6 hours
def schedule_cache_clearing():
    """Schedule periodic cache clearing"""
    clear_caches()
    # Schedule next run in 6 hours
    threading.Timer(6 * 60 * 60, schedule_cache_clearing).start()

# Start the cache clearing schedule
if not app.config['DEBUG']:
    threading.Timer(6 * 60 * 60, schedule_cache_clearing).start()
    logger.info("Scheduled cache clearing every 6 hours")

if __name__ == '__main__':
    # Log startup information
    logger.info(f"Starting application in {'debug' if app.config['DEBUG'] else 'production'} mode")
    logger.info(f"Data folder: {app.config['DATA_FOLDER']}")
    logger.info(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    logger.info(f"Static folder: {app.config['STATIC_FOLDER']}")

    # Run the application
    app.run(
        host=os.getenv('FLASK_HOST', '0.0.0.0'),
        port=int(os.getenv('FLASK_PORT', 5000)),
        debug=app.config['DEBUG']
    )