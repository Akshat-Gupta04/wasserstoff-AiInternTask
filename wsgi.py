"""
WSGI entry point for Gunicorn to serve the Flask application

This file serves as the entry point for Gunicorn when deploying to production
environments like Render. It initializes the Flask application with production
settings and ensures all necessary directories and databases are created.
"""

import os
import logging
from app import app, init_database, init_chromadb

# Configure logging for production
if not app.debug:
    # Set up production logging
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)

    # Log startup information
    app.logger.info("Starting application in production mode")
    app.logger.info(f"Data folder: {app.config['DATA_FOLDER']}")
    app.logger.info(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    app.logger.info(f"Static folder: {app.config['STATIC_FOLDER']}")

    # Ensure required directories exist
    os.makedirs(app.config['DATA_FOLDER'], exist_ok=True)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(os.path.join(app.config['STATIC_FOLDER'], 'graphs'), exist_ok=True)

    # Initialize database and vector store if needed
    try:
        init_database(reset=False)
        init_chromadb(reset=False)
        app.logger.info("Database and vector store initialized successfully")
    except Exception as e:
        app.logger.error(f"Error initializing database or vector store: {str(e)}")

# Application instance for Gunicorn
if __name__ == "__main__":
    app.run()
