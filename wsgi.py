"""
WSGI entry point for Gunicorn to serve the Flask application
"""

from app import app

if __name__ == "__main__":
    app.run()
