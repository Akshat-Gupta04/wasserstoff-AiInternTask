"""
WSGI entry point for Gunicorn to serve the Flask application

This file serves as the entry point for Gunicorn when deploying to production
environments like Render.
"""

from app import app

# Application instance for Gunicorn
if __name__ == "__main__":
    app.run()
