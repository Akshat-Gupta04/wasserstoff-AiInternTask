# Production Deployment Guide

This document provides instructions for deploying the Document Research & Theme Identification Chatbot in a production environment.

## Repository Information

- **GitHub Repository**: [https://github.com/Akshat-Gupta04/wasserstoff-AiInternTask](https://github.com/Akshat-Gupta04/wasserstoff-AiInternTask)
- **Application Type**: Flask web application with Gunicorn WSGI server
- **Python Version**: 3.9.0

## Prerequisites

- Docker and Docker Compose
- OpenAI API key
- 4GB+ RAM recommended
- 10GB+ disk space

## Deployment Steps

### 1. Prepare Environment

1. Clone the repository:
   ```
   git clone https://github.com/Akshat-Gupta04/wasserstoff-AiInternTask.git
   cd wasserstoff-AiInternTask
   ```

2. Create environment file:
   ```
   cp .env.example .env
   ```

3. Edit the `.env` file and set your OpenAI API key and other configuration options:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   SECRET_KEY=generate_a_secure_random_string
   ```

### 2. Build and Start the Application

1. Build and start the Docker container:
   ```
   docker-compose up -d
   ```

2. Check the logs to ensure everything started correctly:
   ```
   docker-compose logs -f
   ```

3. Access the application at `http://your-server-ip:5000`

### 3. Render Deployment (Recommended)

The application is configured for easy deployment on Render.com:

1. Sign up for a Render account at [render.com](https://render.com)

2. Fork or clone the repository to your GitHub account:
   ```
   git clone https://github.com/Akshat-Gupta04/wasserstoff-AiInternTask.git
   ```

3. In the Render dashboard, click "New" and select "Blueprint"

4. Connect to your GitHub repository

5. Render will detect the `render.yaml` file and configure the service automatically

6. Add your `OPENAI_API_KEY` when prompted

7. Click "Apply" to deploy

8. Once deployed, access your application at the provided URL

### 4. Production Configuration

For a production environment, consider the following:

1. Use a reverse proxy (Nginx, Apache) to handle SSL termination
2. Set up proper authentication if needed
3. Configure regular backups of the `data` directory
4. Set up monitoring using the `/health` endpoint

Example Nginx configuration:

```nginx
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 5. Maintenance

1. Update the application:
   ```
   git pull
   docker-compose down
   docker-compose up -d --build
   ```

2. Backup data:
   ```
   tar -czf backup-$(date +%Y%m%d).tar.gz data/
   ```

3. Monitor the application:
   ```
   curl http://localhost:5000/health
   ```

### 6. Troubleshooting

1. Check application logs:
   ```
   docker-compose logs -f
   ```

2. Reset the application if needed:
   - Access the application and use the "Reset All Data" button
   - Or manually delete the data directory and restart:
     ```
     docker-compose down
     rm -rf data/*
     docker-compose up -d
     ```

3. If the application becomes unresponsive:
   ```
   docker-compose restart
   ```

## Security Considerations

1. Always use HTTPS in production
2. Regularly update dependencies
3. Limit file upload size and types
4. Implement proper authentication if needed
5. Regularly backup your data
6. Monitor the application for unusual activity
