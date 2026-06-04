FROM python:3.11-slim

WORKDIR /srv

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from parent directory
COPY ../requirements.txt /srv/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . /srv

# Expose port
EXPOSE 3000

# Default command
CMD ["streamlit", "run", "streamlit_app.py", "--server.port", "3000", "--server.headless", "true", "--server.address", "0.0.0.0"]
