# Use official Python image
FROM python:3.12-slim

# Install git (required for pip to install from git+https) and libgomp1
RUN apt-get update && apt-get install -y git libgomp1 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create the components directory
RUN mkdir -p components
# Copy application code
COPY main.py .  
COPY components/ ./components/

# Define default command
CMD ["python", "main.py"]