# Use official Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Create the components directory
RUN mkdir -p components

# Copy application code
COPY main.py .  
COPY components/ ./components/

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Define default command
CMD ["python", "main.py"]
