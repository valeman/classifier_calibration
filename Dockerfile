# Use official Python image
FROM python:3.12-slim

# Install reqs
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl git build-essential g++ libgomp1 \
 && rm -rf /var/lib/apt/lists/*
 
# Set working directory
WORKDIR /app

# Install dependencies
COPY LICENSE .
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir autogluon.tabular[all]==1.3.1
RUN pip install --no-cache-dir git+https://github.com/autogluon/tabrepo.git@3396519469875a85e8a8090ee96821e409e09740#egg=tabrepo[benchmark]
RUN pip install --upgrade "scipy>=1.11.4,<1.13"

# Copy application code
COPY src/ .  

# Define default command
CMD ["sh", "-c", "python main.py && python analyse_results.py"]