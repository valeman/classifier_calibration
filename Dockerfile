# Use official Python image
FROM python:3.12-slim

# Install git (required for pip to install from git+https) and libgomp1
RUN apt-get update && apt-get install -y git libgomp1 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir autogluon.tabular[all]==1.3.1
RUN pip install --no-cache-dir git+https://github.com/autogluon/tabrepo.git@3396519469875a85e8a8090ee96821e409e09740#egg=tabrepo[benchmark]

# Copy application code
COPY src/ .  

# Define default command
CMD ["sh", "-c", "python main.py && python analyse_results.py"]