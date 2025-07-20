FROM ghcr.io/mlflow/mlflow:v3.1.0

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install from requirements file
RUN pip install --no-cache-dir -r requirements.txt