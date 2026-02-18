FROM rocm/pytorch:latest

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib C Library
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xvzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib-0.4.0-src.tar.gz ta-lib/

WORKDIR /app

# Copy requirements first to leverage cache
COPY requirements.txt .

# Install dependencies (ignoring torch in requirements if possible, or ensuring compatible versions)
# We remove torch from requirements temporarily or rely on pip to not downgrade if already installed
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Set Env
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port (FastAPI API Server)
EXPOSE 8000

# Default command: Launch IARA API Server
CMD ["python", "-m", "src.api.main"]
