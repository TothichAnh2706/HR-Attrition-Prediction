FROM python:3.10-slim

WORKDIR /app

# Cài đặt curl và thư viện phụ trợ
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code và model artifacts
COPY main.py .
COPY model_artifacts/ ./model_artifacts/

# Google Cloud Run tự động inject biến môi trường PORT (mặc định 8080)
ENV PORT=8080
EXPOSE 8080

# Chạy ứng dụng FastAPI qua Uvicorn
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}
