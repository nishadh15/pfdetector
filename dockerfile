FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir \
    --default-timeout=300 \
    --retries=10 \
    -r requirements.txt
COPY app.py .
COPY mltracker.py .
COPY Models/ ./models/
EXPOSE 10000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}"]
