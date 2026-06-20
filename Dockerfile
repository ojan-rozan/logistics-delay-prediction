FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY test/ ./test/
EXPOSE 8000
CMD ["uvicorn", "test.app:app", "--host", "0.0.0.0", "--port", "8000"]
