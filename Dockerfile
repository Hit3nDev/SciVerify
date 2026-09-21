FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Data is not baked into the image (license + size). Run download at
# container start or mount a pre-downloaded data/ volume:
#   docker run -v $(pwd)/data:/app/data sciverify python -m data.download_data
RUN pytest tests/ -v

EXPOSE 8000 8501

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
