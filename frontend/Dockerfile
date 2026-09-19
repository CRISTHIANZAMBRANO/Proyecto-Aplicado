FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_API_URL=http://api:8000

WORKDIR /app

COPY requirements-frontend.txt ./
RUN pip install --no-cache-dir -r requirements-frontend.txt

COPY frontend ./frontend
COPY data ./data
COPY artifacts/model_metadata.json ./artifacts/model_metadata.json

EXPOSE 8501
CMD ["streamlit", "run", "frontend/app.py", "--global.developmentMode=false", "--server.address=0.0.0.0", "--server.port=8501"]
