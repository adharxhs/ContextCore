FROM python:3.12-slim
WORKDIR /srv
ENV PYTHONPATH=/srv PYTHONUNBUFFERED=1
COPY backend/requirements.txt backend/requirements.txt
COPY ml/requirements.txt ml/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt -r ml/requirements.txt
COPY backend backend
COPY ml ml
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
