FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -i https://pypi.org/simple --timeout 120 --retries 5 -r requirements.txt


COPY app ./app

RUN mkdir -p /app/data /app/storage

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host","0.0.0.0","--port","8000"]
