FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y build-essential

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir fastapi uvicorn numpy scipy websockets

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
