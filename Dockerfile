FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libreoffice \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "raganything[all]" \
    "mineru" \
    fastapi \
    "uvicorn[standard]" \
    python-multipart

COPY server.py .

RUN mkdir -p /app/rag_storage /app/output

EXPOSE 9622

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9622"]
