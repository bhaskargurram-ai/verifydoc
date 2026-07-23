# VerifyDoc self-hosted server (REST API + web review UI + messaging bots).
# Build:  docker build -t verifydoc .
# Run:    docker run -p 8000:8000 verifydoc   ->  http://localhost:8000
FROM python:3.12-slim

WORKDIR /app

# system libs for image/PDF handling used by the pdf + ocr extras
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

# server + local extraction (text/PDF/OCR) so documents can be processed on-box
RUN pip install --no-cache-dir -e ".[server,pdf,ocr]"

EXPOSE 8000
ENV VERIFYDOC_HOST=0.0.0.0 \
    VERIFYDOC_PORT=8000

CMD ["verifydoc-server"]
