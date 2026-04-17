FROM python:3.12-slim

WORKDIR /app

# Install system deps for httpx, markdownify, and PDF support
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# AGENTS.md is copied into the build context by CI (from data/AGENTS.md)
# and will be synced to the data volume on startup by the app
COPY AGENTS.md.bundled /app/AGENTS.md.bundled

CMD ["python", "main.py"]
