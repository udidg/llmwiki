#!/usr/bin/env bash
# deploy.sh — Pull latest config + image, migrate data, restart, and tail logs
set -euo pipefail

REPO_RAW="https://raw.githubusercontent.com/udidg/llmwiki/main"

cd "$(dirname "$0")"

# ── Pull latest docker-compose.yml from repo ──────────────────────────────────
echo "⬇️  Fetching latest docker-compose.yml …"
curl -fsSL "$REPO_RAW/docker-compose.yml" -o docker-compose.yml
echo "✅ docker-compose.yml updated"

# ── Ensure ./data directory exists ────────────────────────────────────────────
if [[ ! -d ./data ]]; then
  echo ""
  echo "📂 Creating ./data directory …"
  mkdir -p ./data

  # Migrate data from old Docker volume if the container is running
  CONTAINER_ID=$(docker compose ps -q bot 2>/dev/null || true)
  if [[ -n "$CONTAINER_ID" ]]; then
    echo "📦 Migrating data from Docker volume to ./data …"
    docker cp "$CONTAINER_ID":/data/. ./data/
    echo "✅ Data migrated"
  fi
fi

# ── Pull latest image ────────────────────────────────────────────────────────
echo ""
echo "📥 Pulling latest images…"
docker compose pull

echo ""
echo "🚀 Starting services…"
docker compose up -d --remove-orphans

echo ""
echo "🧹 Cleaning up dangling images…"
docker image prune -f

echo ""
echo "📋 Tailing logs (Ctrl+C to stop)…"
docker compose logs -f bot
