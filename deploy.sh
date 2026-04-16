#!/usr/bin/env bash
# deploy.sh — Pull latest image, restart, clean up, and tail logs
set -euo pipefail

cd "$(dirname "$0")"

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
