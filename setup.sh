#!/usr/bin/env bash
# LLM Wiki — one-shot setup script
# Usage: bash <(curl -fsSL https://raw.githubusercontent.com/udidg/llmwiki/main/setup.sh)
set -euo pipefail

REPO_RAW="https://raw.githubusercontent.com/udidg/llmwiki/main"
WORKDIR="${LLMWIKI_DIR:-$HOME/llmwiki}"

echo "📁  Creating working directory: $WORKDIR"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

# ── Fetch docker-compose.yml ──────────────────────────────────────────────────
echo "⬇️   Fetching docker-compose.yml …"
curl -fsSL "$REPO_RAW/docker-compose.yml" -o docker-compose.yml
echo "✅  docker-compose.yml saved"

# ── Create .env if it doesn't exist ──────────────────────────────────────────
if [[ -f .env ]]; then
  echo "ℹ️   .env already exists — skipping (delete it to reconfigure)"
else
  echo ""
  echo "🔑  Bot configuration"
  echo "────────────────────────────────────────"

  read -rp "  Telegram bot token : " TELEGRAM_TOKEN
  read -rp "  Your Telegram user ID : " TELEGRAM_ALLOWED_USERS
  read -rp "  Gemini API key (https://aistudio.google.com/apikey) : " GEMINI_API_KEY
  read -rp "  Gemini model [gemini-3.1-flash-lite-preview] : " GEMINI_MODEL
  GEMINI_MODEL="${GEMINI_MODEL:-gemini-3.1-flash-lite-preview}"

  cat > .env <<EOF
TELEGRAM_TOKEN=${TELEGRAM_TOKEN}
TELEGRAM_ALLOWED_USERS=${TELEGRAM_ALLOWED_USERS}
GEMINI_API_KEY=${GEMINI_API_KEY}
GEMINI_MODEL=${GEMINI_MODEL}
EOF
  echo "✅  .env saved"
fi

# ── Create data directory ─────────────────────────────────────────────────────
echo ""
echo "📂  Creating data directory …"
mkdir -p "$WORKDIR/data"

# ── Start the stack ───────────────────────────────────────────────────────────
echo ""
echo "🐳  Pulling images and starting stack …"
docker compose pull
docker compose up -d

echo ""
echo "🎉  Done! Open Telegram, find your bot, and send /start"
echo "    Wiki data is stored in: $WORKDIR/data/"
echo "    Logs: docker compose -C $WORKDIR logs -f bot"
