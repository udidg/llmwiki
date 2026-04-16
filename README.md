# LLM Wiki — Personal Knowledge Base via Telegram + Ollama

> Implements the [Karpathy LLM Wiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f):
> a persistent, compounding knowledge base where the LLM incrementally builds and maintains
> structured markdown pages — not a RAG chatbot that re-derives answers from scratch every time.

**Stack:** Telegram bot · Ollama (`gemma4:e4b`) · Docker Compose · BM25 search · Markdown wiki

---

## How it works

```
You (Telegram)
    │  send text / file / URL / command
    ▼
bot/main.py          ← classifies intent
    │
    ▼
bot/wiki.py          ← assembles context:
                        AGENTS.md + wiki/index.md + relevant pages
    │
    ▼
gemma4:e4b (Ollama)  ← reads context, produces answer + file writes
    │
    ▼
bot/wiki.py          ← executes file writes (wiki pages, index, log)
    │
    ▼
You (Telegram)       ← receives streamed answer
```

The wiki is the LLM's **persistent memory**. Every journal entry, article, and podcast note
you send gets compiled into structured markdown pages that grow richer over time.

---

## Quick start

### 1. Prerequisites

- [Docker](https://docs.docker.com/get-docker/) + Docker Compose
- A Telegram bot token (from [@BotFather](https://t.me/BotFather))
- Your Telegram user ID (from [@userinfobot](https://t.me/userinfobot))

### 2. Clone and configure

```bash
git clone <this-repo>
cd llm-wiki
cp .env.example .env
```

Edit `.env`:

```env
TELEGRAM_TOKEN=your_bot_token_here
TELEGRAM_ALLOWED_USERS=123456789
OLLAMA_MODEL=gemma4:e4b
```

### 3. Start

```bash
docker compose up -d
```

On first run, the bot will automatically pull `gemma4:e4b` from Ollama (~3–5 GB).
This takes a few minutes. Check progress with:

```bash
docker compose logs -f bot
```

### 4. Use it

Open Telegram, find your bot, and send `/start`.

---

## Interacting with the bot

### Send content (no command needed)

| What you send | What happens |
|---|---|
| Any text message | Ingested as a journal entry |
| A URL | Fetched, converted to markdown, ingested as article |
| A `.md` or `.txt` file | Ingested as article |

After ingest, the bot replies with a summary of pages created/updated.

### Natural language — no commands required

Just send a message. The bot classifies your intent automatically:

| What you send | What happens |
|---|---|
| `"What did I learn about sleep?"` | Detected as **query** → LLM reads wiki, answers |
| `"Search online for stoicism articles"` | Detected as **web search** → DuckDuckGo results |
| `"I went for a 5km run today, felt great"` | Detected as **journal** → ingested into wiki |
| Any URL | Detected as **fetch** → page fetched + ingested |

### Commands (optional shortcuts)

| Command | Description |
|---|---|
| `/query What have I learned about sleep?` | Ask a question — LLM reads wiki, synthesizes answer |
| `/search cortisol` | BM25 keyword search over all wiki pages |
| `/websearch stoicism articles` | Search the web via DuckDuckGo (no API key) |
| `/fetch https://example.com/article` | Fetch a URL and ingest it into the wiki |
| `/lint` | Health-check: contradictions, orphans, missing pages |
| `/status` | Wiki stats: page counts, recent operations |
| `/index` | Show the full wiki index |
| `/save <slug>` | Save your last query answer as a wiki insight page |
| `/help` | Full command reference |

---

## Project structure

```
llm-wiki/
├── docker-compose.yml       ← two services: ollama + bot
├── .env.example             ← copy to .env and fill in
├── README.md
├── .github/
│   └── workflows/
│       └── docker-publish.yml  ← build + push to GHCR on push/tag
├── bot/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py              ← Telegram bot handlers + intent classifier
│   ├── wiki.py              ← WikiManager: ingest / query / lint
│   ├── ollama.py            ← Ollama HTTP API wrapper
│   ├── fetcher.py           ← URL fetch + DuckDuckGo web search
│   └── search.py            ← BM25 search over wiki pages
└── data/                    ← mounted as Docker volume
    ├── AGENTS.md            ← LLM schema: page formats, workflows
    ├── raw/                 ← your immutable source documents
    │   ├── articles/
    │   ├── journals/
    │   └── podcasts/
    └── wiki/                ← LLM-maintained knowledge base
        ├── index.md         ← catalog of all pages
        ├── log.md           ← append-only operation log
        ├── overview.md      ← high-level synthesis
        ├── sources/         ← one page per ingested source
        ├── people/          ← one page per person
        ├── concepts/        ← one page per concept/topic
        └── insights/        ← saved query answers and analyses
```

---

## CI/CD — GitHub Actions + GHCR

The included workflow (`.github/workflows/docker-publish.yml`) automatically builds and pushes the bot image to **GitHub Container Registry** on every push to `main` or version tag.

### Image location
```
ghcr.io/<your-github-username>/<repo-name>/llm-wiki-bot:latest
```

### Setup
1. Push this repo to GitHub — Actions runs automatically (uses `GITHUB_TOKEN`, no extra secrets needed)
2. Make the package public: GitHub repo → **Packages** → your image → **Change visibility → Public**

### Use the pre-built image (skip local build)

In `docker-compose.yml`, swap `build` for `image`:
```yaml
bot:
  # build: ./bot          ← comment this out
  image: ghcr.io/your-username/your-repo/llm-wiki-bot:latest
```
Then:
```bash
docker compose pull bot && docker compose up -d
```

### Tag a release
```bash
git tag v1.0.0 && git push origin v1.0.0
```
Produces images tagged `v1.0.0`, `1.0`, and `latest`.

---

## Internet access

### Fetch a URL
Send any URL directly or use `/fetch`:
```
https://www.hubermanlab.com/episode/sleep-toolkit
/fetch https://www.hubermanlab.com/episode/sleep-toolkit
```
The bot fetches the page, converts HTML → markdown, and ingests it into the wiki automatically.

### Web search (DuckDuckGo — no API key)
```
Search online for intermittent fasting research
/websearch intermittent fasting benefits
```
Returns top 5 results. Fetch any result: paste the URL or `/fetch <url>`.

---

## The wiki (reading it)

The wiki lives in `data/wiki/` as plain markdown files. You can:

- **Browse it in [Obsidian](https://obsidian.md/)** — open `data/` as a vault.
  Graph view shows connections between pages. `[[wiki-links]]` are clickable.
- **Read it in any text editor** — it's just markdown.
- **Version it with git** — the wiki is a git repo of markdown files.

### Recommended Obsidian plugins

| Plugin | Why |
|---|---|
| **Obsidian Web Clipper** (browser extension) | Clip articles to `data/raw/articles/` as markdown |
| **Dataview** | Query page frontmatter (e.g. list all pages tagged `health`) |
| **Graph View** | Visualize connections — which pages are hubs, which are orphans |

---

## AGENTS.md — the schema file

`data/AGENTS.md` is the most important file. It is loaded as the system prompt on every
Ollama call and tells the LLM exactly how to behave as a wiki maintainer:

- Page formats (source summaries, person pages, concept pages, insight pages)
- Ingest workflow (what to create, what to update, what JSON to return)
- Query workflow (how to navigate the index, how to cite sources)
- Lint workflow (what to check for, what JSON to return)

You can edit `AGENTS.md` to customize the wiki for your domain. The LLM will adapt.

---

## GPU support

If you have an NVIDIA GPU, uncomment the `deploy` section in `docker-compose.yml`:

```yaml
ollama:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

---

## Changing the model

Edit `.env`:

```env
OLLAMA_MODEL=llama3.1:8b
```

Then restart:

```bash
docker compose restart bot
```

The bot will pull the new model automatically on startup.

---

## Useful commands

```bash
# Start everything
docker compose up -d

# View logs
docker compose logs -f bot
docker compose logs -f ollama

# Stop
docker compose down

# Stop and remove volumes (WARNING: deletes all wiki data)
docker compose down -v

# Rebuild bot image after code changes
docker compose build bot && docker compose up -d bot
```

---

## Data persistence

All data is stored in Docker named volumes:

| Volume | Contents |
|---|---|
| `ollama_models` | Downloaded Ollama models (~3–8 GB) |
| `wiki_data` | Your wiki and raw sources |

To back up your wiki:

```bash
docker run --rm -v llm-wiki_wiki_data:/data -v $(pwd):/backup \
  alpine tar czf /backup/wiki-backup.tar.gz /data
```
