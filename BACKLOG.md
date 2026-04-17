# Backlog

## UX / Bot behaviour

- [x] **Typing indicator while thinking** — send `chat action: typing` to Telegram while the bot is waiting for the LLM response, so the user knows it's working and not frozen.
- [x] **Error reporting to user** — when the bot encounters an error (LLM timeout, Gemini API unreachable, etc.) it should send a human-readable error message back to the Telegram chat instead of silently failing.
- [x] **Smart link categorization** — when a user sends a link, fetch its content, generate a description, and add it to a relevant action list (To Buy / To Review / To Read) based on the content. If the intent is ambiguous, ask the user which list to add it to via inline keyboard buttons.
- [x] **Fix web search** — replaced broken DuckDuckGo HTML scraping with the `duckduckgo-search` Python library for reliable results.
- [x] **Instagram post support** — when a user sends an Instagram post/reel URL, extract metadata via yt-dlp (caption, author, thumbnail, hashtags), generate tags and description via LLM, and assign to a reading list (To Buy / To Review / To Read).
- [x] **Watchtower auto-updater** — added Watchtower container to `docker-compose.yml` that polls GHCR every 30 minutes for new bot images and auto-restarts. Uses label-based filtering to only watch the bot container.
- [x] **Cleaner bot messages** — simplified all status messages: removed multi-step progress updates, removed intent classification display messages, kept typing indicators and error messages. Messages are now concise single-line status → final result.
- [x] **Feedback file separation** — moved `feedback.md` from `wiki/` to the data root (`data/feedback.md`), outside the git-tracked wiki directory. AGENTS.md now references the file location. Includes automatic migration of existing feedback data.