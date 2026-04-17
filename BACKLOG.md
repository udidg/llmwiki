# Backlog

## UX / Bot behaviour

- [x] **Typing indicator while thinking** — send `chat action: typing` to Telegram while the bot is waiting for the LLM response, so the user knows it's working and not frozen.
- [x] **Error reporting to user** — when the bot encounters an error (LLM timeout, Gemini API unreachable, etc.) it should send a human-readable error message back to the Telegram chat instead of silently failing.
- [x] **Smart link categorization** — when a user sends a link, fetch its content, generate a description, and add it to a relevant action list (To Buy / To Review / To Read) based on the content. If the intent is ambiguous, ask the user which list to add it to via inline keyboard buttons.
- [x] **Fix web search** — replaced broken DuckDuckGo HTML scraping with the `duckduckgo-search` Python library for reliable results.
- [x] **Instagram post support** — when a user sends an Instagram post/reel URL, extract metadata via yt-dlp (caption, author, thumbnail, hashtags), generate tags and description via LLM, and assign to a reading list (To Buy / To Review / To Read).
- [] Add a watch tower for changes - that pull changes every 30 min. check for docker-compose update pull the docker-compose and apply it.
- [] Make the bot messages a bit more clearly formatted, a bit less verbose but keep important messages (like it is thinking, errors etc) 
- [] Reactions, good and bad responses that improve the LLMshould be added to a separate file as it is edited by the bot itself and not committed to git. AGENTS.md is part of the code base and should reference the file. 