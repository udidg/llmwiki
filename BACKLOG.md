# Backlog

## UX / Bot behaviour

- [ ] **Typing indicator while thinking** — send `chat action: typing` to Telegram while the bot is waiting for the LLM response, so the user knows it's working and not frozen.
- [ ] **Error reporting to user** — when the bot encounters an error (LLM timeout, Gemini API unreachable, etc.) it should send a human-readable error message back to the Telegram chat instead of silently failing.
