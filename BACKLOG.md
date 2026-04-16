# Backlog

## UX / Bot behaviour

- [x] **Typing indicator while thinking** — send `chat action: typing` to Telegram while the bot is waiting for the LLM response, so the user knows it's working and not frozen.
- [x] **Error reporting to user** — when the bot encounters an error (LLM timeout, Gemini API unreachable, etc.) it should send a human-readable error message back to the Telegram chat instead of silently failing.
- [x] **Smart link categorization** — when a user sends a link, fetch its content, generate a description, and add it to a relevant action list (To Buy / To Review / To Read) based on the content. If the intent is ambiguous, ask the user which list to add it to via inline keyboard buttons.
