"""
LLM Wiki — Telegram Bot
=======================
Personal knowledge base powered by Ollama (gemma4:e4b).
Implements the Karpathy LLM Wiki pattern via Telegram.

Commands (all optional — natural language works too):
  /start      — welcome
  /help       — command reference
  /query      — ask a question against the wiki
  /search     — BM25 keyword search over wiki pages
  /websearch  — search the web (DuckDuckGo, no API key)
  /fetch      — fetch a URL and ingest it into the wiki
  /lint       — wiki health check
  /status     — wiki stats
  /index      — show wiki index
  /save       — save last query answer as an insight page

Natural language (no command needed):
  "What did I learn about sleep?"     → query
  "Search for cortisol"               → wiki search
  "Find articles about stoicism"      → web search
  "Fetch https://..."                 → URL fetch + ingest
  "I went for a run today..."         → journal ingest
  Any URL in message                  → auto-fetch + ingest
  .md / .txt file                     → ingest
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from telegram import Document, Message, Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from fetcher import FetchResult, fetch_url, web_search
from ollama import OllamaClient
from search import WikiSearch
from wiki import WikiManager, now_slug, slugify

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

load_dotenv()

TELEGRAM_TOKEN: str = os.environ["TELEGRAM_TOKEN"]
ALLOWED_USERS: set[int] = {
    int(uid.strip())
    for uid in os.getenv("TELEGRAM_ALLOWED_USERS", "").split(",")
    if uid.strip()
}
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "gemma4:e4b")
DATA_DIR: str = os.getenv("DATA_DIR", "./data")

# ── Globals (initialized in main) ─────────────────────────────────────────────

ollama: OllamaClient
wiki: WikiManager
wiki_search: WikiSearch

# Per-user state: last query result for /save
_last_query: dict[int, dict] = {}

# ── Access control ────────────────────────────────────────────────────────────


def is_allowed(update: Update) -> bool:
    if not ALLOWED_USERS:
        return True  # no restriction configured
    return update.effective_user.id in ALLOWED_USERS


async def deny(update: Update) -> None:
    await update.message.reply_text("⛔ Access denied.")


# ── Ingest helper ─────────────────────────────────────────────────────────────


async def do_ingest(
    message: Message,
    content: str,
    source_type: str,
    filename: str,
) -> None:
    """Run ingest and reply with a formatted summary."""
    await message.chat.send_action(ChatAction.TYPING)
    status_msg = await message.reply_text("⏳ Ingesting… this may take a minute.")

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, wiki.ingest, content, source_type, filename
        )
        wiki_search.rebuild_index()

        created_list = "\n".join(f"  📄 `{p}`" for p in result.created) or "  (none)"
        updated_list = "\n".join(f"  ✏️ `{p}`" for p in result.updated) or "  (none)"

        reply = (
            f"✅ *Ingested:* {result.title}\n\n"
            f"*Created:*\n{created_list}\n\n"
            f"*Updated:*\n{updated_list}\n\n"
            f"_{result.summary}_"
        )
        await status_msg.edit_text(reply, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logger.exception("Ingest failed")
        await status_msg.edit_text(f"❌ Ingest failed: {e}")


# ── Command handlers ──────────────────────────────────────────────────────────


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_allowed(update):
        return await deny(update)
    await update.message.reply_text(
        "👋 *LLM Wiki Bot*\n\n"
        "Your personal knowledge base, powered by Ollama.\n\n"
        "*Send me:*\n"
        "• A text message → journal entry\n"
        "• A URL → fetched and ingested\n"
        "• A `.md` or `.txt` file → ingested\n\n"
        "*Commands:*\n"
        "/query — ask a question\n"
        "/search — keyword search (wiki)\n"
        "/websearch — search the web\n"
        "/fetch — fetch a URL into the wiki\n"
        "/lint — wiki health check\n"
        "/status — wiki stats\n"
        "/index — show wiki index\n"
        "/help — full reference",
        parse_mode=ParseMode.MARKDOWN,
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_allowed(update):
        return await deny(update)
    await update.message.reply_text(
        "*LLM Wiki — Command Reference*\n\n"
        "*/query <question>*\n"
        "Ask a question against your wiki. The LLM reads relevant pages and synthesizes an answer.\n\n"
        "*/search <terms>*\n"
        "BM25 keyword search over all wiki pages.\n\n"
        "*/websearch <query>*\n"
        "Search the web via DuckDuckGo (no API key). Returns top results with snippets.\n\n"
        "*/fetch <url>*\n"
        "Fetch a URL, convert to markdown, and ingest it into the wiki.\n\n"
        "*/lint*\n"
        "Health-check the wiki: contradictions, orphan pages, missing concepts.\n\n"
        "*/status*\n"
        "Show wiki stats: page counts, last operations.\n\n"
        "*/index*\n"
        "Show the wiki index.\n\n"
        "*/save <slug>*\n"
        "Save your last query answer as a wiki insight page.\n\n"
        "*Ingest (no command needed):*\n"
        "• Send any text → journal entry\n"
        "• Send a URL → fetched and ingested as article\n"
        "• Send a `.md` or `.txt` file → ingested\n",
        parse_mode=ParseMode.MARKDOWN,
    )


async def cmd_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_allowed(update):
        return await deny(update)

    question = " ".join(context.args or []).strip()
    if not question:
        await update.message.reply_text("Usage: /query <your question>")
        return

    await update.message.chat.send_action(ChatAction.TYPING)

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, wiki.query, question
        )

        sources_text = ""
        if result.sources_consulted:
            sources_text = "\n\n📚 *Sources consulted:* " + ", ".join(
                f"`{s}`" for s in result.sources_consulted
            )

        save_hint = ""
        if result.save_as:
            save_hint = f"\n\n💾 Save as insight? `/save {result.save_as}`"

        reply = result.answer[:3800] + sources_text + save_hint

        _last_query[update.effective_user.id] = {
            "answer": result.answer,
            "save_as": result.save_as,
            "question": question,
        }

        await update.message.reply_text(reply, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logger.exception("Query failed")
        await update.message.reply_text(f"❌ Query failed: {e}")


async def cmd_search(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_allowed(update):
        return await deny(update)

    terms = " ".join(context.args or []).strip()
    if not terms:
        await update.message.reply_text("Usage: /search <keywords>")
        return

    results = wiki_search.search(terms, top_k=5)
    if not results:
        await update.message.reply_text("🔍 No results found in wiki.")
        return

    lines = [f"🔍 *Wiki search:* `{terms}`\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. *{r.title}*\n   `{r.path}` (score: {r.score})\n   _{r.snippet}_\n")

    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


async def cmd_websearch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Search the web via DuckDuckGo and return results."""
    if not is_allowed(update):
        return await deny(update)

    query = " ".join(context.args or []).strip()
    if not query:
        await update.message.reply_text("Usage: /websearch <query>")
        return

    status_msg = await update.message.reply_text(f"🌐 Searching the web for: `{query}`…", parse_mode=ParseMode.MARKDOWN)

    try:
        results = await web_search(query, max_results=5)
        if not results:
            await status_msg.edit_text("🔍 No web results found.")
            return

        lines = [f"🌐 *Web search:* `{query}`\n"]
        for i, r in enumerate(results, 1):
            lines.append(
                f"{i}. *{r.title}*\n"
                f"   {r.url}\n"
                f"   _{r.snippet}_\n"
            )
        lines.append("\n💡 Fetch any result: `/fetch <url>`")

        await status_msg.edit_text("\n".join(lines)[:4000], parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logger.exception("Web search failed")
        await status_msg.edit_text(f"❌ Web search failed: {e}")


async def cmd_fetch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Fetch a URL, convert to markdown, and ingest into the wiki."""
    if not is_allowed(update):
        return await deny(update)

    url = " ".join(context.args or []).strip()
    if not url or not url.startswith("http"):
        await update.message.reply_text("Usage: /fetch <url>\nExample: /fetch https://example.com/article")
        return

    status_msg = await update.message.reply_text(f"🌐 Fetching `{url}`…", parse_mode=ParseMode.MARKDOWN)

    try:
        fetch_result: FetchResult = await fetch_url(url)
        await status_msg.edit_text(
            f"✅ Fetched: *{fetch_result.title}*\n"
            f"📝 {fetch_result.word_count} words\n\n"
            f"⏳ Ingesting into wiki…",
            parse_mode=ParseMode.MARKDOWN,
        )

        # Save raw and ingest
        wiki.save_raw(fetch_result.content, "articles", fetch_result.filename)
        await do_ingest(update.message, fetch_result.content, "article", fetch_result.filename)

    except Exception as e:
        logger.exception("Fetch failed")
        await status_msg.edit_text(f"❌ Fetch failed: {e}")


async def cmd_lint(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_allowed(update):
        return await deny(update)

    status_msg = await update.message.reply_text("🔍 Running wiki health check… this may take a moment.")

    try:
        result = await asyncio.get_event_loop().run_in_executor(None, wiki.lint)

        lines = ["🔍 *Wiki Health Check*\n"]

        if result.contradictions:
            lines.append("⚠️ *Contradictions:*")
            for c in result.contradictions:
                pages = ", ".join(f"`{p}`" for p in c.get("pages", []))
                lines.append(f"  • {pages}: {c.get('description', '')}")
            lines.append("")

        if result.orphans:
            lines.append("🔗 *Orphan pages (no inbound links):*")
            for o in result.orphans:
                lines.append(f"  • `{o}`")
            lines.append("")

        if result.missing_pages:
            lines.append("📭 *Concepts mentioned but no page:*")
            for m in result.missing_pages:
                concept = m.get("concept", "")
                mentioned = ", ".join(m.get("mentioned_in", []))
                lines.append(f"  • *{concept}* (in {mentioned})")
            lines.append("")

        if result.stale:
            lines.append("🕰️ *Stale content:*")
            for s in result.stale:
                lines.append(f"  • `{s.get('page', '')}`: {s.get('reason', '')}")
            lines.append("")

        if result.suggestions:
            lines.append("💡 *Suggestions:*")
            for s in result.suggestions:
                lines.append(f"  • {s}")
            lines.append("")

        if not any([result.contradictions, result.orphans, result.missing_pages, result.stale]):
            lines.append("✅ Wiki looks healthy!")

        await status_msg.edit_text("\n".join(lines)[:4000], parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logger.exception("Lint failed")
        await status_msg.edit_text(f"❌ Lint failed: {e}")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_allowed(update):
        return await deny(update)

    result = wiki.get_status(OLLAMA_MODEL)
    log_lines = "\n".join(f"  `{e}`" for e in result.last_log_entries) or "  (empty)"

    reply = (
        f"📊 *Wiki Status*\n\n"
        f"*Model:* `{result.model}`\n"
        f"*Total pages:* {result.total_pages}\n"
        f"  • Sources: {result.sources}\n"
        f"  • People: {result.people}\n"
        f"  • Concepts: {result.concepts}\n"
        f"  • Insights: {result.insights}\n\n"
        f"*Recent operations:*\n{log_lines}"
    )
    await update.message.reply_text(reply, parse_mode=ParseMode.MARKDOWN)


async def cmd_index(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_allowed(update):
        return await deny(update)

    index_content = wiki.read_index()
    if not index_content.strip():
        await update.message.reply_text("📋 Wiki index is empty. Start by sending me some content to ingest!")
        return

    chunks = [index_content[i : i + 4000] for i in range(0, len(index_content), 4000)]
    for chunk in chunks:
        await update.message.reply_text(f"```\n{chunk}\n```", parse_mode=ParseMode.MARKDOWN)


async def cmd_save(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_allowed(update):
        return await deny(update)

    user_id = update.effective_user.id
    last = _last_query.get(user_id)
    if not last:
        await update.message.reply_text("No recent query to save. Run /query first.")
        return

    slug_arg = " ".join(context.args or []).strip()
    slug = slugify(slug_arg) if slug_arg else last.get("save_as") or slugify(last["question"][:40])

    today = datetime.now().strftime("%Y-%m-%d")
    content = (
        f"---\n"
        f"title: {last['question']}\n"
        f"type: insight\n"
        f"date: {today}\n"
        f"---\n\n"
        f"## Question\n\n{last['question']}\n\n"
        f"## Answer\n\n{last['answer']}\n"
    )

    try:
        path = wiki.save_insight(slug, content)
        wiki_search.rebuild_index()
        await update.message.reply_text(f"💾 Saved as `{path}`", parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        await update.message.reply_text(f"❌ Save failed: {e}")


# ── Message handlers ──────────────────────────────────────────────────────────

URL_RE = re.compile(r"https?://\S+")

# ── Intent classification ─────────────────────────────────────────────────────

# Fast regex patterns — checked before LLM classification
_QUERY_RE = re.compile(
    r"^(what|how|why|when|who|where|which|tell me|explain|summarize|show me|"
    r"give me|list|find in wiki|search wiki|look up|do i have|have i|"
    r"what did i|what do i know|what have i)",
    re.IGNORECASE,
)
_WEBSEARCH_RE = re.compile(
    r"^(search (the web|online|internet|google|duckduckgo)|"
    r"find (articles?|pages?|info|information|news) (about|on)|"
    r"look up online|google|web search)",
    re.IGNORECASE,
)
_FETCH_RE = re.compile(
    r"^(fetch|get|download|read|ingest|import|add)\s+https?://",
    re.IGNORECASE,
)


def _classify_intent_fast(text: str) -> str | None:
    """
    Fast regex-based intent classification.
    Returns: 'query' | 'websearch' | 'fetch' | 'journal' | None (ambiguous)
    """
    stripped = text.strip()

    # URL-only message → fetch
    if URL_RE.fullmatch(stripped.split()[0] if stripped.split() else ""):
        return "fetch"

    # Contains a URL → fetch
    if URL_RE.search(stripped):
        return "fetch"

    # Explicit fetch command words + URL
    if _FETCH_RE.match(stripped):
        return "fetch"

    # Web search intent
    if _WEBSEARCH_RE.match(stripped):
        return "websearch"

    # Question patterns → query wiki
    if _QUERY_RE.match(stripped) and "?" in stripped:
        return "query"

    # Short messages (< 20 words) that are questions → query
    words = stripped.split()
    if len(words) < 20 and stripped.endswith("?"):
        return "query"

    return None  # ambiguous — use LLM classifier


def _classify_intent_llm(text: str) -> str:
    """
    Use the LLM to classify intent for ambiguous messages.
    Returns: 'query' | 'websearch' | 'journal'
    """
    system = (
        "You are an intent classifier for a personal knowledge base bot. "
        "Classify the user's message into exactly one of these intents:\n"
        "- query: the user is asking a question about their existing wiki/notes\n"
        "- websearch: the user wants to search the internet for new information\n"
        "- journal: the user is logging something (a thought, event, note, observation)\n\n"
        "Reply with ONLY one word: query, websearch, or journal. No explanation."
    )
    messages = [{"role": "user", "content": text}]
    response = ollama.chat(system, messages).strip().lower()
    if response in ("query", "websearch", "journal"):
        return response
    # Default to journal if LLM gives unexpected output
    return "journal"


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle plain text messages with smart intent detection.
    No command required — the bot figures out what you mean:

      URL in message          → fetch + ingest
      "What did I learn..."   → query wiki
      "Search online for..."  → web search
      "I went for a run..."   → journal ingest
    """
    if not is_allowed(update):
        return await deny(update)

    text = (update.message.text or "").strip()
    if not text:
        return

    # ── 1. URL detection (highest priority) ──────────────────────────────────
    url_match = URL_RE.search(text)
    if url_match:
        url = url_match.group(0)
        status_msg = await update.message.reply_text(
            f"🌐 Fetching `{url}`…", parse_mode=ParseMode.MARKDOWN
        )
        try:
            fetch_result = await fetch_url(url)
            await status_msg.edit_text(
                f"✅ Fetched: *{fetch_result.title}* ({fetch_result.word_count} words)\n⏳ Ingesting…",
                parse_mode=ParseMode.MARKDOWN,
            )
            wiki.save_raw(fetch_result.content, "articles", fetch_result.filename)
            await do_ingest(update.message, fetch_result.content, "article", fetch_result.filename)
        except Exception as e:
            await status_msg.edit_text(f"❌ Could not fetch URL: {e}")
        return

    # ── 2. Fast regex classification ─────────────────────────────────────────
    intent = _classify_intent_fast(text)

    # ── 3. LLM classification for ambiguous messages ──────────────────────────
    if intent is None:
        thinking = await update.message.reply_text("🤔 …")
        intent = await asyncio.get_event_loop().run_in_executor(
            None, _classify_intent_llm, text
        )
        await thinking.delete()

    # ── 4. Route to the right handler ────────────────────────────────────────
    if intent == "query":
        # Treat as a wiki query
        await update.message.chat.send_action(ChatAction.TYPING)
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, wiki.query, text
            )
            sources_text = ""
            if result.sources_consulted:
                sources_text = "\n\n📚 " + ", ".join(f"`{s}`" for s in result.sources_consulted)
            save_hint = f"\n\n💾 `/save {result.save_as}`" if result.save_as else ""
            _last_query[update.effective_user.id] = {
                "answer": result.answer,
                "save_as": result.save_as,
                "question": text,
            }
            await update.message.reply_text(
                result.answer[:3800] + sources_text + save_hint,
                parse_mode=ParseMode.MARKDOWN,
            )
        except Exception as e:
            logger.exception("Query failed")
            await update.message.reply_text(f"❌ Query failed: {e}")

    elif intent == "websearch":
        # Extract the search query (strip leading "search for", "find", etc.)
        query = re.sub(
            r"^(search (the web|online|internet|for)|find (articles?|info|information|news) (about|on)|"
            r"look up online|google|web search)\s*",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip() or text

        status_msg = await update.message.reply_text(
            f"🌐 Searching the web for: `{query}`…", parse_mode=ParseMode.MARKDOWN
        )
        try:
            results = await web_search(query, max_results=5)
            if not results:
                await status_msg.edit_text("🔍 No web results found.")
                return
            lines = [f"🌐 *Web search:* `{query}`\n"]
            for i, r in enumerate(results, 1):
                lines.append(f"{i}. *{r.title}*\n   {r.url}\n   _{r.snippet}_\n")
            lines.append("\n💡 Fetch any result: `/fetch <url>` or just paste the URL")
            await status_msg.edit_text("\n".join(lines)[:4000], parse_mode=ParseMode.MARKDOWN)
        except Exception as e:
            logger.exception("Web search failed")
            await status_msg.edit_text(f"❌ Web search failed: {e}")

    else:
        # journal — ingest as a journal entry
        ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
        filename = f"journal-{ts}.md"
        wiki.save_raw(text, "journals", filename)
        await do_ingest(update.message, text, "journal", filename)


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle document uploads: .md and .txt files."""
    if not is_allowed(update):
        return await deny(update)

    doc: Document = update.message.document
    fname = doc.file_name or "upload.txt"
    ext = Path(fname).suffix.lower()

    if ext not in (".md", ".txt"):
        await update.message.reply_text(
            f"⚠️ Unsupported file type `{ext}`. Send `.md` or `.txt` files.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    await update.message.reply_text(f"📥 Downloading `{fname}`…", parse_mode=ParseMode.MARKDOWN)

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tg_file = await doc.get_file()
        await tg_file.download_to_drive(tmp.name)
        content = Path(tmp.name).read_text(encoding="utf-8", errors="replace")

    wiki.save_raw(content, "articles", fname)
    await do_ingest(update.message, content, "article", fname)


# ── Startup ───────────────────────────────────────────────────────────────────


async def post_init(application: Application) -> None:
    """Called after the bot is initialized — pull model if needed."""
    logger.info("Waiting for Ollama…")
    ollama.wait_until_ready()

    if not ollama.model_is_pulled():
        logger.info("Model %s not found locally — pulling…", OLLAMA_MODEL)

        def _pull_progress(status: str) -> None:
            logger.info("Pull: %s", status)

        await asyncio.get_event_loop().run_in_executor(
            None, lambda: ollama.pull_model(_pull_progress)
        )
        logger.info("Model pull complete.")
    else:
        logger.info("Model %s already available.", OLLAMA_MODEL)

    wiki_search.rebuild_index()
    logger.info("Bot ready.")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    global ollama, wiki, wiki_search

    ollama = OllamaClient(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
    wiki = WikiManager(data_dir=DATA_DIR, ollama=ollama)
    wiki_search = WikiSearch(wiki_dir=str(Path(DATA_DIR) / "wiki"))

    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .post_init(post_init)
        .build()
    )

    # Commands
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("query", cmd_query))
    app.add_handler(CommandHandler("search", cmd_search))
    app.add_handler(CommandHandler("websearch", cmd_websearch))
    app.add_handler(CommandHandler("fetch", cmd_fetch))
    app.add_handler(CommandHandler("lint", cmd_lint))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("index", cmd_index))
    app.add_handler(CommandHandler("save", cmd_save))

    # Message handlers
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    logger.info("Starting bot (polling)…")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
