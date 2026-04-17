"""
LLM Wiki — Telegram Bot
=======================
Personal knowledge base powered by Google Gemini API.
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
  Instagram URL                       → extract post, tag, categorize + ingest
  Any URL in message                  → auto-fetch + ingest
  .md / .txt file                     → ingest
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
import traceback
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator

from dotenv import load_dotenv
from telegram import Document, InlineKeyboardButton, InlineKeyboardMarkup, Message, Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from fetcher import (
    FetchResult,
    InstagramPost,
    extract_instagram_post,
    fetch_url,
    is_instagram_url,
    web_search,
)
from gemini import GeminiClient
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
GEMINI_API_KEY: str = os.environ["GEMINI_API_KEY"]
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
DATA_DIR: str = os.getenv("DATA_DIR", "./data")

# ── Globals (initialized in main) ─────────────────────────────────────────────

llm: GeminiClient
wiki: WikiManager
wiki_search: WikiSearch

# Per-user state: last query result for /save
_last_query: dict[int, dict] = {}

# Per-user state: map message_id → query data for emoji feedback tracking
_query_answer_messages: dict[int, dict] = {}

# Positive feedback emoji set
_POSITIVE_EMOJI = {"👍", "👌", "❤️", "🔥", "💯", "🙏"}

# ── Access control ────────────────────────────────────────────────────────────


def is_allowed(update: Update) -> bool:
    if not ALLOWED_USERS:
        return True  # no restriction configured
    return update.effective_user.id in ALLOWED_USERS


async def deny(update: Update) -> None:
    await update.message.reply_text("⛔ Access denied.")


# ── Typing indicator helper ───────────────────────────────────────────────────


@asynccontextmanager
async def typing_indicator(chat_id: int, bot) -> AsyncIterator[None]:
    """
    Context manager that sends ChatAction.TYPING every 4 seconds
    while the wrapped code is running. Telegram typing indicators
    expire after ~5 seconds, so we re-send periodically.
    """
    stop = asyncio.Event()

    async def _keep_typing() -> None:
        while not stop.is_set():
            try:
                await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            except Exception:
                pass  # best-effort — don't crash if typing indicator fails
            try:
                await asyncio.wait_for(stop.wait(), timeout=4.0)
            except asyncio.TimeoutError:
                pass

    task = asyncio.create_task(_keep_typing())
    try:
        yield
    finally:
        stop.set()
        await task


# ── Error formatting helper ──────────────────────────────────────────────────


def _format_user_error(error: Exception) -> str:
    """
    Convert an exception into a human-readable error message for the user.
    Categorizes errors by type and provides actionable information.
    """
    error_str = str(error)
    error_type = type(error).__name__

    # Timeout errors
    if any(kw in error_type.lower() for kw in ("timeout", "timedout")):
        return (
            "⏱️ *Request timed out*\n\n"
            "The operation took too long to complete. This usually means:\n"
            "• The LLM is overloaded — try again in a minute\n"
            "• The content was too large to process\n\n"
            f"_Technical: {error_type}_"
        )
    if "timeout" in error_str.lower():
        return (
            "⏱️ *Request timed out*\n\n"
            "The server didn't respond in time. Please try again.\n\n"
            f"_Technical: {error_type}: {error_str[:120]}_"
        )

    # Connection / network errors
    if any(kw in error_type.lower() for kw in ("connection", "connect", "network", "dns")):
        return (
            "🌐 *Connection error*\n\n"
            "Could not reach the external service. This could mean:\n"
            "• The API service is temporarily down\n"
            "• Network connectivity issues\n\n"
            "Please try again in a few moments.\n\n"
            f"_Technical: {error_type}_"
        )

    # HTTP errors
    if "status" in error_str.lower() and any(code in error_str for code in ("401", "403")):
        return (
            "🔑 *Authentication error*\n\n"
            "The API key appears to be invalid or expired. "
            "Please check the bot configuration.\n\n"
            f"_Technical: {error_str[:150]}_"
        )
    if "429" in error_str or "rate" in error_str.lower():
        return (
            "🚦 *Rate limit exceeded*\n\n"
            "Too many requests to the API. Please wait a minute and try again.\n\n"
            f"_Technical: {error_str[:150]}_"
        )
    if "500" in error_str or "502" in error_str or "503" in error_str:
        return (
            "🔧 *Service temporarily unavailable*\n\n"
            "The external service is experiencing issues. Please try again later.\n\n"
            f"_Technical: {error_str[:150]}_"
        )

    # Gemini-specific errors
    if "gemini" in error_str.lower() or "google" in error_str.lower():
        return (
            "🤖 *Gemini API error*\n\n"
            "The AI model encountered an issue processing your request.\n"
            "Please try again. If the problem persists, the content may be "
            "too large or contain unsupported formatting.\n\n"
            f"_Technical: {error_str[:150]}_"
        )

    # JSON parsing errors (LLM returned bad output)
    if "json" in error_type.lower() or "json" in error_str.lower():
        return (
            "📋 *Processing error*\n\n"
            "The AI model returned an unexpected response format. "
            "Please try again — this is usually a one-off issue.\n\n"
            f"_Technical: {error_type}_"
        )

    # Generic fallback
    return (
        f"❌ *Something went wrong*\n\n"
        f"An unexpected error occurred. Please try again.\n\n"
        f"_Technical: {error_type}: {error_str[:200]}_"
    )


# ── Markdown safety helpers ───────────────────────────────────────────────────


def _escape_code_block(text: str) -> str:
    """Escape backticks in text so it can be safely wrapped in a Markdown code block.

    Telegram's Markdown v1 parser breaks when the content inside a ```
    code fence contains backticks.  We replace them with the visually
    similar Unicode character ʻ (modifier letter turned comma / okina).
    """
    return text.replace("`", "ʻ")


# ── Verbose status helper ─────────────────────────────────────────────────────


async def _update_status(msg: Message, text: str) -> Message:
    """Edit an existing status message, appending a new line of progress."""
    try:
        await msg.edit_text(text, parse_mode=ParseMode.MARKDOWN)
    except Exception:
        # If edit fails (e.g. message unchanged), just continue
        pass
    return msg


# ── Ingest helper ─────────────────────────────────────────────────────────────


async def do_ingest(
    message: Message,
    content: str,
    source_type: str,
    filename: str,
) -> None:
    """Run ingest and reply with a formatted summary showing each step."""
    await message.chat.send_action(ChatAction.TYPING)
    word_count = len(content.split())

    # Step 1: Starting
    status_msg = await message.reply_text(
        f"⏳ *Ingesting* `{filename}`\n"
        f"📊 Source: {source_type} • {word_count} words\n\n"
        f"🔄 Step 1/4: Assembling context (index + log)…",
        parse_mode=ParseMode.MARKDOWN,
    )

    try:
        # Step 2: Sending to LLM
        await _update_status(
            status_msg,
            f"⏳ *Ingesting* `{filename}`\n"
            f"📊 Source: {source_type} • {word_count} words\n\n"
            f"✅ Step 1/4: Context assembled\n"
            f"🔄 Step 2/4: Sending to Gemini (`{GEMINI_MODEL}`)…\n"
            f"_LLM is reading the source and deciding what wiki pages to create/update_",
        )

        async with typing_indicator(message.chat_id, message.get_bot()):
            result = await asyncio.get_event_loop().run_in_executor(
                None, wiki.ingest, content, source_type, filename
            )

        # Step 3: File writes done
        created_list = "\n".join(f"  📄 `{p}`" for p in result.created) or "  (none)"
        updated_list = "\n".join(f"  ✏️ `{p}`" for p in result.updated) or "  (none)"

        await _update_status(
            status_msg,
            f"⏳ *Ingesting* `{filename}`\n"
            f"📊 Source: {source_type} • {word_count} words\n\n"
            f"✅ Step 1/4: Context assembled\n"
            f"✅ Step 2/4: Gemini processed\n"
            f"✅ Step 3/4: Files written\n"
            f"🔄 Step 4/4: Rebuilding search index…",
        )

        wiki_search.rebuild_index()

        # Step 4: Done — build inline keyboard for viewing created/updated files
        all_paths = result.created + result.updated
        buttons = []
        for p in all_paths:
            # Skip index.md and log.md — they're bookkeeping, not interesting
            if p.endswith(("index.md", "log.md")):
                continue
            label = p.replace("wiki/", "", 1)  # shorter label
            buttons.append([InlineKeyboardButton(f"📖 {label}", callback_data=f"view:{p}")])

        reply_markup = InlineKeyboardMarkup(buttons) if buttons else None

        reply = (
            f"✅ *Ingested:* {result.title}\n\n"
            f"*Created:*\n{created_list}\n\n"
            f"*Updated:*\n{updated_list}\n\n"
            f"_{result.summary}_"
        )
        await status_msg.edit_text(reply, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)
    except Exception as e:
        logger.exception("Ingest failed")
        await status_msg.edit_text(_format_user_error(e), parse_mode=ParseMode.MARKDOWN)


# ── Callback query handler (inline keyboard buttons) ─────────────────────────


async def handle_view_page(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline keyboard button presses to view wiki page content."""
    query = update.callback_query
    await query.answer()  # acknowledge the button press

    data = query.data or ""
    if not data.startswith("view:"):
        return

    rel_path = data[len("view:"):]

    # Security: only allow reading wiki/ files
    if not rel_path.startswith("wiki/"):
        await query.message.reply_text("⛔ Cannot read files outside wiki/.")
        return

    content = wiki.read_page(rel_path)
    if not content:
        await query.message.reply_text(f"📭 File not found: `{rel_path}`", parse_mode=ParseMode.MARKDOWN)
        return

    # Send the file content, chunked if needed (Telegram 4096 char limit)
    header = f"📖 *{rel_path}*\n\n"
    max_chunk = 4000 - len(header)

    safe_content = _escape_code_block(content)
    if len(safe_content) <= max_chunk:
        await query.message.reply_text(
            f"{header}```\n{safe_content}\n```",
            parse_mode=ParseMode.MARKDOWN,
        )
    else:
        # Send in chunks
        chunks = [safe_content[i : i + max_chunk] for i in range(0, len(safe_content), max_chunk)]
        for i, chunk in enumerate(chunks, 1):
            chunk_header = f"📖 *{rel_path}* (part {i}/{len(chunks)})\n\n"
            await query.message.reply_text(
                f"{chunk_header}```\n{chunk}\n```",
                parse_mode=ParseMode.MARKDOWN,
            )


# ── Command handlers ──────────────────────────────────────────────────────────


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_allowed(update):
        return await deny(update)
    await update.message.reply_text(
        "👋 *LLM Wiki Bot*\n\n"
        "Your personal knowledge base, powered by Google Gemini.\n\n"
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

    # Verbose: show query expansion + BM25 search steps
    status_msg = await update.message.reply_text(
        f"🔍 *Query:* _{question}_\n\n"
        f"🔄 Step 1/2: Expanding query + BM25 searching wiki…\n"
        f"_LLM generates alternative search terms, then BM25 finds relevant pages_",
        parse_mode=ParseMode.MARKDOWN,
    )

    try:
        await _update_status(
            status_msg,
            f"🔍 *Query:* _{question}_\n\n"
            f"✅ Step 1/2: Query expanded + pages found\n"
            f"🔄 Step 2/2: Loading pages → asking Gemini (`{GEMINI_MODEL}`)…\n"
            f"_LLM is reading your wiki and composing an answer_",
        )

        async with typing_indicator(update.message.chat_id, update.message.get_bot()):
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

        answer_msg = await status_msg.edit_text(reply, parse_mode=ParseMode.MARKDOWN)

        # Track this message for emoji feedback
        _query_answer_messages[answer_msg.message_id] = {
            "question": question,
            "answer": result.answer[:300],
            "user_id": update.effective_user.id,
        }
    except Exception as e:
        logger.exception("Query failed")
        await status_msg.edit_text(_format_user_error(e), parse_mode=ParseMode.MARKDOWN)


async def cmd_search(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_allowed(update):
        return await deny(update)

    terms = " ".join(context.args or []).strip()
    if not terms:
        await update.message.reply_text("Usage: /search <keywords>")
        return

    logger.info("cmd_search: terms=%r", terms)
    results = wiki_search.search(terms, top_k=5)
    logger.info("cmd_search: %d result(s) found", len(results))

    if not results:
        await update.message.reply_text(
            f"🔍 *Search:* `{terms}`\n\n"
            f"No results found in wiki.\n"
            f"_Searched across all wiki pages using BM25 ranking._",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    lines = [
        f"🔍 *Wiki search:* `{terms}`\n"
        f"_{len(results)} result(s) found via BM25 ranking_\n"
    ]
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

    status_msg = await update.message.reply_text(
        f"🌐 *Web search:* `{query}`\n\n"
        f"🔄 Querying DuckDuckGo (no API key, HTML scrape)…",
        parse_mode=ParseMode.MARKDOWN,
    )

    try:
        async with typing_indicator(update.message.chat_id, update.message.get_bot()):
            results = await web_search(query, max_results=5)
        if not results:
            await status_msg.edit_text(
                f"🌐 *Web search:* `{query}`\n\n"
                f"🔍 No results found.\n"
                f"_DuckDuckGo returned no matching pages._",
                parse_mode=ParseMode.MARKDOWN,
            )
            return

        lines = [f"🌐 *Web search:* `{query}`\n_{len(results)} result(s)_\n"]
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
        await status_msg.edit_text(_format_user_error(e), parse_mode=ParseMode.MARKDOWN)


async def cmd_fetch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Fetch a URL, convert to markdown, and ingest into the wiki."""
    if not is_allowed(update):
        return await deny(update)

    url = " ".join(context.args or []).strip()
    if not url or not url.startswith("http"):
        await update.message.reply_text("Usage: /fetch <url>\nExample: /fetch https://example.com/article")
        return

    status_msg = await update.message.reply_text(
        f"🌐 *Fetching URL*\n`{url}`\n\n"
        f"🔄 Step 1/3: Downloading page…",
        parse_mode=ParseMode.MARKDOWN,
    )

    try:
        async with typing_indicator(update.message.chat_id, update.message.get_bot()):
            fetch_result: FetchResult = await fetch_url(url)

        await _update_status(
            status_msg,
            f"🌐 *Fetching URL*\n`{url}`\n\n"
            f"✅ Step 1/3: Downloaded — *{fetch_result.title}*\n"
            f"📝 {fetch_result.word_count} words extracted\n"
            f"🔄 Step 2/3: Saving raw content…",
        )

        # Save raw and ingest
        wiki.save_raw(fetch_result.content, "articles", fetch_result.filename)

        await _update_status(
            status_msg,
            f"🌐 *Fetching URL*\n`{url}`\n\n"
            f"✅ Step 1/3: Downloaded — *{fetch_result.title}*\n"
            f"✅ Step 2/3: Raw saved as `{fetch_result.filename}`\n"
            f"🔄 Step 3/3: Ingesting into wiki…",
        )

        await do_ingest(update.message, fetch_result.content, "article", fetch_result.filename)

    except Exception as e:
        logger.exception("Fetch failed")
        await status_msg.edit_text(_format_user_error(e), parse_mode=ParseMode.MARKDOWN)


async def cmd_lint(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_allowed(update):
        return await deny(update)

    page_count = len(wiki.list_pages())
    status_msg = await update.message.reply_text(
        f"🔍 *Wiki Health Check*\n\n"
        f"🔄 Step 1/2: Loading all {page_count} wiki page(s)…",
        parse_mode=ParseMode.MARKDOWN,
    )

    try:
        await _update_status(
            status_msg,
            f"🔍 *Wiki Health Check*\n\n"
            f"✅ Step 1/2: Loaded {page_count} page(s)\n"
            f"🔄 Step 2/2: Asking Gemini (`{GEMINI_MODEL}`) to analyze…\n"
            f"_LLM is checking for contradictions, orphans, missing concepts, stale content_",
        )

        async with typing_indicator(update.message.chat_id, update.message.get_bot()):
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
        await status_msg.edit_text(_format_user_error(e), parse_mode=ParseMode.MARKDOWN)


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_allowed(update):
        return await deny(update)

    result = wiki.get_status(GEMINI_MODEL)
    log_lines = "\n".join(f"  `{e}`" for e in result.last_log_entries) or "  (empty)"

    reply = (
        f"📊 *Wiki Status*\n\n"
        f"*Model:* `{result.model}`\n"
        f"*API:* Google Gemini\n"
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
        await update.message.reply_text(f"```\n{_escape_code_block(chunk)}\n```", parse_mode=ParseMode.MARKDOWN)


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
        await update.message.reply_text(
            f"💾 *Saved insight*\n\n"
            f"📄 `{path}`\n"
            f"🔍 Search index rebuilt",
            parse_mode=ParseMode.MARKDOWN,
        )
    except Exception as e:
        logger.exception("Save failed")
        await update.message.reply_text(_format_user_error(e), parse_mode=ParseMode.MARKDOWN)


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
    logger.info("  → LLM intent classification for: %r", text[:80])
    system = (
        "You are an intent classifier for a personal knowledge base bot. "
        "Classify the user's message into exactly one of these intents:\n"
        "- query: the user is asking a question about their existing wiki/notes\n"
        "- websearch: the user wants to search the internet for new information\n"
        "- journal: the user is logging something (a thought, event, note, observation)\n\n"
        "Reply with ONLY one word: query, websearch, or journal. No explanation."
    )
    messages = [{"role": "user", "content": text}]
    response = llm.chat(system, messages).strip().lower()
    logger.info("  ← LLM classified intent as: %r", response)
    if response in ("query", "websearch", "journal"):
        return response
    # Default to journal if LLM gives unexpected output
    logger.warning("  LLM returned unexpected intent %r — defaulting to 'journal'", response)
    return "journal"


# ── Intent → emoji mapping ────────────────────────────────────────────────────

_INTENT_EMOJI = {
    "query": "❓",
    "websearch": "🌐",
    "fetch": "📥",
    "journal": "📝",
}

_INTENT_LABEL = {
    "query": "Wiki Query",
    "websearch": "Web Search",
    "fetch": "URL Fetch + Ingest",
    "journal": "Journal Entry",
}


def _classify_link(url: str, content: str, title: str) -> dict:
    """
    Use the LLM to classify a link into an action list category.
    Returns a dict with keys: category, description, confidence.
    category is one of: 'to_buy', 'to_review', 'to_read', 'ambiguous'
    """
    logger.info("  → LLM link classification for: %r (title=%r)", url[:80], title[:60])
    system = (
        "You are a link categorizer for a personal knowledge base. "
        "Given a URL and its content, classify it into exactly one action list:\n"
        "- to_buy: products, items to purchase, shopping links, wishlists, deals\n"
        "- to_review: tools, apps, services, courses, or things to evaluate/try\n"
        "- to_read: articles, blog posts, papers, documentation, tutorials, books\n"
        "- ambiguous: if you genuinely cannot determine the category\n\n"
        "Reply with ONLY a JSON object (no markdown fences):\n"
        '{"category": "to_buy|to_review|to_read|ambiguous", '
        '"description": "One-sentence description of the link content", '
        '"confidence": 0.0-1.0}\n\n'
        "Be decisive — only use 'ambiguous' if the content truly fits multiple categories equally."
    )
    # Send a truncated version of the content to avoid token limits
    truncated = content[:3000] if len(content) > 3000 else content
    user_msg = f"URL: {url}\nTitle: {title}\n\nContent (truncated):\n{truncated}"
    messages = [{"role": "user", "content": user_msg}]
    response = llm.chat(system, messages).strip()
    logger.info("  ← LLM link classification response: %r", response[:200])

    from gemini import extract_json as _extract_json
    data = _extract_json(response)
    if data and "category" in data:
        return data
    # Fallback
    logger.warning("  LLM returned unexpected link classification — defaulting to ambiguous")
    return {"category": "ambiguous", "description": title or "Link", "confidence": 0.0}


def _classify_instagram_post(post: InstagramPost) -> dict:
    """
    Use the LLM to generate tags, description, and category for an Instagram post.
    Returns a dict with keys: tags, description, category, confidence.
    """
    logger.info("  → LLM Instagram classification for: @%s post %s", post.author, post.url[:60])
    system = (
        "You are a content tagger for a personal knowledge base. "
        "Given an Instagram post, generate:\n"
        "- tags: a list of 3-7 relevant tags for this content (lowercase, no #)\n"
        "- description: A one-sentence description of what this post is about\n"
        "- category: one of to_buy, to_review, to_read\n"
        "  - to_buy: products, items, shopping recommendations, deals\n"
        "  - to_review: tools, apps, services, places, restaurants to try\n"
        "  - to_read: educational content, articles, tutorials, inspiration, motivation\n"
        "- confidence: 0.0-1.0\n\n"
        "Reply with ONLY a JSON object (no markdown fences):\n"
        '{"tags": ["tag1", "tag2"], "description": "...", '
        '"category": "to_buy|to_review|to_read|ambiguous", "confidence": 0.9}'
    )

    # Build context from the post
    parts = [f"Instagram Post URL: {post.url}"]
    if post.author:
        parts.append(f"Author: @{post.author}")
    if post.caption:
        # Truncate very long captions
        caption = post.caption[:2000] if len(post.caption) > 2000 else post.caption
        parts.append(f"Caption:\n{caption}")
    if post.hashtags:
        parts.append(f"Hashtags: {', '.join('#' + h for h in post.hashtags)}")
    if post.is_video:
        parts.append("Type: Video/Reel")
    else:
        parts.append("Type: Photo")
    if post.like_count is not None:
        parts.append(f"Likes: {post.like_count}")
    if post.comment_count is not None:
        parts.append(f"Comments: {post.comment_count}")

    user_msg = "\n".join(parts)
    messages = [{"role": "user", "content": user_msg}]
    response = llm.chat(system, messages).strip()
    logger.info("  ← LLM Instagram classification response: %r", response[:200])

    from gemini import extract_json as _extract_json
    data = _extract_json(response)
    if data and "category" in data:
        # Ensure tags is a list
        if "tags" not in data or not isinstance(data["tags"], list):
            data["tags"] = post.hashtags or []
        return data

    # Fallback
    logger.warning("  LLM returned unexpected Instagram classification — defaulting to ambiguous")
    return {
        "tags": post.hashtags or [],
        "description": post.caption[:100] if post.caption else "Instagram post",
        "category": "ambiguous",
        "confidence": 0.0,
    }


def _build_instagram_content(post: InstagramPost, classification: dict) -> str:
    """Build enriched markdown content for an Instagram post to be ingested."""
    tags = classification.get("tags", post.hashtags or [])
    description = classification.get("description", "")
    category = classification.get("category", "")
    label = _CATEGORY_LABEL.get(category, category)

    lines = [
        "---",
        f"url: {post.url}",
        f"title: Instagram Post by @{post.author}" if post.author else "title: Instagram Post",
        "source_type: instagram",
        f"author: \"@{post.author}\"" if post.author else "author: unknown",
        f"date_ingested: {datetime.now().strftime('%Y-%m-%d')}",
    ]
    if post.timestamp:
        lines.append(f"date_posted: {post.timestamp}")
    if tags:
        lines.append(f"tags: [{', '.join(tags)}]")
    if label:
        lines.append(f"action_list: {label}")
    if description:
        lines.append(f"description: \"{description}\"")
    if post.thumbnail_url:
        lines.append(f"thumbnail: {post.thumbnail_url}")
    lines.append("---")
    lines.append("")

    if post.caption:
        lines.append("## Caption")
        lines.append("")
        lines.append(post.caption)
        lines.append("")

    lines.append("## Metadata")
    lines.append("")
    if post.author:
        lines.append(f"- Author: @{post.author}")
    if post.timestamp:
        lines.append(f"- Posted: {post.timestamp}")
    if post.like_count is not None:
        lines.append(f"- Likes: {post.like_count:,}")
    if post.comment_count is not None:
        lines.append(f"- Comments: {post.comment_count:,}")
    lines.append(f"- Type: {'Video/Reel' if post.is_video else 'Photo'}")
    if post.hashtags:
        lines.append(f"- Hashtags: {', '.join('#' + h for h in post.hashtags)}")
    lines.append("")

    return "\n".join(lines)


# Per-user state: pending link categorization for inline keyboard
_pending_links: dict[int, dict] = {}

# Per-user state: pending Instagram post for inline keyboard
_pending_instagram: dict[int, dict] = {}

_CATEGORY_EMOJI = {
    "to_buy": "🛒",
    "to_review": "🔍",
    "to_read": "📖",
}

_CATEGORY_LABEL = {
    "to_buy": "To Buy",
    "to_review": "To Review",
    "to_read": "To Read",
}


async def handle_link_category_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline keyboard button presses for link categorization."""
    query = update.callback_query
    await query.answer()

    data = query.data or ""
    if not data.startswith("linkcat:"):
        return

    category = data[len("linkcat:"):]
    user_id = update.effective_user.id
    pending = _pending_links.pop(user_id, None)

    if not pending:
        await query.message.reply_text("⚠️ Link data expired. Please send the link again.")
        return

    url = pending["url"]
    description = pending["description"]
    content = pending["content"]
    filename = pending["filename"]
    title = pending["title"]

    emoji = _CATEGORY_EMOJI.get(category, "📋")
    label = _CATEGORY_LABEL.get(category, category)

    # Save raw and ingest with category metadata
    wiki.save_raw(content, "articles", filename)

    # Update the message to show the chosen category
    await query.message.edit_text(
        f"🔗 *Link categorized:* {emoji} {label}\n\n"
        f"*Title:* {title}\n"
        f"*Description:* _{description}_\n"
        f"📥 `{url}`\n\n"
        f"🔄 Ingesting into wiki…",
        parse_mode=ParseMode.MARKDOWN,
    )

    # Ingest with category info prepended
    enriched_content = (
        f"---\n"
        f"url: {url}\n"
        f"title: {title}\n"
        f"action_list: {label}\n"
        f"description: {description}\n"
        f"---\n\n"
        f"{content}"
    )
    await do_ingest(query.message, enriched_content, "article", filename)


async def handle_instagram_category_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline keyboard button presses for Instagram post categorization."""
    query = update.callback_query
    await query.answer()

    data = query.data or ""
    if not data.startswith("igcat:"):
        return

    category = data[len("igcat:"):]
    user_id = update.effective_user.id
    pending = _pending_instagram.pop(user_id, None)

    if not pending:
        await query.message.reply_text("⚠️ Instagram data expired. Please send the link again.")
        return

    post: InstagramPost = pending["post"]
    classification: dict = pending["classification"]

    # Override category with user's choice
    classification["category"] = category

    emoji = _CATEGORY_EMOJI.get(category, "📋")
    label = _CATEGORY_LABEL.get(category, category)
    description = classification.get("description", post.caption[:80] if post.caption else "Instagram post")
    tags = classification.get("tags", post.hashtags or [])

    # Update the message to show the chosen category
    await query.message.edit_text(
        f"📸 *Instagram post categorized:* {emoji} {label}\n\n"
        f"*Author:* @{post.author}\n"
        f"*Description:* _{description}_\n"
        f"🏷️ Tags: {', '.join(tags) if tags else '(none)'}\n\n"
        f"🔄 Ingesting into wiki…",
        parse_mode=ParseMode.MARKDOWN,
    )

    # Build enriched content and ingest
    enriched_content = _build_instagram_content(post, classification)
    slug = slugify(f"instagram-{post.author}-{now_slug()}" if post.author else f"instagram-{now_slug()}")
    filename = f"{slug}.md"

    wiki.save_raw(enriched_content, "articles", filename)
    await do_ingest(query.message, enriched_content, "instagram", filename)


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle plain text messages with smart intent detection.
    No command required — the bot figures out what you mean:

      Instagram URL           → extract post, tag, categorize + ingest
      URL in message          → smart link categorization + ingest
      "What did I learn..."   → query wiki
      "Search online for..."  → web search
      "I went for a run..."   → journal ingest
    """
    if not is_allowed(update):
        return await deny(update)

    text = (update.message.text or "").strip()
    if not text:
        return

    # ── 1. URL detection (highest priority) ───────────────────────────────────
    url_match = URL_RE.search(text)
    if url_match:
        url = url_match.group(0)

        # ── 1a. Instagram URL — special handling ──────────────────────────────
        if is_instagram_url(url):
            status_msg = await update.message.reply_text(
                f"📸 *Instagram Post Detected*\n"
                f"_Extracting post data…_\n\n"
                f"🔄 Step 1/3: Extracting metadata via yt-dlp…",
                parse_mode=ParseMode.MARKDOWN,
            )
            try:
                async with typing_indicator(update.message.chat_id, update.message.get_bot()):
                    post = await extract_instagram_post(url)

                if not post:
                    await status_msg.edit_text(
                        f"📸 *Instagram Post*\n\n"
                        f"❌ Could not extract post data.\n"
                        f"The post may be private or Instagram may be blocking access.\n\n"
                        f"_Try sending a public post URL._",
                        parse_mode=ParseMode.MARKDOWN,
                    )
                    return

                caption_preview = (post.caption[:100] + "…") if post.caption and len(post.caption) > 100 else (post.caption or "(no caption)")

                await _update_status(
                    status_msg,
                    f"📸 *Instagram Post Detected*\n\n"
                    f"✅ Step 1/3: Extracted — @{post.author or 'unknown'}\n"
                    f"📝 _{caption_preview}_\n"
                    f"🔄 Step 2/3: Generating tags & categorizing…\n"
                    f"_LLM is analyzing the post content_",
                )

                # Classify the Instagram post using LLM
                async with typing_indicator(update.message.chat_id, update.message.get_bot()):
                    classification = await asyncio.get_event_loop().run_in_executor(
                        None, _classify_instagram_post, post
                    )

                category = classification.get("category", "ambiguous")
                description = classification.get("description", caption_preview)
                confidence = classification.get("confidence", 0.0)
                tags = classification.get("tags", post.hashtags or [])

                logger.info(
                    "instagram classified: url=%s category=%s confidence=%.2f tags=%s",
                    url[:60], category, confidence, tags[:5],
                )

                if category == "ambiguous" or confidence < 0.6:
                    # Ambiguous — ask the user via inline keyboard
                    _pending_instagram[update.effective_user.id] = {
                        "post": post,
                        "classification": classification,
                    }

                    buttons = [
                        [InlineKeyboardButton("🛒 To Buy", callback_data="igcat:to_buy")],
                        [InlineKeyboardButton("🔍 To Review", callback_data="igcat:to_review")],
                        [InlineKeyboardButton("📖 To Read", callback_data="igcat:to_read")],
                    ]
                    reply_markup = InlineKeyboardMarkup(buttons)

                    await status_msg.edit_text(
                        f"📸 *Instagram Post by @{post.author or 'unknown'}*\n\n"
                        f"📝 _{description}_\n"
                        f"🏷️ Tags: {', '.join(tags) if tags else '(none)'}\n\n"
                        f"🤔 I'm not sure which list this belongs to.\n"
                        f"*Which reading list should I add it to?*",
                        parse_mode=ParseMode.MARKDOWN,
                        reply_markup=reply_markup,
                    )
                else:
                    # Confident classification — auto-categorize and ingest
                    emoji = _CATEGORY_EMOJI.get(category, "📋")
                    label = _CATEGORY_LABEL.get(category, category)

                    await _update_status(
                        status_msg,
                        f"📸 *Instagram Post Detected*\n\n"
                        f"✅ Step 1/3: Extracted — @{post.author or 'unknown'}\n"
                        f"✅ Step 2/3: Categorized → {emoji} {label}\n"
                        f"📝 _{description}_\n"
                        f"🏷️ Tags: {', '.join(tags)}\n\n"
                        f"🔄 Step 3/3: Ingesting into wiki…",
                    )

                    enriched_content = _build_instagram_content(post, classification)
                    slug = slugify(
                        f"instagram-{post.author}-{now_slug()}"
                        if post.author
                        else f"instagram-{now_slug()}"
                    )
                    filename = f"{slug}.md"

                    wiki.save_raw(enriched_content, "articles", filename)
                    await do_ingest(update.message, enriched_content, "instagram", filename)

            except Exception as e:
                logger.exception("Instagram processing failed")
                await status_msg.edit_text(_format_user_error(e), parse_mode=ParseMode.MARKDOWN)
            return

        # ── 1b. Generic URL — smart link categorization ───────────────────────
        status_msg = await update.message.reply_text(
            f"🔗 *Smart Link Processing*\n"
            f"_Detected URL in message_\n\n"
            f"🔄 Step 1/3: Fetching `{url}`…",
            parse_mode=ParseMode.MARKDOWN,
        )
        try:
            async with typing_indicator(update.message.chat_id, update.message.get_bot()):
                fetch_result = await fetch_url(url)

            await _update_status(
                status_msg,
                f"🔗 *Smart Link Processing*\n\n"
                f"✅ Step 1/3: Fetched — *{fetch_result.title}* ({fetch_result.word_count} words)\n"
                f"🔄 Step 2/3: Analyzing content & categorizing…\n"
                f"_LLM is reading the page and deciding which action list it belongs to_",
            )

            # Classify the link using LLM
            async with typing_indicator(update.message.chat_id, update.message.get_bot()):
                classification = await asyncio.get_event_loop().run_in_executor(
                    None, _classify_link, url, fetch_result.content, fetch_result.title
                )

            category = classification.get("category", "ambiguous")
            description = classification.get("description", fetch_result.title)
            confidence = classification.get("confidence", 0.0)

            logger.info(
                "link classified: url=%s category=%s confidence=%.2f desc=%r",
                url[:60], category, confidence, description[:80],
            )

            if category == "ambiguous" or confidence < 0.6:
                # Ambiguous — ask the user via inline keyboard
                _pending_links[update.effective_user.id] = {
                    "url": url,
                    "description": description,
                    "content": fetch_result.content,
                    "filename": fetch_result.filename,
                    "title": fetch_result.title,
                }

                buttons = [
                    [InlineKeyboardButton("🛒 To Buy", callback_data="linkcat:to_buy")],
                    [InlineKeyboardButton("🔍 To Review", callback_data="linkcat:to_review")],
                    [InlineKeyboardButton("📖 To Read", callback_data="linkcat:to_read")],
                ]
                reply_markup = InlineKeyboardMarkup(buttons)

                await status_msg.edit_text(
                    f"🔗 *Link fetched:* *{fetch_result.title}*\n\n"
                    f"📝 _{description}_\n\n"
                    f"🤔 I'm not sure which list this belongs to.\n"
                    f"*Which action list should I add it to?*",
                    parse_mode=ParseMode.MARKDOWN,
                    reply_markup=reply_markup,
                )
            else:
                # Confident classification — auto-categorize and ingest
                emoji = _CATEGORY_EMOJI.get(category, "📋")
                label = _CATEGORY_LABEL.get(category, category)

                await _update_status(
                    status_msg,
                    f"🔗 *Smart Link Processing*\n\n"
                    f"✅ Step 1/3: Fetched — *{fetch_result.title}*\n"
                    f"✅ Step 2/3: Categorized → {emoji} {label}\n"
                    f"📝 _{description}_\n\n"
                    f"🔄 Step 3/3: Saving raw + ingesting into wiki…",
                )

                wiki.save_raw(fetch_result.content, "articles", fetch_result.filename)

                # Ingest with category info prepended
                enriched_content = (
                    f"---\n"
                    f"url: {url}\n"
                    f"title: {fetch_result.title}\n"
                    f"action_list: {label}\n"
                    f"description: {description}\n"
                    f"---\n\n"
                    f"{fetch_result.content}"
                )
                await do_ingest(
                    update.message, enriched_content, "article", fetch_result.filename
                )

        except Exception as e:
            logger.exception("Smart link processing failed")
            await status_msg.edit_text(_format_user_error(e), parse_mode=ParseMode.MARKDOWN)
        return

    # ── 2. Fast regex classification ─────────────────────────────────────────
    intent = _classify_intent_fast(text)

    if intent:
        # Show the fast classification result
        emoji = _INTENT_EMOJI.get(intent, "🤖")
        label = _INTENT_LABEL.get(intent, intent)
        thinking = await update.message.reply_text(
            f"🧠 *Intent:* {emoji} {label}\n"
            f"_Classified via fast regex pattern match (no LLM needed)_",
            parse_mode=ParseMode.MARKDOWN,
        )
        logger.info("intent classified (fast regex): %r → %s", text[:60], intent)
        # Delete the thinking message after a short delay
        await asyncio.sleep(1.5)
        try:
            await thinking.delete()
        except Exception:
            pass
    else:
        # ── 3. LLM classification for ambiguous messages ──────────────────────
        thinking = await update.message.reply_text(
            f"🧠 *Classifying intent…*\n"
            f"_Message is ambiguous — asking Gemini (`{GEMINI_MODEL}`) to classify_\n\n"
            f"🔄 Sending to LLM intent classifier…",
            parse_mode=ParseMode.MARKDOWN,
        )
        logger.info("intent ambiguous — using LLM classifier for: %r", text[:60])

        async with typing_indicator(update.message.chat_id, update.message.get_bot()):
            intent = await asyncio.get_event_loop().run_in_executor(
                None, _classify_intent_llm, text
            )

        emoji = _INTENT_EMOJI.get(intent, "🤖")
        label = _INTENT_LABEL.get(intent, intent)
        await _update_status(
            thinking,
            f"🧠 *Intent:* {emoji} {label}\n"
            f"_Classified by Gemini LLM (message was ambiguous for regex)_",
        )
        await asyncio.sleep(1.5)
        try:
            await thinking.delete()
        except Exception:
            pass

    # ── 4. Route to the right handler ────────────────────────────────────────
    if intent == "query":
        # Treat as a wiki query
        await update.message.chat.send_action(ChatAction.TYPING)

        status_msg = await update.message.reply_text(
            f"🔍 *Query:* _{text[:80]}_\n\n"
            f"🔄 Step 1/2: Expanding query + BM25 searching wiki…\n"
            f"_LLM generates alternative search terms, then BM25 finds relevant pages_",
            parse_mode=ParseMode.MARKDOWN,
        )

        try:
            await _update_status(
                status_msg,
                f"🔍 *Query:* _{text[:80]}_\n\n"
                f"✅ Step 1/2: Query expanded + pages found\n"
                f"🔄 Step 2/2: Loading pages → asking Gemini (`{GEMINI_MODEL}`)…\n"
                f"_LLM is reading your wiki and composing an answer_",
            )

            async with typing_indicator(update.message.chat_id, update.message.get_bot()):
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
            answer_msg = await status_msg.edit_text(
                result.answer[:3800] + sources_text + save_hint,
                parse_mode=ParseMode.MARKDOWN,
            )

            # Track this message for emoji feedback
            _query_answer_messages[answer_msg.message_id] = {
                "question": text,
                "answer": result.answer[:300],
                "user_id": update.effective_user.id,
            }
        except Exception as e:
            logger.exception("Query failed")
            await status_msg.edit_text(_format_user_error(e), parse_mode=ParseMode.MARKDOWN)

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
            f"🌐 *Web search:* `{query}`\n\n"
            f"🔄 Querying DuckDuckGo…",
            parse_mode=ParseMode.MARKDOWN,
        )
        try:
            async with typing_indicator(update.message.chat_id, update.message.get_bot()):
                results = await web_search(query, max_results=5)
            if not results:
                await status_msg.edit_text(
                    f"🌐 *Web search:* `{query}`\n\n🔍 No results found.",
                    parse_mode=ParseMode.MARKDOWN,
                )
                return
            lines = [f"🌐 *Web search:* `{query}`\n_{len(results)} result(s)_\n"]
            for i, r in enumerate(results, 1):
                lines.append(f"{i}. *{r.title}*\n   {r.url}\n   _{r.snippet}_\n")
            lines.append("\n💡 Fetch any result: `/fetch <url>` or just paste the URL")
            await status_msg.edit_text("\n".join(lines)[:4000], parse_mode=ParseMode.MARKDOWN)
        except Exception as e:
            logger.exception("Web search failed")
            await status_msg.edit_text(_format_user_error(e), parse_mode=ParseMode.MARKDOWN)

    else:
        # journal — ingest as a journal entry
        ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
        filename = f"journal-{ts}.md"
        logger.info("journal ingest: saving raw + ingesting as %s", filename)
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

    status_msg = await update.message.reply_text(
        f"📥 *File upload:* `{fname}`\n\n"
        f"🔄 Step 1/3: Downloading from Telegram…",
        parse_mode=ParseMode.MARKDOWN,
    )

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tg_file = await doc.get_file()
        await tg_file.download_to_drive(tmp.name)
        content = Path(tmp.name).read_text(encoding="utf-8", errors="replace")

    word_count = len(content.split())
    await _update_status(
        status_msg,
        f"📥 *File upload:* `{fname}`\n\n"
        f"✅ Step 1/3: Downloaded ({word_count} words)\n"
        f"🔄 Step 2/3: Saving raw content…",
    )

    wiki.save_raw(content, "articles", fname)

    await _update_status(
        status_msg,
        f"📥 *File upload:* `{fname}`\n\n"
        f"✅ Step 1/3: Downloaded ({word_count} words)\n"
        f"✅ Step 2/3: Raw saved\n"
        f"🔄 Step 3/3: Ingesting into wiki…",
    )

    await do_ingest(update.message, content, "article", fname)


# ── Emoji feedback handler ────────────────────────────────────────────────


async def handle_emoji_feedback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Detect when the user replies to a query answer with a positive emoji
    (👍, 👌, ❤️, 🔥, 💯, 🙏) and log it as positive feedback.
    """
    if not is_allowed(update):
        return

    text = (update.message.text or "").strip()
    if not text:
        return

    # Check if this is a positive emoji
    # Strip variation selectors and whitespace for comparison
    clean = text.replace("\ufe0f", "").strip()
    if clean not in _POSITIVE_EMOJI:
        return

    # Check if this is a reply to a tracked query answer message
    reply = update.message.reply_to_message
    if not reply:
        return

    query_data = _query_answer_messages.get(reply.message_id)
    if not query_data:
        return

    # Log the positive feedback
    logger.info(
        "emoji feedback: user reacted with %r to query=%r",
        text, query_data["question"][:60],
    )
    try:
        wiki.append_feedback(query_data["question"], query_data["answer"])
        await update.message.reply_text(
            "✅ Thanks! I'll remember this was a good answer.",
        )
    except Exception as e:
        logger.exception("Failed to save feedback")


# ── Startup ───────────────────────────────────────────────────────────────────


async def post_init(application: Application) -> None:
    """Called after the bot is initialized — verify Gemini API access."""
    logger.info("Checking Gemini API…")
    llm.wait_until_ready()
    logger.info("Gemini API ready — model=%s", GEMINI_MODEL)

    wiki_search.rebuild_index()
    logger.info("Bot ready.")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    global llm, wiki, wiki_search

    llm = GeminiClient(api_key=GEMINI_API_KEY, model=GEMINI_MODEL)
    wiki_search = WikiSearch(wiki_dir=str(Path(DATA_DIR) / "wiki"))
    wiki = WikiManager(data_dir=DATA_DIR, llm=llm, search=wiki_search)

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

    # Callback query handlers (inline keyboard buttons)
    app.add_handler(CallbackQueryHandler(handle_view_page, pattern=r"^view:"))
    app.add_handler(CallbackQueryHandler(handle_link_category_callback, pattern=r"^linkcat:"))
    app.add_handler(CallbackQueryHandler(handle_instagram_category_callback, pattern=r"^igcat:"))

    # Message handlers — emoji feedback must come before general text handler
    # so that emoji replies to query answers are caught first
    app.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND & filters.REPLY,
        handle_emoji_feedback,
    ), group=0)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text), group=1)
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document), group=1)

    logger.info("Starting bot (polling)…")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
