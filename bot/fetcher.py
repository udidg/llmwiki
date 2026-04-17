"""
Internet fetch tool for the LLM Wiki bot.

Provides:
  - fetch_url(url)                → clean markdown from any URL
  - web_search(query)             → DuckDuckGo search results (via duckduckgo-search lib)
  - search_and_fetch_top(query)   → convenience: search + fetch top result
  - is_instagram_url(url)         → check if URL is an Instagram post/reel
  - extract_instagram_post(url)   → extract Instagram post metadata via yt-dlp

Uses httpx for HTTP, markdownify for HTML→markdown, duckduckgo-search for
web search, and yt-dlp for Instagram post extraction.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import subprocess
import time
from dataclasses import dataclass, field

import httpx
from markdownify import markdownify

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; LLMWikiBot/1.0; +https://github.com/llm-wiki)"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

TIMEOUT = httpx.Timeout(30.0, connect=10.0)


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class FetchResult:
    url: str
    title: str
    content: str        # clean markdown
    filename: str       # suggested raw/ filename
    word_count: int


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


@dataclass
class InstagramPost:
    """Metadata extracted from an Instagram post via yt-dlp."""
    url: str
    caption: str              # post caption/text
    author: str               # Instagram username
    thumbnail_url: str        # URL to post image/thumbnail
    timestamp: str            # post date if available (ISO format)
    like_count: int | None
    comment_count: int | None
    is_video: bool
    hashtags: list[str] = field(default_factory=list)  # extracted from caption


# ── Instagram URL detection ──────────────────────────────────────────────────

INSTAGRAM_RE = re.compile(
    r"https?://(?:www\.)?instagram\.com/"
    r"(?:p|reel|tv|reels)/([A-Za-z0-9_-]+)"
)


def is_instagram_url(url: str) -> bool:
    """Check if a URL is an Instagram post, reel, or IGTV link."""
    return bool(INSTAGRAM_RE.search(url))


def _extract_hashtags(text: str) -> list[str]:
    """Extract hashtags from text."""
    return re.findall(r"#(\w+)", text)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _slugify(text: str, max_len: int = 60) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text[:max_len].strip("-")


def _extract_title_from_html(html: str) -> str:
    """Extract <title> tag content from raw HTML."""
    m = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Fallback: first H1
    m = re.search(r"<h1[^>]*>([^<]+)</h1>", html, re.IGNORECASE)
    if m:
        return re.sub(r"<[^>]+>", "", m.group(1)).strip()
    return "Untitled"


def _clean_markdown(md: str) -> str:
    """Remove excessive blank lines and navigation noise from converted markdown."""
    # Collapse 3+ blank lines to 2
    md = re.sub(r"\n{3,}", "\n\n", md)
    # Remove lines that are just punctuation/symbols (nav artifacts)
    lines = [l for l in md.splitlines() if not re.match(r"^[|•·▸►▼▲→←\-=_]{3,}$", l.strip())]
    return "\n".join(lines).strip()


def _url_to_filename(url: str, title: str) -> str:
    """Generate a filename from URL or title."""
    slug = _slugify(title) if title and title != "Untitled" else _slugify(url.split("/")[-1] or "article")
    return f"{slug}.md"


def _extract_og_meta(html: str) -> dict[str, str]:
    """Extract Open Graph meta tags from HTML."""
    meta: dict[str, str] = {}
    for m in re.finditer(
        r'<meta\s+(?:property|name)=["\']og:(\w+)["\']\s+content=["\']([^"\']*)["\']',
        html,
        re.IGNORECASE,
    ):
        meta[m.group(1)] = m.group(2)
    # Also try reversed attribute order
    for m in re.finditer(
        r'<meta\s+content=["\']([^"\']*)["\']\s+(?:property|name)=["\']og:(\w+)["\']',
        html,
        re.IGNORECASE,
    ):
        meta[m.group(2)] = m.group(1)
    return meta


# ── Core fetch ────────────────────────────────────────────────────────────────


async def fetch_url(url: str) -> FetchResult:
    """
    Fetch a URL and return clean markdown content.
    Handles HTML pages and plain text/markdown files.
    """
    logger.info("▶ fetch_url: GET %s", url)
    t0 = time.time()

    async with httpx.AsyncClient(
        timeout=TIMEOUT,
        follow_redirects=True,
        headers=HEADERS,
    ) as client:
        resp = await client.get(url)
        resp.raise_for_status()

    elapsed = time.time() - t0
    content_type = resp.headers.get("content-type", "").lower()
    raw_bytes = len(resp.content)
    raw = resp.text

    logger.info(
        "  HTTP %d  content-type=%r  raw_size=%s  elapsed=%.2fs",
        resp.status_code,
        content_type,
        _fmt_bytes(raw_bytes),
        elapsed,
    )

    if "html" in content_type:
        title = _extract_title_from_html(raw)
        logger.info("  extracted title: %r", title)
        logger.info("  converting HTML → markdown (stripping script/style/nav/footer/header) …")
        md = markdownify(raw, heading_style="ATX", strip=["script", "style", "nav", "footer", "header"])
        raw_md_len = len(md)
        md = _clean_markdown(md)
        logger.info(
            "  markdown conversion: raw_md=%d chars → cleaned=%d chars (removed %d chars of noise)",
            raw_md_len, len(md), raw_md_len - len(md),
        )
    else:
        # Plain text or markdown — use as-is
        title = url.split("/")[-1] or "Document"
        md = raw
        logger.info("  plain text/markdown content — using as-is  title=%r", title)

    filename = _url_to_filename(url, title)
    word_count = len(md.split())

    logger.info(
        "✓ fetch_url done  title=%r  filename=%s  words=%d  total_elapsed=%.2fs",
        title, filename, word_count, time.time() - t0,
    )
    return FetchResult(
        url=url,
        title=title,
        content=md,
        filename=filename,
        word_count=word_count,
    )


# ── DuckDuckGo search (via duckduckgo-search library) ─────────────────────────


def _web_search_sync(query: str, max_results: int = 5) -> list[SearchResult]:
    """
    Synchronous web search using the duckduckgo-search library.
    Called from async context via run_in_executor.
    """
    from duckduckgo_search import DDGS

    results: list[SearchResult] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                title = r.get("title", "")
                url = r.get("href", "")
                snippet = r.get("body", "")
                if title and url:
                    results.append(SearchResult(title=title, url=url, snippet=snippet))
    except Exception as e:
        logger.error("duckduckgo-search failed: %s: %s", type(e).__name__, e)
    return results


async def web_search(query: str, max_results: int = 5) -> list[SearchResult]:
    """
    Search the web using DuckDuckGo via the duckduckgo-search library.
    Returns up to max_results results. No API key required.
    """
    logger.info("▶ web_search: query=%r  max_results=%d", query, max_results)
    t0 = time.time()

    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(None, _web_search_sync, query, max_results)

    logger.info(
        "✓ web_search done  query=%r  results=%d  elapsed=%.2fs",
        query, len(results), time.time() - t0,
    )
    return results


# ── Convenience: search + fetch top result ────────────────────────────────────


async def search_and_fetch_top(query: str) -> FetchResult | None:
    """Search for query and fetch the top result."""
    logger.info("search_and_fetch_top: query=%r", query)
    results = await web_search(query, max_results=1)
    if not results:
        logger.info("search_and_fetch_top: no results found")
        return None
    logger.info("search_and_fetch_top: fetching top result → %s", results[0].url)
    return await fetch_url(results[0].url)


# ── Instagram post extraction (via yt-dlp) ────────────────────────────────────


def _extract_instagram_sync(url: str) -> InstagramPost | None:
    """
    Synchronous Instagram post extraction using yt-dlp.
    Called from async context via run_in_executor.
    """
    logger.info("▶ _extract_instagram_sync: %s", url)
    t0 = time.time()

    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "--dump-json",
                "--no-download",
                "--no-warnings",
                "--quiet",
                url,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            logger.warning(
                "yt-dlp failed (exit %d): stderr=%s",
                result.returncode,
                result.stderr[:300] if result.stderr else "(empty)",
            )
            return None

        if not result.stdout.strip():
            logger.warning("yt-dlp returned empty output for %s", url)
            return None

        data = json.loads(result.stdout)

        caption = data.get("description", "") or data.get("title", "") or ""
        author = data.get("uploader", "") or data.get("uploader_id", "") or data.get("channel", "") or ""
        thumbnail = data.get("thumbnail", "") or ""
        timestamp_epoch = data.get("timestamp")
        timestamp_str = ""
        if timestamp_epoch:
            from datetime import datetime, timezone
            timestamp_str = datetime.fromtimestamp(timestamp_epoch, tz=timezone.utc).strftime("%Y-%m-%d")
        elif data.get("upload_date"):
            # yt-dlp gives upload_date as YYYYMMDD
            ud = data["upload_date"]
            if len(ud) == 8:
                timestamp_str = f"{ud[:4]}-{ud[4:6]}-{ud[6:8]}"

        like_count = data.get("like_count")
        comment_count = data.get("comment_count")
        is_video = data.get("ext", "") in ("mp4", "webm") or bool(data.get("duration"))

        hashtags = _extract_hashtags(caption)

        post = InstagramPost(
            url=url,
            caption=caption,
            author=author,
            thumbnail_url=thumbnail,
            timestamp=timestamp_str,
            like_count=like_count,
            comment_count=comment_count,
            is_video=is_video,
            hashtags=hashtags,
        )

        logger.info(
            "✓ _extract_instagram_sync done  author=%r  caption_len=%d  "
            "hashtags=%d  is_video=%s  elapsed=%.2fs",
            author, len(caption), len(hashtags), is_video, time.time() - t0,
        )
        return post

    except subprocess.TimeoutExpired:
        logger.error("yt-dlp timed out for %s", url)
        return None
    except json.JSONDecodeError as e:
        logger.error("yt-dlp returned invalid JSON for %s: %s", url, e)
        return None
    except FileNotFoundError:
        logger.error("yt-dlp not found — is it installed?")
        return None
    except Exception as e:
        logger.error("yt-dlp extraction failed for %s: %s: %s", url, type(e).__name__, e)
        return None


async def _extract_instagram_fallback(url: str) -> InstagramPost | None:
    """
    Fallback Instagram extraction using HTTP fetch + OG meta tags.
    Used when yt-dlp fails.
    """
    logger.info("▶ _extract_instagram_fallback: %s", url)
    t0 = time.time()

    try:
        async with httpx.AsyncClient(
            timeout=TIMEOUT,
            follow_redirects=True,
            headers=HEADERS,
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()

        html = resp.text
        og = _extract_og_meta(html)

        title = og.get("title", "")
        description = og.get("description", "")
        image = og.get("image", "")

        # Try to extract author from title (Instagram titles are often "Author on Instagram: ...")
        author = ""
        if " on Instagram" in title:
            author = title.split(" on Instagram")[0].strip()
        elif "@" in title:
            m = re.search(r"@(\w+)", title)
            if m:
                author = m.group(1)

        caption = description or title
        hashtags = _extract_hashtags(caption)

        post = InstagramPost(
            url=url,
            caption=caption,
            author=author,
            thumbnail_url=image,
            timestamp="",
            like_count=None,
            comment_count=None,
            is_video=False,
            hashtags=hashtags,
        )

        logger.info(
            "✓ _extract_instagram_fallback done  author=%r  caption_len=%d  "
            "elapsed=%.2fs",
            author, len(caption), time.time() - t0,
        )
        return post

    except Exception as e:
        logger.error("Instagram fallback extraction failed for %s: %s: %s", url, type(e).__name__, e)
        return None


async def extract_instagram_post(url: str) -> InstagramPost | None:
    """
    Extract Instagram post metadata.
    Tries yt-dlp first, falls back to OG meta tags if that fails.
    """
    logger.info("▶ extract_instagram_post: %s", url)

    # Try yt-dlp first (richer data)
    loop = asyncio.get_event_loop()
    post = await loop.run_in_executor(None, _extract_instagram_sync, url)

    if post:
        return post

    # Fallback to OG meta tags
    logger.info("  yt-dlp failed — trying OG meta tag fallback")
    return await _extract_instagram_fallback(url)


# ── Utilities ─────────────────────────────────────────────────────────────────


def _fmt_bytes(n: int) -> str:
    """Human-readable byte size."""
    if n == 0:
        return "0 B"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n //= 1024
    return f"{n:.1f} TB"
