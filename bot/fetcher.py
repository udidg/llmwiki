"""
Internet fetch tool for the LLM Wiki bot.

Provides:
  - fetch_url(url)         → clean markdown from any URL
  - web_search(query)      → DuckDuckGo search results (no API key needed)
  - fetch_and_ingest(url)  → convenience: fetch + return content + filename

Uses httpx for HTTP, markdownify for HTML→markdown, and DuckDuckGo's
lite HTML interface for search (no API key required).
"""

from __future__ import annotations

import logging
import re
import time
import urllib.parse
from dataclasses import dataclass

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


# ── DuckDuckGo search (no API key) ────────────────────────────────────────────


async def web_search(query: str, max_results: int = 5) -> list[SearchResult]:
    """
    Search the web using DuckDuckGo's HTML interface.
    Returns up to max_results results. No API key required.
    """
    logger.info("▶ web_search: query=%r  max_results=%d", query, max_results)
    t0 = time.time()

    encoded = urllib.parse.quote_plus(query)
    url = f"https://html.duckduckgo.com/html/?q={encoded}"

    async with httpx.AsyncClient(
        timeout=TIMEOUT,
        follow_redirects=True,
        headers=HEADERS,
    ) as client:
        resp = await client.get(url)
        resp.raise_for_status()

    elapsed = time.time() - t0
    logger.info(
        "  DDG HTTP %d  response_size=%s  elapsed=%.2fs",
        resp.status_code, _fmt_bytes(len(resp.content)), elapsed,
    )

    html = resp.text
    results: list[SearchResult] = []

    # Parse result blocks from DDG HTML
    result_blocks = re.findall(
        r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>.*?'
        r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
        html,
        re.DOTALL,
    )

    logger.info("  parsed %d raw result block(s) from DDG HTML", len(result_blocks))

    for href, title_html, snippet_html in result_blocks[:max_results]:
        title = re.sub(r"<[^>]+>", "", title_html).strip()
        snippet = re.sub(r"<[^>]+>", "", snippet_html).strip()

        # DDG wraps URLs — extract the actual URL
        actual_url = href
        if "uddg=" in href:
            m = re.search(r"uddg=([^&]+)", href)
            if m:
                actual_url = urllib.parse.unquote(m.group(1))

        if title and actual_url:
            results.append(SearchResult(title=title, url=actual_url, snippet=snippet))
            logger.debug("  result: %r  url=%s", title, actual_url)

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
