"""
BM25 search over wiki markdown files.
No embeddings, no vector DB — pure on-device keyword search.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    path: str          # relative path from wiki dir, e.g. "sources/huberman-sleep.md"
    title: str         # extracted from frontmatter or filename
    snippet: str       # first ~200 chars of content after frontmatter
    score: float


def _extract_title(content: str, fallback: str) -> str:
    """Extract title from YAML frontmatter or first H1."""
    for line in content.splitlines():
        if line.startswith("title:"):
            return line.split(":", 1)[1].strip().strip('"').strip("'")
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def _strip_frontmatter(content: str) -> str:
    """Remove YAML frontmatter block."""
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            return content[end + 3:].strip()
    return content


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer, lowercased."""
    return re.findall(r"\b\w+\b", text.lower())


def _snippet(content: str, max_chars: int = 200) -> str:
    body = _strip_frontmatter(content)
    # Remove markdown headings and links for cleaner snippet
    body = re.sub(r"#+\s+", "", body)
    body = re.sub(r"\[\[([^\]]+)\]\]", r"\1", body)
    body = body.strip()
    return body[:max_chars] + ("…" if len(body) > max_chars else "")


class WikiSearch:
    """BM25 search index over all wiki markdown files."""

    def __init__(self, wiki_dir: str) -> None:
        self.wiki_dir = Path(wiki_dir)
        self._paths: list[str] = []
        self._contents: list[str] = []
        self._bm25: BM25Okapi | None = None
        self.rebuild_index()

    def rebuild_index(self) -> None:
        """Scan wiki directory and rebuild the BM25 index."""
        paths: list[str] = []
        contents: list[str] = []

        for p in sorted(self.wiki_dir.rglob("*.md")):
            if p.name in ("log.md",):
                continue
            rel = str(p.relative_to(self.wiki_dir))
            content = p.read_text(encoding="utf-8")
            paths.append(rel)
            contents.append(content)

        self._paths = paths
        self._contents = contents

        if contents:
            tokenized = [_tokenize(c) for c in contents]
            self._bm25 = BM25Okapi(tokenized)
            logger.info("BM25 index built: %d pages", len(paths))
        else:
            self._bm25 = None
            logger.info("BM25 index empty (no wiki pages yet)")

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Return top_k results for query, sorted by BM25 score descending."""
        if self._bm25 is None or not self._paths:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        results: list[SearchResult] = []
        for idx, score in ranked:
            if score <= 0:
                continue
            path = self._paths[idx]
            content = self._contents[idx]
            title = _extract_title(content, fallback=Path(path).stem.replace("-", " ").title())
            results.append(
                SearchResult(
                    path=path,
                    title=title,
                    snippet=_snippet(content),
                    score=round(float(score), 3),
                )
            )
        return results
