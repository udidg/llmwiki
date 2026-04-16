"""
WikiManager — core wiki operations: ingest, query, lint, status.

The LLM (via GeminiClient) is the agent. This module is its "hands":
it assembles context, calls the LLM, then executes the file writes
the LLM decides on.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from gemini import GeminiClient, extract_json, parse_file_blocks

logger = logging.getLogger(__name__)

# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class IngestResult:
    slug: str
    title: str
    created: list[str]
    updated: list[str]
    summary: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    answer: str
    sources_consulted: list[str]
    save_as: str


@dataclass
class LintResult:
    contradictions: list[dict]
    orphans: list[str]
    missing_pages: list[dict]
    stale: list[dict]
    suggestions: list[str]


@dataclass
class StatusResult:
    total_pages: int
    sources: int
    people: int
    concepts: int
    insights: int
    last_log_entries: list[str]
    model: str


# ── Helpers ───────────────────────────────────────────────────────────────────


def slugify(text: str) -> str:
    """Convert text to a URL-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text[:80]


def today() -> str:
    return date.today().isoformat()


def now_slug() -> str:
    return datetime.now().strftime("%Y-%m-%d-%H-%M")


# ── WikiManager ───────────────────────────────────────────────────────────────


class WikiManager:
    """
    Orchestrates all wiki operations.

    data_dir/
      AGENTS.md          ← system prompt / schema
      raw/               ← immutable sources
      wiki/              ← LLM-maintained pages
        index.md
        log.md
        overview.md
        sources/
        people/
        concepts/
        insights/
    """

    def __init__(self, data_dir: str, llm: GeminiClient) -> None:
        self.data = Path(data_dir)
        self.raw = self.data / "raw"
        self.wiki = self.data / "wiki"
        self.llm = llm
        self._ensure_dirs()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _ensure_dirs(self) -> None:
        for sub in ["articles", "journals", "podcasts", "assets"]:
            (self.raw / sub).mkdir(parents=True, exist_ok=True)
        for sub in ["sources", "people", "concepts", "insights"]:
            (self.wiki / sub).mkdir(parents=True, exist_ok=True)

        # Initialize index and log if missing
        index = self.wiki / "index.md"
        if not index.exists():
            index.write_text(
                "# Wiki Index\n\nLast updated: —\nTotal pages: 0\n\n"
                "## Sources\n| Page | Summary | Date | Tags |\n|------|---------|------|------|\n\n"
                "## People\n| Page | Description |\n|------|-------------|\n\n"
                "## Concepts\n| Page | Description |\n|------|-------------|\n\n"
                "## Insights\n| Page | Description |\n|------|-------------|\n"
            )

        log = self.wiki / "log.md"
        if not log.exists():
            log.write_text(
                "# Wiki Log\n\nAppend-only chronological record of all operations.\n"
                "Format: `## [YYYY-MM-DD] <operation> | <title>`\n\n---\n"
            )

    # ── File I/O ──────────────────────────────────────────────────────────────

    def read_page(self, rel_path: str) -> str:
        """Read a wiki or data file. rel_path is relative to data_dir."""
        p = self.data / rel_path
        if p.exists():
            return p.read_text(encoding="utf-8")
        return ""

    def write_page(self, rel_path: str, content: str) -> None:
        """Write a wiki page. Creates parent dirs as needed."""
        p = self.data / rel_path
        existed = p.exists()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        action = "updated" if existed else "created"
        logger.info("  ✎ file %s → %s (%d chars, %d lines)", action, rel_path, len(content), content.count("\n"))

    def append_log(self, entry: str) -> None:
        log = self.wiki / "log.md"
        existing = log.read_text(encoding="utf-8")
        log.write_text(existing + "\n" + entry + "\n", encoding="utf-8")

    def list_pages(self) -> list[str]:
        """Return all wiki page paths relative to data_dir."""
        pages = []
        for p in self.wiki.rglob("*.md"):
            pages.append(str(p.relative_to(self.data)))
        return sorted(pages)

    def count_pages_by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {"sources": 0, "people": 0, "concepts": 0, "insights": 0, "other": 0}
        for p in self.wiki.rglob("*.md"):
            parts = p.relative_to(self.wiki).parts
            if parts[0] in counts:
                counts[parts[0]] += 1
            elif p.name not in ("index.md", "log.md", "overview.md"):
                counts["other"] += 1
        return counts

    def last_log_entries(self, n: int = 5) -> list[str]:
        log = self.wiki / "log.md"
        if not log.exists():
            return []
        lines = log.read_text(encoding="utf-8").splitlines()
        entries = [l for l in lines if l.startswith("## [")]
        return entries[-n:]

    # ── System prompt ─────────────────────────────────────────────────────────

    def _system_prompt(self) -> str:
        agents = self.data / "AGENTS.md"
        if agents.exists():
            return agents.read_text(encoding="utf-8")
        return "You are a wiki maintainer. Read sources and write structured markdown pages."

    # ── Context assembly ──────────────────────────────────────────────────────

    def _base_context(self) -> str:
        """Always-included context: index + recent log."""
        index = self.read_page("wiki/index.md")
        log_tail = "\n".join(self.last_log_entries(10))
        return f"## Current Wiki Index\n\n{index}\n\n## Recent Log\n\n{log_tail}"

    def _load_pages_by_tags(self, tags: list[str]) -> str:
        """Load wiki pages whose frontmatter contains any of the given tags."""
        loaded: list[str] = []
        matched_paths: list[str] = []
        for p in self.wiki.rglob("*.md"):
            if p.name in ("index.md", "log.md"):
                continue
            content = p.read_text(encoding="utf-8")
            if any(tag.lower() in content.lower() for tag in tags):
                rel = str(p.relative_to(self.data))
                loaded.append(f"## {rel}\n\n{content}")
                matched_paths.append(rel)
        if matched_paths:
            logger.info("  loaded %d relevant page(s): %s", len(matched_paths), matched_paths)
        else:
            logger.info("  no wiki pages matched tags: %s", tags)
        return "\n\n---\n\n".join(loaded)

    def _load_all_wiki_pages(self) -> str:
        """Load all wiki pages (for lint)."""
        pages: list[str] = []
        paths: list[str] = []
        for p in self.wiki.rglob("*.md"):
            if p.name == "log.md":
                continue
            rel = str(p.relative_to(self.data))
            pages.append(f"## {rel}\n\n{p.read_text(encoding='utf-8')}")
            paths.append(rel)
        logger.info("  loaded %d wiki page(s) for lint: %s", len(paths), paths)
        return "\n\n---\n\n".join(pages)

    # ── Execute LLM file writes ───────────────────────────────────────────────

    def _execute_file_writes(self, llm_response: str) -> tuple[list[str], list[str]]:
        """
        Parse FILE: blocks from LLM response and write them to disk.
        Returns (created_paths, updated_paths).
        """
        file_blocks = parse_file_blocks(llm_response)
        created: list[str] = []
        updated: list[str] = []

        if not file_blocks:
            logger.info("  _execute_file_writes: no FILE: blocks in LLM response — nothing to write")
            return created, updated

        logger.info("  _execute_file_writes: processing %d file block(s)", len(file_blocks))

        # Known wiki subdirectories and root files — used to auto-fix missing wiki/ prefix
        _WIKI_SUBDIRS = ("sources/", "people/", "concepts/", "insights/")
        _WIKI_ROOT_FILES = ("index.md", "log.md", "overview.md")

        for rel_path, content in file_blocks.items():
            # Auto-fix: if LLM omitted the wiki/ prefix but path starts with
            # a known wiki subdirectory or is a known root file, prepend it.
            if not rel_path.startswith("wiki/") and (
                any(rel_path.startswith(sub) for sub in _WIKI_SUBDIRS)
                or rel_path in _WIKI_ROOT_FILES
            ):
                fixed = f"wiki/{rel_path}"
                logger.info(
                    "  🔧 auto-fix: LLM omitted wiki/ prefix: '%s' → '%s'",
                    rel_path, fixed,
                )
                rel_path = fixed

            # Security: only allow writes inside wiki/
            if not rel_path.startswith("wiki/"):
                logger.warning(
                    "  ⛔ SECURITY: LLM tried to write outside wiki/: '%s' — skipped",
                    rel_path,
                )
                continue
            full = self.data / rel_path
            existed = full.exists()
            self.write_page(rel_path, content)
            if existed:
                updated.append(rel_path)
                logger.info("  ✏️  updated: %s", rel_path)
            else:
                created.append(rel_path)
                logger.info("  📄 created: %s", rel_path)

        logger.info(
            "  _execute_file_writes done — created=%d  updated=%d",
            len(created), len(updated),
        )
        return created, updated

    # ── Operations ────────────────────────────────────────────────────────────

    def ingest(self, source_content: str, source_type: str, filename: str) -> IngestResult:
        """
        Ingest a source document.
        source_content: full text of the source
        source_type: 'journal' | 'article' | 'podcast' | 'note'
        filename: original filename (used for slug hint)
        """
        word_count = len(source_content.split())
        logger.info(
            "━━ ingest START  type=%s  filename=%s  source_words=%d",
            source_type, filename, word_count,
        )

        system = self._system_prompt()
        context = self._base_context()

        logger.info("  assembling prompt (system=%d chars, context=%d chars, source=%d chars)",
                    len(system), len(context), len(source_content))

        user_msg = (
            f"{context}\n\n"
            f"---\n\n"
            f"## Task: Ingest\n\n"
            f"Source type: {source_type}\n"
            f"Original filename: {filename}\n"
            f"Today's date: {today()}\n\n"
            f"## Source Content\n\n{source_content}\n\n"
            f"---\n\n"
            f"Follow the Ingest Workflow from AGENTS.md exactly. "
            f"Return the JSON summary and all FILE: blocks."
        )

        messages = [{"role": "user", "content": user_msg}]

        logger.info("  → sending ingest prompt to Gemini …")
        response = self.llm.chat(system, messages)
        logger.info("  ← Gemini response received (%d chars)", len(response))

        # Execute file writes
        logger.info("  executing file writes from LLM response …")
        created, updated = self._execute_file_writes(response)

        # Parse JSON summary
        logger.info("  parsing JSON summary from response …")
        data = extract_json(response) or {}
        if data:
            logger.info("  JSON summary keys: %s", list(data.keys()))
        else:
            logger.warning("  no JSON summary found in response — using fallback values")

        result = IngestResult(
            slug=data.get("slug", slugify(filename)),
            title=data.get("title", filename),
            created=data.get("created", created),
            updated=data.get("updated", updated),
            summary=data.get("summary", ""),
            raw=data,
        )

        # Append to log
        log_entry = (
            f"## [{today()}] ingest | {result.title}\n"
            + "".join(f"- Created: {p}\n" for p in result.created)
            + "".join(f"- Updated: {p}\n" for p in result.updated)
        )
        self.append_log(log_entry)

        logger.info(
            "━━ ingest DONE  title=%r  created=%d  updated=%d  summary=%r",
            result.title, len(result.created), len(result.updated),
            (result.summary[:80] + "…") if len(result.summary) > 80 else result.summary,
        )
        return result

    def query(self, question: str) -> QueryResult:
        """Answer a question using the wiki as context."""
        logger.info("━━ query START  question=%r", question[:120])

        system = self._system_prompt()
        context = self._base_context()

        # Load pages likely relevant to the question (simple keyword match)
        keywords = [w for w in question.lower().split() if len(w) > 3]
        logger.info("  extracted %d keyword(s) for page matching: %s", len(keywords), keywords)

        relevant = self._load_pages_by_tags(keywords) if keywords else ""
        relevant_chars = len(relevant)
        logger.info(
            "  relevant context: %d chars%s",
            relevant_chars,
            " (empty — no matching pages)" if not relevant_chars else "",
        )

        user_msg = (
            f"{context}\n\n"
            + (f"## Relevant Wiki Pages\n\n{relevant}\n\n---\n\n" if relevant else "")
            + f"## Task: Query\n\n"
            f"Today's date: {today()}\n\n"
            f"Question: {question}\n\n"
            f"Follow the Query Workflow from AGENTS.md. "
            f"Return the JSON object with your answer and sources_consulted."
        )

        total_prompt_chars = len(system) + len(user_msg)
        logger.info(
            "  → sending query prompt to Gemini  total_prompt=%d chars (~%d tokens) …",
            total_prompt_chars, total_prompt_chars // 4,
        )

        messages = [{"role": "user", "content": user_msg}]
        response = self.llm.chat(system, messages)
        logger.info("  ← Gemini response received (%d chars)", len(response))

        # If LLM also wrote any insight pages, execute them
        logger.info("  checking for any FILE: blocks in query response …")
        self._execute_file_writes(response)

        logger.info("  parsing JSON answer from response …")
        data = extract_json(response) or {}
        answer = data.get("answer", response)
        sources = data.get("sources_consulted", [])
        save_as = data.get("save_as", "")

        logger.info(
            "  query result — sources=%s  save_as=%r  answer_chars=%d",
            sources, save_as, len(answer),
        )

        # Log the query
        self.append_log(
            f"## [{today()}] query | {question[:60]}\n"
            f"- Pages consulted: {', '.join(sources) or 'none'}\n"
        )

        logger.info("━━ query DONE")
        return QueryResult(answer=answer, sources_consulted=sources, save_as=save_as)

    def lint(self) -> LintResult:
        """Health-check the wiki."""
        logger.info("━━ lint START")

        system = self._system_prompt()

        logger.info("  loading all wiki pages …")
        all_pages = self._load_all_wiki_pages()
        logger.info("  total wiki content for lint: %d chars", len(all_pages))

        user_msg = (
            f"## Task: Lint\n\n"
            f"Today's date: {today()}\n\n"
            f"## All Wiki Pages\n\n{all_pages}\n\n"
            f"---\n\n"
            f"Follow the Lint Workflow from AGENTS.md. "
            f"Return the JSON lint report."
        )

        logger.info("  → sending lint prompt to Gemini …")
        messages = [{"role": "user", "content": user_msg}]
        response = self.llm.chat(system, messages)
        logger.info("  ← Gemini response received (%d chars)", len(response))

        logger.info("  parsing JSON lint report …")
        data = extract_json(response) or {}

        result = LintResult(
            contradictions=data.get("contradictions", []),
            orphans=data.get("orphans", []),
            missing_pages=data.get("missing_pages", []),
            stale=data.get("stale", []),
            suggestions=data.get("suggestions", []),
        )

        logger.info(
            "━━ lint DONE — contradictions=%d  orphans=%d  missing_pages=%d  stale=%d  suggestions=%d",
            len(result.contradictions), len(result.orphans),
            len(result.missing_pages), len(result.stale), len(result.suggestions),
        )

        self.append_log(f"## [{today()}] lint | Wiki health check\n")
        return result

    def save_insight(self, slug: str, content: str) -> str:
        """Save a query answer as an insight page."""
        rel_path = f"wiki/insights/{slug}.md"
        logger.info("save_insight: writing %s (%d chars)", rel_path, len(content))
        self.write_page(rel_path, content)
        self.append_log(f"## [{today()}] query | Saved insight: {slug}\n- Created: {rel_path}\n")
        logger.info("save_insight: done → %s", rel_path)
        return rel_path

    def get_status(self, model: str) -> StatusResult:
        counts = self.count_pages_by_type()
        total = sum(counts.values())
        logger.info("get_status: total=%d  by_type=%s", total, counts)
        return StatusResult(
            total_pages=total,
            sources=counts["sources"],
            people=counts["people"],
            concepts=counts["concepts"],
            insights=counts["insights"],
            last_log_entries=self.last_log_entries(5),
            model=model,
        )

    def read_index(self) -> str:
        return self.read_page("wiki/index.md")

    def save_raw(self, content: str, subdir: str, filename: str) -> Path:
        """Save raw source content to data/raw/<subdir>/<filename>."""
        dest = self.raw / subdir / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
        logger.info("save_raw: wrote %s (%d chars)", dest, len(content))
        return dest
