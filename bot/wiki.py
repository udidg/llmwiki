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
from typing import TYPE_CHECKING, Any

from gemini import GeminiClient, extract_json, parse_file_blocks

if TYPE_CHECKING:
    from search import WikiSearch

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

    def __init__(self, data_dir: str, llm: GeminiClient, search: "WikiSearch | None" = None) -> None:
        self.data = Path(data_dir)
        self.raw = self.data / "raw"
        self.wiki = self.data / "wiki"
        self.llm = llm
        self.search = search
        self._ensure_dirs()

    # ── Setup ─────────────────────────────────────────────────────────────────

    # Path where the Docker image bundles AGENTS.md (copied by CI)
    _BUNDLED_AGENTS = Path("/app/AGENTS.md.bundled")

    def _ensure_dirs(self) -> None:
        for sub in ["articles", "journals", "podcasts", "assets"]:
            (self.raw / sub).mkdir(parents=True, exist_ok=True)
        for sub in ["sources", "people", "concepts", "insights"]:
            (self.wiki / sub).mkdir(parents=True, exist_ok=True)

        # Sync bundled AGENTS.md to the data volume (always overwrite —
        # AGENTS.md is code/config, not user data, so it should stay
        # in sync with the deployed image version)
        agents_dest = self.data / "AGENTS.md"
        if self._BUNDLED_AGENTS.exists():
            import shutil
            shutil.copy2(self._BUNDLED_AGENTS, agents_dest)
            logger.info("synced AGENTS.md from bundled image → %s", agents_dest)

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

        feedback = self.wiki / "feedback.md"
        if not feedback.exists():
            feedback.write_text(
                "# Query Feedback\n\n"
                "Positively-rated Q&A pairs (user reacted with 👍/👌/❤️/🔥).\n"
                "The LLM uses these as examples of good answers.\n\n---\n"
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

    # ── Feedback ──────────────────────────────────────────────────────────────

    def append_feedback(self, question: str, answer_snippet: str) -> None:
        """Append a positively-rated Q&A pair to the feedback log."""
        feedback = self.wiki / "feedback.md"
        snippet = answer_snippet[:300].replace("\n", " ").strip()
        entry = (
            f"\n## [{today()}] ✅ Good answer\n"
            f"**Q:** {question}\n"
            f"**A:** {snippet}\n"
        )
        existing = feedback.read_text(encoding="utf-8")
        feedback.write_text(existing + entry, encoding="utf-8")
        logger.info("append_feedback: logged positive feedback for question=%r", question[:60])

    def _load_recent_feedback(self, n: int = 5) -> str:
        """Load the last n positively-rated Q&A pairs for LLM context."""
        feedback = self.wiki / "feedback.md"
        if not feedback.exists():
            return ""
        content = feedback.read_text(encoding="utf-8")
        # Parse feedback entries (each starts with "## [")
        entries = re.split(r"(?=\n## \[)", content)
        # Filter to actual entries (skip the header)
        entries = [e.strip() for e in entries if e.strip().startswith("## [")]
        if not entries:
            return ""
        recent = entries[-n:]
        logger.info("  loaded %d recent feedback example(s)", len(recent))
        return "\n\n".join(recent)

    # ── Context assembly ──────────────────────────────────────────────────────

    def _base_context(self) -> str:
        """Always-included context: index + recent log."""
        index = self.read_page("wiki/index.md")
        log_tail = "\n".join(self.last_log_entries(10))
        return f"## Current Wiki Index\n\n{index}\n\n## Recent Log\n\n{log_tail}"

    def _expand_query(self, question: str) -> list[str]:
        """Use the LLM to expand a user question into multiple search queries.

        This bridges the vocabulary gap between how users ask questions
        and how wiki pages are written. For example:
          "What helps me fall asleep faster?"
          → ["sleep onset", "falling asleep", "sleep hygiene", "insomnia", "melatonin"]

        Returns a list of 3-5 search query strings (always includes the original).
        """
        logger.info("  _expand_query: expanding question=%r", question[:80])

        system = (
            "You are a search query expander for a personal knowledge base. "
            "Given a user's question, generate 3-5 alternative search queries that "
            "would help find relevant wiki pages. Think about:\n"
            "- Synonyms and related terms\n"
            "- Technical/scientific terms for colloquial language\n"
            "- Specific concepts the question might relate to\n"
            "- Key entities (people, topics) mentioned or implied\n\n"
            "Reply with ONLY a JSON array of strings. No explanation.\n"
            'Example: ["sleep onset latency", "falling asleep tips", "melatonin dosage", "sleep hygiene"]'
        )
        messages = [{"role": "user", "content": question}]

        try:
            response = self.llm.chat(system, messages).strip()
            logger.info("  _expand_query: LLM response=%r", response[:200])

            # Parse JSON array from response
            import json
            # Try to find a JSON array in the response
            start = response.find("[")
            end = response.rfind("]")
            if start != -1 and end != -1:
                queries = json.loads(response[start:end + 1])
                if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
                    # Always include the original question as the first query
                    result = [question] + [q for q in queries if q.lower() != question.lower()]
                    logger.info("  _expand_query: expanded to %d queries: %s", len(result), result)
                    return result

            logger.warning("  _expand_query: could not parse LLM response — using original query only")
        except Exception as e:
            logger.warning("  _expand_query: LLM call failed (%s) — using original query only", e)

        return [question]

    def _load_pages_by_search(self, query: str) -> tuple[str, list[str]]:
        """Use LLM query expansion + BM25 search to find and load relevant wiki pages.

        1. Expands the user's question into multiple search queries via LLM
        2. Runs BM25 search for each expanded query
        3. Merges results, keeping the highest score per page
        4. Returns formatted context and list of paths consulted

        Returns (formatted_context, list_of_paths_consulted).
        """
        if self.search is None:
            logger.warning("  _load_pages_by_search: no WikiSearch instance — falling back to tag match")
            return self._load_pages_by_tags_legacy(query.split()), []

        from search import SearchResultWithContent

        # Step 1: Expand the query into multiple search terms
        expanded_queries = self._expand_query(query)

        # Step 2: Run BM25 for each query and merge results
        best_by_path: dict[str, SearchResultWithContent] = {}

        for q in expanded_queries:
            results = self.search.search_with_content(q, top_k=5, min_score=0.5, fallback_k=0)
            for r in results:
                existing = best_by_path.get(r.path)
                if existing is None or r.score > existing.score:
                    best_by_path[r.path] = r

        logger.info(
            "  _load_pages_by_search: %d unique page(s) found across %d expanded queries",
            len(best_by_path), len(expanded_queries),
        )

        # Step 3: If nothing found via expansion, try the original query with fallback
        if not best_by_path:
            results = self.search.search_with_content(query, top_k=8, min_score=1.0, fallback_k=3)
            for r in results:
                best_by_path[r.path] = r

        if not best_by_path:
            logger.info("  _load_pages_by_search: no results for query=%r", query[:60])
            return "", []

        # Step 4: Sort by score descending, cap at 8 pages
        sorted_results = sorted(best_by_path.values(), key=lambda r: r.score, reverse=True)[:8]

        loaded: list[str] = []
        paths: list[str] = []
        for r in sorted_results:
            loaded.append(
                f"### wiki/{r.path}  (relevance score: {r.score})\n\n{r.content}"
            )
            paths.append(f"wiki/{r.path}")

        logger.info(
            "  _load_pages_by_search: loaded %d page(s) for query=%r: %s",
            len(paths), query[:60], paths,
        )
        return "\n\n---\n\n".join(loaded), paths

    def _load_pages_by_tags_legacy(self, tags: list[str]) -> str:
        """Legacy fallback: load wiki pages whose content contains any of the given tags."""
        loaded: list[str] = []
        matched_paths: list[str] = []
        for p in self.wiki.rglob("*.md"):
            if p.name in ("index.md", "log.md", "feedback.md"):
                continue
            content = p.read_text(encoding="utf-8")
            if any(tag.lower() in content.lower() for tag in tags):
                rel = str(p.relative_to(self.data))
                loaded.append(f"## {rel}\n\n{content}")
                matched_paths.append(rel)
        if matched_paths:
            logger.info("  loaded %d relevant page(s) (legacy tag match): %s", len(matched_paths), matched_paths)
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
        logger.info("  _execute_file_writes: response length = %d chars", len(llm_response))
        logger.info(
            "  _execute_file_writes: 'FILE:' appears %d time(s), 'END_FILE' appears %d time(s), "
            "code fences: %d",
            llm_response.count("FILE:"), llm_response.count("END_FILE"),
            llm_response.count("```"),
        )

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
        """Answer a question using the wiki as context.

        Uses BM25 search to find the most relevant pages, loads their full
        content into the LLM prompt (ranked by relevance score), and includes
        recent positive feedback examples so the LLM learns the user's
        preferred answer style.
        """
        logger.info("━━ query START  question=%r", question[:120])

        system = self._system_prompt()
        context = self._base_context()

        # BM25 search for relevant pages (replaces naive keyword matching)
        logger.info("  running BM25 search for relevant wiki pages …")
        relevant, searched_paths = self._load_pages_by_search(question)
        relevant_chars = len(relevant)
        logger.info(
            "  relevant context: %d chars, %d page(s)%s",
            relevant_chars, len(searched_paths),
            " (empty — no matching pages)" if not relevant_chars else "",
        )

        # Load recent positive feedback examples
        feedback = self._load_recent_feedback(n=5)
        feedback_section = ""
        if feedback:
            feedback_section = (
                f"## Examples of Answers the User Liked\n\n"
                f"The following Q&A pairs were positively rated by the user. "
                f"Use them as style and quality examples.\n\n{feedback}\n\n---\n\n"
            )

        user_msg = (
            f"{context}\n\n"
            + (f"## Relevant Wiki Pages (BM25-ranked, highest relevance first)\n\n"
               f"These pages were pre-searched and ranked by relevance to the question. "
               f"Prioritize higher-scored pages but consider all provided context.\n\n"
               f"{relevant}\n\n---\n\n" if relevant else
               "## Relevant Wiki Pages\n\nNo wiki pages matched this query.\n\n---\n\n")
            + feedback_section
            + f"## Task: Query\n\n"
            f"Today's date: {today()}\n\n"
            f"Question: {question}\n\n"
            f"Follow the Query Workflow from AGENTS.md. "
            f"Return the JSON object with your answer and sources_consulted. "
            f"Only cite pages that were actually provided above — do NOT invent sources."
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
        sources = data.get("sources_consulted", searched_paths if not data.get("sources_consulted") else data["sources_consulted"])
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
