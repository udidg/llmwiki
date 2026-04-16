"""
Ollama HTTP API wrapper with streaming support.
Handles connection retries, model pulling, and token streaming.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ── Response parsing ──────────────────────────────────────────────────────────

FILE_BLOCK_START = "FILE:"
FILE_BLOCK_SEP = "---"
FILE_BLOCK_END = "END_FILE"


def parse_file_blocks(text: str) -> dict[str, str]:
    """
    Parse LLM output for embedded file blocks in the format:

        FILE: wiki/sources/slug.md
        ---
        <content>
        ---
        END_FILE

    Returns a dict mapping path → content.
    """
    files: dict[str, str] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith(FILE_BLOCK_START):
            path = line[len(FILE_BLOCK_START):].strip()
            # skip separator
            i += 1
            if i < len(lines) and lines[i].strip() == FILE_BLOCK_SEP:
                i += 1
            content_lines: list[str] = []
            while i < len(lines):
                if lines[i].strip() == FILE_BLOCK_SEP and i + 1 < len(lines) and lines[i + 1].strip() == FILE_BLOCK_END:
                    i += 2  # skip --- and END_FILE
                    break
                if lines[i].strip() == FILE_BLOCK_END:
                    i += 1
                    break
                content_lines.append(lines[i])
                i += 1
            files[path] = "\n".join(content_lines)
            logger.debug("  parsed FILE block → %s (%d lines)", path, len(content_lines))
        else:
            i += 1

    if files:
        logger.info("parse_file_blocks: found %d file block(s): %s", len(files), list(files.keys()))
    else:
        logger.debug("parse_file_blocks: no FILE: blocks found in response")

    return files


def extract_json(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object found in text."""
    start = text.find("{")
    if start == -1:
        logger.debug("extract_json: no JSON object found in response")
        return None
    depth = 0
    for idx in range(start, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                try:
                    result = json.loads(text[start : idx + 1])
                    logger.debug("extract_json: parsed JSON with keys: %s", list(result.keys()))
                    return result
                except json.JSONDecodeError as e:
                    logger.warning("extract_json: JSON parse error: %s", e)
                    return None
    logger.warning("extract_json: unbalanced braces — could not extract JSON")
    return None


# ── Client ────────────────────────────────────────────────────────────────────


class OllamaClient:
    """Thin async-compatible wrapper around the Ollama HTTP API."""

    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.Client(timeout=300.0)
        logger.info("OllamaClient initialised — base_url=%s  model=%s", base_url, model)

    # ── Availability ──────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        try:
            r = self._client.get(f"{self.base_url}/api/tags", timeout=5.0)
            return r.status_code == 200
        except Exception:
            return False

    def wait_until_ready(self, retries: int = 30, delay: float = 5.0) -> None:
        """Block until Ollama is reachable or raise RuntimeError."""
        logger.info("Waiting for Ollama at %s …", self.base_url)
        for attempt in range(retries):
            if self.is_available():
                logger.info("Ollama is ready after %d attempt(s).", attempt + 1)
                return
            logger.info("  attempt %d/%d — not ready yet, retrying in %.0fs", attempt + 1, retries, delay)
            time.sleep(delay)
        raise RuntimeError(f"Ollama not available at {self.base_url} after {retries} attempts")

    # ── Model management ──────────────────────────────────────────────────────

    def list_local_models(self) -> list[str]:
        r = self._client.get(f"{self.base_url}/api/tags")
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        logger.debug("list_local_models: %s", models)
        return models

    def model_is_pulled(self) -> bool:
        try:
            models = self.list_local_models()
            found = any(
                m == self.model or m.split(":")[0] == self.model.split(":")[0]
                for m in models
            )
            logger.info(
                "model_is_pulled: model=%s  found=%s  available_models=%s",
                self.model, found, models,
            )
            return found
        except Exception as e:
            logger.warning("model_is_pulled: check failed (%s) — assuming not pulled", e)
            return False

    def pull_model(self, progress_callback=None) -> None:
        """
        Pull the model from Ollama registry.
        Calls progress_callback(status_str) periodically if provided.
        """
        logger.info("▶ Pulling model %s from Ollama registry …", self.model)
        t0 = time.time()
        last_status: str = ""
        bytes_total: int = 0
        bytes_done: int = 0

        with self._client.stream(
            "POST",
            f"{self.base_url}/api/pull",
            json={"name": self.model},
            timeout=3600.0,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                status = data.get("status", "")
                total = data.get("total", 0)
                completed = data.get("completed", 0)

                if total:
                    bytes_total = total
                if completed:
                    bytes_done = completed

                # Log meaningful status changes
                if status != last_status:
                    if bytes_total:
                        pct = (bytes_done / bytes_total * 100) if bytes_total else 0
                        logger.info(
                            "  pull [%s] %.1f%% (%s / %s)",
                            status,
                            pct,
                            _fmt_bytes(bytes_done),
                            _fmt_bytes(bytes_total),
                        )
                    else:
                        logger.info("  pull [%s]", status)
                    last_status = status

                if progress_callback:
                    progress_callback(status)
                if data.get("error"):
                    raise RuntimeError(f"Pull error: {data['error']}")

        elapsed = time.time() - t0
        logger.info(
            "✓ Model %s pull complete in %.1fs (downloaded ~%s)",
            self.model, elapsed, _fmt_bytes(bytes_total),
        )

    # ── Chat ──────────────────────────────────────────────────────────────────

    def chat_stream(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
    ) -> Iterator[str]:
        """
        Stream chat tokens from Ollama.
        Yields text chunks as they arrive.
        """
        prompt_chars = len(system_prompt) + sum(len(m.get("content", "")) for m in messages)
        prompt_tokens_est = prompt_chars // 4  # rough estimate: ~4 chars/token
        logger.info(
            "▶ ollama.chat_stream  model=%s  prompt≈%d tokens (%d chars)",
            self.model, prompt_tokens_est, prompt_chars,
        )
        t0 = time.time()
        total_chunks = 0
        total_chars = 0

        payload = {
            "model": self.model,
            "stream": True,
            "messages": [{"role": "system", "content": system_prompt}] + messages,
        }
        with self._client.stream(
            "POST",
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=300.0,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                chunk = data.get("message", {}).get("content", "")
                if chunk:
                    total_chunks += 1
                    total_chars += len(chunk)
                    yield chunk
                if data.get("done"):
                    eval_count = data.get("eval_count", 0)
                    prompt_eval_count = data.get("prompt_eval_count", 0)
                    elapsed = time.time() - t0
                    logger.info(
                        "✓ ollama.chat_stream done  elapsed=%.1fs  "
                        "prompt_tokens=%s  response_tokens=%s  response_chars=%d",
                        elapsed,
                        prompt_eval_count or f"~{prompt_tokens_est}",
                        eval_count or total_chunks,
                        total_chars,
                    )
                    break

    def chat(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
    ) -> str:
        """Non-streaming chat — returns full response string."""
        logger.info("▶ ollama.chat (non-streaming)  model=%s", self.model)
        t0 = time.time()
        result = "".join(self.chat_stream(system_prompt, messages))
        elapsed = time.time() - t0
        logger.info(
            "✓ ollama.chat complete  elapsed=%.1fs  response_chars=%d  response_lines=%d",
            elapsed, len(result), result.count("\n"),
        )
        return result


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
