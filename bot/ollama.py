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
        else:
            i += 1
    return files


def extract_json(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object found in text."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for idx in range(start, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : idx + 1])
                except json.JSONDecodeError:
                    return None
    return None


# ── Client ────────────────────────────────────────────────────────────────────


class OllamaClient:
    """Thin async-compatible wrapper around the Ollama HTTP API."""

    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.Client(timeout=300.0)

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
                logger.info("Ollama is ready.")
                return
            logger.info("  attempt %d/%d — not ready yet, retrying in %.0fs", attempt + 1, retries, delay)
            time.sleep(delay)
        raise RuntimeError(f"Ollama not available at {self.base_url} after {retries} attempts")

    # ── Model management ──────────────────────────────────────────────────────

    def list_local_models(self) -> list[str]:
        r = self._client.get(f"{self.base_url}/api/tags")
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    def model_is_pulled(self) -> bool:
        try:
            models = self.list_local_models()
            # Match exact name or name without tag
            return any(
                m == self.model or m.split(":")[0] == self.model.split(":")[0]
                for m in models
            )
        except Exception:
            return False

    def pull_model(self, progress_callback=None) -> None:
        """
        Pull the model from Ollama registry.
        Calls progress_callback(status_str) periodically if provided.
        """
        logger.info("Pulling model %s …", self.model)
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
                if progress_callback:
                    progress_callback(status)
                if data.get("error"):
                    raise RuntimeError(f"Pull error: {data['error']}")
        logger.info("Model %s ready.", self.model)

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
                    yield chunk
                if data.get("done"):
                    break

    def chat(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
    ) -> str:
        """Non-streaming chat — returns full response string."""
        return "".join(self.chat_stream(system_prompt, messages))
