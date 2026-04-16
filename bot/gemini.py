"""
Google Gemini API wrapper with streaming support.
Replaces the Ollama client — uses the google-genai SDK.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from typing import Any

from google import genai
from google.genai import types

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


class GeminiClient:
    """Wrapper around the Google Gemini API (google-genai SDK)."""

    def __init__(self, api_key: str, model: str) -> None:
        self.model = model
        self._client = genai.Client(api_key=api_key)
        logger.info("GeminiClient initialised — model=%s", model)

    # ── Availability ──────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        try:
            # Quick check: list models to verify API key works
            self._client.models.list(config={"page_size": 1})
            return True
        except Exception:
            return False

    def wait_until_ready(self, retries: int = 5, delay: float = 3.0) -> None:
        """Verify the Gemini API is reachable or raise RuntimeError."""
        logger.info("Checking Gemini API availability …")
        for attempt in range(retries):
            if self.is_available():
                logger.info("Gemini API is ready after %d attempt(s).", attempt + 1)
                return
            logger.info("  attempt %d/%d — not ready yet, retrying in %.0fs", attempt + 1, retries, delay)
            time.sleep(delay)
        raise RuntimeError(f"Gemini API not available after {retries} attempts — check your API key")

    # ── Chat ──────────────────────────────────────────────────────────────────

    def chat_stream(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
    ) -> Iterator[str]:
        """
        Stream chat tokens from Gemini.
        Yields text chunks as they arrive.
        """
        prompt_chars = len(system_prompt) + sum(len(m.get("content", "")) for m in messages)
        prompt_tokens_est = prompt_chars // 4
        logger.info(
            "▶ gemini.chat_stream  model=%s  prompt≈%d tokens (%d chars)",
            self.model, prompt_tokens_est, prompt_chars,
        )
        t0 = time.time()
        total_chunks = 0
        total_chars = 0

        # Build contents list for Gemini
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            # Gemini uses "user" and "model" roles (not "assistant")
            if role == "assistant":
                role = "model"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=msg["content"])],
                )
            )

        response = self._client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.7,
                max_output_tokens=8192,
            ),
        )

        for chunk in response:
            if chunk.text:
                total_chunks += 1
                total_chars += len(chunk.text)
                yield chunk.text

        elapsed = time.time() - t0
        logger.info(
            "✓ gemini.chat_stream done  elapsed=%.1fs  "
            "prompt_tokens≈%d  response_chars=%d  chunks=%d",
            elapsed, prompt_tokens_est, total_chars, total_chunks,
        )

    def chat(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
    ) -> str:
        """Non-streaming chat — returns full response string."""
        logger.info("▶ gemini.chat (non-streaming)  model=%s", self.model)
        t0 = time.time()
        result = "".join(self.chat_stream(system_prompt, messages))
        elapsed = time.time() - t0
        logger.info(
            "✓ gemini.chat complete  elapsed=%.1fs  response_chars=%d  response_lines=%d",
            elapsed, len(result), result.count("\n"),
        )
        return result
