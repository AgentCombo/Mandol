"""Application-layer retry logic for LLM JSON response parsing.

When the LLM returns a response that cannot be parsed as valid JSON
(e.g., wrapped in markdown code fences, truncated, or malformed),
this module provides retry logic to re-prompt the LLM before falling
back to degraded results.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Callable, List, Optional, TypeVar

from ..ports.llm_provider import ChatMessage, LLMProvider

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_DELAY = 1.0

# Regex patterns for JSON cleaning
# Single-line comments: // ... or # ...
_RE_SINGLE_LINE_COMMENT = re.compile(r"(?<!:)//.*$|(?<!:)#.*$", re.MULTILINE)
# Multi-line comments: /* ... */
_RE_MULTI_LINE_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
# Trailing commas before ] or }
_RE_TRAILING_COMMA = re.compile(r",(\s*[}\]])")
# Markdown json fences
_RE_MD_FENCE = re.compile(r"```(?:json|JSON)?\s*")
# Locate the first JSON object or array
_RE_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_RE_JSON_ARRAY = re.compile(r"\[.*\]", re.DOTALL)


def strip_json_fences(text: str) -> str:
    """Remove markdown fences, comments, and non-JSON preamble/suffix.

    Handles:
    - Markdown code fences (```json / ```JSON / ```)
    - JavaScript single-line comments (//)
    - Python single-line comments (#)
    - Multi-line comments (/* */)
    - Trailing commas before ] or }
    - Preamble text before the JSON object/array
    - Suffix text after the JSON object/array
    """
    if not text or not text.strip():
        return text

    text = text.strip()

    # Step 1: Strip markdown fences
    text = _RE_MD_FENCE.sub("", text)
    # Remove trailing ``` (including those after JSON)
    text = text.replace("```", "")

    # Step 2: Extract the JSON object or array from surrounding text
    # Try to find the outermost {} or []
    obj_match = _RE_JSON_OBJECT.search(text)
    arr_match = _RE_JSON_ARRAY.search(text)

    extracted = None
    if obj_match and arr_match:
        # Use whichever starts first
        if obj_match.start() <= arr_match.start():
            extracted = obj_match.group(0)
        else:
            extracted = arr_match.group(0)
    elif obj_match:
        extracted = obj_match.group(0)
    elif arr_match:
        extracted = arr_match.group(0)

    if extracted is not None:
        text = extracted

    # Step 3: Remove multi-line comments first (before single-line,
    # so that // inside /* */ doesn't interfere)
    text = _RE_MULTI_LINE_COMMENT.sub("", text)

    # Step 4: Remove single-line comments (// and #)
    # Process line by line to avoid removing // inside strings
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        # Simple state machine to avoid removing comments inside strings
        cleaned = _remove_comments_from_line(line)
        cleaned_lines.append(cleaned)
    text = "\n".join(cleaned_lines)

    # Step 5: Remove trailing commas (JSON spec doesn't allow them)
    text = _RE_TRAILING_COMMA.sub(r"\1", text)

    # Step 6: Collapse multiple blank lines
    text = re.sub(r"\n\s*\n", "\n", text)

    return text.strip()


def _remove_comments_from_line(line: str) -> str:
    """Remove trailing comments from a single line, respecting string context.

    Handles:
    - ``// comment`` (not inside a string)
    - ``# comment`` (not inside a string)
    - ``"key": "value", // this is a comment"``  → strips the comment
    - ``"key": "value // not a comment"``  → preserves the // inside the string
    """
    result: List[str] = []
    i = 0
    in_double = False
    in_single = False

    while i < len(line):
        ch = line[i]

        if ch == "\\" and (in_double or in_single) and i + 1 < len(line):
            # Escaped character inside string — skip both chars
            result.append(ch)
            result.append(line[i + 1])
            i += 2
            continue

        if ch == '"' and not in_single:
            in_double = not in_double
            result.append(ch)
            i += 1
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            result.append(ch)
            i += 1
            continue

        if not in_double and not in_single:
            # Check for // comment
            if ch == "/" and i + 1 < len(line) and line[i + 1] == "/":
                # Found // comment outside string — stop here
                trailing = line[i:].rstrip()
                if trailing != "//":
                    # Keep the comment as part of the result if it might be meaningful
                    # but trim trailing whitespace from the result
                    pass
                break
            # Check for # comment (Python style)
            if ch == "#":
                # Make sure it's not part of a URL or something
                prev_char = line[i - 1] if i > 0 else " "
                if prev_char in (" ", ",", ";", ":", ""):
                    break

        result.append(ch)
        i += 1

    return "".join(result)


def retry_llm_json_call(
    llm: LLMProvider,
    messages: List[ChatMessage],
    parse_fn: Callable[[str], T],
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
    temperature: float = 0.1,
    max_tokens: int = 2048,
    response_format: Optional[dict] = None,
    context_label: str = "",
) -> T:
    total_chars = sum(len(str(m.get("content", ""))) for m in messages)
    est_tokens = total_chars // 3
    logger.info(
        "LLM call [%s]: prompt_size=%d chars (~%d tokens), output_max=%d tokens.",
        context_label, total_chars, est_tokens, max_tokens,
    )
    for i, m in enumerate(messages):
        role = m.get("role", "unknown")
        content = str(m.get("content", ""))
        logger.info(
            "LLM call [%s] message[%d] role=%s:\n%s",
            context_label, i, role, content,
        )

    last_error: Optional[json.JSONDecodeError] = None

    for attempt in range(max_retries + 1):
        response = llm.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )

        logger.info(
            "LLM response [%s] attempt=%d: usage=%s, content:\n%s",
            context_label, attempt + 1,
            response.usage if hasattr(response, "usage") else "N/A",
            response.content,
        )

        content = strip_json_fences(response.content)

        try:
            return parse_fn(content)
        except json.JSONDecodeError as e:
            last_error = e
            if attempt < max_retries:
                delay = retry_delay * (2 ** attempt)
                logger.warning(
                    "JSON parse failed for %s (attempt %d/%d), retrying in %.1fs: %.200s",
                    context_label,
                    attempt + 1,
                    max_retries + 1,
                    delay,
                    response.content,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "[FALLBACK] JSON parse FAILED for '%s' after %d attempts (all retries exhausted). "
                    "Last response (truncated): %.200s",
                    context_label,
                    max_retries + 1,
                    response.content,
                )

    raise last_error  # type: ignore[misc]
