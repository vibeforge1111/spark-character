"""Provider-agnostic web_search adapter.

Closes the live-data gap from ROADMAP. Z.AI's coding/paas/v4/ endpoint
ignores tools=[{type: web_search, ...}] (verified by raw API probe -
no tool_calls in response). The persona handles the no-tool case
correctly today by pointing users at sources, but a real fetch is a
strict improvement.

This module provides a pluggable client-side search:

1. detect_needs_live_data(prompt): heuristic that decides whether a
   prompt benefits from live data (current X, latest Y, today's Z,
   price of, status of, news about, etc).

2. search(query): hits a search backend and returns short text
   snippets with source URLs. Default backend is DuckDuckGo HTML
   scrape (no API key, no auth). Pluggable via search_fn parameter.

3. attach_search_context(prompt, query=None, ...): does the detection
   + fetch + injection in one call. Returns a wrapped prompt with
   '[Live search results for ...]' before the user message, so the
   persona can answer with cited sources instead of refusing.

Usage:

    from spark_character import attach_search_context, generate
    wrapped = attach_search_context(user_message)
    result = generate(wrapped, provider=...)

Or in evolve / pulse drivers:

    from spark_character.search_adapter import search_results_for
    snippets = search_results_for("current btc price")
    # use snippets directly, or wrap into a prompt manually
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Callable
from urllib.parse import quote_plus

import httpx


_LIVE_DATA_PATTERNS = (
    r"\b(?:current|today's|latest|right now|recent|breaking)\s+",
    r"\b(?:what(?:'s| is) the (?:current|latest|today's)\s+\w+)",
    r"\b(?:price|stock|exchange rate|quote|rate)\s+(?:of|for|on)\s+\w+",
    r"\b(?:news|headlines|update[s]?)\s+(?:about|on|for|from)\s+\w+",
    r"\b(?:status|outage|incident)\s+(?:of|for|at)\s+",
    r"\b(?:released|announced|launched|shipped)\s+(?:today|yesterday|this week)",
    r"\b(?:weather|forecast)\s+(?:in|at|for)\s+",
)
_LIVE_DATA_RE = re.compile("|".join(_LIVE_DATA_PATTERNS), re.IGNORECASE)


@dataclass(frozen=True)
class SearchResult:
    title: str
    snippet: str
    url: str


def detect_needs_live_data(prompt: str) -> bool:
    """Heuristic for whether a prompt benefits from a live search."""
    if not prompt:
        return False
    return bool(_LIVE_DATA_RE.search(prompt))


def extract_search_query(prompt: str) -> str:
    """Extract a search query from a user prompt. v1: just returns the
    prompt itself, trimmed to a reasonable length. Future versions can
    use a small LLM call to extract the actual query intent."""
    text = prompt.strip()
    if len(text) > 200:
        text = text[:200]
    return text


def search_results_for(
    query: str,
    *,
    max_results: int = 5,
    timeout_seconds: float = 8.0,
    search_fn: Callable[[str], list[SearchResult]] | None = None,
) -> list[SearchResult]:
    """Search the web for `query` and return up to `max_results` SearchResults.

    Default backend: DuckDuckGo HTML endpoint. Pluggable via search_fn.
    Soft-fails: returns [] on any error so the caller can fall through.
    """
    if not query.strip():
        return []
    fn = search_fn or _duckduckgo_html_search
    try:
        results = fn(query)
        return results[:max_results]
    except Exception:
        return []


def attach_search_context(
    user_message: str,
    *,
    query: str | None = None,
    max_results: int = 4,
    search_fn: Callable[[str], list[SearchResult]] | None = None,
    only_if_needed: bool = True,
) -> str:
    """Return a prompt with live search context attached when relevant.

    only_if_needed=True (default) runs detect_needs_live_data first;
    if the prompt does not need live data, returns the original. Set
    False to always fetch.
    """
    if only_if_needed and not detect_needs_live_data(user_message):
        return user_message
    q = query or extract_search_query(user_message)
    results = search_results_for(q, max_results=max_results, search_fn=search_fn)
    if not results:
        return user_message
    context_lines = ["[Live search results, treat as ground truth for current data]"]
    for i, r in enumerate(results, 1):
        context_lines.append(f"{i}. {r.title}")
        if r.snippet:
            context_lines.append(f"   {r.snippet}")
        if r.url:
            context_lines.append(f"   source: {r.url}")
    context_lines.append("")
    context_lines.append("[User message]")
    context_lines.append(user_message)
    return "\n".join(context_lines)


def _duckduckgo_html_search(query: str) -> list[SearchResult]:
    """Default backend: DuckDuckGo HTML scrape. No auth, no key.

    Hits html.duckduckgo.com (the result-serving subdomain) with a
    GET. The bare duckduckgo.com/html/ root returns the home page on
    POST and is bot-rate-limited.

    Returns up to ~10 results parsed from the HTML. Best-effort: the
    HTML format may change, in which case this returns []. Callers
    should not rely on it for production-critical paths until paired
    with a stable API search backend (Brave, Serper, SerpAPI).
    """
    url = "https://html.duckduckgo.com/html/"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        ),
    }
    with httpx.Client(timeout=8.0, follow_redirects=True) as client:
        resp = client.get(url, params={"q": query}, headers=headers)
        resp.raise_for_status()
        html_text = resp.text
    return _parse_duckduckgo_html(html_text)


_RESULT_BLOCK_RE = re.compile(
    r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
_SNIPPET_RE = re.compile(
    r'<a[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)


def _parse_duckduckgo_html(text: str) -> list[SearchResult]:
    results: list[SearchResult] = []
    titles = _RESULT_BLOCK_RE.findall(text or "")
    snippets = _SNIPPET_RE.findall(text or "")
    for i, (raw_url, raw_title) in enumerate(titles[:10]):
        clean_title = _strip_tags(html.unescape(raw_title)).strip()
        clean_snippet = ""
        if i < len(snippets):
            clean_snippet = _strip_tags(html.unescape(snippets[i])).strip()
        url = raw_url
        if url.startswith("//duckduckgo.com/l/?uddg="):
            try:
                from urllib.parse import unquote, urlparse, parse_qs
                parsed = urlparse(url)
                qs = parse_qs(parsed.query)
                if qs.get("uddg"):
                    url = unquote(qs["uddg"][0])
            except Exception:
                pass
        if not clean_title and not clean_snippet:
            continue
        results.append(SearchResult(title=clean_title, snippet=clean_snippet, url=url))
    return results


def _strip_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)
