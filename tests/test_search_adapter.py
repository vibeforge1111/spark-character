"""search_adapter pure-function tests. No network."""

from __future__ import annotations

import httpx
import pytest

from spark_character.search_adapter import (
    SearchResult,
    _parse_duckduckgo_html,
    _strip_tags,
    attach_search_context,
    detect_needs_live_data,
    search_results_for,
)


def test_detect_live_data_positive() -> None:
    assert detect_needs_live_data("What's the current price of BTC?")
    assert detect_needs_live_data("Latest news about OpenAI")
    assert detect_needs_live_data("Today's weather in Dubai")
    assert detect_needs_live_data("Recent updates on the OpenCUA repo")


def test_detect_live_data_negative() -> None:
    assert not detect_needs_live_data("How do I write a Python decorator?")
    assert not detect_needs_live_data("What does TVL mean in DeFi?")
    assert not detect_needs_live_data("I'm anxious about the launch tomorrow.")


def test_strip_tags_removes_html() -> None:
    assert _strip_tags("<b>hello</b> <i>world</i>") == "hello world"
    assert _strip_tags("plain text") == "plain text"
    assert _strip_tags("<a href='x'>link</a>") == "link"


def test_attach_search_context_only_if_needed_skips_irrelevant() -> None:
    out = attach_search_context(
        "How do I write a Python decorator?",
        search_fn=lambda q: [SearchResult("title", "snippet", "https://x.com")],
        only_if_needed=True,
    )
    # not detected as live-data, returns prompt unchanged
    assert out == "How do I write a Python decorator?"


def test_attach_search_context_injects_when_relevant() -> None:
    fake = lambda q: [
        SearchResult("Bitcoin price", "BTC at $X today", "https://coingecko.com"),
        SearchResult("Crypto markets", "BTC up 2%", "https://example.com"),
    ]
    out = attach_search_context(
        "What's the current price of BTC?",
        search_fn=fake,
        only_if_needed=True,
    )
    assert "[Live search results" in out
    assert "Bitcoin price" in out
    assert "https://coingecko.com" in out
    assert "[User message]" in out
    assert "What's the current price of BTC?" in out


def test_attach_search_context_force_fetch_when_off() -> None:
    fake = lambda q: [SearchResult("x", "y", "https://x.com")]
    out = attach_search_context(
        "Tell me a joke",
        search_fn=fake,
        only_if_needed=False,
    )
    assert "[Live search results" in out


def test_attach_search_context_no_results_returns_original() -> None:
    out = attach_search_context(
        "What's the latest BTC price?",
        search_fn=lambda q: [],
        only_if_needed=True,
    )
    assert out == "What's the latest BTC price?"


def test_parse_duckduckgo_html_minimal() -> None:
    html_text = """
    <html>
    <a class="result__a" href="https://example.com">Example Title</a>
    <a class="result__snippet">A snippet about something</a>
    <a class="result__a" href="https://second.com">Second Title</a>
    <a class="result__snippet">Second snippet</a>
    </html>
    """
    results = _parse_duckduckgo_html(html_text)
    assert len(results) == 2
    assert results[0].title == "Example Title"
    assert results[0].snippet == "A snippet about something"
    assert results[0].url == "https://example.com"
    assert results[1].title == "Second Title"


def test_search_results_soft_fails_expected_backend_errors() -> None:
    def failing_search(_query: str) -> list[SearchResult]:
        raise httpx.ReadTimeout("search timed out")

    assert search_results_for("current btc price", search_fn=failing_search) == []


def test_search_results_surfaces_unexpected_programming_errors() -> None:
    def broken_search(_query: str) -> list[SearchResult]:
        raise RuntimeError("programmer bug")

    with pytest.raises(RuntimeError, match="programmer bug"):
        search_results_for("current btc price", search_fn=broken_search)


def test_parse_duckduckgo_redirect_decodes_target_url() -> None:
    html_text = """
    <html>
    <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fsafe">Example</a>
    <a class="result__snippet">A snippet</a>
    </html>
    """

    results = _parse_duckduckgo_html(html_text)

    assert results[0].url == "https://example.com/safe"


def test_parse_duckduckgo_malformed_redirect_keeps_raw_url(monkeypatch) -> None:
    def empty_uddg(_query: str) -> dict[str, list[str]]:
        return {"uddg": []}

    monkeypatch.setattr("spark_character.search_adapter.parse_qs", empty_uddg)
    raw_url = "//duckduckgo.com/l/?uddg="
    html_text = f"""
    <html>
    <a class="result__a" href="{raw_url}">Example</a>
    <a class="result__snippet">A snippet</a>
    </html>
    """

    results = _parse_duckduckgo_html(html_text)

    assert results[0].url == raw_url
