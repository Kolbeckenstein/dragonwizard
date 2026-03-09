"""
Unit tests for WikidotScraper.

Tests cover URL filtering, link extraction, and BFS traversal logic.
All tests that would make network calls use mocked sessions.
"""

from __future__ import annotations

import pytest

from dragonwizard.rag.sources.web.scraper import WikidotScraper


# ---------------------------------------------------------------------------
# HTML fixtures for link extraction
# ---------------------------------------------------------------------------

INDEX_HTML = """
<html><body>
  <div id="page-content">
    <a href="/spell:fireball">Fireball</a>
    <a href="/spell:shield">Shield</a>
    <a href="/classes">Classes</a>
    <a href="https://dnd5e.wikidot.com/rogue">Rogue</a>
    <a href="https://other-site.com/page">External</a>
    <a href="/system:list-all-pages">Sitemap</a>
    <a href="/forum:general">Forum</a>
    <a href="/spell:fireball#at-higher-levels">Fragment</a>
    <a href="/spell:fireball">Fireball again</a>
  </div>
</body></html>
"""

EMPTY_HTML = "<html><body><p>No links here.</p></body></html>"


# ---------------------------------------------------------------------------
# _is_skip tests (pure unit — no I/O)
# ---------------------------------------------------------------------------

class TestIsSkip:
    def setup_method(self):
        self.scraper = WikidotScraper("https://dnd5e.wikidot.com/")

    def test_allows_spell_pages(self):
        assert not self.scraper._is_skip("/spell:fireball")

    def test_allows_class_pages(self):
        assert not self.scraper._is_skip("/rogue")

    def test_allows_root(self):
        assert not self.scraper._is_skip("/")

    def test_filters_system_pages(self):
        assert self.scraper._is_skip("/system:list-all-pages")

    def test_filters_forum_pages(self):
        assert self.scraper._is_skip("/forum:thread-1")

    def test_filters_talk_pages(self):
        assert self.scraper._is_skip("/talk:spell:fireball")

    def test_filters_user_pages(self):
        assert self.scraper._is_skip("/user:someone")

    def test_filters_admin_pages(self):
        assert self.scraper._is_skip("/admin:manage")

    def test_filters_site_manager_pages(self):
        assert self.scraper._is_skip("/site-manager:settings")

    def test_filters_urls_with_fragments(self):
        assert self.scraper._is_skip("/spell:fireball#at-higher-levels")

    def test_filters_fragment_only(self):
        assert self.scraper._is_skip("#toc")


# ---------------------------------------------------------------------------
# _extract_links tests (pure unit — no I/O)
# ---------------------------------------------------------------------------

class TestExtractLinks:
    def setup_method(self):
        self.scraper = WikidotScraper("https://dnd5e.wikidot.com/")

    def test_converts_relative_links_to_absolute(self):
        links = self.scraper._extract_links(INDEX_HTML)
        assert "https://dnd5e.wikidot.com/spell:fireball" in links

    def test_includes_same_domain_absolute_links(self):
        links = self.scraper._extract_links(INDEX_HTML)
        assert "https://dnd5e.wikidot.com/rogue" in links

    def test_excludes_external_domain_links(self):
        links = self.scraper._extract_links(INDEX_HTML)
        assert not any("other-site.com" in link for link in links)

    def test_excludes_system_pages(self):
        links = self.scraper._extract_links(INDEX_HTML)
        assert not any("system:" in link for link in links)

    def test_excludes_forum_pages(self):
        links = self.scraper._extract_links(INDEX_HTML)
        assert not any("forum:" in link for link in links)

    def test_excludes_fragment_urls(self):
        links = self.scraper._extract_links(INDEX_HTML)
        assert not any("#" in link for link in links)

    def test_deduplicates_same_href_appearing_twice(self):
        links = self.scraper._extract_links(INDEX_HTML)
        # /spell:fireball appears twice in the fixture
        fireball_links = [l for l in links if l.endswith("/spell:fireball")]
        assert len(fireball_links) == 1, "Duplicate hrefs should appear once"

    def test_returns_empty_list_when_no_links(self):
        links = self.scraper._extract_links(EMPTY_HTML)
        assert links == []


# ---------------------------------------------------------------------------
# discover_urls — BFS with mocked session
# ---------------------------------------------------------------------------

def _make_session_with_pages(pages: dict[str, str]):
    """
    Build a mock aiohttp.ClientSession that serves HTML from a dict
    keyed by URL. Unknown URLs return empty HTML.
    """
    from unittest.mock import AsyncMock, MagicMock

    def get_side_effect(url, **kwargs):
        html = pages.get(url, EMPTY_HTML)
        response = AsyncMock()
        response.text = AsyncMock(return_value=html)
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=response)
        ctx.__aexit__ = AsyncMock(return_value=False)
        return ctx

    session = MagicMock()
    session.get = MagicMock(side_effect=get_side_effect)
    return session


class TestDiscoverUrls:
    @pytest.mark.asyncio
    async def test_yields_seed_url_first(self):
        pages = {"https://dnd5e.wikidot.com/": EMPTY_HTML}
        session = _make_session_with_pages(pages)
        scraper = WikidotScraper("https://dnd5e.wikidot.com/", delay=0, max_pages=1)
        urls = [url async for url in scraper.discover_urls(session)]
        assert urls[0] == "https://dnd5e.wikidot.com/"

    @pytest.mark.asyncio
    async def test_max_pages_limits_output(self):
        # Homepage links to 3 pages
        homepage = """<html><body>
            <a href="/page1">P1</a><a href="/page2">P2</a><a href="/page3">P3</a>
        </body></html>"""
        pages = {
            "https://example.com/": homepage,
            "https://example.com/page1": EMPTY_HTML,
            "https://example.com/page2": EMPTY_HTML,
            "https://example.com/page3": EMPTY_HTML,
        }
        session = _make_session_with_pages(pages)
        scraper = WikidotScraper("https://example.com/", delay=0, max_pages=2)
        urls = [url async for url in scraper.discover_urls(session)]
        assert len(urls) == 2

    @pytest.mark.asyncio
    async def test_does_not_revisit_urls(self):
        # Both pages link back to each other — should not loop
        page_a = '<html><body><a href="/page-b">B</a></body></html>'
        page_b = '<html><body><a href="/">A</a></body></html>'
        pages = {
            "https://example.com/": page_a,
            "https://example.com/page-b": page_b,
        }
        session = _make_session_with_pages(pages)
        scraper = WikidotScraper("https://example.com/", delay=0)
        urls = [url async for url in scraper.discover_urls(session)]
        # Should visit exactly 2 pages, not loop forever
        assert len(urls) == 2
        assert len(set(urls)) == 2  # no duplicates

    @pytest.mark.asyncio
    async def test_skips_system_links_found_during_crawl(self):
        homepage = '<html><body><a href="/system:list">Sys</a><a href="/rogue">Rogue</a></body></html>'
        pages = {
            "https://dnd5e.wikidot.com/": homepage,
            "https://dnd5e.wikidot.com/rogue": EMPTY_HTML,
        }
        session = _make_session_with_pages(pages)
        scraper = WikidotScraper("https://dnd5e.wikidot.com/", delay=0)
        urls = [url async for url in scraper.discover_urls(session)]
        assert not any("system:" in u for u in urls)

    @pytest.mark.asyncio
    async def test_continues_after_failed_fetch(self):
        """A broken page should not stop the crawl."""
        from unittest.mock import AsyncMock, MagicMock

        homepage = '<html><body><a href="/good">Good</a><a href="/bad">Bad</a></body></html>'

        def get_side_effect(url, **kwargs):
            if "bad" in url:
                ctx = AsyncMock()
                ctx.__aenter__ = AsyncMock(side_effect=Exception("connection error"))
                ctx.__aexit__ = AsyncMock(return_value=False)
                return ctx
            html = {"https://example.com/": homepage}.get(url, EMPTY_HTML)
            response = AsyncMock()
            response.text = AsyncMock(return_value=html)
            ctx = AsyncMock()
            ctx.__aenter__ = AsyncMock(return_value=response)
            ctx.__aexit__ = AsyncMock(return_value=False)
            return ctx

        session = MagicMock()
        session.get = MagicMock(side_effect=get_side_effect)
        scraper = WikidotScraper("https://example.com/", delay=0)
        urls = [url async for url in scraper.discover_urls(session)]
        # Homepage + /good both visited; /bad yielded but fetch failed (silently)
        assert "https://example.com/" in urls
        assert "https://example.com/bad" in urls  # still yielded before the failed fetch
