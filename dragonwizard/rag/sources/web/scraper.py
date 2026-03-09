"""
Wikidot site crawler for the RAG ingestion pipeline.

Discovers content pages via BFS from a seed URL, skipping wikidot
administrative and meta pages (system:, forum:, talk:, etc.).

Can be used standalone or driven by IngestionPipeline.ingest_url() in a loop:

    scraper = WikidotScraper("https://dnd5e.wikidot.com/", delay=1.0)
    async with aiohttp.ClientSession() as session:
        async for url in scraper.discover_urls(session):
            await pipeline.ingest_url(url, session, edition="5e")

Design decisions:
- BFS from seed URL — gets everything without needing to know URL patterns
- URL normalisation strips query strings and fragments before deduplication
- Errors during page fetch are caught and logged; crawl continues
- delay=0 is allowed for unit tests; production should use >=1.0 s
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import AsyncIterator
from urllib.parse import urlparse, urlunparse

import aiohttp
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Wikidot path prefixes that are administrative / not D&D content
_SKIP_PREFIXES: tuple[str, ...] = (
    "/system:",
    "/forum:",
    "/talk:",
    "/user:",
    "/account:",
    "/site-manager:",
    "/admin:",
)


def _normalise_url(url: str) -> str:
    """Strip fragment and query string; keep scheme + host + path only."""
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))


class WikidotScraper:
    """
    BFS crawler for wikidot sites.

    Starts from a seed URL (typically the homepage), follows internal links,
    and yields each content URL once. Skips administrative pages and respects
    a polite request delay.

    Args:
        base_url: Seed URL (e.g. "https://dnd5e.wikidot.com/")
        delay: Seconds to wait between requests (default: 1.0 for politeness)
        max_pages: Stop after yielding this many URLs (None = no limit)

    Example:
        >>> scraper = WikidotScraper("https://dnd5e.wikidot.com/", delay=1.0)
        >>> async with aiohttp.ClientSession() as session:
        ...     async for url in scraper.discover_urls(session):
        ...         print(url)
    """

    def __init__(
        self,
        base_url: str,
        delay: float = 1.0,
        max_pages: int | None = None,
    ) -> None:
        self._base_url = base_url
        self._delay = delay
        self._max_pages = max_pages
        parsed = urlparse(base_url)
        self._scheme = parsed.scheme
        self._host = parsed.netloc

    async def discover_urls(
        self, session: aiohttp.ClientSession
    ) -> AsyncIterator[str]:
        """
        BFS from base_url; yield each content URL once.

        Args:
            session: Active aiohttp.ClientSession (caller manages lifecycle)

        Yields:
            Absolute URLs for content pages
        """
        seen: set[str] = set()
        queue: deque[str] = deque([self._base_url])
        count = 0

        while queue:
            if self._max_pages is not None and count >= self._max_pages:
                return

            url = queue.popleft()
            normalised = _normalise_url(url)
            if normalised in seen:
                continue
            seen.add(normalised)

            yield url
            count += 1

            # Fetch to discover more links; errors are non-fatal
            try:
                async with session.get(url, allow_redirects=True) as resp:
                    html = await resp.text()

                for link in self._extract_links(html):
                    link_norm = _normalise_url(link)
                    if link_norm not in seen:
                        queue.append(link)

            except Exception as exc:
                logger.debug(f"Failed to fetch {url} for link discovery: {exc}")

            if self._delay:
                await asyncio.sleep(self._delay)

    def _extract_links(self, html: str) -> list[str]:
        """
        Extract internal content links from an HTML page.

        Returns absolute URLs on the same host, deduplicated, with
        system/admin/forum pages removed.

        Args:
            html: Raw HTML string

        Returns:
            List of absolute URLs (may contain duplicates across calls)
        """
        soup = BeautifulSoup(html, "lxml")
        seen_in_page: set[str] = set()
        links: list[str] = []

        for a in soup.find_all("a", href=True):
            href: str = a["href"]
            parsed = urlparse(href)

            # Relative link: /spell:fireball
            if not parsed.netloc and href.startswith("/"):
                if self._is_skip(href):
                    continue
                absolute = f"{self._scheme}://{self._host}{parsed.path}"

            # Absolute link on same host: https://dnd5e.wikidot.com/rogue
            elif parsed.netloc == self._host:
                if self._is_skip(parsed.path):
                    continue
                absolute = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

            else:
                continue  # external domain

            if absolute not in seen_in_page:
                seen_in_page.add(absolute)
                links.append(absolute)

        return links

    def _is_skip(self, path: str) -> bool:
        """
        Return True if the path should be excluded from crawling.

        Skips wikidot administrative pages and fragment-only anchors.

        Args:
            path: URL path component (may include fragment)

        Returns:
            True if the URL should not be crawled
        """
        if "#" in path:
            return True
        return any(path.startswith(prefix) for prefix in _SKIP_PREFIXES)
