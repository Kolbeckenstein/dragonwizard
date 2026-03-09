"""
Web page loader for the RAG pipeline.

Fetches a single URL and returns a Document with the page's clean text content.
Designed for wikidot-style pages but works with any static HTML site.

The loader:
- Scopes content to the #page-content div (wikidot convention)
- Falls back to <body> if #page-content is absent
- Strips <script> and <style> tags
- Extracts the page title from <h1 id="page-title"> (wikidot) or the URL path
- Prepends the title to the text so it appears in every chunk

Source metadata:
    source_file  = the URL (used as a clickable citation in Discord)
    source_type  = "web"
    edition      = None (the caller — pipeline or scrape command — sets this
                   based on the site being scraped)
"""

from __future__ import annotations

from urllib.parse import urlparse

import aiohttp
from bs4 import BeautifulSoup

from dragonwizard.config.logging import get_logger
from dragonwizard.rag.base import Document, DocumentMetadata

logger = get_logger(__name__)


class WebPageLoader:
    """
    Fetches a single URL and returns a Document.

    Reuses the same Document / DocumentMetadata models as file-based loaders
    so the ingestion pipeline can handle web pages identically to PDFs and
    markdown files after loading.

    Args:
        session: An active aiohttp.ClientSession (caller manages lifecycle)
        user_agent: User-Agent header sent with every request

    Example:
        >>> async with aiohttp.ClientSession() as session:
        ...     loader = WebPageLoader(session)
        ...     doc = await loader.load("https://dnd5e.wikidot.com/spell:fireball")
        ...     print(doc.metadata.title)   # "Fireball"
        ...     print(doc.metadata.source_file)  # "https://dnd5e.wikidot.com/spell:fireball"
    """

    def __init__(
        self,
        session: aiohttp.ClientSession,
        user_agent: str = "DragonWizard/0.1",
    ) -> None:
        self._session = session
        self._user_agent = user_agent

    async def load(self, url: str) -> Document:
        """
        Fetch a URL and return a Document with extracted text and metadata.

        Args:
            url: Fully-qualified URL to fetch

        Returns:
            Document with clean text and web metadata

        Raises:
            aiohttp.ClientResponseError: On HTTP errors (4xx, 5xx)
            aiohttp.ClientError: On connection/timeout errors
        """
        logger.info(f"Fetching: {url}")
        headers = {"User-Agent": self._user_agent}

        async with self._session.get(url, headers=headers, allow_redirects=True) as resp:
            resp.raise_for_status()
            html = await resp.text()

        title, text = self._extract_text(html, url)
        logger.info(f"Fetched: {title!r} ({len(text)} chars) from {url}")

        return Document(
            text=text,
            metadata=DocumentMetadata(
                source_file=url,
                source_type="web",
                title=title,
                edition=None,  # set by the pipeline after loading
            ),
        )

    def _extract_text(self, html: str, url: str) -> tuple[str, str]:
        """
        Extract (title, clean_text) from raw HTML.

        Content strategy:
        1. Use #page-content if present (wikidot standard), else <body>
        2. Strip <script> and <style> elements
        3. Collapse whitespace with get_text(separator="\\n", strip=True)
        4. Prepend "# {title}" so the title appears in every chunked piece

        Title strategy:
        1. <h1 id="page-title"> (wikidot standard)
        2. URL path as fallback (e.g. "spell:fireball")

        Args:
            html: Raw HTML string
            url: Source URL (used for title fallback)

        Returns:
            Tuple of (title, text_with_title_prepended)
        """
        soup = BeautifulSoup(html, "lxml")

        # --- Title ---
        h1 = soup.find("h1", id="page-title")
        if h1:
            title = h1.get_text(strip=True)
        else:
            # Use the URL path as a human-readable fallback
            path = urlparse(url).path.strip("/")
            title = path if path else url

        # --- Content area ---
        content = soup.find(id="page-content") or soup.body
        if content is None:
            return title, f"# {title}\n"

        # Remove JS and CSS — they add noise and no semantic value
        for tag in content.find_all(["script", "style"]):
            tag.decompose()

        text = content.get_text(separator="\n", strip=True)

        # Prepend title so chunks have context about what page they came from
        return title, f"# {title}\n\n{text}"
