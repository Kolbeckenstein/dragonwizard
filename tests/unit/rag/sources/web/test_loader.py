"""
Unit tests for WebPageLoader.

All HTTP calls are mocked via aiohttp test helpers — no network access.
Tests follow Red → Green → Refactor: this file is written before the
implementation so each test starts failing, then passes once the code is added.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from dragonwizard.rag.sources.web.loader import WebPageLoader


# ---------------------------------------------------------------------------
# HTML fixtures
# ---------------------------------------------------------------------------

WIKIDOT_PAGE = """
<html>
<head><title>Fireball - D&D 5e</title></head>
<body>
  <div id="side-bar">Nav stuff</div>
  <div id="page-content">
    <h1 id="page-title">Fireball</h1>
    <p>3rd-level evocation</p>
    <p>A bright streak flashes from your pointing finger...</p>
    <script>alert('ad')</script>
    <style>body { color: red }</style>
  </div>
</body>
</html>
"""

NO_PAGE_CONTENT = """
<html>
<body>
  <h1 id="page-title">Simple Page</h1>
  <p>Body only content.</p>
</body>
</html>
"""

NO_H1_PAGE = """
<html>
<body>
  <div id="page-content">
    <p>Content without a heading.</p>
  </div>
</body>
</html>
"""

SCRIPT_HEAVY_PAGE = """
<html>
<body>
  <div id="page-content">
    <script>trackUser()</script>
    <style>.ad { display: none }</style>
    <p>Clean content here.</p>
  </div>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session(html: str, status: int = 200):
    """Return a mock aiohttp.ClientSession that serves the given HTML."""
    response = AsyncMock()
    response.status = status
    response.text = AsyncMock(return_value=html)
    if status >= 400:
        from aiohttp import ClientResponseError
        response.raise_for_status = MagicMock(
            side_effect=ClientResponseError(request_info=MagicMock(), history=())
        )
    else:
        response.raise_for_status = MagicMock()

    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=response)
    ctx.__aexit__ = AsyncMock(return_value=False)

    session = MagicMock()
    session.get = MagicMock(return_value=ctx)
    return session


# ---------------------------------------------------------------------------
# Tests: _extract_text (pure unit — no network)
# ---------------------------------------------------------------------------

class TestExtractText:
    def setup_method(self):
        self.loader = WebPageLoader.__new__(WebPageLoader)

    def test_extracts_title_from_h1_page_title(self):
        title, _ = self.loader._extract_text(WIKIDOT_PAGE, "https://example.com/spell:fireball")
        assert title == "Fireball"

    def test_falls_back_to_url_path_when_no_h1(self):
        title, _ = self.loader._extract_text(NO_H1_PAGE, "https://dnd5e.wikidot.com/spell:fireball")
        assert title == "spell:fireball"

    def test_uses_page_content_div_when_present(self):
        _, text = self.loader._extract_text(WIKIDOT_PAGE, "https://example.com/")
        assert "3rd-level evocation" in text
        # Sidebar content should NOT appear
        assert "Nav stuff" not in text

    def test_falls_back_to_body_when_no_page_content(self):
        _, text = self.loader._extract_text(NO_PAGE_CONTENT, "https://example.com/simple")
        assert "Body only content." in text

    def test_removes_script_tags(self):
        _, text = self.loader._extract_text(SCRIPT_HEAVY_PAGE, "https://example.com/")
        assert "trackUser" not in text

    def test_removes_style_tags(self):
        _, text = self.loader._extract_text(SCRIPT_HEAVY_PAGE, "https://example.com/")
        assert ".ad" not in text
        assert "display: none" not in text

    def test_title_prepended_to_text(self):
        title, text = self.loader._extract_text(WIKIDOT_PAGE, "https://example.com/")
        assert text.startswith(f"# {title}")

    def test_content_follows_title(self):
        _, text = self.loader._extract_text(WIKIDOT_PAGE, "https://example.com/")
        assert "3rd-level evocation" in text
        assert "bright streak" in text


# ---------------------------------------------------------------------------
# Tests: load() — mocked session
# ---------------------------------------------------------------------------

class TestLoad:
    @pytest.mark.asyncio
    async def test_returns_document_with_web_source_type(self):
        session = _make_session(WIKIDOT_PAGE)
        loader = WebPageLoader(session)
        doc = await loader.load("https://dnd5e.wikidot.com/spell:fireball")
        assert doc.metadata.source_type == "web"

    @pytest.mark.asyncio
    async def test_source_file_is_the_url(self):
        url = "https://dnd5e.wikidot.com/spell:fireball"
        session = _make_session(WIKIDOT_PAGE)
        loader = WebPageLoader(session)
        doc = await loader.load(url)
        assert doc.metadata.source_file == url

    @pytest.mark.asyncio
    async def test_title_extracted_from_page(self):
        session = _make_session(WIKIDOT_PAGE)
        loader = WebPageLoader(session)
        doc = await loader.load("https://dnd5e.wikidot.com/spell:fireball")
        assert doc.metadata.title == "Fireball"

    @pytest.mark.asyncio
    async def test_document_text_is_non_empty(self):
        session = _make_session(WIKIDOT_PAGE)
        loader = WebPageLoader(session)
        doc = await loader.load("https://dnd5e.wikidot.com/spell:fireball")
        assert doc.text.strip()

    @pytest.mark.asyncio
    async def test_raises_on_http_error(self):
        from aiohttp import ClientResponseError
        session = _make_session("<html/>", status=404)
        loader = WebPageLoader(session)
        with pytest.raises(ClientResponseError):
            await loader.load("https://dnd5e.wikidot.com/nonexistent")

    @pytest.mark.asyncio
    async def test_user_agent_header_sent(self):
        session = _make_session(WIKIDOT_PAGE)
        loader = WebPageLoader(session, user_agent="TestAgent/1.0")
        await loader.load("https://example.com/")
        call_kwargs = session.get.call_args[1]
        assert call_kwargs["headers"]["User-Agent"] == "TestAgent/1.0"

    @pytest.mark.asyncio
    async def test_edition_is_none_by_default(self):
        """Loader does not infer edition — that's the pipeline's job."""
        session = _make_session(WIKIDOT_PAGE)
        loader = WebPageLoader(session)
        doc = await loader.load("https://dnd5e.wikidot.com/spell:fireball")
        assert doc.metadata.edition is None
