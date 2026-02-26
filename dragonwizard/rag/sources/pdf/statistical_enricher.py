"""
StatisticalHeadingEnricher — font-size heuristic heading detection.

Detects headings by comparing each text span's font size to the document's
body font size (computed as the mode weighted by character count). Spans that
are significantly larger, or that are bold and short, are treated as headings.

Detected headings are injected as "[Section: <heading>]\\n" prefixes into
chunks that immediately follow them in the document.

Heading Filter Pipeline
-----------------------
Heading candidates pass through a list of filter functions before being
accepted.  Each filter has the signature::

    (text: str, font_ratio: float) -> bool

Returning ``True`` means the candidate *passes* (keep it); ``False`` means
it should be rejected.

Six built-in filters are provided and collected in ``DEFAULT_HEADING_FILTERS``.
Pass a custom ``heading_filters`` list to the constructor to replace them.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

import fitz  # PyMuPDF

from dragonwizard.config.logging import get_logger
from dragonwizard.rag.base import Chunk, ChunkEnricher, Document

logger = get_logger(__name__)

# fitz flag bit for bold text
_BOLD_FLAG = 16

# ---------------------------------------------------------------------------
# Module-level compiled regexes used by built-in filters
# ---------------------------------------------------------------------------

# Control characters, tilde, or "::" sequences that appear in OCR artefacts
_RE_ARTIFACTS = re.compile(r'[~\x00-\x1f]|::')

# Single all-caps "word" that contains a digit adjacent to a capital letter
# e.g. "0ROG", "3SPELL" — indicative of OCR mis-reads or page numbers
_RE_DIGIT_IN_CAPS = re.compile(r'[0-9][A-Z]|[A-Z][0-9]')


# ---------------------------------------------------------------------------
# Built-in filter functions
# ---------------------------------------------------------------------------

def filter_no_artifacts(text: str, font_ratio: float) -> bool:
    """Reject text that contains control characters, tildes, or '::' sequences."""
    return not bool(_RE_ARTIFACTS.search(text))


def filter_not_all_lowercase(text: str, font_ratio: float) -> bool:
    """Reject single-word candidates that are entirely lowercase (e.g. 'ints')."""
    words = text.split()
    if len(words) != 1:
        return True
    return not words[0].islower()


def filter_no_trailing_punctuation(text: str, font_ratio: float) -> bool:
    """Reject candidates that end with sentence-ending punctuation."""
    return not text.endswith(('.', ',', ';'))


def filter_font_ratio_ceiling(text: str, font_ratio: float) -> bool:
    """Reject spans with font_ratio >= 2.0 (chapter ornaments, drop-caps, logos)."""
    return font_ratio < 2.0


def filter_no_digit_artifacts(text: str, font_ratio: float) -> bool:
    """Reject single all-caps words that mix digits and letters (e.g. '0ROG')."""
    words = text.split()
    if len(words) != 1:
        return True
    word = words[0]
    if not (word.isupper() and len(word) > 1):
        return True
    return not bool(_RE_DIGIT_IN_CAPS.search(word))


def filter_max_length(text: str, font_ratio: float) -> bool:
    """Reject spans longer than 80 characters (body text that happens to be large)."""
    return len(text) <= 80


# ---------------------------------------------------------------------------
# Default filter set — applied in order; all must return True to keep candidate
# ---------------------------------------------------------------------------

DEFAULT_HEADING_FILTERS: list[Callable[[str, float], bool]] = [
    filter_no_artifacts,
    filter_not_all_lowercase,
    filter_no_trailing_punctuation,
    filter_font_ratio_ceiling,
    filter_no_digit_artifacts,
    filter_max_length,
]


@dataclass
class _HeadingCandidate:
    """An identified heading candidate with its position and confidence."""
    text: str
    char_offset: int   # approximate offset in document.text
    font_ratio: float  # candidate_size / body_size
    confidence: float  # min(1.0, font_ratio / 2.0)


class StatisticalHeadingEnricher(ChunkEnricher):
    """
    Enricher that injects detected section headings as chunk prefixes.

    Algorithm:
        1. Re-open the source PDF with PyMuPDF in dict mode to get per-span
           font size and bold flags (not available in the "blocks" tuples).
        2. Compute body_size as the mode font size weighted by character count.
        3. Identify heading candidates: spans with font_ratio >= threshold,
           OR bold spans within max_bold_length characters.
        4. For each chunk with a known char_start, find the nearest preceding
           heading within max_section_gap characters.
        5. Return a new chunk with "[Section: <heading>]\\n" prepended.

    Non-PDF documents skip steps 1-3 and return chunks unchanged.

    Attributes:
        font_ratio_threshold: Minimum size ratio (candidate/body) to qualify.
        max_bold_length: Maximum characters for a bold span to qualify.
        max_section_gap: Maximum char distance between heading and chunk start.
        heading_filters: List of ``(text, font_ratio) -> bool`` callables applied
            in order; a candidate is kept only if all filters return True.
            Defaults to ``DEFAULT_HEADING_FILTERS``.
    """

    def __init__(
        self,
        font_ratio_threshold: float = 1.15,
        max_bold_length: int = 60,
        max_section_gap: int = 5000,
        heading_filters: list[Callable[[str, float], bool]] | None = None,
    ) -> None:
        self.font_ratio_threshold = font_ratio_threshold
        self.max_bold_length = max_bold_length
        self.max_section_gap = max_section_gap
        self.heading_filters = (
            list(heading_filters) if heading_filters is not None
            else list(DEFAULT_HEADING_FILTERS)
        )

    def _detect_headings(self, document: Document) -> list[_HeadingCandidate]:
        """
        Detect heading candidates from the source PDF's font metadata.

        Returns an empty list for non-PDF documents.
        """
        if document.metadata.source_type != "pdf":
            return []

        return self._detect_headings_from_file(document)

    def _detect_headings_from_file(self, document: Document) -> list[_HeadingCandidate]:
        """Open the PDF and extract heading candidates using font statistics."""
        try:
            with fitz.open(document.metadata.source_file) as pdf:
                return self._extract_candidates(pdf, document.text)
        except Exception as e:
            logger.warning(f"Heading detection failed for {document.metadata.source_file}: {e}")
            return []

    def _extract_candidates(self, pdf: fitz.Document, doc_text: str) -> list[_HeadingCandidate]:
        """Extract all heading candidates from a fitz PDF document."""
        # Step 1: collect all spans and compute body size
        all_spans: list[dict] = []
        for page in pdf:
            page_dict = page.get_text("dict")
            for block in page_dict["blocks"]:
                if block.get("type", -1) != 0:
                    continue  # skip image blocks
                for line in block["lines"]:
                    for span in line["spans"]:
                        all_spans.append(span)

        if not all_spans:
            return []

        body_size = self._compute_body_size(all_spans)
        if body_size <= 0:
            return []

        # Step 2: identify candidates
        candidates: list[_HeadingCandidate] = []
        search_start = 0

        for span in all_spans:
            text = span.get("text", "").strip()
            if not text:
                continue

            size = span.get("size", 0.0)
            flags = span.get("flags", 0)
            is_bold = bool(flags & _BOLD_FLAG)
            font_ratio = size / body_size

            # Heading criteria
            is_large = font_ratio >= self.font_ratio_threshold
            is_bold_short = is_bold and len(text) <= self.max_bold_length

            if not (is_large or is_bold_short):
                continue

            # Apply the composable filter pipeline
            if not all(f(text, font_ratio) for f in self.heading_filters):
                continue

            # Locate the span text in the document
            pos = doc_text.find(text, search_start)
            if pos == -1:
                pos = doc_text.find(text)  # broader search if not found forward
            if pos == -1:
                continue

            confidence = min(1.0, font_ratio / 2.0)
            candidates.append(_HeadingCandidate(
                text=text,
                char_offset=pos,
                font_ratio=font_ratio,
                confidence=confidence,
            ))

        return candidates

    @staticmethod
    def _compute_body_size(spans: list[dict]) -> float:
        """
        Compute the body font size as the mode weighted by character count.

        Rounds font sizes to 1 decimal place to bucket near-identical values.
        """
        size_char_counts: Counter[float] = Counter()
        for span in spans:
            text = span.get("text", "")
            size = round(span.get("size", 0.0), 1)
            size_char_counts[size] += len(text)

        if not size_char_counts:
            return 0.0

        return size_char_counts.most_common(1)[0][0]

    def _format_prefix(self, heading: _HeadingCandidate) -> str:
        """Format the heading prefix string to prepend to a chunk."""
        return f"[Section: {heading.text}]\n"

    async def enrich(self, chunks: list[Chunk], document: Document) -> list[Chunk]:
        """
        Inject heading prefixes into chunks that follow a detected heading.

        Non-PDF documents are returned unchanged. Chunks with char_start=None
        are skipped (position unknown). Chunks more than max_section_gap
        characters away from the nearest heading are left unchanged.

        Returns a new list — input chunks are never mutated.
        """
        headings = self._detect_headings(document)
        if not headings:
            return chunks

        result: list[Chunk] = []
        for chunk in chunks:
            enriched = self._enrich_chunk(chunk, headings)
            result.append(enriched)
        return result

    def _enrich_chunk(self, chunk: Chunk, headings: list[_HeadingCandidate]) -> Chunk:
        """Return an enriched chunk (or the original if no heading applies)."""
        if chunk.metadata.char_start is None:
            return chunk

        char_start = chunk.metadata.char_start

        # Find the nearest preceding heading within max_section_gap
        best: _HeadingCandidate | None = None
        for heading in headings:
            if heading.char_offset >= char_start:
                continue  # heading must precede the chunk
            gap = char_start - heading.char_offset
            if gap > self.max_section_gap:
                continue
            if best is None or heading.char_offset > best.char_offset:
                best = heading

        if best is None:
            return chunk

        prefix = self._format_prefix(best)
        return chunk.model_copy(update={"text": prefix + chunk.text})
