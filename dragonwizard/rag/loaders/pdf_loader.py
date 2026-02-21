"""
PDF document loader with optional OCR fallback for scanned pages.

Loads .pdf files using PyMuPDF for text extraction. When a page yields
insufficient text but contains image content (the signature of a scanned page),
automatically falls back to Tesseract OCR via PyMuPDF's built-in integration.

OCR Detection Strategy:
    A page is considered "scanned" when BOTH conditions are true:
    1. Extracted text is < 50 characters (after stripping whitespace)
    2. Page contains at least one image object

    Requiring images prevents wasting OCR cycles on legitimately blank pages
    (separators, intentional whitespace). The 50-char threshold is permissive
    enough for chapter-title pages while catching fully scanned pages.

Graceful Degradation:
    If Tesseract isn't installed or OCR fails on a page, the loader continues
    with whatever text was extractable rather than crashing. Individual page
    failures never abort the entire document load.
"""

import subprocess
from pathlib import Path

import fitz  # PyMuPDF

from dragonwizard.config.logging import get_logger
from dragonwizard.rag.base import Document, DocumentLoader, DocumentMetadata

logger = get_logger(__name__)

# Pages with fewer extracted characters than this AND containing images
# are treated as scanned and routed through OCR.
_OCR_TEXT_THRESHOLD = 50


class PDFLoadError(Exception):
    """Raised when PDF loading fails."""
    pass


class PDFLoader(DocumentLoader):
    """
    Loader for PDF files (.pdf).

    Uses PyMuPDF (fitz) for text extraction, with an optional OCR fallback
    for scanned pages (requires tesseract-ocr to be installed on the host).

    Each page in the returned Document records its ``extraction_method``
    (``"text"`` or ``"ocr"``) so callers can audit ingestion quality.

    Args:
        ocr_enabled: Attempt OCR on scanned pages (default: True).
                     Set False for faster processing or when Tesseract
                     isn't available and you want to suppress the probe.

    Example:
        >>> loader = PDFLoader()
        >>> doc = await loader.load(Path("srd.pdf"))
        >>> print(doc.metadata.page_count)
        241
        >>> print(doc.pages[0]['extraction_method'])
        text
    """

    def __init__(self, *, ocr_enabled: bool = True):
        self._ocr_enabled = ocr_enabled
        # Lazily populated on first OCR attempt; avoids subprocess calls when
        # every page has sufficient text.
        self._ocr_available: bool | None = None

    def _check_ocr_available(self) -> bool:
        """
        Probe whether Tesseract is installed (result cached per instance).

        Runs ``tesseract --version`` once and caches the result so repeated
        page checks within the same load() call don't spawn extra processes.
        """
        if self._ocr_available is None:
            try:
                result = subprocess.run(
                    ["tesseract", "--version"],
                    capture_output=True,
                    timeout=5,
                )
                self._ocr_available = result.returncode == 0
            except (FileNotFoundError, subprocess.TimeoutExpired):
                self._ocr_available = False
                logger.warning(
                    "Tesseract not found — OCR disabled. "
                    "Install tesseract-ocr to process scanned PDFs."
                )
        return self._ocr_available

    def _page_needs_ocr(self, page: fitz.Page, extracted_text: str) -> bool:
        """
        Decide whether a page should be sent through OCR.

        Returns True only when text is suspiciously short AND the page
        contains image objects (i.e. it looks like a scan, not a blank page).
        """
        if len(extracted_text.strip()) >= _OCR_TEXT_THRESHOLD:
            return False
        return bool(page.get_images(0))

    def _ocr_page(self, page: fitz.Page) -> str:
        """
        Extract text from a scanned page via Tesseract OCR.

        Uses PyMuPDF's built-in Tesseract integration (``get_textpage_ocr``),
        which calls Tesseract through C bindings — no subprocess overhead per
        page, and no extra Python dependencies beyond PyMuPDF itself.

        dpi=300 is the standard archival-quality resolution; lower values
        degrade accuracy on small fonts.
        """
        textpage = page.get_textpage_ocr(dpi=300, full=True)
        return page.get_text(textpage=textpage)

    def supports_format(self, file_path: Path) -> bool:
        """Check if this loader can handle the file."""
        return file_path.suffix.lower() == ".pdf"

    async def load(self, file_path: Path) -> Document:
        """
        Load a PDF file, falling back to OCR on scanned pages.

        Args:
            file_path: Path to the .pdf file

        Returns:
            Document with extracted text, metadata, and per-page data.
            Each page dict includes ``page_num``, ``text``, and
            ``extraction_method`` (``"text"`` or ``"ocr"``).

        Raises:
            FileNotFoundError: If the file doesn't exist
            PDFLoadError: If the PDF cannot be opened or is encrypted
            ValueError: If no text could be extracted from any page
        """
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        logger.info(f"Loading PDF file: {file_path}")

        try:
            pdf_doc = fitz.open(str(file_path))

            if pdf_doc.is_encrypted:
                pdf_doc.close()
                raise PDFLoadError(f"PDF is encrypted and cannot be read: {file_path}")

            pages = []
            all_text_parts = []
            failed_pages = []

            for page_num in range(len(pdf_doc)):
                try:
                    page = pdf_doc[page_num]
                    page_text = page.get_text()
                    extraction_method = "text"

                    # OCR fallback: if text looks insufficient and page has images,
                    # it's probably a scanned page — try Tesseract.
                    if self._ocr_enabled and self._page_needs_ocr(page, page_text):
                        if self._check_ocr_available():
                            try:
                                page_text = self._ocr_page(page)
                                extraction_method = "ocr"
                            except Exception as e:
                                logger.warning(
                                    f"OCR failed for page {page_num + 1} of "
                                    f"{file_path.name}: {e}"
                                )
                                # page_text remains as-is; likely empty → skipped below

                    if page_text.strip():
                        pages.append({
                            "page_num": page_num + 1,  # 1-indexed for humans
                            "text": page_text,
                            "extraction_method": extraction_method,
                        })
                        all_text_parts.append(page_text)
                    else:
                        logger.warning(f"Page {page_num + 1} has no text: {file_path.name}")

                except Exception as e:
                    failed_pages.append(page_num + 1)
                    logger.warning(
                        f"Failed to extract text from page {page_num + 1} of {file_path.name}: {e}"
                    )

            pdf_doc.close()

            if not all_text_parts:
                error_msg = f"PDF has no extractable text: {file_path}"
                if failed_pages:
                    error_msg += f" (failed pages: {failed_pages})"
                raise ValueError(error_msg)

            full_text = "\n\n".join(all_text_parts)

            metadata = DocumentMetadata(
                source_file=str(file_path),
                source_type="pdf",
                title=file_path.stem,
                author=None,
                page_count=len(pages)
            )

            if failed_pages:
                logger.warning(
                    f"Successfully loaded {len(pages)} pages from {file_path.name}, "
                    f"but {len(failed_pages)} pages failed: {failed_pages}"
                )
            else:
                logger.info(
                    f"Loaded PDF: {file_path.name} "
                    f"({len(pages)} pages, {len(full_text)} characters)"
                )

            return Document(
                text=full_text,
                metadata=metadata,
                pages=pages
            )

        except fitz.FileDataError as e:
            logger.error(f"Invalid or corrupted PDF file: {file_path}")
            raise PDFLoadError(f"Invalid or corrupted PDF: {file_path}") from e

        except PDFLoadError:
            raise

        except ValueError:
            raise

        except Exception as e:
            logger.error(f"Failed to load PDF: {e}")
            raise PDFLoadError(f"Could not load PDF '{file_path}': {e}") from e
