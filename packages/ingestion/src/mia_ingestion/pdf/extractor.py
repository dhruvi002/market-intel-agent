"""PDF and HTML text extraction for SEC filings.

Extraction strategy:
  1. If the file is .htm / .html → BeautifulSoup (SEC filings are often HTML)
  2. If the file is .pdf:
       a. Try Docling (IBM open-source) — best table extraction, returns Markdown
       b. Fallback to PyMuPDF — faster, simpler, plain text only

Section detection uses heuristic regexes keyed on EDGAR Item numbers:
  mda           → Item 7 — Management's Discussion and Analysis
  market_risk   → Item 7A — Quantitative and Qualitative Disclosures About Market Risk
  risk_factors  → Item 1A — Risk Factors
  business      → Item 1 — Business
  financials    → Item 8 — Financial Statements
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# (regex pattern, section key) — order matters; earlier patterns win on overlap
_SECTION_PATTERNS: list[tuple[str, str]] = [
    (r"(?i)item\s+1a[\.\s].*?risk\s+factors", "risk_factors"),
    (r"(?i)item\s+1[\.\s].*?business", "business"),
    (r"(?i)item\s+7a[\.\s].*?quantitative", "market_risk"),
    (r"(?i)item\s+7[\.\s].*?management.{0,40}discussion", "mda"),
    (r"(?i)item\s+8[\.\s].*?financial\s+statements", "financials"),
]

# Cap how many chars to keep per section (prevents huge rows in the DB)
_SECTION_MAX_CHARS = 80_000


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class ExtractedDocument:
    """Structured output of the extraction pipeline."""

    text: str                                   # Full plain text
    sections: dict[str, str] = field(default_factory=dict)  # section key → text
    tables: list[dict] = field(default_factory=list)        # list of {rows: [...]}
    page_count: int = 0
    char_count: int = 0
    metadata: dict = field(default_factory=dict)
    extraction_backend: str = "unknown"


# ── Main extractor ────────────────────────────────────────────────────────────

class PDFExtractor:
    """Extract text, tables, and sections from SEC filing documents.

    Handles both HTML filings (most 10-Ks filed after 1996) and PDFs.
    All I/O-heavy work is wrapped in asyncio.to_thread so the event loop
    is never blocked.
    """

    async def extract(self, file_path: Path) -> ExtractedDocument:
        """Dispatch to the appropriate backend based on file extension."""
        suffix = file_path.suffix.lower()
        if suffix in (".htm", ".html"):
            return await self._extract_html(file_path)
        return await self._extract_pdf(file_path)

    # ── PDF ───────────────────────────────────────────────────────────────────

    async def _extract_pdf(self, file_path: Path) -> ExtractedDocument:
        """Try Docling; fall back to PyMuPDF on any error."""
        try:
            return await asyncio.to_thread(self._docling_sync, file_path)
        except Exception as exc:
            logger.warning(
                "Docling failed for %s (%s); falling back to PyMuPDF",
                file_path.name,
                type(exc).__name__,
            )
            return await asyncio.to_thread(self._pymupdf_sync, file_path)

    def _docling_sync(self, file_path: Path) -> ExtractedDocument:
        from docling.document_converter import DocumentConverter  # heavy import

        converter = DocumentConverter()
        result = converter.convert(str(file_path))
        doc = result.document
        text: str = doc.export_to_markdown()
        tables: list[dict] = []
        for tbl in doc.tables:
            try:
                rows = tbl.export_to_dataframe().to_dict(orient="records")
            except Exception:
                rows = []
            tables.append({"rows": rows})

        sections = _extract_sections(text)
        return ExtractedDocument(
            text=text,
            sections=sections,
            tables=tables,
            page_count=getattr(doc, "num_pages", 0),
            char_count=len(text),
            metadata={"source": str(file_path)},
            extraction_backend="docling",
        )

    def _pymupdf_sync(self, file_path: Path) -> ExtractedDocument:
        import fitz  # PyMuPDF

        doc = fitz.open(str(file_path))
        pages: list[str] = [page.get_text() for page in doc]
        text = "\n".join(pages)
        sections = _extract_sections(text)
        return ExtractedDocument(
            text=text,
            sections=sections,
            tables=[],
            page_count=len(doc),
            char_count=len(text),
            metadata={"source": str(file_path)},
            extraction_backend="pymupdf",
        )

    # ── HTML ──────────────────────────────────────────────────────────────────

    async def _extract_html(self, file_path: Path) -> ExtractedDocument:
        return await asyncio.to_thread(self._html_sync, file_path)

    def _html_sync(self, file_path: Path) -> ExtractedDocument:
        from bs4 import BeautifulSoup

        html = file_path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html, "lxml")

        # Strip noise
        for tag in soup(["script", "style", "head", "meta", "link"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)

        # Basic table extraction — captures financial statement tables
        tables: list[dict] = []
        for tbl in soup.find_all("table"):
            rows: list[list[str]] = []
            for row in tbl.find_all("tr"):
                cells = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
                if any(cells):
                    rows.append(cells)
            if rows:
                tables.append({"rows": rows})

        sections = _extract_sections(text)
        return ExtractedDocument(
            text=text,
            sections=sections,
            tables=tables,
            page_count=0,
            char_count=len(text),
            metadata={"source": str(file_path)},
            extraction_backend="beautifulsoup",
        )


# ── Section detection ─────────────────────────────────────────────────────────

def _extract_sections(text: str) -> dict[str, str]:
    """Heuristically split filing text into named sections using EDGAR Item headers."""
    # Find all pattern match positions
    positions: list[tuple[int, str]] = []
    for pattern, name in _SECTION_PATTERNS:
        for m in re.finditer(pattern, text):
            positions.append((m.start(), name))

    if not positions:
        return {}

    positions.sort()
    sections: dict[str, str] = {}
    for idx, (start, name) in enumerate(positions):
        end = positions[idx + 1][0] if idx + 1 < len(positions) else start + _SECTION_MAX_CHARS
        section_text = text[start:end].strip()
        # Only store the first occurrence of each section name
        if name not in sections:
            sections[name] = section_text[:_SECTION_MAX_CHARS]

    return sections
