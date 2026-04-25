"""
src/agents/scripting/rag/pdf_parser.py

Multimodal PDF parser using pypdf (Pure Python).
This avoids Windows "Application Control policy" DLL blocks.

Extracts:
  - TextElement  : raw text from pages
  - ImageElement : extracted images using pypdf's image extraction
  - TableElement : basic table detection from text patterns
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

# ──────────────────────────── Data classes ────────────────────────────────────


@dataclass
class TextElement:
    """A contiguous text block extracted from a PDF page."""
    page_number: int
    section_hint: str
    text: str
    original_pdf: str


@dataclass
class ImageElement:
    """A raster image extracted from a PDF page."""
    page_number: int
    original_pdf: str
    caption: str              # approximated from surrounding text
    image_bytes: bytes
    image_index: int


@dataclass
class TableElement:
    """A detected table extracted from a PDF page as Markdown."""
    page_number: int
    original_pdf: str
    markdown: str
    table_index: int


@dataclass
class ParsedPDF:
    """Container for all elements extracted from a single PDF file."""
    source_path: Path
    texts: list[TextElement] = field(default_factory=list)
    images: list[ImageElement] = field(default_factory=list)
    tables: list[TableElement] = field(default_factory=list)

    @property
    def stats(self) -> dict[str, int]:
        return {
            "texts": len(self.texts),
            "images": len(self.images),
            "tables": len(self.tables),
        }


# ──────────────────────────── Parser ─────────────────────────────────────────


class PDFParser:
    """
    Parse a PDF using pypdf (Pure Python).
    
    Since pypdf doesn't give precise bounding boxes for text easily,
    we use a simpler heuristic for captions and tables.
    """

    def __init__(
        self,
        *,
        min_image_width: int = 50,
        min_image_height: int = 50,
        min_text_len: int = 20,
    ) -> None:
        self.min_image_width = min_image_width
        self.min_image_height = min_image_height
        self.min_text_len = min_text_len

    def parse(self, pdf_path: Path) -> ParsedPDF:
        """Parse a PDF file and return a ParsedPDF container."""
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ImportError(
                "pypdf is required: pip install pypdf"
            ) from exc

        result = ParsedPDF(source_path=pdf_path)
        pdf_name = pdf_path.name

        try:
            reader = PdfReader(str(pdf_path))
        except Exception as exc:
            logger.error("PDFParser: cannot open %s — %s", pdf_name, exc)
            return result

        for page_number, page in enumerate(reader.pages, start=1):
            self._process_page(page, page_number, pdf_name, result)

        logger.info(
            "PDFParser: %s → %d texts, %d images, %d tables",
            pdf_name, len(result.texts), len(result.images), len(result.tables),
        )
        return result

    def _process_page(
        self,
        page,
        page_number: int,
        pdf_name: str,
        result: ParsedPDF,
    ) -> None:
        """Extract elements from a page using pypdf methods."""
        
        # 1. Extract Text with multiple attempts
        raw_text = ""
        try:
            # Try basic extraction
            raw_text = page.extract_text() or ""
            # If empty, try with layout processing (if available in this version)
            if not raw_text.strip():
                raw_text = page.extract_text(extraction_mode="layout") or ""
        except:
            raw_text = page.extract_text() or ""

        lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
        
        # Simple table detection: lines with multiple '|' or lots of spaces
        current_text_blocks: list[str] = []
        table_idx = 0
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Heuristic for table: multiple columns separated by 3+ spaces or '|'
            if self._is_table_line(line):
                table_lines = []
                while i < len(lines) and (self._is_table_line(lines[i]) or not lines[i]):
                    if lines[i]:
                        table_lines.append(lines[i])
                    i += 1
                
                if len(table_lines) >= 2:
                    md = self._to_markdown_table(table_lines)
                    result.tables.append(
                        TableElement(
                            page_number=page_number,
                            original_pdf=pdf_name,
                            markdown=md,
                            table_index=table_idx,
                        )
                    )
                    table_idx += 1
                continue
            
            current_text_blocks.append(line)
            i += 1

        # Save concatenated text
        final_text = "\n".join(current_text_blocks)
        if len(final_text) >= self.min_text_len:
            result.texts.append(
                TextElement(
                    page_number=page_number,
                    section_hint="",  # hard to detect reliably with pypdf
                    text=final_text,
                    original_pdf=pdf_name,
                )
            )

        # 2. Extract Images
        try:
            image_idx = 0
            for img_file in page.images:
                img_bytes = img_file.data
                
                # Filter small images
                try:
                    from PIL import Image as PILImage
                    with PILImage.open(io.BytesIO(img_bytes)) as pil_img:
                        w, h = pil_img.size
                        if w < self.min_image_width or h < self.min_image_height:
                            continue
                except:
                    pass

                result.images.append(
                    ImageElement(
                        page_number=page_number,
                        original_pdf=pdf_name,
                        caption="",  # will be hard to find without coordinates
                        image_bytes=img_bytes,
                        image_index=image_idx,
                    )
                )
                image_idx += 1
        except Exception as exc:
            logger.debug("PDFParser: image extraction failed on p%s — %s", page_number, exc)

    @staticmethod
    def _is_table_line(line: str) -> bool:
        if line.count('|') >= 2:
            return True
        # Detect multiple columns by looking for wide gaps
        if re.search(r'\s{3,}', line):
            return True
        return False

    @staticmethod
    def _to_markdown_table(lines: list[str]) -> str:
        rows = []
        for line in lines:
            # Split by | or 3+ spaces
            cells = re.split(r'\||\s{3,}', line)
            cells = [c.strip() for c in cells if c.strip()]
            if cells:
                rows.append(cells)
        
        if not rows: return ""
        max_cols = max(len(r) for r in rows)
        
        md = []
        # Header
        md.append("| " + " | ".join(rows[0] + [""] * (max_cols - len(rows[0]))) + " |")
        md.append("| " + " | ".join(["---"] * max_cols) + " |")
        # Body
        for r in rows[1:]:
            md.append("| " + " | ".join(r + [""] * (max_cols - len(r))) + " |")
            
        return "\n".join(md)
