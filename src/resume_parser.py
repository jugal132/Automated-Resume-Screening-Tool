"""Resume parser module for extracting raw text from TXT, PDF, and DOCX files.

This module provides dedicated extractors for common resume file formats
and a centralized dispatcher function with comprehensive validation and error handling.
"""

from pathlib import Path
from typing import Union
import docx
import pdfplumber


class ResumeParserError(Exception):
    """Base exception for resume parsing errors."""


class UnsupportedFormatError(ResumeParserError, ValueError):
    """Raised when an unsupported file format or extension is provided."""


class CorruptFileError(ResumeParserError, RuntimeError):
    """Raised when a file cannot be parsed due to corruption or malformed structure."""


class EmptyFileError(ResumeParserError, ValueError):
    """Raised when extracted text is empty or contains only whitespace."""


# Supported file extensions mapped to their standard formats
SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}


def extract_text_from_txt(file_path: Union[str, Path]) -> str:
    """Extract raw text from a plain text (.txt) file.

    Attempts UTF-8 decoding first with fallback to latin-1 to safely handle
    different text encodings without losing content.

    Args:
        file_path: Path to the target text file.

    Returns:
        The extracted raw text string.

    Raises:
        FileNotFoundError: If the file does not exist.
        CorruptFileError: If the file cannot be read or decoded.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"TXT file not found at: {path}")

    try:
        # Try UTF-8 with BOM handling first
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        try:
            # Fallback to Latin-1 for legacy/Windows-encoded files
            return path.read_text(encoding="latin-1")
        except Exception as exc:
            raise CorruptFileError(f"Failed to decode text file '{path}': {exc}") from exc
    except Exception as exc:
        raise CorruptFileError(f"Unable to read text file '{path}': {exc}") from exc


def extract_text_from_pdf(file_path: Union[str, Path]) -> str:
    """Extract raw text page by page from a PDF (.pdf) file using pdfplumber.

    Pages where text extraction returns None (e.g. blank pages or scanned images)
    are handled gracefully by skipping empty content.

    Args:
        file_path: Path to the target PDF file.

    Returns:
        The extracted text combined across all pages.

    Raises:
        FileNotFoundError: If the file does not exist.
        CorruptFileError: If the PDF is corrupted, encrypted, or unreadable.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found at: {path}")

    extracted_pages = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    extracted_pages.append(page_text.strip())
    except Exception as exc:
        raise CorruptFileError(f"Failed to extract text from PDF '{path}': {exc}") from exc

    return "\n\n".join(extracted_pages)


def extract_text_from_docx(file_path: Union[str, Path]) -> str:
    """Extract raw text from a Microsoft Word (.docx) document using python-docx.

    Extracts text from all non-empty paragraphs and table cells, preserving
    readable document structure.

    Args:
        file_path: Path to the target DOCX file.

    Returns:
        The extracted document text.

    Raises:
        FileNotFoundError: If the file does not exist.
        CorruptFileError: If the document is corrupted or malformed.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"DOCX file not found at: {path}")

    try:
        doc = docx.Document(str(path))
        extracted_elements = []

        # Extract text from paragraphs
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if text:
                extracted_elements.append(text)

        # Extract text from tables if present
        for table in doc.tables:
            for row in table.rows:
                row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_cells:
                    # Remove duplicate adjacent cell texts resulting from merged cells
                    deduped_cells = []
                    for cell in row_cells:
                        if not deduped_cells or deduped_cells[-1] != cell:
                            deduped_cells.append(cell)
                    extracted_elements.append(" | ".join(deduped_cells))

        return "\n\n".join(extracted_elements)
    except Exception as exc:
        raise CorruptFileError(f"Failed to extract text from DOCX '{path}': {exc}") from exc


def extract_resume_text(file_path: Union[str, Path]) -> str:
    """Validate, route, and extract text from a resume file.

    Primary public interface for the resume parser module. Supports .txt,
    .pdf, and .docx formats. Performs path validation, case-insensitive extension
    detection, extraction routing, and empty content detection.

    Args:
        file_path: Path to the resume file to extract.

    Returns:
        Extracted raw text as a non-empty string.

    Raises:
        FileNotFoundError: If the supplied path does not point to an existing file.
        UnsupportedFormatError: If the file extension is not supported.
        EmptyFileError: If extracted text is empty or contains only whitespace.
        CorruptFileError: If the file structure is corrupted or unreadable.
    """
    if file_path is None:
        raise ValueError("File path cannot be None.")

    path = Path(file_path)

    # 1. Validate that the path exists and is a file
    if not path.exists():
        raise FileNotFoundError(f"Resume file not found: '{path}'")
    if not path.is_file():
        raise ValueError(f"Path is not a regular file: '{path}'")

    # 2. Case-insensitive extension detection
    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(
            f"Unsupported file format '{path.suffix}'. "
            f"Supported formats are: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    # 3. Route to the appropriate extractor
    if extension == ".txt":
        raw_text = extract_text_from_txt(path)
    elif extension == ".pdf":
        raw_text = extract_text_from_pdf(path)
    elif extension == ".docx":
        raw_text = extract_text_from_docx(path)
    else:  # Defensive fallback
        raise UnsupportedFormatError(f"Unsupported file extension: '{extension}'")

    # 4. Validate that extracted text is non-empty
    cleaned_text = raw_text.strip()
    if not cleaned_text:
        raise EmptyFileError(
            f"Extracted text from '{path.name}' is empty or contains only whitespace. "
            "The file may be blank, unreadable, or contain scanned images without OCR."
        )

    return cleaned_text
