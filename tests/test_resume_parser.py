"""Unit tests for the resume parser module (src/resume_parser.py).

Covers TXT, DOCX, and PDF extraction, error handling, edge cases,
and dispatcher routing using Python's built-in unittest framework.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import docx

from src.resume_parser import (
    CorruptFileError,
    EmptyFileError,
    ResumeParserError,
    UnsupportedFormatError,
    extract_resume_text,
    extract_text_from_docx,
    extract_text_from_pdf,
    extract_text_from_txt,
)


class TestResumeParser(unittest.TestCase):
    """Test suite for resume text extraction functions."""

    def setUp(self):
        """Create a temporary directory for test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up temporary directory and files."""
        self.temp_dir.cleanup()

    # -------------------------------------------------------------------------
    # 1. TXT Extraction Tests
    # -------------------------------------------------------------------------

    def test_extract_text_from_txt_valid_utf8(self):
        """Verify successful extraction of UTF-8 encoded text file."""
        sample_text = "Candidate Profile: Junior Python Developer\nSkills: Python, SQL"
        txt_file = self.test_dir / "valid_sample.txt"
        txt_file.write_text(sample_text, encoding="utf-8")

        result = extract_text_from_txt(txt_file)
        self.assertIn("Junior Python Developer", result)
        self.assertIn("Python, SQL", result)

    def test_extract_text_from_txt_latin1_fallback(self):
        """Verify fallback decoding for Latin-1 encoded text files."""
        sample_text = "Détails du candidat: Ingénieur Python résumé"
        txt_file = self.test_dir / "latin1_sample.txt"
        txt_file.write_bytes(sample_text.encode("latin-1"))

        result = extract_text_from_txt(txt_file)
        self.assertIn("Détails du candidat", result)

    # -------------------------------------------------------------------------
    # 2. DOCX Extraction Tests
    # -------------------------------------------------------------------------

    def test_extract_text_from_docx_valid(self):
        """Verify extraction from a valid DOCX document with paragraphs and tables."""
        docx_file = self.test_dir / "sample_resume.docx"
        doc = docx.Document()
        doc.add_heading("Candidate Resume", level=1)
        doc.add_paragraph("Summary: Experienced in Python, Flask, and SQLite.")

        # Add a table to test table extraction
        table = doc.add_table(rows=2, cols=2)
        table.cell(0, 0).text = "Skill"
        table.cell(0, 1).text = "Proficiency"
        table.cell(1, 0).text = "Python"
        table.cell(1, 1).text = "Advanced"

        doc.save(str(docx_file))

        result = extract_text_from_docx(docx_file)
        self.assertIn("Candidate Resume", result)
        self.assertIn("Experienced in Python, Flask, and SQLite.", result)
        self.assertIn("Python | Advanced", result)

    def test_extract_text_from_docx_corrupted(self):
        """Verify that a corrupted DOCX file raises CorruptFileError."""
        corrupted_docx = self.test_dir / "corrupted.docx"
        corrupted_docx.write_bytes(b"This is not a valid zip/docx archive.")

        with self.assertRaises(CorruptFileError):
            extract_text_from_docx(corrupted_docx)

    # -------------------------------------------------------------------------
    # 3. PDF Extraction Tests (using safe mock of pdfplumber)
    # -------------------------------------------------------------------------

    @patch("src.resume_parser.pdfplumber.open")
    def test_extract_text_from_pdf_valid(self, mock_pdf_open):
        """Verify extraction from a multi-page PDF using a pdfplumber mock."""
        pdf_file = self.test_dir / "sample_resume.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 mock content")

        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Page 1: Senior Python Architect"
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Page 2: Projects and Experience"

        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page1, mock_page2]
        mock_pdf_open.return_value.__enter__.return_value = mock_pdf

        result = extract_text_from_pdf(pdf_file)
        self.assertIn("Page 1: Senior Python Architect", result)
        self.assertIn("Page 2: Projects and Experience", result)

    @patch("src.resume_parser.pdfplumber.open")
    def test_extract_text_from_pdf_with_none_pages(self, mock_pdf_open):
        """Verify that pages returning None (e.g. blank or image-only) are skipped."""
        pdf_file = self.test_dir / "blank_pages.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 mock content")

        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = None  # Scanned or image page
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Valid text on page 2"

        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page1, mock_page2]
        mock_pdf_open.return_value.__enter__.return_value = mock_pdf

        result = extract_text_from_pdf(pdf_file)
        self.assertEqual(result, "Valid text on page 2")

    @patch("src.resume_parser.pdfplumber.open")
    def test_extract_text_from_pdf_corrupted(self, mock_pdf_open):
        """Verify that an unreadable or corrupted PDF raises CorruptFileError."""
        pdf_file = self.test_dir / "damaged.pdf"
        pdf_file.write_bytes(b"broken bytes")
        mock_pdf_open.side_effect = Exception("Invalid PDF syntax stream")

        with self.assertRaises(CorruptFileError):
            extract_text_from_pdf(pdf_file)

    # -------------------------------------------------------------------------
    # 4. Missing File Tests
    # -------------------------------------------------------------------------

    def test_extract_resume_text_missing_file(self):
        """Verify that non-existent files raise FileNotFoundError."""
        non_existent = self.test_dir / "does_not_exist.txt"
        with self.assertRaises(FileNotFoundError):
            extract_resume_text(non_existent)

    # -------------------------------------------------------------------------
    # 5. Unsupported File Extension Tests
    # -------------------------------------------------------------------------

    def test_extract_resume_text_unsupported_extension(self):
        """Verify that unsupported file extensions raise UnsupportedFormatError."""
        for ext in [".png", ".jpg", ".rtf", ".odt", ".csv"]:
            invalid_file = self.test_dir / f"resume{ext}"
            invalid_file.write_text("Some text", encoding="utf-8")
            with self.assertRaises(UnsupportedFormatError):
                extract_resume_text(invalid_file)

    # -------------------------------------------------------------------------
    # 6. Empty File Tests
    # -------------------------------------------------------------------------

    def test_extract_resume_text_empty_txt(self):
        """Verify that a 0-byte or whitespace-only TXT file raises EmptyFileError."""
        empty_txt = self.test_dir / "empty.txt"
        empty_txt.write_text("", encoding="utf-8")
        with self.assertRaises(EmptyFileError):
            extract_resume_text(empty_txt)

        whitespace_txt = self.test_dir / "whitespace.txt"
        whitespace_txt.write_text("   \n\t  \n  ", encoding="utf-8")
        with self.assertRaises(EmptyFileError):
            extract_resume_text(whitespace_txt)

    # -------------------------------------------------------------------------
    # 7. PDF with No Extractable Text
    # -------------------------------------------------------------------------

    @patch("src.resume_parser.pdfplumber.open")
    def test_extract_resume_text_pdf_no_extractable_text(self, mock_pdf_open):
        """Verify that a PDF where all pages yield no text raises EmptyFileError."""
        pdf_file = self.test_dir / "scanned_image_only.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 mock")

        mock_page = MagicMock()
        mock_page.extract_text.return_value = None  # Image without OCR
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf_open.return_value.__enter__.return_value = mock_pdf

        with self.assertRaises(EmptyFileError):
            extract_resume_text(pdf_file)

    # -------------------------------------------------------------------------
    # 8. Case-Insensitive Extension Detection Tests
    # -------------------------------------------------------------------------

    def test_case_insensitive_extension_detection(self):
        """Verify that uppercase and mixed-case extensions are handled properly."""
        uppercase_txt = self.test_dir / "sample.TXT"
        uppercase_txt.write_text("Content from uppercase TXT", encoding="utf-8")
        result_txt = extract_resume_text(uppercase_txt)
        self.assertEqual(result_txt, "Content from uppercase TXT")

        # Test DOCX with mixed casing
        mixed_docx = self.test_dir / "sample.DocX"
        doc = docx.Document()
        doc.add_paragraph("Content from mixed-case DocX")
        doc.save(str(mixed_docx))
        result_docx = extract_resume_text(mixed_docx)
        self.assertEqual(result_docx, "Content from mixed-case DocX")

    # -------------------------------------------------------------------------
    # 9. Dispatcher Routing & Synthetic Dataset Validation
    # -------------------------------------------------------------------------

    def test_extract_resume_text_routes_synthetic_txt(self):
        """Verify that extract_resume_text parses an actual synthetic resume correctly."""
        candidate_file = Path("resumes/sample_resumes/CAND-001.txt")
        self.assertTrue(candidate_file.exists(), "CAND-001.txt fixture must exist")

        extracted = extract_resume_text(candidate_file)
        self.assertIsInstance(extracted, str)
        self.assertIn("CANDIDATE ID: CAND-001", extracted)
        self.assertIn("Python", extracted)
        self.assertIn("FastAPI", extracted)


if __name__ == "__main__":
    unittest.main()
