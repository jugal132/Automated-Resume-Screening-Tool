"""Unit tests for the text cleaning and normalization module (src/text_cleaner.py).

Covers whitespace normalization, bullet standardization, technical token preservation,
empty/invalid input handling, and realistic resume text cleaning using unittest.
"""

import unittest
from src.text_cleaner import clean_text, normalize_bullets, normalize_whitespace


class TestTextCleaner(unittest.TestCase):
    """Test suite for text cleaner functions."""

    # -------------------------------------------------------------------------
    # 1. Lowercase Normalization
    # -------------------------------------------------------------------------

    def test_lowercase_normalization(self):
        """Verify that uppercase and mixed-case text is converted to lowercase."""
        raw = "SENIOR Python DEVELOPER with PostgreSql and Flask"
        cleaned = clean_text(raw)
        self.assertEqual(cleaned, "senior python developer with postgresql and flask")

    # -------------------------------------------------------------------------
    # 2. Whitespace Normalization (Spaces, Tabs, Newlines, Leading/Trailing)
    # -------------------------------------------------------------------------

    def test_multiple_spaces(self):
        """Verify that consecutive horizontal spaces are collapsed to a single space."""
        raw = "Python     FastAPI       Docker"
        self.assertEqual(clean_text(raw), "python fastapi docker")
        self.assertEqual(normalize_whitespace(raw), "Python FastAPI Docker")

    def test_tabs_normalization(self):
        """Verify that tabs are collapsed and converted to single spaces."""
        raw = "Python\t\tPandas\tNumPy"
        self.assertEqual(clean_text(raw), "python pandas numpy")
        self.assertEqual(normalize_whitespace(raw), "Python Pandas NumPy")

    def test_multiple_newlines(self):
        """Verify that excessive newlines are normalized without merging words."""
        raw = "Header Section\n\n\n\n\nBody Paragraph"
        # In normalize_whitespace, 3+ newlines reduce to 2
        normalized = normalize_whitespace(raw)
        self.assertEqual(normalized, "Header Section\n\nBody Paragraph")

        # In clean_text, all whitespace collapses to single spaces for flat tokenization
        cleaned = clean_text(raw)
        self.assertEqual(cleaned, "header section body paragraph")

    def test_leading_and_trailing_whitespace(self):
        """Verify that leading and trailing whitespace is stripped."""
        raw = "   \t  Python Developer  \n\n   "
        self.assertEqual(clean_text(raw), "python developer")
        self.assertEqual(normalize_whitespace(raw), "Python Developer")

    # -------------------------------------------------------------------------
    # 3. Bullet Normalization
    # -------------------------------------------------------------------------

    def test_bullet_normalization(self):
        """Verify standardization of various resume bullet characters."""
        raw = (
            "• Python\n"
            "▪ SQL\n"
            "● Pandas\n"
            "◦ NumPy\n"
            "- Git\n"
            "scikit-learn and full-stack"
        )
        normalized = normalize_bullets(raw)
        expected = (
            "- Python\n"
            "- SQL\n"
            "- Pandas\n"
            "- NumPy\n"
            "- Git\n"
            "scikit-learn and full-stack"
        )
        self.assertEqual(normalized, expected)

    def test_bullet_removal_in_clean_text(self):
        """Verify that bullet glyphs and standalone bullet hyphens are stripped in clean_text."""
        raw = "Skills: • Python ▪ SQL ● Pandas ◦ NumPy - Git"
        cleaned = clean_text(raw)
        self.assertEqual(cleaned, "skills: python sql pandas numpy git")

    # -------------------------------------------------------------------------
    # 4. Unnecessary Punctuation Removal
    # -------------------------------------------------------------------------

    def test_unnecessary_punctuation_removal(self):
        """Verify removal of exclamation points, question marks, commas, ellipses, and brackets."""
        raw = "Expertise: Python, SQL, and FastAPI!!! (Top 5%?) [Verified] \"Gold\"... End;"
        cleaned = clean_text(raw)
        self.assertNotIn("!", cleaned)
        self.assertNotIn("?", cleaned)
        self.assertNotIn(",", cleaned)
        self.assertNotIn(";", cleaned)
        self.assertNotIn("...", cleaned)
        self.assertNotIn("(", cleaned)
        self.assertNotIn(")", cleaned)
        self.assertNotIn("[", cleaned)
        self.assertNotIn("]", cleaned)
        self.assertNotIn('"', cleaned)
        self.assertEqual(cleaned, "expertise: python sql and fastapi top 5% verified gold end")

    # -------------------------------------------------------------------------
    # 5. Technical Terminology Preservation
    # -------------------------------------------------------------------------

    def test_preservation_of_cpp(self):
        """Verify that 'C++' is preserved as 'c++' and not stripped to 'c'."""
        raw = "Proficient in C++ and Modern C++17 design."
        cleaned = clean_text(raw)
        self.assertIn("c++", cleaned)
        self.assertEqual(cleaned, "proficient in c++ and modern c++17 design")

    def test_preservation_of_csharp(self):
        """Verify that 'C#' is preserved as 'c#' and not stripped to 'c'."""
        raw = "Experienced in C# backend development with .NET."
        cleaned = clean_text(raw)
        self.assertIn("c#", cleaned)
        self.assertIn(".net", cleaned)

    def test_preservation_of_dotnet(self):
        """Verify that '.NET' and 'ASP.NET' are preserved properly."""
        raw = "Built enterprise microservices using .NET Core and ASP.NET."
        cleaned = clean_text(raw)
        self.assertIn(".net", cleaned)
        self.assertIn("asp.net", cleaned)

    def test_preservation_of_nodejs(self):
        """Verify that 'Node.js' and related *.js runtimes are preserved."""
        raw = "Full-stack developer with Node.js, Vue.js, and React.js."
        cleaned = clean_text(raw)
        self.assertIn("node.js", cleaned)
        self.assertIn("vue.js", cleaned)
        self.assertIn("react.js", cleaned)

    def test_preservation_of_cicd(self):
        """Verify that 'CI/CD' is preserved as 'ci/cd'."""
        raw = "Automated deployment pipelines with CI/CD and Docker."
        cleaned = clean_text(raw)
        self.assertIn("ci/cd", cleaned)
        self.assertEqual(cleaned, "automated deployment pipelines with ci/cd and docker")

    def test_preservation_of_html_css(self):
        """Verify that 'HTML/CSS' is preserved as 'html/css'."""
        raw = "Front-end layouts using HTML/CSS and JavaScript."
        cleaned = clean_text(raw)
        self.assertIn("html/css", cleaned)
        self.assertIn("javascript", cleaned)

    def test_preservation_of_hyphenated_compound_terms(self):
        """Verify that technical compound words like scikit-learn are preserved."""
        raw = "Trained ML models using scikit-learn and object-oriented architectures."
        cleaned = clean_text(raw)
        self.assertIn("scikit-learn", cleaned)
        self.assertIn("object-oriented", cleaned)

    def test_preservation_of_decimals(self):
        """Verify that decimal numbers such as GPA 3.8 and Python 3.11 are preserved."""
        raw = "Python 3.11 developer with GPA: 3.85 / 4.0."
        cleaned = clean_text(raw)
        self.assertIn("3.11", cleaned)
        self.assertIn("3.85", cleaned)
        self.assertIn("4.0", cleaned)

    # -------------------------------------------------------------------------
    # 6. Empty and Whitespace-Only Strings
    # -------------------------------------------------------------------------

    def test_empty_string(self):
        """Verify that empty strings return empty strings across all functions."""
        self.assertEqual(clean_text(""), "")
        self.assertEqual(normalize_whitespace(""), "")
        self.assertEqual(normalize_bullets(""), "")

    def test_whitespace_only_string(self):
        """Verify that whitespace-only strings return empty strings."""
        self.assertEqual(clean_text("   \n\t  \r\n  "), "")
        self.assertEqual(normalize_whitespace("   \n\t  \r\n  "), "")
        self.assertEqual(normalize_bullets("   \n\t  \r\n  "), "")

    # -------------------------------------------------------------------------
    # 7. Invalid Input Types
    # -------------------------------------------------------------------------

    def test_invalid_input_types(self):
        """Verify that non-string inputs raise TypeError."""
        for invalid in [None, 123, 45.67, ["python"], {"text": "python"}, False]:
            with self.subTest(invalid=invalid):
                with self.assertRaises(TypeError):
                    clean_text(invalid)
                with self.assertRaises(TypeError):
                    normalize_whitespace(invalid)
                with self.assertRaises(TypeError):
                    normalize_bullets(invalid)

    # -------------------------------------------------------------------------
    # 8. Prompt's Exact Specification Example
    # -------------------------------------------------------------------------

    def test_prompt_example(self):
        """Verify exact behavior required by the prompt's reference example."""
        raw = "  PYTHON   Developer!!! • Experience: 2 YEARS \n\n Skills: Pandas, NumPy, SQL... "
        expected = "python developer experience: 2 years skills: pandas numpy sql"
        self.assertEqual(clean_text(raw), expected)

    # -------------------------------------------------------------------------
    # 9. Realistic Synthetic Resume Text
    # -------------------------------------------------------------------------

    def test_realistic_synthetic_resume_snippet(self):
        """Verify cleaning on a realistic multi-line resume excerpt."""
        resume_snippet = """
        ================================================================================
        CANDIDATE ID: CAND-001
        ================================================================================

        PROFESSIONAL SUMMARY
        Results-driven Junior Python Developer with 1+ year experience in FastAPI, Flask!
        • Wrote complex SQL queries in PostgreSQL.
        • Cleaned datasets using Pandas & NumPy.
        • Proficient in Git/GitHub, CI/CD, and C++.
        """
        cleaned = clean_text(resume_snippet)
        self.assertIn("candidate id: cand-001", cleaned)
        self.assertIn("fastapi", cleaned)
        self.assertIn("flask", cleaned)
        self.assertIn("postgresql", cleaned)
        self.assertIn("pandas", cleaned)
        self.assertIn("numpy", cleaned)
        self.assertIn("ci/cd", cleaned)
        self.assertIn("c++", cleaned)
        self.assertNotIn("•", cleaned)
        self.assertNotIn("!", cleaned)
        self.assertNotIn("&", cleaned)


if __name__ == "__main__":
    unittest.main()
