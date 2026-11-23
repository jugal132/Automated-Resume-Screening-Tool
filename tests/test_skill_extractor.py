"""Unit tests for technical skill extraction and alias normalization (src/skill_extractor.py).

Covers skill database loading, boundary-aware matching, alias normalization,
false-positive prevention (Java/JavaScript, SQL/NoSQL, C, Go), punctuation preservation,
category grouping, and integration with the Phase 6 text cleaner.
"""

import json
from pathlib import Path
import tempfile
import unittest

from src.resume_parser import extract_resume_text
from src.skill_extractor import (
    extract_skills,
    get_skill_categories,
    load_skill_database,
    normalize_skill_name,
)
from src.text_cleaner import clean_text


class TestSkillExtractor(unittest.TestCase):
    """Test suite for the skill extractor module."""

    @classmethod
    def setUpClass(cls):
        """Load default database once for tests."""
        cls.db = load_skill_database()

    def setUp(self):
        """Create temporary directory for isolated database tests."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_path = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    # -------------------------------------------------------------------------
    # 1. Database Loading & Validation
    # -------------------------------------------------------------------------

    def test_load_skill_database_valid(self):
        """Verify successful loading of the default skill database."""
        db = load_skill_database()
        self.assertIsInstance(db, dict)
        self.assertIn("Programming Languages", db)
        self.assertIn("Python", db["Programming Languages"])
        self.assertIn("SQL", db["Data"])

    def test_load_skill_database_missing_file(self):
        """Verify that a nonexistent database path raises FileNotFoundError."""
        missing = self.test_path / "nonexistent_skills.json"
        with self.assertRaises(FileNotFoundError):
            load_skill_database(missing)

    def test_load_skill_database_invalid_json(self):
        """Verify that malformed JSON raises ValueError."""
        invalid_json = self.test_path / "corrupted.json"
        invalid_json.write_text("{ broken json content ...", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_skill_database(invalid_json)

    def test_load_skill_database_invalid_schema(self):
        """Verify that invalid schema (e.g. empty or non-dict) raises ValueError."""
        empty_json = self.test_path / "empty.json"
        empty_json.write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_skill_database(empty_json)

    # -------------------------------------------------------------------------
    # 2. Skill Normalization Tests
    # -------------------------------------------------------------------------

    def test_normalize_skill_name_canonical_and_aliases(self):
        """Verify normalization of aliases to their canonical skill names."""
        test_cases = [
            ("sklearn", "Scikit-learn"),
            ("scikit learn", "Scikit-learn"),
            ("Postgres", "PostgreSQL"),
            ("JS", "JavaScript"),
            ("TS", "TypeScript"),
            ("RESTful APIs", "REST APIs"),
            ("Fast API", "FastAPI"),
            ("ML", "Machine Learning"),
            ("Excel", "Microsoft Excel"),
            ("CPP", "C++"),
            ("C Sharp", "C#"),
            ("Git Hub", "GitHub"),
            ("Python 3", "Python"),
            ("Python", "Python"),
            ("python", "Python"),
        ]
        for alias, expected in test_cases:
            with self.subTest(alias=alias):
                self.assertEqual(normalize_skill_name(alias, db=self.db), expected)

    def test_normalize_skill_name_unknown(self):
        """Verify that unknown skills return the stripped input string."""
        self.assertEqual(normalize_skill_name("QuantumComputingXYZ", db=self.db), "QuantumComputingXYZ")
        self.assertEqual(normalize_skill_name("   ", db=self.db), "")

    def test_normalize_skill_name_invalid_type(self):
        """Verify that non-string input raises TypeError."""
        with self.assertRaises(TypeError):
            normalize_skill_name(12345, db=self.db)

    # -------------------------------------------------------------------------
    # 3. Canonical and Alias Detection
    # -------------------------------------------------------------------------

    def test_canonical_skill_detection(self):
        """Verify detection of canonical skill names."""
        text = "Skilled in Python, SQL, Pandas, NumPy, and Git."
        skills = extract_skills(text, db=self.db)
        expected = ["Git", "NumPy", "Pandas", "Python", "SQL"]
        self.assertEqual(sorted(skills), expected)

    def test_alias_detection(self):
        """Verify that aliases are extracted and mapped to canonical names."""
        text = "Built ML models with sklearn and visualized Postgres data using PowerBI."
        skills = extract_skills(text, db=self.db)
        self.assertIn("Machine Learning", skills)
        self.assertIn("Scikit-learn", skills)
        self.assertIn("PostgreSQL", skills)
        self.assertIn("Power BI", skills)

    def test_case_insensitive_detection(self):
        """Verify that skill detection is case-insensitive."""
        text = "Experienced in PYTHON, fastAPI, and FLASK."
        skills = extract_skills(text, db=self.db)
        self.assertIn("Python", skills)
        self.assertIn("FastAPI", skills)
        self.assertIn("Flask", skills)

    def test_duplicate_removal(self):
        """Verify that repeated mentions result in a single canonical entry."""
        text = "Python developer writing Python 3 code with Python3 scripts and Python library."
        skills = extract_skills(text, db=self.db)
        self.assertEqual(skills.count("Python"), 1)

    # -------------------------------------------------------------------------
    # 4. Boundary-Aware & False-Positive Prevention Tests
    # -------------------------------------------------------------------------

    def test_java_vs_javascript(self):
        """Verify that 'JavaScript' does not falsely trigger 'Java'."""
        text = "Frontend engineer specializing in JavaScript and React."
        skills = extract_skills(text, db=self.db)
        self.assertIn("JavaScript", skills)
        self.assertNotIn("Java", skills)

    def test_sql_vs_nosql(self):
        """Verify that 'NoSQL' does not falsely trigger 'SQL'."""
        text = "Extensive experience managing MongoDB in a NoSQL database cluster."
        skills = extract_skills(text, db=self.db)
        self.assertIn("NoSQL", skills)
        self.assertIn("MongoDB", skills)
        self.assertNotIn("SQL", skills)

    def test_c_false_positive_prevention(self):
        """Verify that words containing 'c' do not falsely trigger language C."""
        text = "Senior developer with excellent communication skills and customer care experience."
        skills = extract_skills(text, db=self.db)
        self.assertNotIn("C", skills)

    def test_c_isolated_detection(self):
        """Verify that language 'C' is detected when explicitly mentioned in technical context."""
        text = "Proficient in C, Python, and C++."
        skills = extract_skills(text, db=self.db)
        self.assertIn("C", skills)
        self.assertIn("C++", skills)
        self.assertIn("Python", skills)

    def test_go_false_positive_prevention(self):
        """Verify that 'good' or 'ongoing' does not falsely trigger 'Go'."""
        text = "Good programming practices and ongoing code maintenance."
        skills = extract_skills(text, db=self.db)
        self.assertNotIn("Go", skills)

    def test_go_detection(self):
        """Verify that 'Go' or 'Golang' is detected when explicitly mentioned."""
        text = "Backend services written in Golang and Docker."
        skills = extract_skills(text, db=self.db)
        self.assertIn("Go", skills)
        self.assertIn("Docker", skills)

    # -------------------------------------------------------------------------
    # 5. Technical Skills with Punctuation
    # -------------------------------------------------------------------------

    def test_technical_skills_with_punctuation(self):
        """Verify detection of skills containing punctuation (C++, C#, HTML/CSS, etc.)."""
        text = "Expertise in C++, C#, HTML/CSS, and REST APIs."
        skills = extract_skills(text, db=self.db)
        self.assertIn("C++", skills)
        self.assertIn("C#", skills)
        self.assertIn("HTML", skills)
        self.assertIn("CSS", skills)
        self.assertIn("REST APIs", skills)

    # -------------------------------------------------------------------------
    # 6. Input Validation & Edge Cases
    # -------------------------------------------------------------------------

    def test_empty_and_whitespace_text(self):
        """Verify that empty or whitespace-only strings return empty lists."""
        self.assertEqual(extract_skills("", db=self.db), [])
        self.assertEqual(extract_skills("   \n\t  ", db=self.db), [])
        self.assertEqual(get_skill_categories("", db=self.db), {})

    def test_invalid_input_type(self):
        """Verify that non-string text raises TypeError."""
        for invalid in [None, 12345, ["Python"], {"text": "Python"}]:
            with self.subTest(invalid=invalid):
                with self.assertRaises(TypeError):
                    extract_skills(invalid, db=self.db)
                with self.assertRaises(TypeError):
                    get_skill_categories(invalid, db=self.db)

    # -------------------------------------------------------------------------
    # 7. Category Grouping Tests
    # -------------------------------------------------------------------------

    def test_get_skill_categories(self):
        """Verify that get_skill_categories groups detected skills accurately."""
        text = "Python developer using FastAPI, PostgreSQL, and Pandas."
        categories = get_skill_categories(text, db=self.db)
        self.assertIsInstance(categories, dict)
        self.assertIn("Programming Languages", categories)
        self.assertIn("Python", categories["Programming Languages"])
        self.assertIn("Backend", categories)
        self.assertIn("FastAPI", categories["Backend"])
        self.assertIn("Databases", categories)
        self.assertIn("PostgreSQL", categories["Databases"])
        self.assertIn("Data", categories)
        self.assertIn("Pandas", categories["Data"])
        # Unmatched categories should not be present
        self.assertNotIn("Frontend", categories)

    # -------------------------------------------------------------------------
    # 8. Integration with Phase 6 Clean Text Output
    # -------------------------------------------------------------------------

    def test_extraction_against_phase6_cleaned_text(self):
        """Verify that skills extract reliably from Phase 6 clean_text output."""
        raw = "  PYTHON   Developer!!! • Experience: 2 YEARS \n\n Skills: Pandas, NumPy, SQL... "
        cleaned = clean_text(raw)
        skills = extract_skills(cleaned, db=self.db)
        self.assertEqual(skills, ["NumPy", "Pandas", "Python", "SQL"])

    def test_extraction_from_realistic_synthetic_resume(self):
        """Verify extraction from actual synthetic resume CAND-001.txt."""
        cand_path = Path("resumes/sample_resumes/CAND-001.txt")
        self.assertTrue(cand_path.exists())
        raw_text = extract_resume_text(cand_path)
        skills = extract_skills(raw_text, db=self.db)

        # Expected core competencies from CAND-001
        expected_subset = ["Python", "SQL", "PostgreSQL", "FastAPI", "Flask", "Pandas", "NumPy", "Git", "GitHub", "Docker"]
        for expected in expected_subset:
            self.assertIn(expected, skills, f"CAND-001 must contain {expected}")


if __name__ == "__main__":
    unittest.main()
