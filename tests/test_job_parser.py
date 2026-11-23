"""Unit tests for the job description and requirements parser (src/job_parser.py).

Covers TXT loading, JSON requirements loading, schema and type validation,
alias normalization, duplicate removal, overlap detection, experience validation,
and standardized job profile assembly.
"""

import json
from pathlib import Path
import tempfile
import unittest

from src.job_parser import (
    get_standardized_job_profile,
    load_job_description,
    load_job_requirements,
    parse_job_requirements,
)


class TestJobParser(unittest.TestCase):
    """Test suite for the job description and requirements parser module."""

    def setUp(self):
        """Create a temporary directory for test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_path = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    # -------------------------------------------------------------------------
    # 1. Job Description (TXT) Loading
    # -------------------------------------------------------------------------

    def test_load_job_description_valid_default(self):
        """Verify that the default job description loads successfully."""
        desc = load_job_description()
        self.assertIsInstance(desc, str)
        self.assertIn("Junior Python Developer", desc)
        self.assertIn("RESPONSIBILITIES", desc)
        self.assertIn("REQUIRED TECHNICAL SKILLS", desc)

    def test_load_job_description_custom_file(self):
        """Verify loading a valid custom job description file."""
        custom_txt = self.test_path / "custom_job.txt"
        custom_txt.write_text("Custom Job Title\nDescription text.", encoding="utf-8")
        result = load_job_description(custom_txt)
        self.assertEqual(result, "Custom Job Title\nDescription text.")

    def test_load_job_description_missing_file(self):
        """Verify that a nonexistent job description path raises FileNotFoundError."""
        missing = self.test_path / "missing_job.txt"
        with self.assertRaises(FileNotFoundError):
            load_job_description(missing)

    def test_load_job_description_empty_file(self):
        """Verify that an empty or whitespace-only job description raises ValueError."""
        empty_txt = self.test_path / "empty_job.txt"
        empty_txt.write_text("   \n\t  \n", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_job_description(empty_txt)

    def test_load_job_description_invalid_path_type(self):
        """Verify that a non-str non-Path argument raises TypeError."""
        with self.assertRaises(TypeError):
            load_job_description(12345)

    # -------------------------------------------------------------------------
    # 2. Job Requirements (JSON) Loading & Syntax
    # -------------------------------------------------------------------------

    def test_load_job_requirements_valid_default(self):
        """Verify that the default requirements JSON loads and validates."""
        reqs = load_job_requirements()
        self.assertEqual(reqs["job_title"], "Junior Python Developer")
        self.assertEqual(reqs["required_skills"], ["Python", "SQL", "Pandas", "Git"])
        self.assertEqual(
            reqs["preferred_skills"],
            ["NumPy", "FastAPI", "Flask", "REST APIs", "PostgreSQL", "SQLite", "Unit Testing", "Scikit-learn"],
        )
        self.assertEqual(reqs["minimum_experience_years"], 0)
        self.assertEqual(reqs["preferred_experience_years"], 1)

    def test_load_job_requirements_missing_file(self):
        """Verify that a missing requirements file raises FileNotFoundError."""
        missing = self.test_path / "missing_requirements.json"
        with self.assertRaises(FileNotFoundError):
            load_job_requirements(missing)

    def test_load_job_requirements_invalid_json_syntax(self):
        """Verify that malformed JSON syntax raises ValueError."""
        corrupted = self.test_path / "corrupted.json"
        corrupted.write_text("{ \"job_title\": \"Incomplete JSON...", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_job_requirements(corrupted)

    # -------------------------------------------------------------------------
    # 3. Schema & Type Validation
    # -------------------------------------------------------------------------

    def test_parse_job_requirements_invalid_root_type(self):
        """Verify that non-dict raw_data raises TypeError."""
        for invalid in ["string", [1, 2, 3], None, 42]:
            with self.subTest(invalid=invalid):
                with self.assertRaises(TypeError):
                    parse_job_requirements(invalid)

    def test_parse_job_requirements_missing_fields(self):
        """Verify that missing mandatory fields raise ValueError."""
        incomplete = {
            "job_title": "Developer",
            "required_skills": ["Python"],
            # Missing preferred_skills, experience, etc.
        }
        with self.assertRaises(ValueError) as ctx:
            parse_job_requirements(incomplete)
        self.assertIn("Missing mandatory requirements fields", str(ctx.exception))

    def test_parse_job_requirements_invalid_job_title(self):
        """Verify that empty or non-string job_title raises ValueError."""
        payload = {
            "job_title": "   ",
            "required_skills": ["Python"],
            "preferred_skills": ["FastAPI"],
            "minimum_experience_years": 0,
            "preferred_experience_years": 1,
            "education_preferred": ["CS"],
            "experience_evidence": ["Internship"],
        }
        with self.assertRaises(ValueError):
            parse_job_requirements(payload)

    def test_parse_job_requirements_invalid_skills_type(self):
        """Verify that non-list or list with non-string items raises TypeError."""
        payload = {
            "job_title": "Dev",
            "required_skills": "Python, SQL",  # Should be list
            "preferred_skills": ["FastAPI"],
            "minimum_experience_years": 0,
            "preferred_experience_years": 1,
            "education_preferred": ["CS"],
            "experience_evidence": ["Internship"],
        }
        with self.assertRaises(TypeError):
            parse_job_requirements(payload)

        payload["required_skills"] = ["Python", 123]  # Non-string item
        with self.assertRaises(TypeError):
            parse_job_requirements(payload)

    def test_parse_job_requirements_empty_skill_item(self):
        """Verify that whitespace-only skill items raise ValueError."""
        payload = {
            "job_title": "Dev",
            "required_skills": ["Python", "  "],
            "preferred_skills": ["FastAPI"],
            "minimum_experience_years": 0,
            "preferred_experience_years": 1,
            "education_preferred": ["CS"],
            "experience_evidence": ["Internship"],
        }
        with self.assertRaises(ValueError):
            parse_job_requirements(payload)

    # -------------------------------------------------------------------------
    # 4. Skill Normalization & Deduplication
    # -------------------------------------------------------------------------

    def test_skill_alias_normalization(self):
        """Verify that aliases in requirements are normalized to canonical names."""
        payload = {
            "job_title": "ML Engineer",
            "required_skills": ["Python 3", "sklearn", "Postgres"],
            "preferred_skills": ["Fast API", "JS"],
            "minimum_experience_years": 0,
            "preferred_experience_years": 1,
            "education_preferred": ["CS"],
            "experience_evidence": ["Academic project"],
        }
        parsed = parse_job_requirements(payload)
        self.assertEqual(parsed["required_skills"], ["Python", "Scikit-learn", "PostgreSQL"])
        self.assertEqual(parsed["preferred_skills"], ["FastAPI", "JavaScript"])

    def test_skill_deduplication(self):
        """Verify that duplicate skills in the same list are consolidated case-insensitively."""
        payload = {
            "job_title": "Backend Dev",
            "required_skills": ["Python", "python", "Python 3", "SQL", "sql"],
            "preferred_skills": ["Flask", "FLASK"],
            "minimum_experience_years": 0,
            "preferred_experience_years": 1,
            "education_preferred": ["CS"],
            "experience_evidence": ["Internship"],
        }
        parsed = parse_job_requirements(payload)
        self.assertEqual(parsed["required_skills"], ["Python", "SQL"])
        self.assertEqual(parsed["preferred_skills"], ["Flask"])

    def test_required_and_preferred_overlap_detection(self):
        """Verify that overlapping skills between required and preferred raise ValueError."""
        payload = {
            "job_title": "Fullstack Dev",
            "required_skills": ["Python", "SQL"],
            "preferred_skills": ["sql", "FastAPI"],  # 'sql' overlaps with required 'SQL'
            "minimum_experience_years": 0,
            "preferred_experience_years": 1,
            "education_preferred": ["CS"],
            "experience_evidence": ["Internship"],
        }
        with self.assertRaises(ValueError) as ctx:
            parse_job_requirements(payload)
        self.assertIn("Overlapping skills", str(ctx.exception))

        # Test overlap via alias resolution: "Postgres" vs "PostgreSQL"
        payload2 = {
            "job_title": "Backend Dev",
            "required_skills": ["PostgreSQL"],
            "preferred_skills": ["Postgres"],
            "minimum_experience_years": 0,
            "preferred_experience_years": 1,
            "education_preferred": ["CS"],
            "experience_evidence": ["Internship"],
        }
        with self.assertRaises(ValueError) as ctx2:
            parse_job_requirements(payload2)
        self.assertIn("Overlapping skills", str(ctx2.exception))

    # -------------------------------------------------------------------------
    # 5. Experience Validation
    # -------------------------------------------------------------------------

    def test_experience_validation_negative(self):
        """Verify that negative minimum experience raises ValueError."""
        payload = {
            "job_title": "Dev",
            "required_skills": ["Python"],
            "preferred_skills": ["Flask"],
            "minimum_experience_years": -1,
            "preferred_experience_years": 1,
            "education_preferred": ["CS"],
            "experience_evidence": ["Internship"],
        }
        with self.assertRaises(ValueError):
            parse_job_requirements(payload)

    def test_experience_validation_preferred_less_than_minimum(self):
        """Verify that preferred experience less than minimum experience raises ValueError."""
        payload = {
            "job_title": "Dev",
            "required_skills": ["Python"],
            "preferred_skills": ["Flask"],
            "minimum_experience_years": 3,
            "preferred_experience_years": 1,
            "education_preferred": ["CS"],
            "experience_evidence": ["Internship"],
        }
        with self.assertRaises(ValueError):
            parse_job_requirements(payload)

    def test_experience_validation_invalid_type(self):
        """Verify that non-numeric or boolean experience values raise TypeError."""
        payload = {
            "job_title": "Dev",
            "required_skills": ["Python"],
            "preferred_skills": ["Flask"],
            "minimum_experience_years": "zero",  # String instead of number
            "preferred_experience_years": 1,
            "education_preferred": ["CS"],
            "experience_evidence": ["Internship"],
        }
        with self.assertRaises(TypeError):
            parse_job_requirements(payload)

        payload["minimum_experience_years"] = True  # bool instead of int
        with self.assertRaises(TypeError):
            parse_job_requirements(payload)

    # -------------------------------------------------------------------------
    # 6. Standardized Job Profile Assembly
    # -------------------------------------------------------------------------

    def test_get_standardized_job_profile_default(self):
        """Verify end-to-end assembly of the standardized job profile."""
        profile = get_standardized_job_profile()
        self.assertIsInstance(profile, dict)

        expected_keys = [
            "job_title",
            "job_description",
            "required_skills",
            "preferred_skills",
            "minimum_experience",
            "preferred_experience",
            "education",
            "responsibilities",
            "experience_evidence",
        ]
        for key in expected_keys:
            self.assertIn(key, profile, f"Missing key in standardized profile: {key}")

        self.assertEqual(profile["job_title"], "Junior Python Developer")
        self.assertEqual(profile["required_skills"], ["Python", "SQL", "Pandas", "Git"])
        self.assertEqual(
            profile["preferred_skills"],
            ["NumPy", "FastAPI", "Flask", "REST APIs", "PostgreSQL", "SQLite", "Unit Testing", "Scikit-learn"],
        )
        self.assertEqual(profile["minimum_experience"], 0)
        self.assertEqual(profile["preferred_experience"], 1)
        self.assertEqual(
            profile["education"],
            ["Computer Science", "Information Technology", "Software Engineering", "Data Science"],
        )
        self.assertGreaterEqual(len(profile["responsibilities"]), 5)
        self.assertIn("Write modular, clean, and well-documented Python code", profile["responsibilities"][0])


if __name__ == "__main__":
    unittest.main()
